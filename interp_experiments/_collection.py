"""
Forward-pass data collection — KV-cache batching + paired direct/meta passes.

Three entrypoints:
  - `collect_paired_data`     — direct + meta forward passes for every question
  - `collect_meta_only`       — meta only, when direct data is being reused
  - `collect_other_confidence` — control task ("how confident would a peer be")

All three return a dict that the runner shapes into the final paired_data.json.

Per-iteration state — which meta task is active, the active dataset name —
lives on `_io.ctx`; static knobs (CONFIDENCE_SCALE, DELEGATE_PROMPT_DESIGN,
TEAMMATE_ACCURACY, BASE_DELEGATE_MODE, AVAILABLE_METRICS) come from
`IntrospectionExperimentConfig`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from experiment_config import IntrospectionExperimentConfig as _C
from core.model_utils import DEVICE
from prompts import (
    format_delegate_prompt,
    format_direct_prompt,
    format_direct_prompt_base,
    format_meta_prompt,
    format_meta_prompt_base,
    format_other_confidence_prompt,
    get_meta_options,
    get_other_confidence_signal,
    option_token_id_groups,
)

from _extractor import BatchedExtractor
from _io import ctx as _ctx
from core.metrics import compute_metrics_single as compute_uncertainty_metrics


# Default batch size for processing (adjust based on GPU memory).
BATCH_SIZE = 4


def find_common_prefix_length(input_ids_list: List[List[int]]) -> int:
    """Find the length of the common token prefix across all sequences."""
    if not input_ids_list:
        return 0
    ref_ids = input_ids_list[0]
    min_len = min(len(ids) for ids in input_ids_list)
    common_len = 0
    for i in range(min_len):
        if all(ids[i] == ref_ids[i] for ids in input_ids_list):
            common_len += 1
        else:
            break
    return common_len


def process_prompts_with_prefix_cache(
    prompts: List[str],
    options_list: List[List[str]],
    tokenizer,
    extractor: BatchedExtractor,
    batch_size: int,
    desc: str,
    collect_activations: bool = True,
    add_special_tokens: bool = False,
) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray], List[Dict], List[str]]:
    """Run a forward pass over `prompts` efficiently with prefix caching.

    When prompts share a common prefix (same system message + setup),
    computes the KV cache for the prefix once and reuses it for every
    suffix. Per-option logits are aggregated via logsumexp over both the
    leading-space and no-space sub-tokens (matches `run_evaluations.run_mcq_forward_pass`).

    Returns:
        (layer_activations_per_item, probs, logits, metrics, responses)
    """
    encodings = tokenizer(prompts, add_special_tokens=add_special_tokens)
    input_ids_list = encodings["input_ids"]

    common_len = find_common_prefix_length(input_ids_list)
    MIN_PREFIX_FOR_CACHE = 20
    if common_len > MIN_PREFIX_FOR_CACHE:
        print(f"  Found common prefix ({common_len} tokens). Computing prefix cache...")
        prefix_ids = torch.tensor([input_ids_list[0][:common_len]], device=DEVICE)
        prefix_cache = extractor.compute_prefix_cache(prefix_ids)
        suffixes = [ids[common_len:] for ids in input_ids_list]
        use_cache = True
    else:
        if common_len > 0:
            print(f"  Common prefix too short ({common_len} tokens). Using standard processing.")
        suffixes = input_ids_list
        prefix_cache = None
        use_cache = False

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    results_acts: List[Dict] = []
    results_probs: List[np.ndarray] = []
    results_logits: List[np.ndarray] = []
    results_metrics: List[Dict] = []
    results_responses: List[str] = []

    for b in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_suffixes = suffixes[b:b + batch_size]
        batch_opts = options_list[b:b + batch_size]
        actual_batch_size = len(batch_suffixes)

        max_len = max(len(s) for s in batch_suffixes)
        padded = torch.full((actual_batch_size, max_len), pad_token_id, dtype=torch.long, device=DEVICE)
        for i, s in enumerate(batch_suffixes):
            padded[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long, device=DEVICE)

        # Per-option token-id groups (matches eval pipeline aggregation).
        # Assume all items in batch share the same options.
        opt_ids = option_token_id_groups(tokenizer, batch_opts[0])

        if use_cache:
            acts, probs, logits, metrics = extractor.extract_batch_with_cache(
                padded, prefix_cache, opt_ids, pad_token_id=pad_token_id
            )
        else:
            mask = (padded != pad_token_id).long()
            acts, probs, logits, metrics = extractor.extract_batch(padded, mask, opt_ids)

        if collect_activations:
            results_acts.extend(acts)
        results_probs.extend(probs)
        results_logits.extend(logits)
        results_metrics.extend(metrics)

        for i, p in enumerate(probs):
            results_responses.append(batch_opts[i][np.argmax(p)])

    return results_acts, results_probs, results_logits, results_metrics, results_responses


def _load_mc_data_for_reuse(paired_data_path: Path, acts_path: Path) -> Tuple[Dict, Dict]:
    """Load confidence-mode direct data for reuse in a delegate-only run.

    Returns (mc_data, paired). mc_data matches the shape `collect_meta_only`
    expects (direct_activations dict, metadata list of per-item
    {probabilities, logits}, optional direct_metrics dict). `paired` is the
    raw paired_data.json so the caller can sanity-check question alignment.
    """
    with open(paired_data_path) as f:
        paired = json.load(f)

    with np.load(acts_path, allow_pickle=True) as f:
        direct_activations = {
            int(k.split("_")[1]): f[k].astype(np.float32)
            for k in f.files if k.startswith("layer_")
        }
        npz_run_id = str(f["run_id"]) if "run_id" in f.files else None
        npz_qids = [str(x) for x in f["question_ids"]] if "question_ids" in f.files else None

    pd_run_id = (paired.get("config") or {}).get("run_id")
    if npz_run_id is not None and pd_run_id is not None and npz_run_id != pd_run_id:
        raise ValueError(
            "Reuse aborted: direct_activations.npz and paired_data.json carry "
            f"different run_ids (npz={npz_run_id!r}, json={pd_run_id!r}). "
            "These files are from different runs and cannot be paired by row."
        )
    if npz_qids is not None:
        json_qids = [str((q or {}).get("id", f"q_{i}")) for i, q in enumerate(paired.get("questions", []))]
        if json_qids and json_qids != npz_qids:
            raise ValueError(
                "Reuse aborted: question_ids in direct_activations.npz do not "
                "match paired_data.json — row alignment is broken."
            )

    direct_probs = paired.get("direct_probs", []) or []
    direct_logits = paired.get("direct_logits", []) or []
    n_items = len(direct_probs) if direct_probs else len(paired.get("questions", []))
    metadata = []
    for i in range(n_items):
        metadata.append({
            "probabilities": direct_probs[i] if i < len(direct_probs) else [],
            "logits": direct_logits[i] if i < len(direct_logits) else [],
        })

    direct_metrics_raw = paired.get("direct_metrics", {}) or {}
    direct_metrics = {
        k: np.asarray(v) for k, v in direct_metrics_raw.items()
        if not isinstance(v, dict) and v is not None
    }

    mc_data = {
        "direct_activations": direct_activations,
        "metadata": metadata,
        "direct_metrics": direct_metrics,
        "_source_run_id": npz_run_id,
    }
    return mc_data, paired


# Subset of metric keys that are scalars (collectible into 1D arrays).
_SCALAR_METRIC_KEYS = list(_C.AVAILABLE_METRICS) + ["entropy_full_vocab", "top2_margin_logprob"]
_EXTENDED_METRIC_KEYS = ("option_logprobs", "entropy_full_vocab", "top20_logits", "top2_margin_logprob")


def collect_paired_data(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool = True,
    batch_size: int = BATCH_SIZE,
    is_base: bool = False,
    few_shot_mode: str = "fixed",
    few_shot_pool: Optional[List[Dict]] = None,
) -> Dict:
    """Collect activations + per-option probs/metrics for direct + meta prompts.

    Reads `_ctx.meta_task` and `_C.DELEGATE_PROMPT_DESIGN / BASE_DELEGATE_MODE`
    to decide which formatter to use.
    """
    print(f"Collecting paired data for {len(questions)} questions (batch_size={batch_size})...")

    extractor = BatchedExtractor(model, num_layers, tokenizer)
    extractor.register_hooks()

    direct_layer_acts = {i: [] for i in range(num_layers)}
    direct_metrics_lists = defaultdict(list)
    direct_probs_list = []
    direct_logits_list = []
    errors = []

    model.eval()
    meta_options = get_meta_options(
        meta_task=_ctx.meta_task,
        scale=_C.CONFIDENCE_SCALE,
        prompt_design=_C.DELEGATE_PROMPT_DESIGN,
    )

    try:
        # ============ PHASE 1: DIRECT PROMPTS ============
        print("\nProcessing direct prompts...")
        num_batches = (len(questions) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Direct prompts"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]

            direct_prompts = []
            direct_options_list = []
            for q in batch_questions:
                if is_base:
                    prompt, options = format_direct_prompt_base(q, mode=few_shot_mode)
                else:
                    prompt, options = format_direct_prompt(q, tokenizer, use_chat_template)
                direct_prompts.append(prompt)
                direct_options_list.append(options)

            first_options = direct_options_list[0]
            all_same_options = all(opts == first_options for opts in direct_options_list)

            try:
                if all_same_options:
                    direct_option_token_ids = option_token_id_groups(tokenizer, first_options)

                    inputs = tokenizer(
                        direct_prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(DEVICE)

                    batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        direct_option_token_ids
                    )

                    for acts, probs, logits, metrics in zip(batch_acts, batch_probs, batch_logits, batch_metrics):
                        for layer_idx, act in acts.items():
                            direct_layer_acts[layer_idx].append(act)
                        direct_probs_list.append(probs.tolist())
                        direct_logits_list.append(logits.tolist())
                        for metric_name, metric_val in metrics.items():
                            direct_metrics_lists[metric_name].append(metric_val)

                    del inputs
                else:
                    for prompt, options in zip(direct_prompts, direct_options_list):
                        option_token_ids = option_token_id_groups(tokenizer, options)
                        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                        batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                            inputs["input_ids"],
                            inputs["attention_mask"],
                            option_token_ids
                        )

                        for layer_idx, act in batch_acts[0].items():
                            direct_layer_acts[layer_idx].append(act)
                        direct_probs_list.append(batch_probs[0].tolist())
                        direct_logits_list.append(batch_logits[0].tolist())
                        for metric_name, metric_val in batch_metrics[0].items():
                            direct_metrics_lists[metric_name].append(metric_val)

                        del inputs
            except Exception as e:
                for idx in range(start_idx, end_idx):
                    errors.append({"phase": "direct", "question_idx": idx, "error": str(e)})
                print(f"\n  Warning: batch {batch_idx} failed ({e}), skipping {end_idx - start_idx} examples")
                torch.cuda.empty_cache()
                continue

            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # ============ PHASE 2: META PROMPTS ============
        print("\nProcessing meta prompts...")
        meta_extended = defaultdict(list)

        if _ctx.meta_task == "delegate":
            meta_layer_acts = {i: [] for i in range(num_layers)}
            meta_probs_list = []
            meta_entropies = []
            meta_responses = []
            meta_mappings = []
            meta_option_token_ids = option_token_id_groups(tokenizer, meta_options)

            for trial_idx, q in enumerate(tqdm(questions, desc="Delegate prompts")):
                prompt, _, mapping = format_delegate_prompt(
                    q, tokenizer, use_chat_template, trial_idx,
                    is_base=is_base,
                    few_shot_mode=_C.BASE_DELEGATE_MODE if is_base else "fixed",
                    few_shot_pool=few_shot_pool,
                    teammate_accuracy=_C.TEAMMATE_ACCURACY,
                    prompt_design=_C.DELEGATE_PROMPT_DESIGN,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    meta_option_token_ids
                )

                for layer_idx, act in batch_acts[0].items():
                    meta_layer_acts[layer_idx].append(act)
                meta_probs_list.append(batch_probs[0].tolist())
                meta_entropies.append(batch_metrics[0]["entropy"])
                for k in _EXTENDED_METRIC_KEYS:
                    if k in batch_metrics[0]:
                        meta_extended[k].append(batch_metrics[0][k])
                meta_responses.append(meta_options[np.argmax(batch_probs[0])])
                meta_mappings.append(mapping)

                del inputs
                if (trial_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        else:
            # Confidence task: use prefix caching (all prompts share same prefix)
            meta_prompts = []
            meta_options_list = []
            for q in questions:
                if is_base:
                    prompt, opts = format_meta_prompt_base(q, mode=few_shot_mode, scale=_C.CONFIDENCE_SCALE)
                else:
                    prompt, opts = format_meta_prompt(q, tokenizer, use_chat_template, scale=_C.CONFIDENCE_SCALE)
                meta_prompts.append(prompt)
                meta_options_list.append(opts)

            meta_acts, meta_probs_raw, _, meta_metrics_raw, meta_responses = process_prompts_with_prefix_cache(
                meta_prompts,
                meta_options_list,
                tokenizer,
                extractor,
                batch_size,
                desc="Meta prompts (with prefix cache)",
                add_special_tokens=is_base,
            )

            meta_layer_acts = {i: [] for i in range(num_layers)}
            for acts in meta_acts:
                for layer_idx, act in acts.items():
                    meta_layer_acts[layer_idx].append(act)

            meta_probs_list = [p.tolist() for p in meta_probs_raw]
            meta_entropies = [m["entropy"] for m in meta_metrics_raw]
            for m in meta_metrics_raw:
                for k in _EXTENDED_METRIC_KEYS:
                    if k in m:
                        meta_extended[k].append(m[k])
            meta_mappings = [None] * len(questions)

    finally:
        extractor.remove_hooks()

    direct_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in direct_layer_acts.items()
    }
    meta_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in meta_layer_acts.items()
    }
    direct_metrics = {}
    direct_extended = {}
    for metric, values in direct_metrics_lists.items():
        if metric in _SCALAR_METRIC_KEYS:
            direct_metrics[metric] = np.array(values)
        else:
            direct_extended[metric] = values

    print(f"\nDirect activations shape (per layer): {direct_activations[0].shape}")
    print(f"Meta activations shape (per layer): {meta_activations[0].shape}")
    print("\nDirect uncertainty metrics:")
    for metric_name, values in direct_metrics.items():
        if isinstance(values, np.ndarray) and values.dtype.kind == 'f':
            print(f"  {metric_name}: range=[{values.min():.3f}, {values.max():.3f}], "
                  f"mean={values.mean():.3f}, std={values.std():.3f}")

    return {
        "direct_activations": direct_activations,
        "meta_activations": meta_activations,
        "direct_metrics": direct_metrics,
        "direct_extended": direct_extended,
        "direct_probs": direct_probs_list,
        "direct_logits": direct_logits_list,
        "meta_entropies": np.array(meta_entropies),
        "meta_extended": dict(meta_extended),
        "meta_probs": meta_probs_list,
        "meta_responses": meta_responses,
        "meta_mappings": meta_mappings,
        "questions": questions,
        "errors": errors,
    }


def collect_meta_only(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool,
    mc_data: Dict,
    batch_size: int = BATCH_SIZE,
    is_base: bool = False,
    few_shot_mode: str = "fixed",
    few_shot_pool: Optional[List[Dict]] = None,
) -> Dict:
    """Run meta prompts only, reusing direct activations from a prior run.

    Much faster than `collect_paired_data` when MC data already exists.
    """
    print(f"Collecting meta data only for {len(questions)} questions (reusing direct activations)...")

    extractor = BatchedExtractor(model, num_layers, tokenizer)
    extractor.register_hooks()
    model.eval()

    meta_options = get_meta_options(
        meta_task=_ctx.meta_task,
        scale=_C.CONFIDENCE_SCALE,
        prompt_design=_C.DELEGATE_PROMPT_DESIGN,
    )
    meta_extended = defaultdict(list)

    try:
        if _ctx.meta_task == "delegate":
            meta_layer_acts = {i: [] for i in range(num_layers)}
            meta_probs_list = []
            meta_entropies = []
            meta_responses = []
            meta_mappings = []
            meta_option_token_ids = option_token_id_groups(tokenizer, meta_options)

            for trial_idx, q in enumerate(tqdm(questions, desc="Delegate prompts")):
                prompt, _, mapping = format_delegate_prompt(
                    q, tokenizer, use_chat_template, trial_idx,
                    is_base=is_base,
                    few_shot_mode=_C.BASE_DELEGATE_MODE if is_base else "fixed",
                    few_shot_pool=few_shot_pool,
                    teammate_accuracy=_C.TEAMMATE_ACCURACY,
                    prompt_design=_C.DELEGATE_PROMPT_DESIGN,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    meta_option_token_ids
                )

                for layer_idx, act in batch_acts[0].items():
                    meta_layer_acts[layer_idx].append(act)
                meta_probs_list.append(batch_probs[0].tolist())
                meta_entropies.append(batch_metrics[0]["entropy"])
                for k in _EXTENDED_METRIC_KEYS:
                    if k in batch_metrics[0]:
                        meta_extended[k].append(batch_metrics[0][k])
                meta_responses.append(meta_options[np.argmax(batch_probs[0])])
                meta_mappings.append(mapping)

                del inputs
                if (trial_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        else:
            meta_prompts = []
            meta_options_list = []
            for q in questions:
                if is_base:
                    prompt, opts = format_meta_prompt_base(q, mode=few_shot_mode, scale=_C.CONFIDENCE_SCALE)
                else:
                    prompt, opts = format_meta_prompt(q, tokenizer, use_chat_template, scale=_C.CONFIDENCE_SCALE)
                meta_prompts.append(prompt)
                meta_options_list.append(opts)

            meta_acts, meta_probs_raw, _, meta_metrics_raw, meta_responses = process_prompts_with_prefix_cache(
                meta_prompts,
                meta_options_list,
                tokenizer,
                extractor,
                batch_size,
                desc="Meta prompts (with prefix cache)",
                add_special_tokens=is_base,
            )

            meta_layer_acts = {i: [] for i in range(num_layers)}
            for acts in meta_acts:
                for layer_idx, act in acts.items():
                    meta_layer_acts[layer_idx].append(act)

            meta_probs_list = [p.tolist() for p in meta_probs_raw]
            meta_entropies = [m["entropy"] for m in meta_metrics_raw]
            for m in meta_metrics_raw:
                for k in _EXTENDED_METRIC_KEYS:
                    if k in m:
                        meta_extended[k].append(m[k])
            meta_mappings = [None] * len(questions)

    finally:
        extractor.remove_hooks()

    meta_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in meta_layer_acts.items()
    }
    print(f"Meta activations shape (per layer): {meta_activations[0].shape}")

    direct_probs_list = [m.get("probabilities", []) for m in mc_data["metadata"]]
    direct_logits_list = [m.get("logits", []) for m in mc_data["metadata"]]

    if "direct_metrics" in mc_data:
        direct_metrics = mc_data["direct_metrics"]
    else:
        direct_metrics = {metric: [] for metric in _C.AVAILABLE_METRICS}
        for m in mc_data["metadata"]:
            probs = np.array(m.get("probabilities", []))
            logits = np.array(m.get("logits", [])) if m.get("logits") else None
            if len(probs) > 0:
                item_metrics = compute_uncertainty_metrics(probs, logits)
                for metric_name, metric_val in item_metrics.items():
                    direct_metrics[metric_name].append(metric_val)
        direct_metrics = {k: np.array(v) for k, v in direct_metrics.items()}

    return {
        "direct_activations": mc_data["direct_activations"],
        "meta_activations": meta_activations,
        "direct_metrics": direct_metrics,
        "direct_extended": {},
        "direct_probs": direct_probs_list,
        "direct_logits": direct_logits_list,
        "meta_entropies": np.array(meta_entropies),
        "meta_extended": dict(meta_extended),
        "meta_probs": meta_probs_list,
        "meta_responses": meta_responses,
        "meta_mappings": meta_mappings,
        "questions": questions,
    }


def collect_other_confidence(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool,
    batch_size: int = BATCH_SIZE,
    collect_activations: bool = False,
) -> Dict:
    """Run the other-confidence (control) task.

    Asks the model to estimate what % of college-educated people would know
    the answer (instead of asking about its own confidence). If the model is
    truly introspecting on its own uncertainty, the SELF-confidence task
    should correlate more strongly with its actual uncertainty metrics than
    this OTHER-confidence task does.

    Uses KV-cache prefix sharing for efficiency.
    """
    print(f"\nCollecting other-confidence (control) data for {len(questions)} questions...")

    extractor = BatchedExtractor(model, num_layers, tokenizer)
    extractor.register_hooks()
    model.eval()

    try:
        other_prompts = []
        other_options_list = []
        for q in questions:
            prompt, opts = format_other_confidence_prompt(q, tokenizer, use_chat_template)
            other_prompts.append(prompt)
            other_options_list.append(opts)

        desc = "Other-confidence (with activations)" if collect_activations else "Other-confidence (with prefix cache)"
        other_activations_raw, other_probs_raw, _, _, other_responses = process_prompts_with_prefix_cache(
            other_prompts,
            other_options_list,
            tokenizer,
            extractor,
            batch_size,
            desc=desc,
            collect_activations=collect_activations,
        )

        other_probs_list = [p.tolist() for p in other_probs_raw]
        other_signals = [get_other_confidence_signal(p) for p in other_probs_raw]

    finally:
        extractor.remove_hooks()

    print(f"Other-confidence signals: mean={np.mean(other_signals):.3f}, std={np.std(other_signals):.3f}")

    result = {
        "other_probs": other_probs_list,
        "other_responses": other_responses,
        "other_signals": np.array(other_signals),
    }

    if collect_activations and other_activations_raw is not None and len(other_activations_raw) > 0:
        n_layers = len(other_activations_raw[0])
        other_layer_acts = {i: [] for i in range(n_layers)}
        for acts in other_activations_raw:
            for layer_idx, act in acts.items():
                other_layer_acts[layer_idx].append(act)
        result["other_activations"] = {
            layer_idx: np.array(acts) for layer_idx, acts in other_layer_acts.items()
        }
        print(f"Collected other-confidence activations: {len(result['other_activations'])} layers, "
              f"shape={result['other_activations'][0].shape}")

    return result
