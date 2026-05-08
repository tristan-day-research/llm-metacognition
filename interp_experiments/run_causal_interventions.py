"""
Causal interventions on contrast directions.

Loads unit-norm contrast directions written by `run_collect_activations.py`
(entropy + stated_confidence) and, for each (direction, intervention type,
layer, multiplier) condition, re-runs the direct MC + meta confidence
forward passes with a residual-stream hook installed at the target layer.

For every condition we record:
  * mean / std direct-task entropy (output entropy on the MC question)
  * mean / std stated confidence (numeric soft signal from meta probs)
  * Spearman ρ between stated confidence and direct entropy across questions
  * mean / std meta entropy (uncertainty over the confidence options)
  * direct accuracy (vs. correct_answer)

The runner reuses helpers from `_collection`, `_runio`, `prompts`, and
`core.model_utils` so the intervention pass goes through the exact same
prompts and option-token aggregation as collection.

Outputs land under
    outputs/causal_interventions/<same subfolder convention as collection>/
along with a params .txt and an _examples.txt that prints one direct +
meta prompt (with chat tags) and the corresponding model response per
condition type.

Usage:
    python run_causal_interventions.py
"""

# --- repo path bootstrap so root-level imports resolve from anywhere ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _preflight import check_hf_login; check_hf_login()


import argparse
import datetime
import json
import random
import subprocess
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from core.model_utils import DEVICE, is_base_model, has_chat_template, load_model_and_tokenizer
from prompts import (
    NUMERIC_CONFIDENCE_OPTIONS,
    format_direct_prompt,
    format_direct_prompt_base,
    format_meta_prompt,
    format_meta_prompt_base,
    get_meta_signal,
    meta_task_type,
    response_to_confidence,
)
from experiment_config import (
    IntrospectionExperimentConfig as _Intro,
    CausalInterventionConfig as _C,
)

import _runio
from _runio import (
    _atomic_savez_compressed,
    _atomic_write_json,
    _dataset_short_name,
    _finetuned_short_tag,
    get_model_short_name,
    load_questions,
    set_run_context,
)
from _collection import BATCH_SIZE as _COLL_BATCH, collect_paired_data


# Seed everything once at import.
np.random.seed(_C.SEED)
torch.manual_seed(_C.SEED)
random.seed(_C.SEED)

_C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Path helpers — mirror the collection-runner naming so subfolders match.
# ============================================================================

def _run_subfolder_name(dataset_name: str) -> str:
    """8b_<model_tag>_<dataset_short>, identical to the collection runner."""
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        model_tag = _finetuned_short_tag(_C.MODEL_NAME)
    elif "Instruct" in _C.BASE_MODEL_NAME:
        model_tag = "instruct"
    else:
        model_tag = "base"
    return f"8b_{model_tag}_{_dataset_short_name(dataset_name)}"


def _intervention_run_dir(dataset_name: str) -> _Path:
    d = _C.OUTPUTS_DIR / _run_subfolder_name(dataset_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_prefix(dataset_name: str) -> str:
    """File-name prefix; mirrors get_directions_prefix conventions."""
    model_short = get_model_short_name(_C.BASE_MODEL_NAME)
    dataset_short = _dataset_short_name(dataset_name)
    scale_suffix = f"_scale-{_C.CONFIDENCE_SCALE}" if _C.CONFIDENCE_SCALE != "letters" else ""
    run_dir = _intervention_run_dir(dataset_name)
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        adapter_short = get_model_short_name(_C.MODEL_NAME)
        return str(run_dir / f"{model_short}_adapter-{adapter_short}_{dataset_short}_interventions{scale_suffix}")
    return str(run_dir / f"{model_short}_{dataset_short}_interventions{scale_suffix}")


def _directions_prefix(dataset_name: str) -> str:
    """Path prefix where contrast direction NPZs were written by the
    collection runner. Reproduces `_runio.get_directions_prefix(metric=None)`
    but rooted at `_C.DIRECTIONS_DIR` so we can read across the two trees
    without monkey-patching `_runio._C`."""
    model_short = get_model_short_name(_C.BASE_MODEL_NAME)
    dataset_short = _dataset_short_name(dataset_name)
    scale_suffix = f"_scale-{_C.CONFIDENCE_SCALE}" if _C.CONFIDENCE_SCALE != "letters" else ""
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        model_tag = _finetuned_short_tag(_C.MODEL_NAME)
    elif "Instruct" in _C.BASE_MODEL_NAME:
        model_tag = "instruct"
    else:
        model_tag = "base"
    subfolder = f"8b_{model_tag}_{dataset_short}"
    run_dir = _C.DIRECTIONS_DIR / subfolder
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        adapter_short = get_model_short_name(_C.MODEL_NAME)
        return str(run_dir / f"{model_short}_adapter-{adapter_short}_{dataset_short}_introspection{scale_suffix}")
    return str(run_dir / f"{model_short}_{dataset_short}_introspection{scale_suffix}")


# ============================================================================
# Direction loading
# ============================================================================

def load_contrast_directions(dataset_name: str, direction_type: str) -> Dict[int, np.ndarray]:
    """Return {layer_idx: (hidden_dim,) unit-norm vector} for one contrast type."""
    prefix = _directions_prefix(dataset_name)
    path = _Path(f"{prefix}_{direction_type}_contrast_directions.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing contrast direction file for '{direction_type}':\n  {path}\n"
            "Run run_collect_activations.py for this (model, dataset, scale) first."
        )
    with np.load(path, allow_pickle=True) as f:
        out = {}
        for k in f.files:
            if k.startswith("layer_"):
                out[int(k.split("_")[1])] = f[k].astype(np.float32)
    return out


# ============================================================================
# Intervention hooks
# ============================================================================

def _resolve_layer_module(model, layer_idx: int):
    """Return the transformer block whose forward output is the residual stream."""
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        return base.model.layers[layer_idx]
    return model.model.layers[layer_idx]


def make_steering_hook(direction: np.ndarray, multiplier: float):
    """Add `multiplier * unit_direction` to every position of the residual."""
    d_unit = direction / (np.linalg.norm(direction) + 1e-12)
    d_cpu = torch.tensor(d_unit, dtype=torch.float32)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            d = d_cpu.to(h.device, dtype=h.dtype)
            h_new = h + multiplier * d
            return (h_new,) + output[1:]
        d = d_cpu.to(output.device, dtype=output.dtype)
        return output + multiplier * d

    return hook


def make_ablation_hook(direction: np.ndarray):
    """Project the residual orthogonal to `direction` (zero-ablate the component)."""
    d_unit = direction / (np.linalg.norm(direction) + 1e-12)
    d_cpu = torch.tensor(d_unit, dtype=torch.float32)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            d = d_cpu.to(h.device, dtype=h.dtype)
            coef = (h * d).sum(dim=-1, keepdim=True)
            return (h - coef * d,) + output[1:]
        d = d_cpu.to(output.device, dtype=output.dtype)
        coef = (output * d).sum(dim=-1, keepdim=True)
        return output - coef * d

    return hook


# ============================================================================
# Per-condition evaluation
# ============================================================================

def _summarise(data: Dict, *, scale: str) -> Dict:
    """Compute per-condition behavioural summary from collect_paired_data output."""
    direct_entropy = np.asarray(data["direct_metrics"]["entropy"], dtype=np.float64)
    meta_probs = data["meta_probs"]
    meta_responses = data["meta_responses"]
    meta_entropy = np.asarray(data["meta_entropies"], dtype=np.float64)

    task_type = meta_task_type(meta_task="confidence", scale=scale)
    stated = np.array([
        response_to_confidence(r, np.asarray(p), None, task_type)
        for r, p in zip(meta_responses, meta_probs)
    ], dtype=np.float64)

    soft_stated = np.array([
        get_meta_signal(np.asarray(p), scale=scale) if p else np.nan
        for p in meta_probs
    ], dtype=np.float64)

    valid = ~np.isnan(stated) & ~np.isnan(direct_entropy)
    if valid.sum() >= 2:
        rho_stated_vs_entropy = float(spearmanr(stated[valid], direct_entropy[valid]).correlation)
    else:
        rho_stated_vs_entropy = float("nan")

    valid_soft = ~np.isnan(soft_stated) & ~np.isnan(direct_entropy)
    if valid_soft.sum() >= 2:
        rho_soft_vs_entropy = float(spearmanr(soft_stated[valid_soft], direct_entropy[valid_soft]).correlation)
    else:
        rho_soft_vs_entropy = float("nan")

    questions = data["questions"]
    direct_probs = data["direct_probs"]
    options_keys = [list(q["options"].keys()) for q in questions]
    direct_responses = [
        opts[int(np.argmax(p))] if p else None
        for opts, p in zip(options_keys, direct_probs)
    ]
    is_correct = [
        bool(resp == q.get("correct_answer")) if resp else False
        for resp, q in zip(direct_responses, questions)
    ]
    accuracy = float(np.mean(is_correct)) if is_correct else float("nan")

    return {
        "n_questions": int(len(meta_responses)),
        "direct_entropy_mean": float(np.nanmean(direct_entropy)),
        "direct_entropy_std": float(np.nanstd(direct_entropy)),
        "stated_confidence_mean": float(np.nanmean(stated)),
        "stated_confidence_std": float(np.nanstd(stated)),
        "soft_stated_confidence_mean": float(np.nanmean(soft_stated)),
        "soft_stated_confidence_std": float(np.nanstd(soft_stated)),
        "meta_entropy_mean": float(np.nanmean(meta_entropy)),
        "meta_entropy_std": float(np.nanstd(meta_entropy)),
        "spearman_stated_vs_direct_entropy": rho_stated_vs_entropy,
        "spearman_soft_stated_vs_direct_entropy": rho_soft_vs_entropy,
        "direct_accuracy": accuracy,
        "stated_confidence_per_question": stated.tolist(),
        "direct_entropy_per_question": direct_entropy.tolist(),
        "meta_responses": meta_responses,
        "direct_responses": direct_responses,
    }


def run_one_condition(
    *, label: str,
    hook_specs: List[Tuple[int, callable]],
    questions, model, tokenizer, num_layers, use_chat_template, is_base,
) -> Dict:
    """Install hooks → run paired collection → strip hooks → summarise.

    `hook_specs` is a list of (layer_idx, hook_fn). An empty list runs the
    baseline (no intervention).
    """
    print(f"\n>>> Condition: {label}")
    handles = []
    try:
        for layer_idx, hook_fn in hook_specs:
            mod = _resolve_layer_module(model, layer_idx)
            handles.append(mod.register_forward_hook(hook_fn))

        data = collect_paired_data(
            questions, model, tokenizer, num_layers, use_chat_template,
            batch_size=_C.BATCH_SIZE, is_base=is_base,
            few_shot_mode=_C.FEW_SHOT_MODE, few_shot_pool=None,
        )
    finally:
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

    summary = _summarise(data, scale=_C.CONFIDENCE_SCALE)
    summary["label"] = label
    return summary, data


# ============================================================================
# Example-prompt rendering
# ============================================================================

def _render_example_prompts(question, tokenizer, *, is_base, use_chat_template):
    if is_base:
        direct_prompt, _ = format_direct_prompt_base(question, mode=_C.FEW_SHOT_MODE)
        meta_prompt, _ = format_meta_prompt_base(question, mode=_C.FEW_SHOT_MODE, scale=_C.CONFIDENCE_SCALE)
    else:
        direct_prompt, _ = format_direct_prompt(question, tokenizer, use_chat_template)
        meta_prompt, _ = format_meta_prompt(question, tokenizer, use_chat_template, scale=_C.CONFIDENCE_SCALE)
    return direct_prompt, meta_prompt


def _write_examples_txt(path: str, *, conditions, questions, tokenizer, is_base, use_chat_template):
    """One direct + meta prompt with chat tags + responses, per condition."""
    q0 = questions[0]
    direct_prompt, meta_prompt = _render_example_prompts(
        q0, tokenizer, is_base=is_base, use_chat_template=use_chat_template,
    )
    lines = []
    lines.append("=" * 80)
    lines.append("Causal-intervention example prompts and responses")
    lines.append("=" * 80)
    lines.append(f"Question id: {q0.get('id')}")
    lines.append(f"Question:    {q0.get('question')}")
    lines.append(f"Options:     {q0.get('options')}")
    lines.append(f"Correct:     {q0.get('correct_answer')}")
    lines.append("")
    lines.append("--- DIRECT PROMPT (verbatim, with chat tags) ---")
    lines.append(direct_prompt)
    lines.append("--- END DIRECT PROMPT ---")
    lines.append("")
    lines.append("--- META PROMPT (verbatim, with chat tags) ---")
    lines.append(meta_prompt)
    lines.append("--- END META PROMPT ---")
    lines.append("")
    for cond in conditions:
        lines.append("-" * 60)
        lines.append(f"Condition: {cond['label']}")
        d_resp = cond["direct_responses"][0] if cond["direct_responses"] else None
        m_resp = cond["meta_responses"][0] if cond["meta_responses"] else None
        d_ent = cond["direct_entropy_per_question"][0] if cond["direct_entropy_per_question"] else None
        s_conf = cond["stated_confidence_per_question"][0] if cond["stated_confidence_per_question"] else None
        lines.append(f"  direct_response (argmax over A/B/C/D): {d_resp}")
        lines.append(f"  direct_entropy: {d_ent}")
        lines.append(f"  meta_response  (argmax confidence opt):  {m_resp}")
        lines.append(f"  stated_confidence: {s_conf}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_params_txt(path: str, *, dataset_name, num_questions, num_layers, run_id, git_commit):
    lines = [
        "Causal interventions — run parameters",
        "=" * 60,
        f"run_id:                {run_id}",
        f"date_run:              {datetime.datetime.now().isoformat()}",
        f"git_commit:            {git_commit}",
        f"device:                {DEVICE}",
        "",
        "[Model]",
        f"BASE_MODEL_NAME:       {_C.BASE_MODEL_NAME}",
        f"MODEL_NAME:            {_C.MODEL_NAME}",
        f"LOAD_IN_4BIT:          {_C.LOAD_IN_4BIT}",
        f"LOAD_IN_8BIT:          {_C.LOAD_IN_8BIT}",
        f"num_layers (detected): {num_layers}",
        "",
        "[Dataset]",
        f"DATASET_NAME:          {dataset_name}",
        f"NUM_QUESTIONS:         {num_questions}",
        f"SEED:                  {_C.SEED}",
        f"FEW_SHOT_MODE:         {_C.FEW_SHOT_MODE}",
        "",
        "[Meta-task]",
        f"CONFIDENCE_SCALE:      {_C.CONFIDENCE_SCALE}",
        "",
        "[Intervention sweep]",
        f"DIRECTION_TYPES:       {_C.DIRECTION_TYPES}",
        f"INTERVENTION_TYPES:    {_C.INTERVENTION_TYPES}",
        f"INTERVENTION_LAYERS:   {_C.INTERVENTION_LAYERS}",
        f"STEERING_MULTIPLIERS:  {_C.STEERING_MULTIPLIERS}",
        f"INCLUDE_BASELINE:      {_C.INCLUDE_BASELINE}",
        f"BATCH_SIZE:            {_C.BATCH_SIZE}",
        "",
        "[Paths]",
        f"DIRECTIONS_DIR:        {_C.DIRECTIONS_DIR}",
        f"OUTPUTS_DIR:           {_C.OUTPUTS_DIR}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Per-dataset orchestrator
# ============================================================================

def run_for_dataset(dataset_name: str, model, tokenizer, num_layers: int):
    num_questions = _C.NUM_QUESTIONS_BY_DATASET.get(
        dataset_name,
        _C.NUM_QUESTIONS_BY_DATASET.get(_dataset_short_name(dataset_name), _C.NUM_QUESTIONS_DEFAULT),
    )
    # _runio's path helpers / ctx are used inside _collection (only for
    # meta_task routing). We aren't sweeping meta tasks here — we always
    # run the confidence prompt — so pin ctx accordingly.
    set_run_context(dataset_name=dataset_name, meta_task="confidence", num_questions=num_questions)

    run_id = uuid.uuid4().hex
    print("\n" + "=" * 80)
    print(f"Causal interventions — dataset: {dataset_name}")
    print(f"run_id: {run_id}")
    print("=" * 80)

    questions = load_questions(dataset_name, num_questions)
    random.seed(_C.SEED)
    random.shuffle(questions)
    print(f"Loaded {len(questions)} questions")

    is_base = is_base_model(_C.BASE_MODEL_NAME)
    use_chat_template = has_chat_template(tokenizer) and not is_base
    print(f"is_base={is_base}  use_chat_template={use_chat_template}")

    # Load all directions up front so a missing file fails fast.
    directions: Dict[str, Dict[int, np.ndarray]] = {}
    for dt in _C.DIRECTION_TYPES:
        directions[dt] = load_contrast_directions(dataset_name, dt)
        print(f"Loaded {dt} directions: {len(directions[dt])} layers, "
              f"hidden_dim={next(iter(directions[dt].values())).shape[0]}")

    # Build the condition list.
    conditions_spec: List[Dict] = []
    if _C.INCLUDE_BASELINE:
        conditions_spec.append({"label": "baseline", "kind": "baseline"})
    for dt in _C.DIRECTION_TYPES:
        for itype in _C.INTERVENTION_TYPES:
            for layer in _C.INTERVENTION_LAYERS:
                if layer not in directions[dt]:
                    print(f"  (skip) {dt}/{itype} layer {layer} — missing in directions npz")
                    continue
                if itype == "ablate":
                    conditions_spec.append({
                        "label": f"{dt}__ablate__L{layer}",
                        "kind": "ablate", "direction_type": dt, "layer": layer,
                    })
                elif itype == "steer":
                    for mult in _C.STEERING_MULTIPLIERS:
                        conditions_spec.append({
                            "label": f"{dt}__steer__L{layer}__a{mult:+g}",
                            "kind": "steer", "direction_type": dt,
                            "layer": layer, "multiplier": float(mult),
                        })

    print(f"\nTotal conditions to run: {len(conditions_spec)}")

    # Run them all.
    summaries: List[Dict] = []
    first_data_per_kind: Dict[str, Dict] = {}
    for spec in conditions_spec:
        if spec["kind"] == "baseline":
            hook_specs = []
        elif spec["kind"] == "ablate":
            d = directions[spec["direction_type"]][spec["layer"]]
            hook_specs = [(spec["layer"], make_ablation_hook(d))]
        else:  # steer
            d = directions[spec["direction_type"]][spec["layer"]]
            hook_specs = [(spec["layer"], make_steering_hook(d, spec["multiplier"]))]

        summary, _ = run_one_condition(
            label=spec["label"], hook_specs=hook_specs,
            questions=questions, model=model, tokenizer=tokenizer,
            num_layers=num_layers, use_chat_template=use_chat_template, is_base=is_base,
        )
        summary.update({k: v for k, v in spec.items() if k != "label"})
        summaries.append(summary)
        first_data_per_kind.setdefault(spec["kind"], summary)
        print(
            f"  {spec['label']:<48s}  "
            f"H_direct={summary['direct_entropy_mean']:.3f}  "
            f"stated={summary['stated_confidence_mean']:.2f}  "
            f"ρ={summary['spearman_stated_vs_direct_entropy']:.3f}  "
            f"acc={summary['direct_accuracy']:.3f}"
        )

    # Save outputs.
    out_prefix = _file_prefix(dataset_name)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    results = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "num_questions": int(num_questions),
        "model_name": _C.MODEL_NAME,
        "base_model_name": _C.BASE_MODEL_NAME,
        "confidence_scale": _C.CONFIDENCE_SCALE,
        "few_shot_mode": _C.FEW_SHOT_MODE if is_base else None,
        "is_base_model": is_base,
        "seed": _C.SEED,
        "intervention_layers": list(_C.INTERVENTION_LAYERS),
        "steering_multipliers": list(_C.STEERING_MULTIPLIERS),
        "direction_types": list(_C.DIRECTION_TYPES),
        "intervention_types": list(_C.INTERVENTION_TYPES),
        "directions_dir": str(_C.DIRECTIONS_DIR),
        "git_commit": git_commit,
        "date_run": datetime.datetime.now().isoformat(),
        "device": str(DEVICE),
        "questions": [
            {"id": q.get("id"), "question": q.get("question"),
             "options": q.get("options"), "correct_answer": q.get("correct_answer")}
            for q in questions
        ],
        "conditions": summaries,
    }

    _atomic_write_json(f"{out_prefix}_intervention_results.json", results)
    print(f"\nSaved results → {out_prefix}_intervention_results.json")

    _write_params_txt(
        f"{out_prefix}_params.txt",
        dataset_name=dataset_name, num_questions=num_questions,
        num_layers=num_layers, run_id=run_id, git_commit=git_commit,
    )
    print(f"Saved params  → {out_prefix}_params.txt")

    _write_examples_txt(
        f"{out_prefix}_examples.txt",
        conditions=summaries, questions=questions, tokenizer=tokenizer,
        is_base=is_base, use_chat_template=use_chat_template,
    )
    print(f"Saved examples → {out_prefix}_examples.txt")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run causal interventions on contrast directions")
    parser.add_argument("--load-in-4bit", action="store_true", default=_C.LOAD_IN_4BIT)
    parser.add_argument("--load-in-8bit", action="store_true", default=_C.LOAD_IN_8BIT)
    parser.add_argument("--batch-size", type=int, default=_C.BATCH_SIZE)
    args = parser.parse_args()

    _C.BATCH_SIZE = args.batch_size

    print(f"Device: {DEVICE}")
    print(f"Datasets to process: {list(_C.DATASETS)}")
    print(f"Model: base={_C.BASE_MODEL_NAME!r}  adapter={_C.MODEL_NAME!r}")

    # Sanity check (same as collection runner).
    print("\nLoading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        _C.BASE_MODEL_NAME,
        adapter_path=_C.MODEL_NAME if _C.MODEL_NAME != _C.BASE_MODEL_NAME else None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    if _C.CONFIDENCE_SCALE == "numeric":
        bad = []
        for opt in NUMERIC_CONFIDENCE_OPTIONS:
            ids = tokenizer.encode(opt, add_special_tokens=False)
            if len(ids) != 1:
                bad.append((opt, ids))
        if bad:
            raise RuntimeError(
                f"Numeric scale requires single-token options. Bad: {bad}"
            )

    for dataset_name in _C.DATASETS:
        run_for_dataset(dataset_name, model, tokenizer, num_layers)

    print("\n" + "=" * 80)
    print("✓ All causal-intervention runs complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
