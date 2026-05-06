"""
Activation patching experiments: Test causal role of full activations.

Unlike steering (which adds a 1D direction), patching swaps the complete
activation pattern from one example into another. This tests whether the
full activation (not just a linear projection) causally determines behavior.

For each question pair (source=low metric, target=high metric):
1. Run source question normally → baseline confidence
2. Run source question with target's activations patched at layer L → patched confidence
3. Measure: Does patching shift source's confidence toward target's confidence?

If patching B's activations into A makes A behave like B, then layer L's
full activation pattern is causally involved in determining confidence.

This is more robust than steering to non-linear encodings. If uncertainty
is encoded categorically (e.g., low=[1,0,0], mid=[0,1,0], high=[0,0,1]),
steering along a probe direction won't work, but patching will.

Usage:
    python run_activation_patching.py --metric logit_gap
    python run_activation_patching.py --metric entropy --n-pairs 200
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import random

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from core.steering import (
    BatchPatchingHook,
    generate_orthogonal_directions,
)
from prompts import (
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    response_to_confidence,
    # Other-confidence task (control)
    OTHER_CONFIDENCE_OPTIONS,
    format_other_confidence_prompt,
    get_other_confidence_signal,
)

# =============================================================================
# CONFIGURATION — edit values in experiment_config.ActivationPatchingConfig
# =============================================================================
from experiment_config import ActivationPatchingConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
MODEL_NAME = _C.MODEL_NAME
DATASET_NAME = _C.DATASET_NAME
META_TASK = _C.META_TASK
NUM_PATCH_PAIRS = _C.NUM_PATCH_PAIRS
PAIRING_METHOD = _C.PAIRING_METHOD
BATCH_SIZE = _C.BATCH_SIZE
PATCHING_LAYERS = _C.PATCHING_LAYERS
AVAILABLE_METRICS = list(_C.AVAILABLE_METRICS)
METRIC = _C.METRIC
OUTPUTS_DIR = _C.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(exist_ok=True)
SEED = _C.SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Option tokens (cached at startup)
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS
_CACHED_TOKEN_IDS = {"meta_options": None, "delegate_options": None}


def initialize_token_cache(tokenizer):
    """Precompute option token IDs once."""
    _CACHED_TOKEN_IDS["meta_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in META_OPTIONS
    ]
    _CACHED_TOKEN_IDS["delegate_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in DELEGATE_OPTIONS
    ]


def get_output_prefix() -> str:
    """Generate output filename prefix."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_patching{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_patching{task_suffix}")


# =============================================================================
# PAIR CREATION
# =============================================================================


def create_patch_pairs(
    metric_values: np.ndarray,
    n_pairs: int,
    method: str = "extremes",
    direction: str = "high_to_low",
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Create source→target pairs for patching.

    Each pair is (source_idx, target_idx) where we'll patch target's
    activations into source and measure the effect.

    Args:
        metric_values: (n_samples,) metric values for each question
        n_pairs: Number of pairs to create
        method: Pairing strategy:
            - "extremes": Pair low-metric sources with high-metric targets
            - "random": Random pairs from different quartiles
            - "quartile": Systematic quartile-to-quartile pairs
        direction: Which direction to patch:
            - "high_to_low": Patch high-metric activations into low-metric questions
            - "low_to_high": Patch low-metric activations into high-metric questions

    Returns:
        List of (source_idx, target_idx) tuples
    """
    rng = np.random.RandomState(seed)
    n = len(metric_values)
    sorted_idx = np.argsort(metric_values)

    pairs = []

    if method == "extremes":
        n_quartile = n // 4
        low_indices = sorted_idx[:n_quartile]
        high_indices = sorted_idx[-n_quartile:]

        if direction == "high_to_low":
            # Source = low metric question, Target = high metric activation
            for _ in range(n_pairs):
                source = rng.choice(low_indices)
                target = rng.choice(high_indices)
                pairs.append((source, target))
        else:  # low_to_high
            # Source = high metric question, Target = low metric activation
            for _ in range(n_pairs):
                source = rng.choice(high_indices)
                target = rng.choice(low_indices)
                pairs.append((source, target))

    elif method == "random":
        n_half = n // 2
        low_half = sorted_idx[:n_half]
        high_half = sorted_idx[n_half:]

        if direction == "high_to_low":
            for _ in range(n_pairs):
                source = rng.choice(low_half)
                target = rng.choice(high_half)
                pairs.append((source, target))
        else:
            for _ in range(n_pairs):
                source = rng.choice(high_half)
                target = rng.choice(low_half)
                pairs.append((source, target))

    elif method == "quartile":
        n_quartile = n // 4
        q1 = sorted_idx[:n_quartile]
        q2 = sorted_idx[n_quartile:2*n_quartile]
        q3 = sorted_idx[2*n_quartile:3*n_quartile]
        q4 = sorted_idx[3*n_quartile:]

        if direction == "high_to_low":
            n_primary = n_pairs // 2
            for _ in range(n_primary):
                pairs.append((rng.choice(q1), rng.choice(q4)))
            n_secondary = (n_pairs - n_primary) // 2
            for _ in range(n_secondary):
                pairs.append((rng.choice(q1), rng.choice(q3)))
            for _ in range(n_pairs - n_primary - n_secondary):
                pairs.append((rng.choice(q2), rng.choice(q4)))
        else:
            n_primary = n_pairs // 2
            for _ in range(n_primary):
                pairs.append((rng.choice(q4), rng.choice(q1)))
            n_secondary = (n_pairs - n_primary) // 2
            for _ in range(n_secondary):
                pairs.append((rng.choice(q4), rng.choice(q2)))
            for _ in range(n_pairs - n_primary - n_secondary):
                pairs.append((rng.choice(q3), rng.choice(q1)))

    else:
        raise ValueError(f"Unknown pairing method: {method}")

    return pairs


def create_random_pairs(n_samples: int, n_pairs: int, seed: int = 42) -> List[Tuple[int, int]]:
    """Create random pairs for control comparison."""
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n_pairs):
        source = rng.randint(0, n_samples)
        target = rng.randint(0, n_samples)
        while target == source:
            target = rng.randint(0, n_samples)
        pairs.append((source, target))
    return pairs


# =============================================================================
# PATCHING EXPERIMENT
# =============================================================================


def format_meta_prompt_for_question(
    question: Dict,
    tokenizer,
    use_chat_template: bool,
    trial_idx: int = 0
) -> Tuple[str, List[str], Optional[Dict]]:
    """Format meta prompt based on META_TASK."""
    if META_TASK == "delegate":
        prompt, options, mapping = format_answer_or_delegate_prompt(
            question, tokenizer, use_chat_template, trial_idx
        )
        return prompt, options, mapping
    else:
        prompt, options = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
        return prompt, options, None


def compute_confidence_from_logits(
    logits: torch.Tensor,
    option_token_ids: List[int],
    mapping: Optional[Dict] = None
) -> float:
    """Compute confidence value from logits over option tokens."""
    option_logits = logits[option_token_ids]
    probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    if META_TASK == "delegate":
        options = DELEGATE_OPTIONS
    else:
        options = META_OPTIONS

    response = options[np.argmax(probs)]
    return response_to_confidence(response, probs, mapping, task_type=META_TASK)


def format_other_meta_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool
) -> str:
    """Format other-confidence prompt (human difficulty estimation)."""
    prompt, _ = format_other_confidence_prompt(question, tokenizer, use_chat_template)
    return prompt


def compute_other_confidence_from_logits(
    logits: torch.Tensor,
    option_token_ids: List[int]
) -> float:
    """Compute other-confidence signal from logits over option tokens."""
    option_logits = logits[option_token_ids]
    probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
    return get_other_confidence_signal(probs)


def run_patching_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    cached_activations: Dict[int, np.ndarray],
    metric_values: np.ndarray,
    layers: List[int],
    patch_pairs: List[Tuple[int, int]],
    use_chat_template: bool,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Run activation patching experiment.

    For each layer and each (source, target) pair:
    1. Run source normally → baseline confidence
    2. Run source with target's activations patched → patched confidence
    3. Compute shift = (patched - baseline) / (target_baseline - baseline)

    Args:
        model: The transformer model
        tokenizer: Tokenizer
        questions: List of question dicts
        cached_activations: {layer_idx: (n_questions, hidden_dim)} pre-extracted activations
        metric_values: (n_questions,) metric values for each question
        layers: Which layers to test
        patch_pairs: List of (source_idx, target_idx) pairs
        use_chat_template: Whether to use chat template
        batch_size: Forward pass batch size

    Returns:
        Dict with results per layer and per pair
    """
    model.eval()

    # Get option token IDs
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]

    # Get access to model layers
    if hasattr(model, 'get_base_model'):
        model_layers = model.get_base_model().model.layers
    else:
        model_layers = model.model.layers

    results = {
        "layers": layers,
        "n_pairs": len(patch_pairs),
        "pairs": [[s, t] for s, t in patch_pairs],  # Convert tuples to lists for JSON
        "metric": METRIC,
        "layer_results": {},
    }

    # Pre-compute all baselines (no patching)
    print("Computing baselines for all questions...")
    all_prompts = []
    all_mappings = []
    for i, q in enumerate(questions):
        prompt, _, mapping = format_meta_prompt_for_question(q, tokenizer, use_chat_template, i)
        all_prompts.append(prompt)
        all_mappings.append(mapping)

    # Batch compute baselines
    all_baseline_confidences = []
    for batch_start in tqdm(range(0, len(questions), batch_size), desc="Baselines"):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_mappings = all_mappings[batch_start:batch_end]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        for i in range(len(batch_prompts)):
            final_logits = outputs.logits[i, -1, :]
            conf = compute_confidence_from_logits(final_logits, option_token_ids, batch_mappings[i])
            all_baseline_confidences.append(conf)

        del inputs, outputs
        torch.cuda.empty_cache()

    all_baseline_confidences = np.array(all_baseline_confidences)
    print(f"Baseline confidence: mean={all_baseline_confidences.mean():.3f}, std={all_baseline_confidences.std():.3f}")

    # Run patching for each layer
    for layer_idx in tqdm(layers, desc="Layers"):
        layer_activations = cached_activations[layer_idx]  # (n_questions, hidden_dim)

        pair_results = []

        # Process pairs in batches
        for batch_start in tqdm(range(0, len(patch_pairs), batch_size),
                                desc=f"Layer {layer_idx}", leave=False):
            batch_end = min(batch_start + batch_size, len(patch_pairs))
            batch_pairs = patch_pairs[batch_start:batch_end]

            # Prepare batch: source prompts and target activations
            batch_prompts = []
            batch_target_acts = []
            batch_mappings = []
            batch_pair_info = []

            for source_idx, target_idx in batch_pairs:
                batch_prompts.append(all_prompts[source_idx])
                batch_target_acts.append(layer_activations[target_idx])
                batch_mappings.append(all_mappings[source_idx])
                batch_pair_info.append({
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "source_metric": float(metric_values[source_idx]),
                    "target_metric": float(metric_values[target_idx]),
                    "source_baseline": float(all_baseline_confidences[source_idx]),
                    "target_baseline": float(all_baseline_confidences[target_idx]),
                })

            batch_target_acts = torch.tensor(np.array(batch_target_acts), dtype=torch.float32)

            # Tokenize batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)

            # Register patching hook
            hook = BatchPatchingHook(batch_target_acts, position="last")
            handle = model_layers[layer_idx].register_forward_hook(hook)

            try:
                with torch.no_grad():
                    outputs = model(**inputs, use_cache=False)

                # Compute patched confidences
                for i in range(len(batch_prompts)):
                    final_logits = outputs.logits[i, -1, :]
                    patched_conf = compute_confidence_from_logits(
                        final_logits, option_token_ids, batch_mappings[i]
                    )

                    info = batch_pair_info[i]
                    info["patched_confidence"] = float(patched_conf)

                    # Compute shift metrics
                    source_base = info["source_baseline"]
                    target_base = info["target_baseline"]
                    gap = target_base - source_base

                    info["confidence_shift"] = float(patched_conf - source_base)
                    if abs(gap) > 0.01:
                        info["normalized_shift"] = float((patched_conf - source_base) / gap)
                    else:
                        info["normalized_shift"] = 0.0

                    pair_results.append(info)

            finally:
                handle.remove()

            del inputs, outputs
            torch.cuda.empty_cache()

        # Aggregate layer results
        shifts = [r["confidence_shift"] for r in pair_results]
        norm_shifts = [r["normalized_shift"] for r in pair_results]

        results["layer_results"][str(layer_idx)] = {
            "pairs": pair_results,
            "mean_shift": float(np.mean(shifts)),
            "std_shift": float(np.std(shifts)),
            "mean_normalized_shift": float(np.mean(norm_shifts)),
            "std_normalized_shift": float(np.std(norm_shifts)),
            "n_positive_shift": int(np.sum(np.array(shifts) > 0)),
            "n_pairs": len(pair_results),
        }

    # Add baseline info
    results["baselines"] = {
        "mean": float(all_baseline_confidences.mean()),
        "std": float(all_baseline_confidences.std()),
        "all": all_baseline_confidences.tolist(),
    }

    return results


def run_other_confidence_patching(
    model,
    tokenizer,
    questions: List[Dict],
    cached_activations: Dict[int, np.ndarray],
    layers: List[int],
    patch_pairs: List[Tuple[int, int]],
    use_chat_template: bool,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Run activation patching on other-confidence task (control).

    Same patching logic but using other-confidence prompts.
    This tests whether patching affects human difficulty estimation
    the same way it affects self-confidence.

    Returns:
        Dict with baseline and patched other-confidence values per layer and pair.
    """
    # Only run for confidence task (not delegate)
    if META_TASK != "confidence":
        return None

    model.eval()
    option_token_ids = _CACHED_TOKEN_IDS["meta_options"]  # Same S-Z scale

    # Get access to model layers
    if hasattr(model, 'get_base_model'):
        model_layers = model.get_base_model().model.layers
    else:
        model_layers = model.model.layers

    results = {
        "layers": layers,
        "n_pairs": len(patch_pairs),
        "layer_results": {},
    }

    # Pre-compute all other-confidence prompts
    print("Formatting other-confidence prompts...")
    all_prompts = []
    for q in questions:
        prompt = format_other_meta_prompt(q, tokenizer, use_chat_template)
        all_prompts.append(prompt)

    # Batch compute baselines (no patching)
    print("Computing other-confidence baselines...")
    all_baseline_signals = []
    for batch_start in tqdm(range(0, len(questions), batch_size), desc="Other baselines"):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_prompts = all_prompts[batch_start:batch_end]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        for i in range(len(batch_prompts)):
            final_logits = outputs.logits[i, -1, :]
            signal = compute_other_confidence_from_logits(final_logits, option_token_ids)
            all_baseline_signals.append(signal)

        del inputs, outputs
        torch.cuda.empty_cache()

    all_baseline_signals = np.array(all_baseline_signals)
    print(f"Other-confidence baseline: mean={all_baseline_signals.mean():.3f}, std={all_baseline_signals.std():.3f}")

    # Run patching for each layer
    for layer_idx in tqdm(layers, desc="Other-conf layers"):
        layer_activations = cached_activations[layer_idx]
        pair_results = []

        for batch_start in tqdm(range(0, len(patch_pairs), batch_size),
                                desc=f"Layer {layer_idx}", leave=False):
            batch_end = min(batch_start + batch_size, len(patch_pairs))
            batch_pairs = patch_pairs[batch_start:batch_end]

            batch_prompts = []
            batch_target_acts = []
            batch_pair_info = []

            for source_idx, target_idx in batch_pairs:
                batch_prompts.append(all_prompts[source_idx])
                batch_target_acts.append(layer_activations[target_idx])
                batch_pair_info.append({
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "source_baseline": float(all_baseline_signals[source_idx]),
                    "target_baseline": float(all_baseline_signals[target_idx]),
                })

            batch_target_acts = torch.tensor(np.array(batch_target_acts), dtype=torch.float32)
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)

            hook = BatchPatchingHook(batch_target_acts, position="last")
            handle = model_layers[layer_idx].register_forward_hook(hook)

            try:
                with torch.no_grad():
                    outputs = model(**inputs, use_cache=False)

                for i in range(len(batch_prompts)):
                    final_logits = outputs.logits[i, -1, :]
                    patched_signal = compute_other_confidence_from_logits(
                        final_logits, option_token_ids
                    )

                    info = batch_pair_info[i]
                    info["patched_signal"] = float(patched_signal)

                    source_base = info["source_baseline"]
                    target_base = info["target_baseline"]
                    gap = target_base - source_base

                    info["signal_shift"] = float(patched_signal - source_base)
                    if abs(gap) > 0.01:
                        info["normalized_shift"] = float((patched_signal - source_base) / gap)
                    else:
                        info["normalized_shift"] = 0.0

                    pair_results.append(info)

            finally:
                handle.remove()

            del inputs, outputs
            torch.cuda.empty_cache()

        # Aggregate layer results
        shifts = [r["signal_shift"] for r in pair_results]
        norm_shifts = [r["normalized_shift"] for r in pair_results]

        results["layer_results"][str(layer_idx)] = {
            "pairs": pair_results,
            "mean_shift": float(np.mean(shifts)),
            "std_shift": float(np.std(shifts)),
            "mean_normalized_shift": float(np.mean(norm_shifts)),
            "std_normalized_shift": float(np.std(norm_shifts)),
            "n_pairs": len(pair_results),
        }

    results["baselines"] = {
        "mean": float(all_baseline_signals.mean()),
        "std": float(all_baseline_signals.std()),
        "all": all_baseline_signals.tolist(),
    }

    return results


def analyze_other_confidence_patching_effect(
    self_results: Dict,
    other_results: Dict,
    layers: List[int],
    significant_layers: List[int] = None
) -> Dict:
    """
    Compare patching effect on self-confidence vs other-confidence.

    Statistical approach:
    1. For each layer, compute mean |shift| for self and other
    2. Bootstrap CI for the difference (self - other)
    3. Permutation test: is self > other significantly?
    4. Focus on significant layers from main analysis

    Args:
        self_results: Main patching results (self-confidence)
        other_results: Other-confidence patching results
        layers: All layers tested
        significant_layers: Layers that showed significant patching effect (focus analysis)

    Returns dict with per-layer comparison and overall assessment.
    """
    from scipy import stats

    if other_results is None:
        return None

    analysis = {
        "layers": layers,
        "significant_layers": significant_layers or [],
        "layer_effects": {},
    }

    for layer_idx in layers:
        layer_str = str(layer_idx)
        self_lr = self_results["layer_results"].get(layer_str, {})
        other_lr = other_results["layer_results"].get(layer_str, {})

        if not self_lr or not other_lr:
            continue

        self_shifts = [p["confidence_shift"] for p in self_lr.get("pairs", [])]
        other_shifts = [p["signal_shift"] for p in other_lr.get("pairs", [])]

        if len(self_shifts) != len(other_shifts):
            continue

        self_shifts = np.array(self_shifts)
        other_shifts = np.array(other_shifts)
        n_pairs = len(self_shifts)

        self_abs = np.abs(self_shifts)
        other_abs = np.abs(other_shifts)

        self_effect = float(np.mean(self_abs))
        other_effect = float(np.mean(other_abs))

        if other_effect > 1e-6:
            ratio = self_effect / other_effect
        else:
            ratio = float('inf') if self_effect > 1e-6 else 1.0

        # Paired difference: self - other (per sample)
        diff = self_abs - other_abs
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff))

        # Bootstrap 95% CI for the difference
        n_bootstrap = 1000
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(n_pairs, size=n_pairs, replace=True)
            boot_diff = np.mean(self_abs[boot_idx] - other_abs[boot_idx])
            bootstrap_diffs.append(boot_diff)
        ci_low = float(np.percentile(bootstrap_diffs, 2.5))
        ci_high = float(np.percentile(bootstrap_diffs, 97.5))

        # Paired t-test: is self effect > other effect?
        t_stat, p_two_sided = stats.ttest_rel(self_abs, other_abs)
        # One-sided: self > other
        p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

        # Permutation test as alternative
        n_perms = 1000
        observed_diff = np.mean(self_abs - other_abs)
        perm_diffs = []
        combined = np.stack([self_abs, other_abs], axis=1)
        for _ in range(n_perms):
            # Randomly swap self/other within each pair
            swaps = np.random.randint(0, 2, size=n_pairs)
            perm_self = np.where(swaps == 0, combined[:, 0], combined[:, 1])
            perm_other = np.where(swaps == 0, combined[:, 1], combined[:, 0])
            perm_diffs.append(np.mean(perm_self - perm_other))
        p_perm = float(np.mean(np.array(perm_diffs) >= observed_diff))

        analysis["layer_effects"][layer_str] = {
            "self_effect_mean_abs": self_effect,
            "self_effect_std": float(np.std(self_abs)),
            "other_effect_mean_abs": other_effect,
            "other_effect_std": float(np.std(other_abs)),
            "self_vs_other_ratio": ratio,
            "diff_mean": diff_mean,  # self - other
            "diff_std": diff_std,
            "diff_ci95": [ci_low, ci_high],
            "t_statistic": float(t_stat),
            "p_value_paired": float(p_two_sided),
            "p_value_one_sided": float(p_one_sided),  # self > other
            "p_value_permutation": p_perm,
            "self_other_correlation": float(np.corrcoef(self_shifts, other_shifts)[0, 1]) if n_pairs > 1 else np.nan,
            "n_pairs": n_pairs,
            "is_significant_layer": layer_idx in (significant_layers or []),
        }

    # Overall summary
    if analysis["layer_effects"]:
        ratios = [e["self_vs_other_ratio"] for e in analysis["layer_effects"].values() if not np.isinf(e["self_vs_other_ratio"])]
        if ratios:
            analysis["mean_ratio"] = float(np.mean(ratios))
        else:
            analysis["mean_ratio"] = float('inf')

        # Focus on significant layers
        sig_effects = [e for l, e in analysis["layer_effects"].items()
                       if e.get("is_significant_layer", False)]
        if sig_effects:
            analysis["significant_layer_summary"] = {
                "mean_self_effect": float(np.mean([e["self_effect_mean_abs"] for e in sig_effects])),
                "mean_other_effect": float(np.mean([e["other_effect_mean_abs"] for e in sig_effects])),
                "mean_ratio": float(np.mean([e["self_vs_other_ratio"] for e in sig_effects if not np.isinf(e["self_vs_other_ratio"])])),
                "any_significant_diff": any(e["p_value_one_sided"] < 0.05 for e in sig_effects),
            }

    return analysis


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================


def analyze_patching_results(results: Dict) -> Dict:
    """Compute summary statistics for patching experiment."""
    analysis = {
        "layers": list(results["layers"]),  # Copy to avoid shared reference
        "n_pairs": results["n_pairs"],
        "metric": results.get("metric", "unknown"),
        "layer_effects": {},
    }

    for layer_idx in results["layers"]:
        lr = results["layer_results"][str(layer_idx)]

        # Effect size: how much did patching shift confidence toward target?
        # normalized_shift of 1.0 = patching made source identical to target
        # normalized_shift of 0.0 = patching had no effect
        mean_norm = lr["mean_normalized_shift"]
        std_norm = lr["std_normalized_shift"]

        # Statistical test: is mean_norm significantly > 0?
        from scipy import stats
        norm_shifts = [p["normalized_shift"] for p in lr["pairs"]]
        t_stat, p_value = stats.ttest_1samp(norm_shifts, 0)

        analysis["layer_effects"][str(layer_idx)] = {
            "mean_shift": float(lr["mean_shift"]),
            "mean_normalized_shift": float(mean_norm),
            "std_normalized_shift": float(std_norm),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_p05": bool(p_value < 0.05),
            "effect_size": float(mean_norm),  # Normalized shift is already effect size
        }

    # Find best layer
    best_layer = max(
        results["layers"],
        key=lambda l: analysis["layer_effects"][str(l)]["mean_normalized_shift"]
    )
    analysis["best_layer"] = int(best_layer) if hasattr(best_layer, 'item') else best_layer
    analysis["best_effect"] = float(analysis["layer_effects"][str(best_layer)]["mean_normalized_shift"])

    return analysis


def print_summary(analysis: Dict):
    """Print patching experiment summary."""
    print("\n" + "=" * 70)
    print("ACTIVATION PATCHING RESULTS")
    print("=" * 70)

    print(f"\nMetric: {analysis['metric']}")
    print(f"Pairs tested per layer: {analysis['n_pairs']}")

    print("\n--- Normalized Shift by Layer ---")
    print("(1.0 = source becomes identical to target, 0.0 = no effect)")
    print(f"{'Layer':<8} {'Mean':<10} {'Std':<10} {'p-value':<10} {'Sig?':<6}")
    print("-" * 50)

    for layer_idx in analysis["layers"]:
        e = analysis["layer_effects"][str(layer_idx)]
        sig = "✓" if e["significant_p05"] else ""
        print(f"{layer_idx:<8} {e['mean_normalized_shift']:<10.3f} "
              f"{e['std_normalized_shift']:<10.3f} {e['p_value']:<10.4f} {sig:<6}")

    best = analysis["best_layer"]
    best_effect = analysis["best_effect"]
    best_p = analysis["layer_effects"][str(best)]["p_value"]

    print(f"\nBest layer: {best} (normalized shift = {best_effect:.3f}, p = {best_p:.4f})")

    if best_effect > 0.3 and best_p < 0.05:
        print("\n✓ STRONG causal effect!")
        print("  Patching high-metric activations into low-metric questions")
        print("  significantly shifts confidence toward the high-metric pattern.")
    elif best_effect > 0.1 and best_p < 0.05:
        print("\n✓ Moderate causal effect detected.")
    elif best_effect > 0:
        print("\n⚠ Weak or non-significant effect.")
    else:
        print("\n✗ No patching effect detected.")


def plot_results(analysis: Dict, results: Dict, output_path: str):
    """Create visualization of patching results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    layers = analysis["layers"]
    n_pairs = analysis["n_pairs"]

    # Plot 1: Normalized shift by layer with 95% CI (SEM-based)
    ax1 = axes[0]
    shifts = [analysis["layer_effects"][str(l)]["mean_normalized_shift"] for l in layers]
    stds = [analysis["layer_effects"][str(l)]["std_normalized_shift"] for l in layers]
    # Use SEM for error bars (std / sqrt(n)), then multiply by 1.96 for 95% CI
    sems = [s / np.sqrt(n_pairs) * 1.96 for s in stds]

    # Color bars by significance
    colors = ['#2ecc71' if analysis["layer_effects"][str(l)]["significant_p05"] and
              analysis["layer_effects"][str(l)]["mean_normalized_shift"] > 0
              else '#e74c3c' if analysis["layer_effects"][str(l)]["significant_p05"]
              else '#95a5a6' for l in layers]

    ax1.bar(range(len(layers)), shifts, yerr=sems, alpha=0.8, color=colors,
            capsize=3, error_kw={'linewidth': 1.5})
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Normalized Shift (95% CI)")
    ax1.set_title("Patching Effect by Layer")
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1.5)
    ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.4, label='Strong effect')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Distribution of normalized shifts for best layer
    ax2 = axes[1]
    best_layer = analysis["best_layer"]
    pairs = results["layer_results"][str(best_layer)]["pairs"]
    norm_shifts = [p["normalized_shift"] for p in pairs]

    # Histogram with KDE
    ax2.hist(norm_shifts, bins=20, alpha=0.6, color='steelblue', edgecolor='white', density=True)

    # Add vertical lines for mean and 0
    mean_shift = np.mean(norm_shifts)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, label='No effect')
    ax2.axvline(x=mean_shift, color='red', linestyle='--', linewidth=2,
                label=f'Mean = {mean_shift:.2f}')
    ax2.axvline(x=1.0, color='green', linestyle=':', linewidth=2, alpha=0.7,
                label='Full transfer')

    ax2.set_xlabel("Normalized Shift")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Layer {best_layer}: Distribution of Effects")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Add text showing fraction positive
    frac_positive = sum(1 for s in norm_shifts if s > 0) / len(norm_shifts)
    ax2.text(0.02, 0.98, f'{frac_positive:.0%} positive', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontweight='bold')

    # Plot 3: Summary text
    ax3 = axes[2]
    ax3.axis('off')

    best_effect = analysis["layer_effects"][str(best_layer)]
    sem = best_effect['std_normalized_shift'] / np.sqrt(n_pairs)

    summary = f"""ACTIVATION PATCHING SUMMARY

Metric: {analysis['metric']}
Pairs per layer: {n_pairs}

Best Layer: {best_layer}
  Mean shift: {best_effect['mean_normalized_shift']:.3f}
  95% CI: [{best_effect['mean_normalized_shift'] - 1.96*sem:.3f}, {best_effect['mean_normalized_shift'] + 1.96*sem:.3f}]
  p-value: {best_effect['p_value']:.2e}
  Significant: {'Yes' if best_effect['significant_p05'] else 'No'}

Legend:
  Green = significant positive shift
  Red = significant negative shift
  Gray = not significant

Interpretation:
"""
    if best_effect['mean_normalized_shift'] > 0.3 and best_effect['significant_p05']:
        summary += """✓ STRONG causal effect
  Patching high-uncertainty activations
  into low-uncertainty questions shifts
  confidence toward uncertainty."""
    elif best_effect['mean_normalized_shift'] > 0.1 and best_effect['significant_p05']:
        summary += """✓ Moderate causal effect
  Activation pattern partially
  determines confidence."""
    elif best_effect['significant_p05']:
        summary += """⚠ Significant but weak effect
  Effect detected but small
  magnitude."""
    else:
        summary += """✗ No significant effect
  Activation pattern may not be
  primary determinant."""

    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def print_bidirectional_summary(all_analyses: Dict[str, Dict]):
    """Print combined summary for bidirectional patching.

    Key insight: normalized_shift should be POSITIVE in BOTH directions when
    patching works. This is because normalized_shift measures movement toward
    the target baseline, regardless of direction.

    - h→l positive: patching high-entropy acts into low-entropy prompts moves
      the output TOWARD high-entropy (the target behavior)
    - l→h positive: patching low-entropy acts into high-entropy prompts moves
      the output TOWARD low-entropy (the target behavior)
    """
    print("\n" + "=" * 70)
    print("BIDIRECTIONAL ACTIVATION PATCHING RESULTS")
    print("=" * 70)

    h2l = all_analyses["high_to_low"]
    l2h = all_analyses["low_to_high"]

    print(f"\nMetric: {h2l['metric']}")
    print(f"Pairs tested per direction: {h2l['n_pairs']}")

    print("\n--- Layer-by-Layer Comparison ---")
    print(f"{'Layer':<8} {'high→low':<12} {'low→high':<12} {'Bidirectional?':<15} {'Interpretation'}")
    print("-" * 75)

    # Threshold for considering an effect "meaningful"
    EFFECT_THRESHOLD = 0.1

    for layer_idx in h2l["layers"]:
        h2l_effect = h2l["layer_effects"][str(layer_idx)]
        l2h_effect = l2h["layer_effects"][str(layer_idx)]

        h2l_shift = h2l_effect["mean_normalized_shift"]
        l2h_shift = l2h_effect["mean_normalized_shift"]

        h2l_sig = "✓" if h2l_effect["significant_p05"] else ""
        l2h_sig = "✓" if l2h_effect["significant_p05"] else ""

        # Interpretation: BOTH should be positive for bidirectional causal effect
        h2l_works = h2l_shift > EFFECT_THRESHOLD
        l2h_works = l2h_shift > EFFECT_THRESHOLD
        h2l_fails = h2l_shift < -EFFECT_THRESHOLD
        l2h_fails = l2h_shift < -EFFECT_THRESHOLD

        if h2l_works and l2h_works:
            bidirectional = "Yes"
            interp = "✓ Bidirectional causal"
        elif h2l_works and not l2h_works and not l2h_fails:
            bidirectional = "Partial"
            interp = "→ h→l only (asymmetric)"
        elif l2h_works and not h2l_works and not h2l_fails:
            bidirectional = "Partial"
            interp = "← l→h only (asymmetric)"
        elif h2l_works and l2h_fails:
            bidirectional = "No"
            interp = "⚠ h→l+, l→h- (conflicting)"
        elif h2l_fails and l2h_works:
            bidirectional = "No"
            interp = "⚠ h→l-, l→h+ (conflicting)"
        elif h2l_fails and l2h_fails:
            bidirectional = "No"
            interp = "✗ Both negative (reversal)"
        else:
            bidirectional = "No"
            interp = "— Weak/no effect"

        print(f"{layer_idx:<8} {h2l_shift:>+.3f}{h2l_sig:<2}     {l2h_shift:>+.3f}{l2h_sig:<2}     {bidirectional:<15} {interp}")

    # Summary statistics
    print("\n--- Summary ---")

    # Count layers by pattern
    n_bidirectional = 0
    n_h2l_only = 0
    n_l2h_only = 0
    n_conflicting = 0

    for layer_idx in h2l["layers"]:
        h2l_shift = h2l["layer_effects"][str(layer_idx)]["mean_normalized_shift"]
        l2h_shift = l2h["layer_effects"][str(layer_idx)]["mean_normalized_shift"]

        h2l_works = h2l_shift > EFFECT_THRESHOLD
        l2h_works = l2h_shift > EFFECT_THRESHOLD
        h2l_fails = h2l_shift < -EFFECT_THRESHOLD
        l2h_fails = l2h_shift < -EFFECT_THRESHOLD

        if h2l_works and l2h_works:
            n_bidirectional += 1
        elif h2l_works and not l2h_works and not l2h_fails:
            n_h2l_only += 1
        elif l2h_works and not h2l_works and not h2l_fails:
            n_l2h_only += 1
        elif (h2l_works and l2h_fails) or (h2l_fails and l2h_works):
            n_conflicting += 1

    total = len(h2l['layers'])
    print(f"Bidirectional causal (both +): {n_bidirectional}/{total}")
    print(f"Asymmetric h→l only:           {n_h2l_only}/{total}")
    print(f"Asymmetric l→h only:           {n_l2h_only}/{total}")
    print(f"Conflicting (one +, one -):    {n_conflicting}/{total}")

    if n_bidirectional > total // 3:
        print("\n✓ Strong bidirectional causal evidence!")
        print("  Patching works in both directions at multiple layers.")
        print("  These activations causally encode the metric.")
    elif n_h2l_only + n_l2h_only > total // 3:
        print("\n~ Asymmetric effects detected.")
        print("  Patching works in one direction but not the other.")
        print("  This suggests directional information flow.")
    elif n_conflicting > total // 3:
        print("\n⚠ Conflicting patterns detected!")
        print("  Opposite signs in different directions suggest")
        print("  complex or confounded encoding.")
    else:
        print("\n— Mixed or weak effects across layers.")


def plot_bidirectional_results(all_analyses: Dict[str, Dict], all_results: Dict[str, Dict], output_path: str):
    """Create visualization comparing both directions.

    Key insight: BOTH directions should show POSITIVE normalized_shift when
    patching works. Positive shift means the patched output moved toward
    the target baseline (the behavior of the source of the patched activations).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    h2l = all_analyses["high_to_low"]
    l2h = all_analyses["low_to_high"]
    layers = h2l["layers"]
    n_pairs = h2l["n_pairs"]

    EFFECT_THRESHOLD = 0.1

    # Plot 1: Side-by-side bar chart of effects
    ax1 = axes[0]
    x = np.arange(len(layers))
    width = 0.35

    h2l_shifts = [h2l["layer_effects"][str(l)]["mean_normalized_shift"] for l in layers]
    l2h_shifts = [l2h["layer_effects"][str(l)]["mean_normalized_shift"] for l in layers]
    h2l_sems = [h2l["layer_effects"][str(l)]["std_normalized_shift"] / np.sqrt(n_pairs) * 1.96 for l in layers]
    l2h_sems = [l2h["layer_effects"][str(l)]["std_normalized_shift"] / np.sqrt(n_pairs) * 1.96 for l in layers]

    bars1 = ax1.bar(x - width/2, h2l_shifts, width, yerr=h2l_sems, label='high→low',
                    color='#e74c3c', alpha=0.8, capsize=2)
    bars2 = ax1.bar(x + width/2, l2h_shifts, width, yerr=l2h_sems, label='low→high',
                    color='#3498db', alpha=0.8, capsize=2)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Normalized Shift (95% CI)")
    ax1.set_title("Bidirectional Patching Effects\n(Both positive = bidirectional causal)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax1.axhline(y=EFFECT_THRESHOLD, color='green', linestyle='--', alpha=0.3)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Bidirectional score by layer (minimum of the two effects)
    ax2 = axes[1]

    # Bidirectional score: min(h2l, l2h) - captures layers where BOTH work
    # Higher = stronger bidirectional effect
    bidir_scores = [min(h, l) for h, l in zip(h2l_shifts, l2h_shifts)]

    colors = ['#2ecc71' if s > EFFECT_THRESHOLD else '#e74c3c' if s < -EFFECT_THRESHOLD else '#95a5a6'
              for s in bidir_scores]
    ax2.bar(x, bidir_scores, color=colors, alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Bidirectional Score (min of both)")
    ax2.set_title("Bidirectional Causal Strength")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(y=EFFECT_THRESHOLD, color='green', linestyle='--', alpha=0.4, label='Threshold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add text explanation
    ax2.text(0.02, 0.98, "Green = both directions\nshow positive shift",
             transform=ax2.transAxes, fontsize=8, verticalalignment='top')

    # Plot 3: Summary text
    ax3 = axes[2]
    ax3.axis('off')

    # Find best layers for each direction (highest positive shift)
    best_h2l_layer = max(layers, key=lambda l: h2l["layer_effects"][str(l)]["mean_normalized_shift"])
    best_l2h_layer = max(layers, key=lambda l: l2h["layer_effects"][str(l)]["mean_normalized_shift"])

    # Count layers by pattern
    n_bidirectional = sum(1 for h, l in zip(h2l_shifts, l2h_shifts)
                          if h > EFFECT_THRESHOLD and l > EFFECT_THRESHOLD)
    n_h2l_only = sum(1 for h, l in zip(h2l_shifts, l2h_shifts)
                     if h > EFFECT_THRESHOLD and abs(l) <= EFFECT_THRESHOLD)
    n_l2h_only = sum(1 for h, l in zip(h2l_shifts, l2h_shifts)
                     if l > EFFECT_THRESHOLD and abs(h) <= EFFECT_THRESHOLD)
    n_conflicting = sum(1 for h, l in zip(h2l_shifts, l2h_shifts)
                        if (h > EFFECT_THRESHOLD and l < -EFFECT_THRESHOLD) or
                           (h < -EFFECT_THRESHOLD and l > EFFECT_THRESHOLD))

    summary = f"""BIDIRECTIONAL PATCHING SUMMARY

Metric: {h2l['metric']}
Pairs per direction: {n_pairs}

High→Low (high-entropy acts → low-entropy prompts):
  Best layer: {best_h2l_layer}
  Effect: {h2l["layer_effects"][str(best_h2l_layer)]["mean_normalized_shift"]:.3f}

Low→High (low-entropy acts → high-entropy prompts):
  Best layer: {best_l2h_layer}
  Effect: {l2h["layer_effects"][str(best_l2h_layer)]["mean_normalized_shift"]:.3f}

Pattern Analysis:
  Bidirectional (both +):  {n_bidirectional}/{len(layers)}
  h→l only (asymmetric):   {n_h2l_only}/{len(layers)}
  l→h only (asymmetric):   {n_l2h_only}/{len(layers)}
  Conflicting (+/-):       {n_conflicting}/{len(layers)}

"""
    if n_bidirectional > len(layers) // 3:
        summary += """Interpretation:
✓ BIDIRECTIONAL causal evidence!
  Patching works in both directions.
  Activations causally encode {metric}.""".format(metric=h2l['metric'])
    elif n_h2l_only + n_l2h_only > len(layers) // 3:
        summary += """Interpretation:
~ Asymmetric effects.
  One direction works, the other doesn't.
  Directional information flow."""
    elif n_conflicting > len(layers) // 3:
        summary += """Interpretation:
⚠ Conflicting patterns!
  Opposite signs suggest complex or
  confounded encoding."""
    else:
        summary += """Interpretation:
— Mixed results across layers.
  Causal role unclear."""

    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBidirectional plot saved to {output_path}")


def plot_other_confidence_comparison(
    other_confidence_analyses: Dict[str, Dict],
    main_analyses: Dict[str, Dict],
    output_path: str
):
    """
    Plot comparison of self-confidence vs other-confidence patching effects.

    Shows:
    1. Bar chart comparing |self| vs |other| effect per layer with CIs
    2. Focus panel on significant layers
    3. Summary statistics with p-values
    """
    # Collect data from both directions
    all_layers = set()
    significant_layers = set()

    for direction, analysis in other_confidence_analyses.items():
        if analysis and analysis.get("layer_effects"):
            for layer_str, effect in analysis["layer_effects"].items():
                all_layers.add(int(layer_str))
                if effect.get("is_significant_layer"):
                    significant_layers.add(int(layer_str))

    if not all_layers:
        print("  No other-confidence data to plot")
        return

    layers = sorted(all_layers)
    sig_layers = sorted(significant_layers)

    # Determine layout based on what we have
    if sig_layers:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Combine data from both directions for each layer
    combined_data = {l: {"self": [], "other": [], "diff": [], "diff_ci": []} for l in layers}

    for direction, analysis in other_confidence_analyses.items():
        if analysis and analysis.get("layer_effects"):
            for layer_str, effect in analysis["layer_effects"].items():
                layer = int(layer_str)
                combined_data[layer]["self"].append(effect["self_effect_mean_abs"])
                combined_data[layer]["other"].append(effect["other_effect_mean_abs"])
                combined_data[layer]["diff"].append(effect["diff_mean"])
                combined_data[layer]["diff_ci"].append(effect["diff_ci95"])

    # Plot 1: Self vs Other effect by layer
    ax1 = axes[0]
    x = np.arange(len(layers))
    width = 0.35

    self_means = [np.mean(combined_data[l]["self"]) for l in layers]
    other_means = [np.mean(combined_data[l]["other"]) for l in layers]
    self_stds = [np.std(combined_data[l]["self"]) if len(combined_data[l]["self"]) > 1 else 0 for l in layers]
    other_stds = [np.std(combined_data[l]["other"]) if len(combined_data[l]["other"]) > 1 else 0 for l in layers]

    # Color significant layers differently
    self_colors = ['steelblue' if l in sig_layers else 'lightsteelblue' for l in layers]
    other_colors = ['coral' if l in sig_layers else 'lightsalmon' for l in layers]

    ax1.bar(x - width/2, self_means, width, yerr=self_stds, label='Self-confidence',
            color=self_colors, alpha=0.8, capsize=2, edgecolor='darkblue', linewidth=0.5)
    ax1.bar(x + width/2, other_means, width, yerr=other_stds, label='Other-confidence',
            color=other_colors, alpha=0.8, capsize=2, edgecolor='darkred', linewidth=0.5)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean |Δ Confidence|")
    ax1.set_title("Self vs Other Confidence: Patching Effect")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Mark significant layers
    for i, l in enumerate(layers):
        if l in sig_layers:
            ax1.annotate('*', (i, max(self_means[i], other_means[i]) * 1.05),
                        ha='center', fontsize=14, fontweight='bold')

    # Plot 2 (or 3): Summary text
    if sig_layers:
        # Plot 2: Focus on significant layers
        ax2 = axes[1]

        sig_x = np.arange(len(sig_layers))
        sig_self = [np.mean(combined_data[l]["self"]) for l in sig_layers]
        sig_other = [np.mean(combined_data[l]["other"]) for l in sig_layers]
        sig_diff = [np.mean(combined_data[l]["diff"]) for l in sig_layers]

        # CI from the analyses
        sig_ci_low = []
        sig_ci_high = []
        for l in sig_layers:
            cis = combined_data[l]["diff_ci"]
            if cis:
                sig_ci_low.append(np.mean([ci[0] for ci in cis]))
                sig_ci_high.append(np.mean([ci[1] for ci in cis]))
            else:
                sig_ci_low.append(0)
                sig_ci_high.append(0)

        # Plot difference (self - other) with CI
        colors = ['green' if d > 0 else 'red' for d in sig_diff]
        yerr_low = [d - ci_l for d, ci_l in zip(sig_diff, sig_ci_low)]
        yerr_high = [ci_h - d for d, ci_h in zip(sig_diff, sig_ci_high)]

        ax2.bar(sig_x, sig_diff, color=colors, alpha=0.7,
                yerr=[yerr_low, yerr_high], capsize=4)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel("Significant Layer")
        ax2.set_ylabel("Difference: |Self| - |Other| (95% CI)")
        ax2.set_title("Self > Other? (Significant Layers Only)")
        ax2.set_xticks(sig_x)
        ax2.set_xticklabels(sig_layers)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add ratio annotations
        for i, l in enumerate(sig_layers):
            ratio = sig_self[i] / sig_other[i] if sig_other[i] > 1e-6 else float('inf')
            ax2.annotate(f'{ratio:.1f}x', (i, sig_diff[i]),
                        ha='center', va='bottom' if sig_diff[i] > 0 else 'top',
                        fontsize=9)

        ax_summary = axes[2]
    else:
        ax_summary = axes[1]

    # Summary panel
    ax_summary.axis('off')

    # Collect p-values from significant layers
    p_values = []
    for direction, analysis in other_confidence_analyses.items():
        if analysis and analysis.get("layer_effects"):
            for layer_str, effect in analysis["layer_effects"].items():
                if int(layer_str) in sig_layers:
                    p_values.append((int(layer_str), direction, effect["p_value_one_sided"]))

    summary = """SELF vs OTHER CONFIDENCE COMPARISON

Question: Does patching affect self-confidence
specifically, or general confidence judgments?

"""
    if sig_layers:
        # Get stats from significant layers
        sig_self_mean = np.mean([np.mean(combined_data[l]["self"]) for l in sig_layers])
        sig_other_mean = np.mean([np.mean(combined_data[l]["other"]) for l in sig_layers])
        sig_ratio = sig_self_mean / sig_other_mean if sig_other_mean > 1e-6 else float('inf')

        summary += f"""Significant Layers: {list(sig_layers)}

Mean effects (significant layers):
  Self:  {sig_self_mean:.3f}
  Other: {sig_other_mean:.3f}
  Ratio: {sig_ratio:.2f}x

"""
        if p_values:
            summary += "P-values (self > other):\n"
            for layer, direction, p in sorted(p_values):
                dir_label = "h→l" if "high" in direction else "l→h"
                sig_marker = "*" if p < 0.05 else ""
                summary += f"  L{layer} {dir_label}: p={p:.3f}{sig_marker}\n"

        summary += "\nInterpretation:\n"
        if sig_ratio > 1.5 and any(p < 0.05 for _, _, p in p_values):
            summary += """✓ Self > Other (significant)
  Patching primarily affects
  SELF-confidence (introspection)."""
        elif sig_ratio > 1.2:
            summary += """~ Self > Other (trend)
  Self-confidence affected more,
  but not significantly so."""
        elif sig_ratio > 0.8:
            summary += """— Similar effects
  Patching affects both self and
  other judgments similarly."""
        else:
            summary += """⚠ Other > Self
  Unexpected: patching affects
  other-confidence more."""
    else:
        summary += """No significant patching layers found
in main analysis. Cannot focus on
specific layers for comparison.

All Layers Summary:
"""
        all_self = np.mean([np.mean(combined_data[l]["self"]) for l in layers])
        all_other = np.mean([np.mean(combined_data[l]["other"]) for l in layers])
        all_ratio = all_self / all_other if all_other > 1e-6 else float('inf')
        summary += f"""  Mean |Self|:  {all_self:.3f}
  Mean |Other|: {all_other:.3f}
  Ratio: {all_ratio:.2f}x"""

    ax_summary.text(0.05, 0.95, summary, transform=ax_summary.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved other-confidence comparison plot to {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    global METRIC

    parser = argparse.ArgumentParser(description="Run activation patching experiments")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Metric for pair selection (default: {METRIC})")
    parser.add_argument("--n-pairs", type=int, default=NUM_PATCH_PAIRS,
                        help=f"Number of patch pairs per layer (default: {NUM_PATCH_PAIRS})")
    parser.add_argument("--method", type=str, default=PAIRING_METHOD,
                        choices=["extremes", "random", "quartile"],
                        help=f"Pairing method (default: {PAIRING_METHOD})")
    args = parser.parse_args()

    METRIC = args.metric
    n_pairs = args.n_pairs
    pairing_method = args.method

    print(f"Device: {DEVICE}")
    print(f"Metric: {METRIC}")
    print(f"Pairs per layer: {n_pairs}")
    print(f"Pairing method: {pairing_method}")
    print(f"Meta-judgment task: {META_TASK}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Paths for input data
    # We load from run_introspection_experiment.py outputs
    introspection_prefix = str(OUTPUTS_DIR / f"{get_model_short_name(BASE_MODEL_NAME)}_{DATASET_NAME}_introspection")
    if META_TASK == "delegate":
        introspection_prefix += "_delegate"

    paired_data_path = f"{introspection_prefix}_paired_data.json"
    direct_activations_path = f"{introspection_prefix}_direct_activations.npz"
    probe_results_path = f"{introspection_prefix}_{METRIC}_results.json"

    # Load paired data
    print(f"\nLoading paired data from {paired_data_path}...")
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)

    questions = paired_data["questions"]
    print(f"Loaded {len(questions)} questions")

    # Load metric values
    if "direct_metrics" in paired_data and METRIC in paired_data["direct_metrics"]:
        metric_values = np.array(paired_data["direct_metrics"][METRIC])
    else:
        raise ValueError(f"Metric {METRIC} not found in paired data")

    print(f"Metric range: [{metric_values.min():.3f}, {metric_values.max():.3f}]")

    # Load cached activations
    print(f"\nLoading cached activations from {direct_activations_path}...")
    acts_data = np.load(direct_activations_path)
    cached_activations = {
        int(k.split("_")[1]): acts_data[k]
        for k in acts_data.files if k.startswith("layer_")
    }
    print(f"Loaded {len(cached_activations)} layers, shape: {cached_activations[0].shape}")

    # Load probe results to select layers
    print(f"\nLoading probe results from {probe_results_path}...")
    with open(probe_results_path, "r") as f:
        probe_results = json.load(f)

    # Select layers with good direct→meta transfer
    layers = PATCHING_LAYERS
    if layers is None:
        layer_candidates = []
        if "probe_results" in probe_results:
            for layer_str, lr in probe_results["probe_results"].items():
                d2m_r2 = lr.get("direct_to_meta_fixed", {}).get("r2", 0)
                if d2m_r2 > 0.1:
                    layer_candidates.append((int(layer_str), d2m_r2))
        layer_candidates.sort(key=lambda x: -x[1])
        layers = [l[0] for l in layer_candidates[:10]]  # Top 10 layers

    if not layers:
        # Fallback: use middle-to-late layers
        all_layers = sorted(cached_activations.keys())
        n_layers = len(all_layers)
        layers = all_layers[n_layers // 3: 2 * n_layers // 3]

    layers = sorted(layers)
    print(f"Selected {len(layers)} layers: {layers}")

    # Load model
    print("\nLoading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
    )
    use_chat_template = should_use_chat_template(BASE_MODEL_NAME, tokenizer)
    initialize_token_cache(tokenizer)

    def json_serializer(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Run bidirectional patching
    all_results = {}
    all_analyses = {}

    for direction in ["high_to_low", "low_to_high"]:
        dir_label = "high→low" if direction == "high_to_low" else "low→high"
        print(f"\n{'='*60}")
        print(f"DIRECTION: {dir_label}")
        print(f"{'='*60}")

        # Create patch pairs for this direction
        print(f"\nCreating {n_pairs} patch pairs using '{pairing_method}' method ({direction})...")
        patch_pairs = create_patch_pairs(
            metric_values, n_pairs, method=pairing_method,
            direction=direction, seed=SEED
        )

        # Print pair statistics
        source_metrics = [metric_values[s] for s, t in patch_pairs]
        target_metrics = [metric_values[t] for s, t in patch_pairs]
        print(f"Source metric: mean={np.mean(source_metrics):.3f}, std={np.std(source_metrics):.3f}")
        print(f"Target metric: mean={np.mean(target_metrics):.3f}, std={np.std(target_metrics):.3f}")

        # Run patching experiment
        print(f"\nRunning patching experiment ({dir_label})...")
        results = run_patching_experiment(
            model, tokenizer, questions, cached_activations, metric_values,
            layers, patch_pairs, use_chat_template, BATCH_SIZE
        )
        results["direction"] = direction

        all_results[direction] = results

        # Save individual results
        results_path = f"{output_prefix}_{METRIC}_patching_{direction}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=json_serializer)
        print(f"\nSaved results to {results_path}")

        # Analyze
        analysis = analyze_patching_results(results)
        analysis["direction"] = direction
        all_analyses[direction] = analysis

        analysis_path = f"{output_prefix}_{METRIC}_patching_{direction}_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis to {analysis_path}")

    # Print combined summary
    print_bidirectional_summary(all_analyses)

    # Plot combined results
    plot_path = f"{output_prefix}_{METRIC}_patching_bidirectional.png"
    plot_bidirectional_results(all_analyses, all_results, plot_path)

    print("\n✓ Activation patching experiment complete!")

    # ==========================================================================
    # OTHER-CONFIDENCE CONTROL (for confidence task only)
    # ==========================================================================
    if META_TASK == "confidence":
        print("\n" + "=" * 60)
        print("OTHER-CONFIDENCE CONTROL EXPERIMENT")
        print("=" * 60)
        print("Testing whether patching affects self-confidence specifically,")
        print("or also affects general confidence-like judgments (human difficulty estimation).")

        # Find significant layers from main analysis (bidirectional: both directions positive + significant)
        significant_layers = []
        for layer in layers:
            h2l_effect = all_analyses["high_to_low"]["layer_effects"].get(str(layer), {})
            l2h_effect = all_analyses["low_to_high"]["layer_effects"].get(str(layer), {})

            h2l_sig = h2l_effect.get("significant_p05", False) and h2l_effect.get("mean_normalized_shift", 0) > 0.1
            l2h_sig = l2h_effect.get("significant_p05", False) and l2h_effect.get("mean_normalized_shift", 0) > 0.1

            # Count as significant if EITHER direction shows strong effect, or BOTH show moderate effect
            if (h2l_sig and l2h_sig) or \
               (h2l_effect.get("mean_normalized_shift", 0) > 0.2 and h2l_effect.get("significant_p05", False)) or \
               (l2h_effect.get("mean_normalized_shift", 0) > 0.2 and l2h_effect.get("significant_p05", False)):
                significant_layers.append(layer)

        if significant_layers:
            print(f"\nFocusing on significant layers from main analysis: {significant_layers}")
        else:
            print("\nNo strongly significant layers found, will analyze all layers.")

        other_confidence_results = {}
        other_confidence_analyses = {}

        for direction in ["high_to_low", "low_to_high"]:
            dir_label = "high→low" if direction == "high_to_low" else "low→high"
            print(f"\n--- Other-confidence: {dir_label} ---")

            # Recreate the same patch pairs used in the main experiment
            patch_pairs = create_patch_pairs(
                metric_values, n_pairs, method=pairing_method,
                direction=direction, seed=SEED
            )

            # Run other-confidence patching
            other_results = run_other_confidence_patching(
                model, tokenizer, questions, cached_activations,
                layers, patch_pairs, use_chat_template, BATCH_SIZE
            )

            if other_results is not None:
                other_confidence_results[direction] = other_results

                # Analyze self vs other effect (with significant layers marked)
                self_results = all_results[direction]
                analysis = analyze_other_confidence_patching_effect(
                    self_results, other_results, layers,
                    significant_layers=significant_layers
                )
                other_confidence_analyses[direction] = analysis

                # Print summary with statistics
                print(f"\n{dir_label} self vs other comparison:")
                print(f"{'Layer':<8} {'|Δself|':<10} {'|Δother|':<10} {'Diff':<10} {'95% CI':<18} {'p(self>other)':<14} {'Sig?':<5}")
                print("-" * 80)

                if analysis and analysis.get("layer_effects"):
                    for layer_str, effect in analysis["layer_effects"].items():
                        self_eff = effect["self_effect_mean_abs"]
                        other_eff = effect["other_effect_mean_abs"]
                        diff = effect["diff_mean"]
                        ci = effect["diff_ci95"]
                        p_val = effect["p_value_one_sided"]
                        is_sig = effect.get("is_significant_layer", False)

                        sig_marker = "*" if is_sig else ""
                        p_marker = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                        print(f"{layer_str + sig_marker:<8} {self_eff:<10.3f} {other_eff:<10.3f} {diff:<+10.3f} {ci_str:<18} {p_val:<14.4f} {p_marker:<5}")

                    if "mean_ratio" in analysis:
                        print(f"\n  Mean ratio (all layers): {analysis['mean_ratio']:.2f}x")

                    # Print significant layer summary if available
                    if analysis.get("significant_layer_summary"):
                        sig_summ = analysis["significant_layer_summary"]
                        print(f"  Significant layers only: self={sig_summ['mean_self_effect']:.3f}, other={sig_summ['mean_other_effect']:.3f}, ratio={sig_summ['mean_ratio']:.2f}x")

        # Overall assessment
        if other_confidence_analyses:
            all_ratios = []
            sig_ratios = []
            sig_p_values = []

            for dir_analysis in other_confidence_analyses.values():
                if dir_analysis and dir_analysis.get("layer_effects"):
                    for layer_str, e in dir_analysis["layer_effects"].items():
                        if not np.isinf(e["self_vs_other_ratio"]):
                            all_ratios.append(e["self_vs_other_ratio"])
                            if e.get("is_significant_layer", False):
                                sig_ratios.append(e["self_vs_other_ratio"])
                                sig_p_values.append(e["p_value_one_sided"])

            print(f"\n" + "=" * 60)
            print("OVERALL OTHER-CONFIDENCE CONTROL SUMMARY")
            print("=" * 60)

            if all_ratios:
                overall_ratio = np.mean(all_ratios)
                print(f"\nAll layers ({len(all_ratios)} measurements):")
                print(f"  Mean self/other ratio: {overall_ratio:.2f}x")

            if sig_ratios:
                sig_ratio = np.mean(sig_ratios)
                n_sig_tests = len(sig_p_values)
                n_sig_p05 = sum(1 for p in sig_p_values if p < 0.05)
                print(f"\nSignificant layers only ({len(sig_ratios)} measurements from layers {significant_layers}):")
                print(f"  Mean self/other ratio: {sig_ratio:.2f}x")
                print(f"  Tests with p<0.05 (self > other): {n_sig_p05}/{n_sig_tests}")

                # Overall interpretation based on significant layers
                print("\nInterpretation:")
                if sig_ratio > 1.5 and n_sig_p05 > 0:
                    print("  ✓ SELF-SPECIFIC: Patching primarily affects self-confidence")
                    print("    The causal effect is specific to introspection, not general confidence.")
                elif sig_ratio > 1.2:
                    print("  ~ TREND: Self-confidence affected more than other-confidence")
                    print("    But the difference may not be statistically significant.")
                elif sig_ratio > 0.8:
                    print("  — GENERAL EFFECT: Patching affects both similarly")
                    print("    The causal effect may not be introspection-specific.")
                else:
                    print("  ⚠ UNEXPECTED: Other-confidence affected more than self")

            elif all_ratios:
                # No significant layers, use all data
                overall_ratio = np.mean(all_ratios)
                print("\nInterpretation (based on all layers):")
                if overall_ratio > 2.0:
                    print("  → Patching primarily affects SELF-confidence (introspection-specific)")
                elif overall_ratio > 1.2:
                    print("  → Patching affects self-confidence more than other-confidence")
                elif overall_ratio > 0.8:
                    print("  → Patching affects self and other confidence similarly (general effect)")
                else:
                    print("  → Patching affects other-confidence more than self (unexpected)")

        # Save other-confidence results
        other_conf_path = f"{output_prefix}_{METRIC}_patching_other_confidence.json"
        with open(other_conf_path, "w") as f:
            json.dump({
                "results": other_confidence_results,
                "analyses": other_confidence_analyses,
                "significant_layers": significant_layers,
            }, f, indent=2, default=json_serializer)
        print(f"\nSaved other-confidence results to {other_conf_path}")

        # Plot other-confidence comparison
        if other_confidence_analyses:
            other_conf_plot_path = f"{output_prefix}_{METRIC}_patching_other_confidence.png"
            plot_other_confidence_comparison(
                other_confidence_analyses, all_analyses, other_conf_plot_path
            )


if __name__ == "__main__":
    main()
