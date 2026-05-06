"""
Ablation causality test for uncertainty directions.

Tests whether directions from identify_mc_correlate.py are causally necessary
for the model's meta-judgments. If ablating a direction degrades the correlation
between stated confidence and actual uncertainty, that's evidence the direction
is causally involved in introspection.

Key features:
- Tests ALL layers by default (no pre-filtering by transfer R²)
- Tests BOTH probe and mean_diff methods in a single run for comparison
- Uses pooled null distribution + FDR correction for robust statistics

Usage:
    python run_ablation_causality.py

Expects outputs from identify_mc_correlate.py (or build_mc_inputs_from_introspection.py):
    outputs/{INPUT_BASE_NAME}_mc_{DIRECTION_METRIC}_directions.npz
    outputs/{INPUT_BASE_NAME}_mc_dataset.json

DIRECTION_METRIC picks the direction file to ablate (e.g. "entropy" or
"stated_confidence"). TARGET_METRIC picks the field the per-question dataset
item is correlated against (e.g. "entropy"). They can differ — e.g. ablate the
stated_confidence direction during the meta pass while measuring whether the
resulting stated-confidence output still correlates with direct-pass entropy.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.model_selection import train_test_split

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from core.steering import generate_orthogonal_directions
from core.steering_experiments import (
    SteeringExperimentConfig,
    BatchAblationHook,
    pretokenize_prompts,
    build_padded_gpu_batches,
    get_kv_cache,
    create_fresh_cache,
    precompute_direction_tensors,
)
from core.metrics import metric_sign_for_confidence
from prompts import (
    format_stated_confidence_prompt,
    format_stated_confidence_prompt_base,
    get_stated_confidence_signal,
    format_answer_or_delegate_prompt,
    get_answer_or_delegate_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
    find_mc_positions,
    format_numeric_confidence_prompt,
    format_numeric_confidence_prompt_base,
    get_numeric_confidence_signal,
    NUMERIC_CONFIDENCE_OPTIONS,
)

# =============================================================================
# CONFIGURATION — edit values in experiment_config.AblationCausalityConfig
# =============================================================================
from experiment_config import AblationCausalityConfig as _C

MODEL = _C.MODEL
ADAPTER = _C.ADAPTER
INPUT_BASE_NAME = _C.INPUT_BASE_NAME
DIRECTION_METRIC = _C.DIRECTION_METRIC
TARGET_METRIC = _C.TARGET_METRIC
META_TASK = _C.META_TASK
CONFIDENCE_SCALE = _C.CONFIDENCE_SCALE
BASE_CONFIDENCE_FEW_SHOT_MODE = _C.BASE_CONFIDENCE_FEW_SHOT_MODE
CONFIDENCE_SIGNAL = _C.CONFIDENCE_SIGNAL
NUM_QUESTIONS = _C.NUM_QUESTIONS
NUM_CONTROLS = _C.NUM_CONTROLS
BATCH_SIZE = _C.BATCH_SIZE
SEED = _C.SEED
USE_TRANSFER_SPLIT = _C.USE_TRANSFER_SPLIT
TRAIN_SPLIT = _C.TRAIN_SPLIT
EXPANDED_BATCH_TARGET = _C.EXPANDED_BATCH_TARGET
LAYERS = _C.LAYERS
METHODS = list(_C.METHODS) if _C.METHODS is not None else None
PROBE_POSITIONS = list(_C.PROBE_POSITIONS)
PRINT_DELTA_DIAGNOSTICS = _C.PRINT_DELTA_DIAGNOSTICS
DELTA_DIAGNOSTIC_TOPK = _C.DELTA_DIAGNOSTIC_TOPK
BOOTSTRAP_N = _C.BOOTSTRAP_N
BOOTSTRAP_SEED = _C.BOOTSTRAP_SEED
BOOTSTRAP_CI_ALPHA = _C.BOOTSTRAP_CI_ALPHA
TRANSFER_R2_THRESHOLD = _C.TRANSFER_R2_THRESHOLD
TRANSFER_RESULTS_PATH = _C.TRANSFER_RESULTS_PATH
NUM_CONTROLS_NONFINAL = _C.NUM_CONTROLS_NONFINAL
LOAD_IN_4BIT = _C.LOAD_IN_4BIT
LOAD_IN_8BIT = _C.LOAD_IN_8BIT
OUTPUT_DIR = _C.OUTPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# TRANSFER RESULTS LOADING (for layer selection)
# =============================================================================

def load_transfer_results(base_name: str, meta_task: str) -> Optional[Dict]:
    """
    Load transfer results JSON to get per-layer R² values.

    Returns None if file not found.
    """
    path = TRANSFER_RESULTS_PATH
    if path is None:
        path = OUTPUT_DIR / f"{base_name}_transfer_{meta_task}_results.json"
    else:
        path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def get_layers_from_transfer(
    transfer_data: Dict,
    metric: str,
    position: str,
    r2_threshold: float,
    method: str = "probe",
) -> List[int]:
    """
    Get layers with transfer R² >= threshold for a given metric and position.

    Args:
        transfer_data: Loaded transfer results JSON
        metric: Which metric to check (e.g., "top_logit", "entropy")
        position: Token position (e.g., "final", "question_mark")
        r2_threshold: Minimum R² to include layer
        method: Direction method - "probe" uses transfer_by_position, "mean_diff" uses mean_diff_by_position

    Returns:
        Sorted list of layer indices meeting threshold
    """
    # Select the appropriate section based on method
    if method == "mean_diff":
        section_key = "mean_diff_by_position"
        legacy_key = None  # No legacy fallback for mean_diff
    else:
        section_key = "transfer_by_position"
        legacy_key = "transfer"

    # Try position-specific data first
    if section_key in transfer_data and position in transfer_data[section_key]:
        pos_data = transfer_data[section_key][position]
    elif legacy_key and legacy_key in transfer_data:
        # Fall back to legacy format (final position only, probe only)
        pos_data = transfer_data[legacy_key]
    else:
        return []

    if metric not in pos_data:
        return []

    metric_data = pos_data[metric]
    per_layer = metric_data.get("per_layer", {})

    selected = []
    for layer_str, layer_data in per_layer.items():
        # Check for centered R² (preferred) or d2m_centered_r2 (legacy)
        r2 = layer_data.get("centered_r2") or layer_data.get("d2m_centered_r2", 0)
        if r2 >= r2_threshold:
            selected.append(int(layer_str))

    return sorted(selected)


# =============================================================================
# DIRECTION LOADING
# =============================================================================

def load_directions(base_name: str, metric: str) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load all direction methods from npz file.

    Returns:
        Dict mapping method name -> {layer: direction_vector}
        e.g., {"probe": {0: arr, 1: arr, ...}, "mean_diff": {0: arr, 1: arr, ...}}
    """
    path = OUTPUT_DIR / f"{base_name}_mc_{metric}_directions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Directions file not found: {path}")

    data = np.load(path)

    methods: Dict[str, Dict[int, np.ndarray]] = {}
    for key in data.files:
        if key.startswith("_"):
            continue  # Skip metadata keys

        # Keys are like "probe_layer_0", "mean_diff_layer_5"
        parts = key.rsplit("_layer_", 1)
        if len(parts) != 2:
            continue

        method, layer_str = parts
        try:
            layer = int(layer_str)
        except ValueError:
            continue

        if method not in methods:
            methods[method] = {}

        # Normalize direction
        direction = data[key].astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        methods[method][layer] = direction

    return methods


def load_dataset(base_name: str) -> Dict:
    """Load dataset with questions and metric values."""
    path = OUTPUT_DIR / f"{base_name}_mc_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# META-TASK HELPERS
# =============================================================================

def _confidence_format_fn(question, tokenizer, use_chat_template=True):
    """Confidence-task prompt picker. Dispatches on CONFIDENCE_SCALE and whether
    the model wants a chat template (proxy for "is this a base model?").

    Matches the dispatch in run_introspection_experiment.format_meta_prompt /
    format_meta_prompt_base so the ablation-time prompt is identical to what
    the introspection pipeline used when producing the direction."""
    if CONFIDENCE_SCALE == "numeric":
        if use_chat_template:
            return format_numeric_confidence_prompt(question, tokenizer, use_chat_template)
        # Base model path: no chat template, few-shot exemplars.
        return format_numeric_confidence_prompt_base(
            question, mode=BASE_CONFIDENCE_FEW_SHOT_MODE, pool=None
        )
    # Letter (S-Z) scale
    if use_chat_template:
        return format_stated_confidence_prompt(question, tokenizer, use_chat_template)
    return format_stated_confidence_prompt_base(
        question, mode=BASE_CONFIDENCE_FEW_SHOT_MODE, pool=None
    )


def get_format_fn(meta_task: str):
    """Get prompt formatting function for meta-task."""
    if meta_task == "confidence":
        return _confidence_format_fn
    elif meta_task == "delegate":
        return format_answer_or_delegate_prompt
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_signal_fn(meta_task: str):
    """Get signal extraction function for meta-task.

    Returns a function with signature (probs, mapping) -> float.
    For confidence task, mapping is ignored.
    """
    if meta_task == "confidence":
        if CONFIDENCE_SCALE == "numeric":
            return lambda p, m: get_numeric_confidence_signal(p)
        return lambda p, m: get_stated_confidence_signal(p)
    elif meta_task == "delegate":
        return get_answer_or_delegate_signal
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_options(meta_task: str) -> List[str]:
    """Get response options for meta-task."""
    if meta_task == "confidence":
        if CONFIDENCE_SCALE == "numeric":
            return list(NUMERIC_CONFIDENCE_OPTIONS.keys())
        return list(STATED_CONFIDENCE_OPTIONS.keys())
    elif meta_task == "delegate":
        return ANSWER_OR_DELEGATE_OPTIONS
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


# =============================================================================
# ABLATION EXPERIMENT
# =============================================================================


# -----------------------------------------------------------------------------
# Confidence signal helpers
# -----------------------------------------------------------------------------
def _extract_probs_logits(out, option_token_ids):
    """Return (probs, logits_np) over the option tokens at the final position."""
    logits = out.logits[:, -1, :][:, option_token_ids]
    logits_np = logits.detach().float().cpu().numpy()
    probs = torch.softmax(logits, dim=-1).detach().float().cpu().numpy()
    return probs, logits_np

def _compute_confidence_used(meta_task: str, probs_row, logits_row, mapping, signal_fn):
    """Return (confidence_used, p_answer, logit_margin)."""
    if meta_task == "delegate":
        # mapping maps "1"/"2" -> "Answer"/"Delegate"
        ans_idx = 0 if mapping.get("1") == "Answer" else 1
        del_idx = 1 - ans_idx
        p_answer = float(probs_row[ans_idx])
        logit_margin = float(logits_row[ans_idx] - logits_row[del_idx])
        sig = str(CONFIDENCE_SIGNAL).lower()
        if sig in {"logit_margin", "margin", "logitdiff", "logit_diff"}:
            return logit_margin, p_answer, logit_margin
        return p_answer, p_answer, logit_margin
    # confidence task: keep the original probability-based signal
    if str(CONFIDENCE_SIGNAL).lower() in {"logit_margin", "margin", "logitdiff", "logit_diff"}:
        # be explicit to avoid silent confusion
        import warnings
        warnings.warn("CONFIDENCE_SIGNAL=logit_margin is only defined for META_TASK=delegate; falling back to prob.")
    conf = float(signal_fn(probs_row, mapping))
    return conf, None, None

def _summarize_dist(values):
    import numpy as _np
    arr = _np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(_np.mean(arr)),
        "std": float(_np.std(arr)),
        "q05": float(_np.quantile(arr, 0.05)),
        "q50": float(_np.quantile(arr, 0.50)),
        "q95": float(_np.quantile(arr, 0.95)),
        "min": float(_np.min(arr)),
        "max": float(_np.max(arr)),
    }

def _report_signal_distributions(results: dict, meta_task: str):
    try:
        layer0 = results["layers"][0]
        baseline_list = results["layer_results"][layer0]["baseline"]
    except Exception:
        return
    conf_vals = [d.get("confidence") for d in baseline_list if isinstance(d, dict) and d.get("confidence") is not None]
    p_answer_vals = [d.get("p_answer") for d in baseline_list if isinstance(d, dict) and d.get("p_answer") is not None]
    margin_vals = [d.get("logit_margin") for d in baseline_list if isinstance(d, dict) and d.get("logit_margin") is not None]
    print() 
    print("  Signal distributions (baseline):")
    if meta_task == "delegate":
        s = _summarize_dist(p_answer_vals)
        if s.get("n", 0) > 0:
            print(f"    P(Answer): n={s['n']} mean={s['mean']:+.4f} std={s['std']:.4f} q05={s['q05']:+.4f} q50={s['q50']:+.4f} q95={s['q95']:+.4f} min={s['min']:+.4f} max={s['max']:+.4f}")
        s = _summarize_dist(margin_vals)
        if s.get("n", 0) > 0:
            print(f"    logit_margin: n={s['n']} mean={s['mean']:+.4f} std={s['std']:.4f} q05={s['q05']:+.4f} q50={s['q50']:+.4f} q95={s['q95']:+.4f} min={s['min']:+.4f} max={s['max']:+.4f}")
    s = _summarize_dist(conf_vals)
    if s.get("n", 0) > 0:
        print(f"    confidence (used): n={s['n']} mean={s['mean']:+.4f} std={s['std']:.4f} q05={s['q05']:+.4f} q50={s['q50']:+.4f} q95={s['q95']:+.4f} min={s['min']:+.4f} max={s['max']:+.4f}")

def run_ablation_for_method(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    directions: Dict[int, np.ndarray],
    num_controls: int,
    meta_task: str,
    use_chat_template: bool,
    layers: Optional[List[int]] = None,
    position: str = "final",
    original_indices: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run ablation experiment for a single direction method at a specific position.

    Uses batched ablation when EXPANDED_BATCH_TARGET is set: multiple directions
    are ablated in a single forward pass by expanding the batch.

    For position="final", uses KV cache optimization.
    For other positions, uses full forward passes with indexed ablation.

    Args:
        position: Token position to ablate at. One of PROBE_POSITIONS:
            - "final": Last token (uses KV cache)
            - "question_mark": Token after "?" in question
            - "question_newline": Newline after question
            - "options_newline": Newline after MC options
        original_indices: Original dataset indices for each question. Used for
            trial_index in delegate task to match prompt formatting with
            test_meta_transfer.py. If None, uses local indices (legacy behavior).

    Returns dict with per-layer results including baseline, ablated, and controls.
    """
    if layers is None:
        layers = sorted(directions.keys())
    else:
        layers = [l for l in layers if l in directions]

    if not layers:
        return {"error": "No layers to test"}

    metric_mean = float(np.mean(metric_values))
    metric_std = float(np.std(metric_values))
    if metric_std < 1e-10:
        metric_std = 1.0

    # Get formatting functions and options
    format_fn = get_format_fn(meta_task)
    signal_fn = get_signal_fn(meta_task)
    options = get_options(meta_task)

    # Tokenize options
    option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options
    ]

    # Format prompts and find token positions
    prompts = []
    mappings = []
    position_indices = []  # Per-prompt token index for intervention
    for q_idx, question in enumerate(questions):
        # Use original dataset index for trial_index to match test_meta_transfer.py
        # This ensures the delegate task uses consistent Answer/Delegate mapping
        trial_idx = int(original_indices[q_idx]) if original_indices is not None else q_idx
        if meta_task == "delegate":
            prompt, _, mapping = format_fn(question, tokenizer, trial_index=trial_idx, use_chat_template=use_chat_template)
        else:
            prompt, _ = format_fn(question, tokenizer, use_chat_template=use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

        # Find token positions for this prompt
        positions = find_mc_positions(prompt, tokenizer, question)
        pos_idx = positions.get(position, -1)
        position_indices.append(pos_idx)

    # Warn if some positions weren't found (will fall back to final token)
    # Note: "final" position is always -1 by design, so don't warn for it
    if position != "final":
        n_valid = sum(1 for idx in position_indices if idx >= 0)
        n_total = len(position_indices)
        if n_valid < n_total:
            print(f"  Warning: {position} position found for {n_valid}/{n_total} prompts (others fall back to final)")

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Check if we can use KV cache (only for "final" position)
    use_kv_cache = (position == "final")

    # Generate control directions for each layer
    print(f"  Generating {num_controls} control directions per layer...")
    controls_by_layer = {}
    for layer in layers:
        controls_by_layer[layer] = generate_orthogonal_directions(
            directions[layer], num_controls, seed=SEED + layer
        )

    # Precompute direction tensors
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    cached_directions = {}
    for layer in layers:
        dir_tensor = torch.tensor(directions[layer], dtype=dtype, device=DEVICE)
        ctrl_tensors = [torch.tensor(c, dtype=dtype, device=DEVICE) for c in controls_by_layer[layer]]
        # Stack all directions: [primary, control_0, control_1, ..., control_N-1]
        all_dirs = torch.stack([dir_tensor] + ctrl_tensors, dim=0)  # (1 + num_controls, hidden_dim)
        cached_directions[layer] = {
            "direction": dir_tensor,
            "controls": ctrl_tensors,
            "all_stacked": all_dirs,
        }

    # Initialize results
    baseline_results = [None] * len(questions)
    layer_results = {}
    for layer in layers:
        layer_results[layer] = {
            "baseline": baseline_results,
            "ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    # Determine batching strategy
    total_directions = 1 + num_controls  # primary + controls
    if EXPANDED_BATCH_TARGET is not None and EXPANDED_BATCH_TARGET > 0:
        directions_per_pass = max(1, EXPANDED_BATCH_TARGET // BATCH_SIZE)
        directions_per_pass = min(directions_per_pass, total_directions)
        use_batched = directions_per_pass > 1
    else:
        directions_per_pass = 1
        use_batched = False

    # Calculate number of passes (same formula for both paths)
    num_passes = (total_directions + directions_per_pass - 1) // directions_per_pass if use_batched else total_directions

    if use_kv_cache:
        # KV cache path: efficient but only works for final position
        if use_batched:
            print(f"  Batched ablation (KV cache): {directions_per_pass} directions per pass, {num_passes} passes per layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (KV cache): 1 direction per pass")
            total_forward_passes = len(gpu_batches) * len(layers) * total_directions
    else:
        # Full forward path: required for non-final positions (also supports batching)
        if use_batched:
            print(f"  Batched ablation (full forward) at '{position}': {directions_per_pass} dirs/pass, {num_passes} passes/layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (full forward) at '{position}': {total_directions} directions per layer")
            total_forward_passes = len(gpu_batches) * len(layers) * total_directions

    print(f"  Total forward passes: {total_forward_passes}")

    pbar = tqdm(total=total_forward_passes, desc=f"  Ablation ({position})")

    for batch_idx, (batch_indices, batch_inputs) in enumerate(gpu_batches):
        B = len(batch_indices)

        if use_kv_cache:
            # KV cache path (position == "final")
            base_step_data = get_kv_cache(model, batch_inputs)
            keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

            inputs_template = {
                "input_ids": base_step_data["input_ids"],
                "attention_mask": base_step_data["attention_mask"],
                "use_cache": True
            }
            if "position_ids" in base_step_data:
                inputs_template["position_ids"] = base_step_data["position_ids"]

            # Compute baseline (no ablation)
            if baseline_results[batch_indices[0]] is None:
                fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                baseline_inputs = inputs_template.copy()
                baseline_inputs["past_key_values"] = fresh_cache

                with torch.inference_mode():
                    out = model(**baseline_inputs)
                    probs, logits_np = _extract_probs_logits(out, option_token_ids)

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                    m_val = metric_values[q_idx]
                    baseline_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": resp,
                        "confidence": float(conf),
                        "metric": float(m_val),
                        "p_answer": (float(p_answer) if p_answer is not None else None),
                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                    }

            # Run ablation for each layer (KV cache path)
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                all_dirs = cached_directions[layer]["all_stacked"]
                hook = BatchAblationHook()
                hook.register(layer_module)

                try:
                    if use_batched:
                        for pass_start in range(0, total_directions, directions_per_pass):
                            pass_end = min(pass_start + directions_per_pass, total_directions)
                            k_dirs = pass_end - pass_start

                            expanded_input_ids = inputs_template["input_ids"].repeat_interleave(k_dirs, dim=0)
                            expanded_attention_mask = inputs_template["attention_mask"].repeat_interleave(k_dirs, dim=0)
                            expanded_inputs = {
                                "input_ids": expanded_input_ids,
                                "attention_mask": expanded_attention_mask,
                                "use_cache": True
                            }
                            if "position_ids" in inputs_template:
                                expanded_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(k_dirs, dim=0)

                            pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_dirs)
                            expanded_inputs["past_key_values"] = pass_cache

                            dirs_for_pass = all_dirs[pass_start:pass_end]
                            dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1).reshape(B * k_dirs, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**expanded_inputs)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
                                    prob_idx = i * k_dirs + j
                                    p = probs[prob_idx]
                                    resp = options[np.argmax(p)]
                                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[prob_idx], mappings[q_idx], signal_fn)
                                    m_val = metric_values[q_idx]
                                    data = {
                                        "question_idx": q_idx,
                                        "response": resp,
                                        "confidence": float(conf),
                                        "metric": float(m_val),
                                        "p_answer": (float(p_answer) if p_answer is not None else None),
                                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                    }
                                    if dir_idx == 0:
                                        layer_results[layer]["ablated"][q_idx] = data
                                    else:
                                        ctrl_key = f"control_{dir_idx - 1}"
                                        layer_results[layer]["controls_ablated"][ctrl_key][q_idx] = data

                            pbar.update(1)
                    else:
                        # Sequential KV cache path
                        def run_single_ablation_kv(direction_tensor, result_list, key=None):
                            pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                            current_inputs = inputs_template.copy()
                            current_inputs["past_key_values"] = pass_cache

                            dirs_batch = direction_tensor.unsqueeze(0).expand(B, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**current_inputs)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                p = probs[i]
                                resp = options[np.argmax(p)]
                                conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                                m_val = metric_values[q_idx]
                                data = {
                                    "question_idx": q_idx,
                                    "response": resp,
                                    "confidence": float(conf),
                                    "metric": float(m_val),
                                    "p_answer": (float(p_answer) if p_answer is not None else None),
                                    "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                }
                                if key:
                                    result_list[key][q_idx] = data
                                else:
                                    result_list[q_idx] = data
                            pbar.update(1)

                        run_single_ablation_kv(cached_directions[layer]["direction"], layer_results[layer]["ablated"])
                        for i_c, ctrl_dir in enumerate(cached_directions[layer]["controls"]):
                            run_single_ablation_kv(ctrl_dir, layer_results[layer]["controls_ablated"], key=f"control_{i_c}")
                finally:
                    hook.remove()

        else:
            # Full forward path (position != "final")
            # Build position indices for this batch (adjusted for left-padding)
            batch_pos_indices = []
            seq_len = batch_inputs["input_ids"].shape[1]
            for i, q_idx in enumerate(batch_indices):
                pos = position_indices[q_idx]
                if pos >= 0:
                    # Adjust for left-padding
                    actual_len = int(batch_inputs["attention_mask"][i].sum())
                    pad_offset = seq_len - actual_len
                    adjusted_pos = pos + pad_offset
                else:
                    adjusted_pos = seq_len - 1  # fallback to final
                batch_pos_indices.append(adjusted_pos)
            batch_pos_tensor = torch.tensor(batch_pos_indices, dtype=torch.long, device=DEVICE)

            # Compute baseline (no ablation) - full forward
            if baseline_results[batch_indices[0]] is None:
                with torch.inference_mode():
                    out = model(**batch_inputs, use_cache=False)
                    probs, logits_np = _extract_probs_logits(out, option_token_ids)

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                    m_val = metric_values[q_idx]
                    baseline_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": resp,
                        "confidence": float(conf),
                        "metric": float(m_val),
                        "p_answer": (float(p_answer) if p_answer is not None else None),
                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                    }

            # Run ablation for each layer (full forward path with batched directions)
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                all_dirs = cached_directions[layer]["all_stacked"]

                if use_batched:
                    # Batched ablation: expand batch by k_dirs directions per pass
                    for pass_start in range(0, total_directions, directions_per_pass):
                        pass_end = min(pass_start + directions_per_pass, total_directions)
                        k_dirs = pass_end - pass_start

                        # Expand inputs by k_dirs
                        expanded_input_ids = batch_inputs["input_ids"].repeat_interleave(k_dirs, dim=0)
                        expanded_attention_mask = batch_inputs["attention_mask"].repeat_interleave(k_dirs, dim=0)
                        expanded_inputs = {
                            "input_ids": expanded_input_ids,
                            "attention_mask": expanded_attention_mask,
                        }

                        # Expand position indices to match expanded batch
                        expanded_pos_tensor = batch_pos_tensor.repeat_interleave(k_dirs)

                        # Build direction tensor: (B * k_dirs, hidden_dim)
                        dirs_for_pass = all_dirs[pass_start:pass_end]
                        dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1).reshape(B * k_dirs, -1)

                        hook = BatchAblationHook(intervention_position="indexed")
                        hook.set_position_indices(expanded_pos_tensor)
                        hook.set_directions(dirs_batch)
                        hook.register(layer_module)

                        try:
                            with torch.inference_mode():
                                out = model(**expanded_inputs, use_cache=False)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            # Store results
                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
                                    prob_idx = i * k_dirs + j
                                    p = probs[prob_idx]
                                    resp = options[np.argmax(p)]
                                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[prob_idx], mappings[q_idx], signal_fn)
                                    m_val = metric_values[q_idx]
                                    data = {
                                        "question_idx": q_idx,
                                        "response": resp,
                                        "confidence": float(conf),
                                        "metric": float(m_val),
                                        "p_answer": (float(p_answer) if p_answer is not None else None),
                                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                    }
                                    if dir_idx == 0:
                                        layer_results[layer]["ablated"][q_idx] = data
                                    else:
                                        ctrl_key = f"control_{dir_idx - 1}"
                                        layer_results[layer]["controls_ablated"][ctrl_key][q_idx] = data
                        finally:
                            hook.remove()

                        pbar.update(1)
                else:
                    # Sequential ablation (one direction per pass)
                    hook = BatchAblationHook(intervention_position="indexed")
                    hook.set_position_indices(batch_pos_tensor)
                    hook.register(layer_module)

                    try:
                        # Primary direction
                        dirs_batch = cached_directions[layer]["direction"].unsqueeze(0).expand(B, -1)
                        hook.set_directions(dirs_batch)

                        with torch.inference_mode():
                            out = model(**batch_inputs, use_cache=False)
                            probs, logits_np = _extract_probs_logits(out, option_token_ids)

                        for i, q_idx in enumerate(batch_indices):
                            p = probs[i]
                            resp = options[np.argmax(p)]
                            conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                            m_val = metric_values[q_idx]
                            layer_results[layer]["ablated"][q_idx] = {
                                "question_idx": q_idx,
                                "response": resp,
                                "confidence": float(conf),
                                "metric": float(m_val),
                                "p_answer": (float(p_answer) if p_answer is not None else None),
                                "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                            }
                        pbar.update(1)

                        # Control directions
                        for i_c, ctrl_dir in enumerate(cached_directions[layer]["controls"]):
                            dirs_batch = ctrl_dir.unsqueeze(0).expand(B, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**batch_inputs, use_cache=False)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                p = probs[i]
                                resp = options[np.argmax(p)]
                                conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                                m_val = metric_values[q_idx]
                                layer_results[layer]["controls_ablated"][f"control_{i_c}"][q_idx] = {
                                    "question_idx": q_idx,
                                    "response": resp,
                                    "confidence": float(conf),
                                    "metric": float(m_val),
                                    "p_answer": (float(p_answer) if p_answer is not None else None),
                                    "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                }
                            pbar.update(1)
                    finally:
                        hook.remove()

    pbar.close()
    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": layer_results,
        "position": position,
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and metric."""
    if len(confidences) < 2 or np.std(confidences) < 1e-10 or np.std(metric_values) < 1e-10:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman (rank) correlation."""
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(spearmanr(x, y).correlation)



def _bh_fdr(pvals_by_layer: Dict[int, float]) -> Dict[int, float]:
    """Benjamini-Hochberg FDR correction.

    Args:
        pvals_by_layer: mapping layer->raw p

    Returns:
        mapping layer->FDR-adjusted p
    """
    items = sorted(pvals_by_layer.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 0:
        return {}

    adj = {}
    for rank, (layer, p) in enumerate(items, 1):
        adj[layer] = min(1.0, (p * n) / rank)

    # enforce monotonicity (non-decreasing in the sorted order)
    prev = 0.0
    for layer, p in items:
        if adj[layer] < prev:
            adj[layer] = prev
        prev = adj[layer]
    return adj


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Vectorized bootstrap Pearson correlation for many resamples.

    x, y are shape (n,). idx is shape (B, n) integer indices.

    Returns: shape (B,) correlations (0.0 where variance degenerates).
    """
    n = x.shape[0]
    if n < 2:
        return np.zeros(idx.shape[0], dtype=np.float32)

    X = x[idx]
    Y = y[idx]

    # center
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)

    denom_n = float(n - 1)
    cov = (Xc * Yc).sum(axis=1) / denom_n
    sx = np.sqrt((Xc * Xc).sum(axis=1) / denom_n)
    sy = np.sqrt((Yc * Yc).sum(axis=1) / denom_n)

    denom = sx * sy
    out = np.zeros_like(cov, dtype=np.float32)
    ok = denom > 1e-12
    out[ok] = (cov[ok] / denom[ok]).astype(np.float32)
    return out


def analyze_ablation_results(results: Dict, metric: str) -> Dict:
    """Compute ablation effect statistics.

    Adds **bootstrap CIs + bootstrap BH-FDR** over questions (cheap, no extra model runs),
    and (when controls exist) retains the existing pooled-control null stats.
    """
    layers = results.get("layers", [])
    num_controls = results.get("num_controls", 0)

    metric_sign = metric_sign_for_confidence(metric)

    analysis = {
        "confidence_signal": results.get("confidence_signal", CONFIDENCE_SIGNAL),
        "layers": layers,
        "num_questions": results.get("num_questions", 0),
        "num_controls": num_controls,
        "metric": metric,
        "metric_sign": metric_sign,
        "bootstrap": {
            "n": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "ci_alpha": BOOTSTRAP_CI_ALPHA,
        },
        "per_layer": {},
    }

    if not layers:
        analysis["summary"] = {
            "pooled_null_size": 0,
            "n_significant_fdr": 0,
            "n_significant_bootstrap_fdr": 0,
            "best_layer": None,
            "best_effect_z": 0.0,
            "best_abs_delta": 0.0,
        }
        return analysis

    # --- Pull baseline arrays once (baseline is identical across layers for a given run) ---
    first_layer = layers[0]
    lr0 = results["layer_results"][first_layer]
    baseline_conf = np.array([r["confidence"] for r in lr0["baseline"]], dtype=np.float32)
    baseline_metric = np.array([r["metric"] for r in lr0["baseline"]], dtype=np.float32)

    # Baseline point estimate
    baseline_corr_point = compute_correlation(baseline_conf, baseline_metric)

    # Bootstrap index matrix (shared across layers)
    n_q = baseline_conf.shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n_q, size=(BOOTSTRAP_N, n_q), dtype=np.int32)

    # Bootstrap baseline corr (shared)
    boot_base = _bootstrap_corr(baseline_conf, baseline_metric, idx)
    lo = BOOTSTRAP_CI_ALPHA / 2.0
    hi = 1.0 - lo
    base_ci = np.quantile(boot_base, [lo, hi]).astype(np.float32)

    # We'll also need signed metric for Δconf diagnostics
    metric_signed = baseline_metric * float(metric_sign)

    # --- If controls exist, build pooled null of corr changes ---
    pooled_null = []

    # First pass: compute per-layer stats and collect pooled null
    layer_data = {}

    for layer in layers:
        lr = results["layer_results"][layer]

        ablated_conf = np.array([r["confidence"] for r in lr["ablated"]], dtype=np.float32)
        ablated_metric = np.array([r["metric"] for r in lr["ablated"]], dtype=np.float32)

        # Point estimates
        baseline_corr = baseline_corr_point
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)
        corr_change = ablated_corr - baseline_corr

        # --- Bootstrap CIs (sampling uncertainty) ---
        boot_ablt = _bootstrap_corr(ablated_conf, baseline_metric, idx)
        boot_delta = boot_ablt - boot_base

        ablt_ci = np.quantile(boot_ablt, [lo, hi]).astype(np.float32)
        delta_ci = np.quantile(boot_delta, [lo, hi]).astype(np.float32)

        # Two-sided "bootstrap sign" p-value for Δcorr != 0
        # (Equivalent to whether 0 is in the bootstrap distribution mass.)
        frac_ge0 = float(np.mean(boot_delta >= 0.0))
        frac_le0 = float(np.mean(boot_delta <= 0.0))
        p_boot = float(min(1.0, 2.0 * min(frac_ge0, frac_le0)))

        # --- Control ablations (null based on random orthogonal directions) ---
        control_corrs = []
        control_corr_changes = []
        control_delta_corrs = []

        if num_controls > 0 and lr.get("controls_ablated"):
            for ctrl_key, ctrl_list in lr["controls_ablated"].items():
                ctrl_conf = np.array([r["confidence"] for r in ctrl_list], dtype=np.float32)
                ctrl_metric = np.array([r["metric"] for r in ctrl_list], dtype=np.float32)
                c_corr = compute_correlation(ctrl_conf, ctrl_metric)
                control_corrs.append(c_corr)
                control_corr_changes.append(c_corr - baseline_corr)

                # Δconf diagnostics: corr(Δconf, signed metric)
                delta_ctrl = ctrl_conf - baseline_conf
                control_delta_corrs.append(compute_correlation(delta_ctrl, metric_signed))

            pooled_null.extend(control_corr_changes)

        # --- Δconf diagnostics (primary) ---
        delta_conf = ablated_conf - baseline_conf
        delta_conf_mean = float(np.mean(delta_conf))
        delta_conf_std = float(np.std(delta_conf))

        delta_corr_metric = compute_correlation(delta_conf, metric_signed)
        delta_spearman_metric = compute_spearman(delta_conf, metric_signed)

        if np.std(baseline_conf) > 1e-10:
            affine_slope, affine_intercept = np.polyfit(baseline_conf, ablated_conf, 1)
        else:
            affine_slope, affine_intercept = 0.0, float(np.mean(ablated_conf))

        baseline_to_ablated_corr = compute_correlation(baseline_conf, ablated_conf)
        resid = ablated_conf - (affine_slope * baseline_conf + affine_intercept)
        residual_corr_metric = compute_correlation(resid, metric_signed)

        pooled_delta_corr = np.array(control_delta_corrs, dtype=np.float32)
        if pooled_delta_corr.size > 0:
            n_worse = int(np.sum(np.abs(pooled_delta_corr) >= abs(delta_corr_metric)))
            p_value_delta_corr_pooled = float((n_worse + 1) / (pooled_delta_corr.size + 1))
            ctrl_delta_mean = float(np.mean(pooled_delta_corr))
            ctrl_delta_std = float(np.std(pooled_delta_corr))
        else:
            p_value_delta_corr_pooled = 1.0
            ctrl_delta_mean = 0.0
            ctrl_delta_std = 0.0

        # Mean Δconf by metric decile
        if np.std(metric_signed) < 1e-10:
            delta_by_decile = [None] * 10
        else:
            edges = np.quantile(metric_signed, np.linspace(0, 1, 11))
            if np.unique(edges).size < 3:
                delta_by_decile = [None] * 10
            else:
                bin_idx = np.digitize(metric_signed, edges[1:-1], right=True)  # 0..9
                delta_by_decile = [
                    float(np.mean(delta_conf[bin_idx == k])) if np.any(bin_idx == k) else None
                    for k in range(10)
                ]

        # Control summary stats (if any)
        if control_corrs:
            ctrl_corr_mean = float(np.mean(control_corrs))
            ctrl_corr_std = float(np.std(control_corrs))
            ctrl_change_mean = float(np.mean(control_corr_changes))
            ctrl_change_std = float(np.std(control_corr_changes))
        else:
            ctrl_corr_mean = baseline_corr
            ctrl_corr_std = 0.0
            ctrl_change_mean = 0.0
            ctrl_change_std = 0.0

        # Effect size vs controls (if any)
        if ctrl_change_std > 1e-10:
            effect_size_z = float((corr_change - ctrl_change_mean) / ctrl_change_std)
            p_value_parametric = float(2 * norm.sf(abs(effect_size_z)))
        else:
            effect_size_z = 0.0
            p_value_parametric = 1.0

        layer_data[layer] = {
            "baseline_corr": baseline_corr,
            "ablated_corr": ablated_corr,
            "corr_change": corr_change,

            # Bootstrap
            "baseline_corr_ci95": [float(base_ci[0]), float(base_ci[1])],
            "ablated_corr_ci95": [float(ablt_ci[0]), float(ablt_ci[1])],
            "delta_corr_ci95": [float(delta_ci[0]), float(delta_ci[1])],
            "p_value_bootstrap_delta": p_boot,

            # Confidence means
            "baseline_conf_mean": float(np.mean(baseline_conf)),
            "ablated_conf_mean": float(np.mean(ablated_conf)),

            # Controls
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
            "control_corr_mean": ctrl_corr_mean,
            "control_corr_std": ctrl_corr_std,
            "control_change_mean": ctrl_change_mean,
            "control_change_std": ctrl_change_std,
            "effect_size_z": float(effect_size_z),
            "p_value_parametric": float(p_value_parametric),

            # Δconf diagnostics
            "delta_conf_mean": delta_conf_mean,
            "delta_conf_std": delta_conf_std,
            "delta_conf_corr_metric": float(delta_corr_metric),
            "delta_conf_spearman_metric": float(delta_spearman_metric),
            "baseline_to_ablated_conf_corr": float(baseline_to_ablated_corr),
            "affine_slope": float(affine_slope),
            "affine_intercept": float(affine_intercept),
            "residual_corr_metric": float(residual_corr_metric),
            "control_delta_conf_corr_metric_mean": ctrl_delta_mean,
            "control_delta_conf_corr_metric_std": ctrl_delta_std,
            "p_value_delta_corr_pooled": float(p_value_delta_corr_pooled),
            "delta_conf_mean_by_metric_decile": delta_by_decile,
        }

    pooled_null = np.array(pooled_null, dtype=np.float32)

    # Second pass: p-values from pooled-null controls (if controls exist)
    raw_p_controls = {}
    for layer in layers:
        ld = layer_data[layer]
        if pooled_null.size > 0:
            n_worse = int(np.sum(np.abs(pooled_null) >= abs(ld["corr_change"])))
            p_val = float((n_worse + 1) / (pooled_null.size + 1))
        else:
            p_val = 1.0
        raw_p_controls[layer] = p_val

    fdr_controls = _bh_fdr(raw_p_controls)

    # Bootstrap BH-FDR
    raw_p_boot = {layer: layer_data[layer]["p_value_bootstrap_delta"] for layer in layers}
    fdr_boot = _bh_fdr(raw_p_boot)

    # Populate analysis[per_layer]
    for layer in layers:
        ld = layer_data[layer]
        analysis["per_layer"][layer] = {
            "baseline_correlation": ld["baseline_corr"],
            "ablated_correlation": ld["ablated_corr"],
            "correlation_change": ld["corr_change"],

            # Bootstrap
            "baseline_corr_ci95": ld["baseline_corr_ci95"],
            "ablated_corr_ci95": ld["ablated_corr_ci95"],
            "delta_corr_ci95": ld["delta_corr_ci95"],
            "p_value_bootstrap_delta": float(ld["p_value_bootstrap_delta"]),
            "p_value_bootstrap_fdr": float(fdr_boot.get(layer, 1.0)),

            # Controls (legacy)
            "control_correlation_mean": float(ld["control_corr_mean"]),
            "control_correlation_std": float(ld["control_corr_std"]),
            "control_correlation_change_mean": float(ld["control_change_mean"]),
            "control_correlation_change_std": float(ld["control_change_std"]),
            "p_value_pooled": float(raw_p_controls[layer]),
            "p_value_fdr": float(fdr_controls.get(layer, 1.0)),
            "p_value_parametric": float(ld["p_value_parametric"]),
            "effect_size_z": float(ld["effect_size_z"]),

            # Δconf diagnostics
            "baseline_confidence_mean": ld["baseline_conf_mean"],
            "ablated_confidence_mean": ld["ablated_conf_mean"],
            "delta_conf_mean": ld["delta_conf_mean"],
            "delta_conf_std": ld["delta_conf_std"],
            "delta_conf_corr_metric": ld["delta_conf_corr_metric"],
            "delta_conf_spearman_metric": ld["delta_conf_spearman_metric"],
            "baseline_to_ablated_conf_corr": ld["baseline_to_ablated_conf_corr"],
            "affine_slope": ld["affine_slope"],
            "affine_intercept": ld["affine_intercept"],
            "residual_corr_metric": ld["residual_corr_metric"],
            "control_delta_conf_corr_metric_mean": ld["control_delta_conf_corr_metric_mean"],
            "control_delta_conf_corr_metric_std": ld["control_delta_conf_corr_metric_std"],
            "p_value_delta_corr_pooled": ld["p_value_delta_corr_pooled"],
            "delta_conf_mean_by_metric_decile": ld["delta_conf_mean_by_metric_decile"],
        }

    # Summary
    per = analysis["per_layer"]

    sig_controls_fdr = [l for l in layers if per[l]["p_value_fdr"] < 0.05]
    sig_boot_fdr = [l for l in layers if per[l]["p_value_bootstrap_fdr"] < 0.05]

    best_layer_z = max(layers, key=lambda l: abs(per[l]["effect_size_z"]))
    best_layer_abs_delta = max(layers, key=lambda l: abs(per[l]["correlation_change"]))

    analysis["summary"] = {
        "pooled_null_size": int(pooled_null.size),
        "significant_layers_fdr": sig_controls_fdr,
        "n_significant_fdr": len(sig_controls_fdr),
        "significant_layers_bootstrap_fdr": sig_boot_fdr,
        "n_significant_bootstrap_fdr": len(sig_boot_fdr),
        "best_layer": int(best_layer_z),
        "best_effect_z": float(per[best_layer_z]["effect_size_z"]),
        "best_layer_abs_delta": int(best_layer_abs_delta),
        "best_abs_delta": float(per[best_layer_abs_delta]["correlation_change"]),
    }

    # Optional: print Δconf diagnostics for biggest +Δcorr and biggest -Δcorr
    if PRINT_DELTA_DIAGNOSTICS and len(layers) > 0:
        top_inc = sorted(layers, key=lambda l: per[l]["correlation_change"], reverse=True)[:DELTA_DIAGNOSTIC_TOPK]
        top_dec = sorted(layers, key=lambda l: per[l]["correlation_change"])[:DELTA_DIAGNOSTIC_TOPK]

        def _fmt_deciles(arr):
            def _fmt_one(x):
                if x is None:
                    return "None"
                try:
                    if x != x:  # NaN
                        return "nan"
                except Exception:
                    return "None"
                return f"{x:+.3f}"
            return "[" + ", ".join(_fmt_one(x) for x in arr) + "]"

        def _print_layer(l):
            d = per[l]
            lo_d, hi_d = d["delta_corr_ci95"]
            print(
                f"    L{l:>3}  corr {d['baseline_correlation']:+.3f} -> {d['ablated_correlation']:+.3f}  "
                f"(Δ={d['correlation_change']:+.3f}, ΔCI=[{lo_d:+.3f},{hi_d:+.3f}], "
                f"bootFDR={d['p_value_bootstrap_fdr']:.3g})"
            )
            print(
                f"         conf mean {d['baseline_confidence_mean']:.3f} -> {d['ablated_confidence_mean']:.3f}  "
                f"(Δmean={d['delta_conf_mean']:+.4f}, Δstd={d['delta_conf_std']:.4f})"
            )
            print(
                f"         corr(Δconf, metric*s) pearson={d['delta_conf_corr_metric']:+.3f} "
                f"spearman={d['delta_conf_spearman_metric']:+.3f}  "
                f"(ctrl mean={d['control_delta_conf_corr_metric_mean']:+.3f}±{d['control_delta_conf_corr_metric_std']:.3f}, "
                f"p_pooled={d['p_value_delta_corr_pooled']:.3g})"
            )
            print(
                f"         ablated≈{d['affine_slope']:.3f}*baseline+{d['affine_intercept']:+.3f}  "
                f"corr(baseline,ablated)={d['baseline_to_ablated_conf_corr']:.4f}  "
                f"corr(resid, metric*s)={d['residual_corr_metric']:+.3f}"
            )
            print(f"         mean Δconf by metric decile: {_fmt_deciles(d['delta_conf_mean_by_metric_decile'])}")

        print("\n  [Δconf diagnostics] Layers with biggest +Δcorr:")
        for l in top_inc:
            _print_layer(l)
        print("\n  [Δconf diagnostics] Layers with biggest -Δcorr:")
        for l in top_dec:
            _print_layer(l)

    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_ablation_results(analysis: Dict, method: str, output_path: Path):
    """Create 3-panel ablation visualization for a single method.

    Uses bootstrap CIs (baseline/ablated/Δcorr) and bootstrap BH-FDR for significance.
    (Control-based Z stats are retained in JSON but not emphasized in plots.)
    """
    layers = analysis.get("layers", [])
    if not layers:
        print(f"  Skipping plot for {method} - no layers")
        return

    fig, axes = plt.subplots(3, 1, figsize=(20, 14))
    # Build a title that identifies the run (model + dataset + direction/target metrics)
    model_short = get_model_short_name(MODEL)
    adapter_tag = f" + {get_model_short_name(ADAPTER)}" if ADAPTER else ""
    title_bits = [
        f"Ablation: {method.upper()} ({DIRECTION_METRIC} direction)",
        f"target={TARGET_METRIC}",
        f"model={model_short}{adapter_tag}",
        f"input={INPUT_BASE_NAME}",
    ]
    fig.suptitle(" | ".join(title_bits), fontsize=12)

    x = np.arange(len(layers))

    per = analysis["per_layer"]

    # Panel 1: Absolute correlations with bootstrap 95% CI bands
    ax1 = axes[0]
    baseline_corrs = np.array([per[l]["baseline_correlation"] for l in layers], dtype=np.float32)
    ablated_corrs = np.array([per[l]["ablated_correlation"] for l in layers], dtype=np.float32)

    base_lo = np.array([per[l]["baseline_corr_ci95"][0] for l in layers], dtype=np.float32)
    base_hi = np.array([per[l]["baseline_corr_ci95"][1] for l in layers], dtype=np.float32)
    ablt_lo = np.array([per[l]["ablated_corr_ci95"][0] for l in layers], dtype=np.float32)
    ablt_hi = np.array([per[l]["ablated_corr_ci95"][1] for l in layers], dtype=np.float32)

    # Marker size: single-layer (e.g. dry-run) needs larger markers — a 1-point
    # line segment has zero length and is invisible without markers.
    _ms = max(4.0, min(10.0, 60.0 / max(len(layers), 1)))
    ax1.plot(
        x, baseline_corrs, "-", label="Baseline", color="blue", linewidth=1.5,
        marker="o", markersize=_ms, markeredgecolor="navy", markeredgewidth=0.4,
    )
    ax1.fill_between(x, base_lo, base_hi, color='blue', alpha=0.15, label='Baseline 95% CI')

    ax1.plot(
        x, ablated_corrs, "-", label=f"{method} ablated", color="red", linewidth=1.5,
        marker="s", markersize=_ms, markeredgecolor="darkred", markeredgewidth=0.4,
    )
    ax1.fill_between(x, ablt_lo, ablt_hi, color='red', alpha=0.15, label='Ablated 95% CI')

    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (confidence vs metric)")
    ax1.set_title("Correlation by Condition (bootstrap 95% CI)")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    all_vals = np.concatenate([base_lo, base_hi, ablt_lo, ablt_hi])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) > 0:
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
        padding = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        ax1.set_ylim(ymin - padding, ymax + padding)

    # Panel 2: Δcorr with bootstrap CI and bootstrap-FDR significance coloring
    ax2 = axes[1]
    delta = np.array([per[l]["correlation_change"] for l in layers], dtype=np.float32)
    d_lo = np.array([per[l]["delta_corr_ci95"][0] for l in layers], dtype=np.float32)
    d_hi = np.array([per[l]["delta_corr_ci95"][1] for l in layers], dtype=np.float32)
    yerr = np.vstack([delta - d_lo, d_hi - delta])

    p_fdr = np.array([per[l]["p_value_bootstrap_fdr"] for l in layers], dtype=np.float32)
    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_fdr]

    _bar_w = min(0.85, 6.0 / max(len(layers), 1))
    ax2.bar(x, delta, width=_bar_w, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.errorbar(x, delta, yerr=yerr, fmt='none', ecolor='black', capsize=2, alpha=0.5, linewidth=1)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("ΔCorrelation (Ablated - Baseline)")
    ax2.set_title("Ablation Effect (bootstrap 95% CI; colors by bootstrap BH-FDR)")
    ax2.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='boot FDR < 0.05'),
        Patch(facecolor='orange', alpha=0.7, edgecolor='black', label='boot FDR < 0.10'),
        Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='n.s.'),
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=9)

    # Panel 3: Summary text
    ax3 = axes[2]
    ax3.axis('off')

    summary = analysis.get("summary", {})
    baseline_corr_mean = float(np.mean(baseline_corrs))

    best_layer = summary.get("best_layer_abs_delta", summary.get("best_layer"))
    best_delta = per[best_layer]["correlation_change"]
    best_ci = per[best_layer]["delta_corr_ci95"]
    best_p = per[best_layer]["p_value_bootstrap_fdr"]

    summary_text = f"""
ABLATION ANALYSIS: {method.upper()}

Metric: {analysis['metric']}
Layers tested: {len(layers)}
Questions: {analysis['num_questions']}
Bootstrap: n={analysis['bootstrap']['n']} resamples

Results (bootstrap-FDR):
  Mean baseline corr: {baseline_corr_mean:.4f}
  Significant layers (boot FDR<0.05): {summary.get('n_significant_bootstrap_fdr', 0)}

Largest |Δcorr| layer: {best_layer}
  Δcorr: {best_delta:+.4f}  CI95: [{best_ci[0]:+.4f}, {best_ci[1]:+.4f}]
  boot FDR: {best_p:.4g}

Note: Negative Δcorr = ablation decreased correlation
      Positive Δcorr = ablation increased correlation
"""

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()
def plot_method_comparison(analyses: Dict[str, Dict], output_path: Path):
    """Comparison plot of different direction methods.

    Shows Δcorr with bootstrap 95% CI bands and marks layers significant under bootstrap BH-FDR.
    """
    methods = list(analyses.keys())
    if len(methods) < 2:
        print("  Skipping comparison plot - need at least 2 methods")
        return

    layers = analyses[methods[0]].get("layers", [])
    if not layers:
        print("  Skipping comparison plot - no layers")
        return

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle("Method Comparison: Ablation Effects (Δcorr)", fontsize=14)

    x = np.arange(len(layers))
    method_colors = {"probe": "tab:blue", "mean_diff": "tab:orange"}

    # Panel 1: Δcorr with CI bands
    ax1 = axes[0]
    for method in methods:
        per = analyses[method]["per_layer"]
        delta = np.array([per[l]["correlation_change"] for l in layers], dtype=np.float32)
        d_lo = np.array([per[l]["delta_corr_ci95"][0] for l in layers], dtype=np.float32)
        d_hi = np.array([per[l]["delta_corr_ci95"][1] for l in layers], dtype=np.float32)
        p_fdr = np.array([per[l]["p_value_bootstrap_fdr"] for l in layers], dtype=np.float32)

        color = method_colors.get(method, "gray")
        ax1.plot(x, delta, "-", label=method, color=color, linewidth=1.8, alpha=0.85)
        ax1.fill_between(x, d_lo, d_hi, color=color, alpha=0.12)

        # Mark significant layers
        sig_x = [i for i, p in enumerate(p_fdr) if p < 0.05]
        sig_y = [delta[i] for i in sig_x]
        if sig_x:
            ax1.scatter(sig_x, sig_y, color=color, s=45, zorder=5, edgecolor="black", linewidth=0.5)

    ax1.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("ΔCorrelation (Ablated - Baseline)")
    ax1.set_title("Δcorr by Method (filled markers = bootstrap FDR<0.05)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Summary
    ax2 = axes[1]
    ax2.axis("off")

    comparison_text = (
        "METHOD COMPARISON (bootstrap-FDR)\n"
        + "=" * 50
        + "\n\n"
    )
    for method in methods:
        summary = analyses[method].get("summary", {})
        comparison_text += f"{method.upper()}:\n"
        comparison_text += f"  Significant layers (boot FDR<0.05): {summary.get('n_significant_bootstrap_fdr', 0)}\n"
        comparison_text += (
            f"  Best |Δ| layer: {summary.get('best_layer_abs_delta')} "
            f"(Δ={summary.get('best_abs_delta', 0.0):+.3f})\n\n"
        )

    best_method = max(methods, key=lambda m: analyses[m].get("summary", {}).get("n_significant_bootstrap_fdr", 0))
    comparison_text += f"Method with more boot-FDR-significant layers: {best_method.upper()}\n"

    ax2.text(
        0.1,
        0.9,
        comparison_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved {output_path}")
    plt.close()


def print_summary(analyses: Dict[str, Dict]):
    """Print summary of ablation results (with bootstrap CIs + bootstrap BH-FDR)."""
    print("\n" + "=" * 90)
    print("ABLATION CAUSALITY TEST RESULTS")
    print("=" * 90)
    print("\nKey question: Does ablating the direction HURT calibration?")
    print("Expected: correlation should DECREASE (negative Δcorr)")
    print("Significance & CIs: bootstrap over questions + BH-FDR across layers")

    for method, analysis in analyses.items():
        layers = analysis.get("layers", [])
        per = analysis.get("per_layer", {})

        print(f"\n{method.upper()} directions ({len(layers)} layers):")
        print("-" * 90)
        print(
            f"{'Layer':>5}  {'Base':>7}  {'BaseCI':>17}  {'Abl':>7}  {'AblCI':>17}  "
            f"{'Δ':>7}  {'ΔCI':>17}  {'bootFDR':>8}  {'Hurt?':>6}"
        )
        print("-" * 90)

        if not layers:
            print("  (no layers)")
            continue

        # Sort by Δ (ascending) for readability
        sorted_layers = sorted(layers, key=lambda l: per[l]["correlation_change"])

        for layer in sorted_layers:
            d = per[layer]
            base = d["baseline_correlation"]
            ablt = d["ablated_correlation"]
            delta = d["correlation_change"]
            bci = d["baseline_corr_ci95"]
            aci = d["ablated_corr_ci95"]
            dci = d["delta_corr_ci95"]
            pfdr = d["p_value_bootstrap_fdr"]

            hurt = "YES" if (delta < 0 and pfdr < 0.05) else "no"
            print(
                f"{layer:>5}  {base:>+7.4f}  [{bci[0]:>+7.4f},{bci[1]:>+7.4f}]  "
                f"{ablt:>+7.4f}  [{aci[0]:>+7.4f},{aci[1]:>+7.4f}]  "
                f"{delta:>+7.4f}  [{dci[0]:>+7.4f},{dci[1]:>+7.4f}]  "
                f"{pfdr:>8.2g}  {hurt:>6}"
            )

        # Summary counts
        n_hurt = sum(
            1
            for l in layers
            if per[l]["correlation_change"] < 0 and per[l]["p_value_bootstrap_fdr"] < 0.05
        )
        n_helped = sum(
            1
            for l in layers
            if per[l]["correlation_change"] > 0 and per[l]["p_value_bootstrap_fdr"] < 0.05
        )

        print("-" * 90)
        print(f"Summary: {n_hurt} layers where ablation HURT calibration (boot FDR<0.05)")
        if n_helped > 0:
            print(f"         {n_helped} layers where ablation HELPED calibration (unexpected)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ABLATION CAUSALITY TEST")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print(f"Adapter: {ADAPTER}")
    print(f"Input: {INPUT_BASE_NAME}")
    print(f"Direction metric (ablated): {DIRECTION_METRIC}")
    print(f"Target metric (correlated): {TARGET_METRIC}")
    print(f"Meta-task: {META_TASK}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Controls: {NUM_CONTROLS} (final), {NUM_CONTROLS_NONFINAL} (non-final)")
    print(f"Bootstrap: {BOOTSTRAP_N} resamples (CI={int((1-BOOTSTRAP_CI_ALPHA)*100)}%)")

    # Load directions
    print("\nLoading directions...")
    all_directions = load_directions(INPUT_BASE_NAME, DIRECTION_METRIC)
    available_methods = list(all_directions.keys())
    print(f"  Found methods: {available_methods}")

    # Filter to requested methods
    if METHODS is not None:
        methods = [m for m in METHODS if m in available_methods]
        if not methods:
            raise ValueError(f"No matching methods found. Available: {available_methods}, requested: {METHODS}")
        print(f"  Using methods: {methods}")
    else:
        methods = available_methods

    for method in methods:
        layers = sorted(all_directions[method].keys())
        print(f"  {method}: {len(layers)} layers ({min(layers)}-{max(layers)})")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(INPUT_BASE_NAME)
    all_data = dataset["data"]

    if USE_TRANSFER_SPLIT:
        # Use same 80/20 split as transfer analysis for apples-to-apples comparison
        n_total = len(all_data)
        indices = np.arange(n_total)
        train_idx, test_idx = train_test_split(
            indices, train_size=TRAIN_SPLIT, random_state=SEED
        )
        data_items = [all_data[i] for i in test_idx]
        # Keep original indices for trial_index in delegate prompt formatting
        original_indices = test_idx
        print(f"  Using transfer test split: {len(data_items)} questions (from {n_total} total, seed={SEED})")
    else:
        # Legacy behavior: first NUM_QUESTIONS
        data_items = all_data[:NUM_QUESTIONS]
        # Original indices are just 0..NUM_QUESTIONS-1
        original_indices = np.arange(len(data_items))
        print(f"  Using first {len(data_items)} questions (legacy mode)")

    # Extract questions (each item has question, options, correct_answer, etc.)
    questions = data_items
    # Extract metric values from each item (correlation target, not ablation target)
    metric_values = np.array([item[TARGET_METRIC] for item in data_items])
    print(f"  Questions: {len(questions)}")
    print(f"  {TARGET_METRIC}: mean={metric_values.mean():.3f}, std={metric_values.std():.3f}")

    # Load transfer results for layer selection (non-final positions)
    transfer_data = load_transfer_results(INPUT_BASE_NAME, META_TASK)
    if transfer_data is not None:
        print(f"\nLoaded transfer results for layer selection")
        # Preview what layers would be selected FOR EACH (POSITION, METHOD) combination
        for pos in PROBE_POSITIONS:
            if pos == "final":
                print(f"  {pos}: all layers (no R² filter)")
            else:
                for method in methods:
                    pos_layers = get_layers_from_transfer(transfer_data, TARGET_METRIC, pos, TRANSFER_R2_THRESHOLD, method)
                    if pos_layers:
                        print(f"  {pos}/{method}: {len(pos_layers)} layers with {TARGET_METRIC} R²≥{TRANSFER_R2_THRESHOLD}: {pos_layers}")
                    else:
                        # Try fallback to final
                        fallback_layers = get_layers_from_transfer(transfer_data, TARGET_METRIC, "final", TRANSFER_R2_THRESHOLD, method)
                        if fallback_layers:
                            print(f"  {pos}/{method}: no position-specific data, using final: {len(fallback_layers)} layers")
                        else:
                            print(f"  {pos}/{method}: WARNING - no layers found, will use ALL layers")
    else:
        expected_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_transfer_{META_TASK}_results.json"
        print(f"\nNo transfer results found - will use all layers for all positions")
        print(f"  Expected: {expected_path}")

    # Determine base layers (all available)
    all_available_layers = sorted(all_directions[methods[0]].keys())

    # Layer selection depends on position - will be set per-position below
    if LAYERS is not None:
        print(f"\nExplicit LAYERS override: {len(LAYERS)} layers")
    else:
        print(f"\nLayer selection: all layers for final, R²≥{TRANSFER_R2_THRESHOLD} for non-final")

    # Load model
    print("\nLoading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    print(f"  Use chat template: {use_chat_template}")
    print(f"  Device: {DEVICE}")

    # Run ablation for each method and position
    # Structure: {position: {method: analysis}}
    all_results_by_pos = {pos: {} for pos in PROBE_POSITIONS}
    all_analyses_by_pos = {pos: {} for pos in PROBE_POSITIONS}

    for position in PROBE_POSITIONS:
        print(f"\n{'='*70}")
        print(f"POSITION: {position}")
        print(f"{'='*70}")

        # Determine number of controls for this position
        position_num_controls = NUM_CONTROLS if position == "final" else NUM_CONTROLS_NONFINAL

        for method in methods:
            print(f"\n{'='*60}")
            print(f"ABLATION EXPERIMENT: {method.upper()} @ {position}")
            print(f"{'='*60}")

            # Determine layers for this position AND method
            if LAYERS is not None:
                # Explicit override applies to all positions/methods
                method_layers = LAYERS
            elif position == "final":
                # Final position: use all layers
                method_layers = all_available_layers
            else:
                # Non-final position: select based on transfer R² for THIS method
                if transfer_data is not None:
                    method_layers = get_layers_from_transfer(
                        transfer_data, TARGET_METRIC, position, TRANSFER_R2_THRESHOLD, method
                    )
                    if not method_layers:
                        # Fall back to "final" position transfer data if position-specific not available
                        method_layers = get_layers_from_transfer(
                            transfer_data, TARGET_METRIC, "final", TRANSFER_R2_THRESHOLD, method
                        )
                else:
                    method_layers = all_available_layers

                if not method_layers:
                    print("\n" + "!"*70)
                    print("!!! WARNING: FALLING BACK TO ALL LAYERS !!!")
                    print(f"!!! No layers meet R²≥{TRANSFER_R2_THRESHOLD} threshold for {method}/{TARGET_METRIC}")
                    print(f"!!! This will test {len(all_available_layers)} layers instead of ~50")
                    print(f"!!! Check that TARGET_METRIC and method match transfer results")
                    print("!"*70)
                    print("Continuing in 3 seconds (Ctrl+C to abort)...")
                    import time
                    time.sleep(3)
                    method_layers = all_available_layers

            print(f"  Layers: {len(method_layers)} (range {min(method_layers)}-{max(method_layers)})")
            print(f"  Controls: {position_num_controls}")

            results = run_ablation_for_method(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                metric_values=metric_values,
                directions=all_directions[method],
                num_controls=position_num_controls,
                meta_task=META_TASK,
                use_chat_template=use_chat_template,
                layers=method_layers,
                position=position,
                original_indices=original_indices,
            )
            all_results_by_pos[position][method] = results

            # Report signal distributions
            _report_signal_distributions(results, META_TASK)

            # Analyze results
            print(f"\n  Analyzing results...")
            analysis = analyze_ablation_results(results, TARGET_METRIC)
            all_analyses_by_pos[position][method] = analysis

            summary = analysis["summary"]
            print(f"  Significant layers (bootstrap FDR): {summary.get('n_significant_bootstrap_fdr', 0)}")
            print(f"  Best |Δcorr| layer: {summary.get('best_layer_abs_delta')} (Δ={summary.get('best_abs_delta', 0.0):+.3f})")

        # Incremental save after each position completes (crash protection)
        model_short = get_model_short_name(MODEL)
        # INPUT_BASE_NAME is like "Llama-3.1-8B-Instruct_SimpleMC_scale-numeric".
        # Drop the model-short prefix (the first chunk) and use the rest as the
        # dataset/scale tag so SimpleMC and TriviaMC end up in different files.
        _parts = INPUT_BASE_NAME.split("_")
        _run_tag = "_".join(_parts[1:]) if len(_parts) > 1 else INPUT_BASE_NAME
        base_output = f"{model_short}_{_run_tag}_ablation_{META_TASK}_{DIRECTION_METRIC}"
        checkpoint_path = OUTPUT_DIR / f"{base_output}_checkpoint.json"
        checkpoint_json = {
            "config": {
                "model": MODEL,
                "adapter": ADAPTER,
                "input_base_name": INPUT_BASE_NAME,
                "direction_metric": DIRECTION_METRIC,
                "target_metric": TARGET_METRIC,
                "meta_task": META_TASK,
                "num_questions": len(questions),
                "use_transfer_split": USE_TRANSFER_SPLIT,
                "positions_completed": [p for p in PROBE_POSITIONS if all_analyses_by_pos[p]],
            },
            "by_position": {},
        }
        for pos in PROBE_POSITIONS:
            if all_analyses_by_pos[pos]:
                checkpoint_json["by_position"][pos] = {}
                for m, analysis in all_analyses_by_pos[pos].items():
                    checkpoint_json["by_position"][pos][m] = {
                        "per_layer": analysis["per_layer"],
                        "summary": analysis["summary"],
                    }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_json, f, indent=2)
        print(f"  Checkpoint saved: {checkpoint_path.name}")

    # For backward compatibility, keep "final" as the default if it was tested
    if "final" in all_analyses_by_pos:
        all_analyses = all_analyses_by_pos["final"]
    else:
        # Use first available position
        all_analyses = all_analyses_by_pos[PROBE_POSITIONS[0]]

    # Generate output filename — include everything after the model-short prefix
    # so SimpleMC and TriviaMC outputs don't collide.
    model_short = get_model_short_name(MODEL)
    _parts = INPUT_BASE_NAME.split("_")
    _run_tag = "_".join(_parts[1:]) if len(_parts) > 1 else INPUT_BASE_NAME
    base_output = f"{model_short}_{_run_tag}_ablation_{META_TASK}_{DIRECTION_METRIC}"

    # Save JSON results
    print("\nSaving results...")
    results_path = OUTPUT_DIR / f"{base_output}_results.json"

    # Load existing results if present, otherwise create new
    if results_path.exists():
        with open(results_path, "r") as f:
            output_json = json.load(f)
        print(f"  Merging with existing results: {results_path.name}")
        # Ensure by_position exists (for older format files)
        if "by_position" not in output_json:
            output_json["by_position"] = {}
    else:
        output_json = {
            "config": {
                "model": MODEL,
                "adapter": ADAPTER,
                "input_base_name": INPUT_BASE_NAME,
                "direction_metric": DIRECTION_METRIC,
                "target_metric": TARGET_METRIC,
                "meta_task": META_TASK,
                "num_questions": len(questions),
                "use_transfer_split": USE_TRANSFER_SPLIT,
                "num_controls_final": NUM_CONTROLS,
                "num_controls_nonfinal": NUM_CONTROLS_NONFINAL,
                "transfer_r2_threshold": TRANSFER_R2_THRESHOLD,
                "methods_tested": [],
                "positions_tested": [],
            },
            "by_position": {},
        }

    # Update config with current run's positions/methods (accumulate)
    existing_positions = set(output_json["config"].get("positions_tested", []))
    existing_methods = set(output_json["config"].get("methods_tested", []))
    output_json["config"]["positions_tested"] = sorted(existing_positions | set(PROBE_POSITIONS))
    output_json["config"]["methods_tested"] = sorted(existing_methods | set(methods))

    # Legacy format: only update top-level keys when "final" position was tested
    if "final" in PROBE_POSITIONS:
        for method, analysis in all_analyses.items():
            output_json[method] = {
                "layers": analysis["layers"],
                "num_questions": analysis["num_questions"],
                "num_controls": analysis["num_controls"],
                "metric": analysis["metric"],
                "per_layer": analysis["per_layer"],
                "summary": analysis["summary"],
            }

    # Merge new results into by_position (overwrites same position/method)
    for position in PROBE_POSITIONS:
        if position not in output_json["by_position"]:
            output_json["by_position"][position] = {}
        for method, analysis in all_analyses_by_pos[position].items():
            output_json["by_position"][position][method] = {
                "layers": analysis["layers"],
                "num_questions": analysis["num_questions"],
                "num_controls": analysis["num_controls"],
                "metric": analysis["metric"],
                "per_layer": analysis["per_layer"],
                "summary": analysis["summary"],
            }

    # Comparison summary (primarily useful when >=2 methods)
    if len(methods) >= 2:
        # For backward compatibility, keep the old keys (control-based) *and* add bootstrap-based keys.
        output_json["comparison"] = {
            method: {
                # Control-null statistics (may be uninformative if NUM_CONTROLS == 0)
                "n_significant_fdr_controls": all_analyses[method]["summary"].get("n_significant_fdr", 0),
                "best_layer_z": all_analyses[method]["summary"].get("best_layer"),
                "best_effect_z": all_analyses[method]["summary"].get("best_effect_z", 0.0),

                # Bootstrap statistics (recommended)
                "n_significant_bootstrap_fdr": all_analyses[method]["summary"].get("n_significant_bootstrap_fdr", 0),
                "best_layer_abs_delta": all_analyses[method]["summary"].get("best_layer_abs_delta"),
                "best_abs_delta": all_analyses[method]["summary"].get("best_abs_delta", 0.0),
            }
            for method in methods
        }

        best_method_boot = max(methods, key=lambda m: all_analyses[m]["summary"].get("n_significant_bootstrap_fdr", 0))
        best_method_ctrl = max(methods, key=lambda m: all_analyses[m]["summary"].get("n_significant_fdr", 0))
        output_json["comparison"]["method_with_more_bootstrap_effect"] = best_method_boot
        output_json["comparison"]["method_with_more_control_effect"] = best_method_ctrl



    with open(results_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Saved {results_path}")

    # Generate plots for each position
    print("\nGenerating plots...")
    for position in PROBE_POSITIONS:
        for method in methods:
            plot_path = OUTPUT_DIR / f"{base_output}_{method}_{position}.png"
            plot_ablation_results(all_analyses_by_pos[position][method], method, plot_path)

        if len(methods) >= 2:
            comparison_path = OUTPUT_DIR / f"{base_output}_comparison_{position}.png"
            plot_method_comparison(all_analyses_by_pos[position], comparison_path)

    # Print summary for each position
    for position in PROBE_POSITIONS:
        print(f"\n--- Position: {position} ---")
        print_summary(all_analyses_by_pos[position])

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {results_path.name}")
    for position in PROBE_POSITIONS:
        for method in methods:
            print(f"  {base_output}_{method}_{position}.png")
        if len(methods) >= 2:
            print(f"  {base_output}_comparison_{position}.png")


if __name__ == "__main__":
    main()
