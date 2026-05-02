"""
Steering causality test for uncertainty directions.

Tests whether directions from identify_mc_correlate.py are causally SUFFICIENT
for the model's meta-judgments. If steering along the direction shifts stated
confidence in the expected direction, that's evidence the direction captures
information the model uses for introspection.

Complements ablation (run_ablation_causality.py) which tests NECESSITY.

Key features:
- Tests ALL layers by default (no pre-filtering by transfer R²)
- Tests BOTH probe and mean_diff methods in a single run for comparison
- Uses pooled null distribution + FDR correction for robust statistics
- KV cache optimization for efficiency
- Batched multiplier sweeps

Usage:
    python run_steering_causality.py

Expects outputs from identify_mc_correlate.py:
    outputs/{INPUT_BASE_NAME}_mc_{METRIC}_directions.npz
    outputs/{INPUT_BASE_NAME}_mc_dataset.json
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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
    BatchSteeringHook,
    pretokenize_prompts,
    build_padded_gpu_batches,
    get_kv_cache,
    create_fresh_cache,
)
# Note: metric_sign_for_confidence exists in core.metrics but we use get_expected_slope_sign locally
from prompts import (
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    format_answer_or_delegate_prompt,
    get_answer_or_delegate_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
    find_mc_positions,
)

# =============================================================================
# CONFIGURATION — edit values in experiment_config.SteeringCausalityConfig
# =============================================================================
from experiment_config import SteeringCausalityConfig as _C

MODEL = _C.MODEL
ADAPTER = _C.ADAPTER
INPUT_BASE_NAME = _C.INPUT_BASE_NAME
METRIC = _C.METRIC
META_TASK = _C.META_TASK
STEERING_MULTIPLIERS = list(_C.STEERING_MULTIPLIERS)
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

def get_format_fn(meta_task: str):
    """Get prompt formatting function for meta-task."""
    if meta_task == "confidence":
        return format_stated_confidence_prompt
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
        # Wrap to match (probs, mapping) signature
        return lambda p, m: get_stated_confidence_signal(p)
    elif meta_task == "delegate":
        return get_answer_or_delegate_signal
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_options(meta_task: str) -> List[str]:
    """Get response options for meta-task."""
    if meta_task == "confidence":
        return list(STATED_CONFIDENCE_OPTIONS.keys())
    elif meta_task == "delegate":
        return ANSWER_OR_DELEGATE_OPTIONS
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_expected_slope_sign(metric: str) -> int:
    """
    Get the expected sign of the confidence slope for a given metric.

    The direction points toward *increasing* the metric value.
    - entropy: HIGH = uncertain -> steering +direction should DECREASE confidence -> slope < 0
    - logit_gap, top_prob, margin, top_logit: HIGH = confident -> slope > 0

    Returns:
        +1 if +direction should increase confidence
        -1 if +direction should decrease confidence
    """
    if metric == "entropy":
        return -1  # +direction = more uncertain = less confident
    else:
        return +1  # +direction = more confident


# =============================================================================
# STEERING EXPERIMENT
# =============================================================================

def run_steering_for_method(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    directions: Dict[int, np.ndarray],
    num_controls: int,
    meta_task: str,
    multipliers: List[float],
    use_chat_template: bool,
    layers: Optional[List[int]] = None,
    position: str = "final",
) -> Dict:
    """
    Run steering experiment for a single direction method.

    Uses KV cache and batched multipliers for efficiency (for position="final").
    For other positions, uses full forward passes with indexed steering hooks.

    Args:
        position: Token position for steering ("final", "question_mark", etc.)

    Returns dict with per-layer results for each multiplier.
    """
    if layers is None:
        layers = sorted(directions.keys())
    else:
        layers = [l for l in layers if l in directions]

    if not layers:
        return {"error": "No layers to test"}

    # Get formatting functions and options
    format_fn = get_format_fn(meta_task)
    signal_fn = get_signal_fn(meta_task)
    options = get_options(meta_task)

    # Tokenize options
    option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options
    ]

    # Format and tokenize prompts, find position indices
    prompts = []
    mappings = []
    position_indices = []  # Per-prompt token index for steering
    for q_idx, question in enumerate(questions):
        if meta_task == "delegate":
            prompt, _, mapping = format_fn(question, tokenizer, trial_index=q_idx, use_chat_template=use_chat_template)
        else:
            prompt, _ = format_fn(question, tokenizer, use_chat_template=use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

        # Find position for this prompt
        positions = find_mc_positions(prompt, tokenizer, question)
        pos_idx = positions.get(position, -1)
        position_indices.append(pos_idx)

    # Warn if some positions weren't found (will fall back to final token)
    n_valid = sum(1 for idx in position_indices if idx >= 0)
    n_total = len(position_indices)
    if n_valid < n_total:
        print(f"  Warning: {position} position found for {n_valid}/{n_total} prompts (others fall back to final)")

    # Determine whether we can use KV cache optimization
    use_kv_cache = (position == "final")

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)

    # Calculate effective batch size based on multiplier expansion
    # When batching k multipliers together, each base batch expands by k
    # So we need to limit base batch size to keep expanded batch within target
    # BATCH_SIZE acts as a safety cap (max base batch regardless of target)
    nonzero_multipliers = [m for m in multipliers if m != 0.0]
    k_mult = len(nonzero_multipliers)
    effective_batch_size = max(1, min(BATCH_SIZE, EXPANDED_BATCH_TARGET // k_mult)) if k_mult > 0 else BATCH_SIZE
    print(f"  Batch sizing: k_mult={k_mult}, effective_batch={effective_batch_size} (expanded={effective_batch_size * k_mult})")

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, effective_batch_size)

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
        cached_directions[layer] = {
            "direction": dir_tensor,
            "controls": ctrl_tensors,
        }

    # Initialize results storage
    # Structure: layer -> {"baseline": [...], "steered": {mult: [...]}, "controls": {ctrl_i: {mult: [...]}}}
    baseline_results = [None] * len(questions)
    layer_results = {}
    for layer in layers:
        layer_results[layer] = {
            "baseline": baseline_results,  # Shared across layers
            "steered": {m: [None] * len(questions) for m in multipliers},
            "controls": {
                f"control_{i}": {m: [None] * len(questions) for m in multipliers}
                for i in range(num_controls)
            },
        }
        # Link multiplier 0 to baseline
        layer_results[layer]["steered"][0.0] = baseline_results
        for ctrl_key in layer_results[layer]["controls"]:
            layer_results[layer]["controls"][ctrl_key][0.0] = baseline_results

    # Calculate total forward passes for progress tracking
    # Per batch: 1 baseline + layers * (1 introspection + num_controls) for all multipliers batched
    total_forward_passes = len(gpu_batches) * (1 + len(layers) * (1 + num_controls))
    print(f"  Total forward passes: {total_forward_passes}")
    print(f"  Processing {len(gpu_batches)} batches, {k_mult} multipliers batched per pass...")

    pbar = tqdm(total=total_forward_passes, desc="  Forward passes")

    if use_kv_cache:
        # =====================================================================
        # KV CACHE PATH (position="final")
        # =====================================================================
        for batch_idx, (batch_indices, batch_inputs) in enumerate(gpu_batches):
            B = len(batch_indices)

            # Compute KV cache once per batch
            base_step_data = get_kv_cache(model, batch_inputs)
            keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

            inputs_template = {
                "input_ids": base_step_data["input_ids"],
                "attention_mask": base_step_data["attention_mask"],
                "use_cache": True
            }
            if "position_ids" in base_step_data:
                inputs_template["position_ids"] = base_step_data["position_ids"]

            # Compute baseline (no steering) - shared across all layers
            if baseline_results[batch_indices[0]] is None:
                fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                baseline_inputs = inputs_template.copy()
                baseline_inputs["past_key_values"] = fresh_cache

                with torch.inference_mode():
                    out = model(**baseline_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf = signal_fn(p, mappings[q_idx])
                    m_val = metric_values[q_idx]
                    baseline_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": resp,
                        "confidence": float(conf),
                        "metric": float(m_val),
                    }

            pbar.update(1)

            # Prepare expanded inputs for batched multiplier sweep
            expanded_input_ids = inputs_template["input_ids"].repeat_interleave(k_mult, dim=0)
            expanded_attention_mask = inputs_template["attention_mask"].repeat_interleave(k_mult, dim=0)
            expanded_inputs_template = {
                "input_ids": expanded_input_ids,
                "attention_mask": expanded_attention_mask,
                "use_cache": True
            }
            if "position_ids" in inputs_template:
                expanded_inputs_template["position_ids"] = inputs_template["position_ids"].repeat_interleave(k_mult, dim=0)

            # Run steering for each layer
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                direction_tensor = cached_directions[layer]["direction"]
                control_tensors = cached_directions[layer]["controls"]

                hook = BatchSteeringHook()
                hook.register(layer_module)

                def run_batched_sweep(dir_vec, result_dict):
                    """Run all multipliers for a direction in one pass."""
                    # Create fresh cache expanded for this pass
                    pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_mult)

                    # Attach cache to inputs
                    current_inputs = expanded_inputs_template.copy()
                    current_inputs["past_key_values"] = pass_cache

                    # Build delta tensor: for each question in batch, apply each multiplier
                    # Shape: (B * k_mult, hidden_dim)
                    deltas = []
                    for _ in range(B):
                        for mult in nonzero_multipliers:
                            deltas.append(dir_vec * mult)
                    delta_bh = torch.stack(deltas, dim=0)
                    hook.set_delta(delta_bh)

                    # Run model
                    with torch.inference_mode():
                        out = model(**current_inputs)
                        logits = out.logits[:, -1, :][:, option_token_ids]
                        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                    # Store results
                    for i, q_idx in enumerate(batch_indices):
                        for j, mult in enumerate(nonzero_multipliers):
                            idx = i * k_mult + j
                            p = probs[idx]
                            resp = options[np.argmax(p)]
                            conf = signal_fn(p, mappings[q_idx])
                            m_val = metric_values[q_idx]
                            result_dict[mult][q_idx] = {
                                "question_idx": q_idx,
                                "response": resp,
                                "confidence": float(conf),
                                "metric": float(m_val),
                            }

                try:
                    # Introspection direction
                    run_batched_sweep(direction_tensor, layer_results[layer]["steered"])
                    pbar.update(1)

                    # Control directions
                    for i_c, ctrl_dir in enumerate(control_tensors):
                        run_batched_sweep(ctrl_dir, layer_results[layer]["controls"][f"control_{i_c}"])
                        pbar.update(1)

                finally:
                    hook.remove()

    else:
        # =====================================================================
        # FULL FORWARD PATH (non-final positions)
        # Batches all multipliers together like KV cache path does
        # =====================================================================
        for batch_idx, (batch_indices, batch_inputs) in enumerate(gpu_batches):
            B = len(batch_indices)
            seq_len = batch_inputs["input_ids"].shape[1]

            # Compute position indices adjusted for left-padding
            batch_pos_indices = []
            for i, q_idx in enumerate(batch_indices):
                pos = position_indices[q_idx]
                if pos >= 0:
                    # Adjust for left-padding: find actual sequence length
                    actual_len = int(batch_inputs["attention_mask"][i].sum())
                    pad_offset = seq_len - actual_len
                    adjusted_pos = pos + pad_offset
                else:
                    # Fallback to final token
                    adjusted_pos = seq_len - 1
                batch_pos_indices.append(adjusted_pos)
            batch_pos_tensor = torch.tensor(batch_pos_indices, dtype=torch.long, device=DEVICE)

            # Compute baseline (no steering) - shared across all layers
            if baseline_results[batch_indices[0]] is None:
                with torch.inference_mode():
                    out = model(**batch_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf = signal_fn(p, mappings[q_idx])
                    m_val = metric_values[q_idx]
                    baseline_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": resp,
                        "confidence": float(conf),
                        "metric": float(m_val),
                    }

            pbar.update(1)

            # Prepare expanded inputs for batched multiplier sweep (same as KV cache path)
            expanded_input_ids = batch_inputs["input_ids"].repeat_interleave(k_mult, dim=0)
            expanded_attention_mask = batch_inputs["attention_mask"].repeat_interleave(k_mult, dim=0)
            expanded_inputs = {
                "input_ids": expanded_input_ids,
                "attention_mask": expanded_attention_mask,
            }

            # Expand position indices to match expanded batch
            expanded_pos_tensor = batch_pos_tensor.repeat_interleave(k_mult)

            # Run steering for each layer (all multipliers batched per direction)
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                direction_tensor = cached_directions[layer]["direction"]
                control_tensors = cached_directions[layer]["controls"]

                def run_batched_sweep(dir_vec, result_dict):
                    """Run all multipliers for a direction in one pass."""
                    # Build delta tensor: for each question in batch, apply each multiplier
                    # Shape: (B * k_mult, hidden_dim)
                    deltas = []
                    for _ in range(B):
                        for mult in nonzero_multipliers:
                            deltas.append(dir_vec * mult)
                    delta_bh = torch.stack(deltas, dim=0)

                    hook = BatchSteeringHook(delta_bh=delta_bh, intervention_position="indexed")
                    hook.set_position_indices(expanded_pos_tensor)
                    hook.register(layer_module)

                    try:
                        with torch.inference_mode():
                            out = model(**expanded_inputs)
                            logits = out.logits[:, -1, :][:, option_token_ids]
                            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                        # Store results
                        for i, q_idx in enumerate(batch_indices):
                            for j, mult in enumerate(nonzero_multipliers):
                                idx = i * k_mult + j
                                p = probs[idx]
                                resp = options[np.argmax(p)]
                                conf = signal_fn(p, mappings[q_idx])
                                m_val = metric_values[q_idx]
                                result_dict[mult][q_idx] = {
                                    "question_idx": q_idx,
                                    "response": resp,
                                    "confidence": float(conf),
                                    "metric": float(m_val),
                                }
                    finally:
                        hook.remove()

                # Introspection direction
                run_batched_sweep(direction_tensor, layer_results[layer]["steered"])
                pbar.update(1)

                # Control directions
                for i_c, ctrl_dir in enumerate(control_tensors):
                    run_batched_sweep(ctrl_dir, layer_results[layer]["controls"][f"control_{i_c}"])
                    pbar.update(1)

    pbar.close()
    return {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": layer_results,
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and metric."""
    if len(confidences) < 2 or np.std(confidences) < 1e-10 or np.std(metric_values) < 1e-10:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def compute_slope(confidences_by_mult: Dict[float, List[Dict]], multipliers: List[float]) -> float:
    """Compute confidence slope across multipliers."""
    mean_confs = []
    for mult in multipliers:
        confs = [r["confidence"] for r in confidences_by_mult[mult]]
        mean_confs.append(np.mean(confs))

    # Linear fit: conf = slope * mult + intercept
    slope, _ = np.polyfit(multipliers, mean_confs, 1)
    return float(slope)


def analyze_steering_results(results: Dict, metric: str) -> Dict:
    """
    Compute steering effect statistics with pooled null + FDR correction.

    Returns analysis dict with per-layer stats and summary.
    """
    layers = results["layers"]
    multipliers = results["multipliers"]
    num_controls = results["num_controls"]

    # Get expected slope sign for interpretation
    expected_sign = get_expected_slope_sign(metric)

    analysis = {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": results["num_questions"],
        "num_controls": num_controls,
        "metric": metric,
        "expected_slope_sign": expected_sign,
        "per_layer": {},
    }

    # First pass: collect all control slopes for pooled null
    all_control_slopes = []
    layer_data = {}

    for layer in layers:
        lr = results["layer_results"][layer]

        # Compute baseline correlation
        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_metric = np.array([r["metric"] for r in lr["baseline"]])
        baseline_corr = compute_correlation(baseline_conf, baseline_metric)

        # Compute introspection slope
        intro_slope = compute_slope(lr["steered"], multipliers)

        # Compute control slopes
        control_slopes = []
        for ctrl_key in lr["controls"]:
            ctrl_slope = compute_slope(lr["controls"][ctrl_key], multipliers)
            control_slopes.append(ctrl_slope)

        all_control_slopes.extend(control_slopes)

        # Get mean confidence at each multiplier for plotting
        intro_mean_conf_by_mult = {}
        ctrl_mean_conf_by_mult = {}
        for mult in multipliers:
            intro_confs = [r["confidence"] for r in lr["steered"][mult]]
            intro_mean_conf_by_mult[mult] = float(np.mean(intro_confs))

            # Average across all controls
            all_ctrl_confs = []
            for ctrl_key in lr["controls"]:
                all_ctrl_confs.extend([r["confidence"] for r in lr["controls"][ctrl_key][mult]])
            ctrl_mean_conf_by_mult[mult] = float(np.mean(all_ctrl_confs))

        layer_data[layer] = {
            "baseline_corr": baseline_corr,
            "baseline_conf_mean": float(np.mean(baseline_conf)),
            "intro_slope": intro_slope,
            "control_slopes": control_slopes,
            "intro_mean_conf_by_mult": intro_mean_conf_by_mult,
            "ctrl_mean_conf_by_mult": ctrl_mean_conf_by_mult,
        }

    # Convert pooled null to array
    pooled_null = np.array(all_control_slopes)
    pooled_null_abs = np.abs(pooled_null)

    # Second pass: compute p-values
    raw_p_values = []

    for layer in layers:
        ld = layer_data[layer]

        intro_slope = ld["intro_slope"]
        intro_slope_abs = abs(intro_slope)
        control_slopes = np.array(ld["control_slopes"])

        # Per-layer statistics
        ctrl_mean = float(np.mean(control_slopes))
        ctrl_std = float(np.std(control_slopes))

        # Pooled p-value: two-tailed test (how many controls have |slope| >= |ours|)
        n_pooled_larger = np.sum(pooled_null_abs >= intro_slope_abs)
        p_value_pooled = (n_pooled_larger + 1) / (len(pooled_null) + 1)

        # Effect size (Z-score vs controls)
        ctrl_abs_mean = float(np.mean(np.abs(control_slopes)))
        ctrl_abs_std = float(np.std(np.abs(control_slopes)))
        if ctrl_abs_std > 1e-10:
            effect_size_z = (intro_slope_abs - ctrl_abs_mean) / ctrl_abs_std
        else:
            effect_size_z = 0.0

        # Check if sign matches expected
        actual_sign = 1 if intro_slope > 0 else -1 if intro_slope < 0 else 0
        sign_matches = (actual_sign == expected_sign)

        raw_p_values.append((layer, p_value_pooled))

        analysis["per_layer"][layer] = {
            "baseline_correlation": ld["baseline_corr"],
            "baseline_confidence_mean": ld["baseline_conf_mean"],
            "introspection_slope": intro_slope,
            "control_slope_mean": ctrl_mean,
            "control_slope_std": ctrl_std,
            "p_value_pooled": float(p_value_pooled),
            "effect_size_z": float(effect_size_z),
            "sign_matches_expected": sign_matches,
            "intro_mean_conf_by_mult": ld["intro_mean_conf_by_mult"],
            "ctrl_mean_conf_by_mult": ld["ctrl_mean_conf_by_mult"],
        }

    # FDR correction (Benjamini-Hochberg)
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer, p_val) in enumerate(sorted_pvals, 1):
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer] = adjusted

    # Make monotonic
    prev_adjusted = 0.0
    for layer, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer] = max(fdr_adjusted[layer], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer]

    # Add FDR p-values
    for layer in layers:
        analysis["per_layer"][layer]["p_value_fdr"] = float(fdr_adjusted[layer])

    # Summary
    significant_pooled = [l for l in layers if analysis["per_layer"][l]["p_value_pooled"] < 0.05]
    significant_fdr = [l for l in layers if analysis["per_layer"][l]["p_value_fdr"] < 0.05]
    sign_correct_fdr = [l for l in significant_fdr if analysis["per_layer"][l]["sign_matches_expected"]]
    sign_correct_pooled = [l for l in significant_pooled if analysis["per_layer"][l]["sign_matches_expected"]]

    # Best layer by effect size Z (slope relative to control variance)
    # This matches ablation's approach and avoids noisy early layers
    best_layer = max(layers, key=lambda l: abs(analysis["per_layer"][l]["effect_size_z"]))

    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_pooled": significant_pooled,
        "significant_layers_fdr": significant_fdr,
        "n_significant_pooled": len(significant_pooled),
        "n_significant_fdr": len(significant_fdr),
        "sign_correct_layers_fdr": sign_correct_fdr,
        "sign_correct_layers_pooled": sign_correct_pooled,
        "n_sign_correct_fdr": len(sign_correct_fdr),
        "n_sign_correct_pooled": len(sign_correct_pooled),
        "best_layer": best_layer,
        "best_slope": analysis["per_layer"][best_layer]["introspection_slope"],
        "best_effect_z": analysis["per_layer"][best_layer]["effect_size_z"],
    }

    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_steering_results(analysis: Dict, method: str, output_path: Path):
    """
    Create 3-panel steering visualization for a single method.
    """
    layers = analysis["layers"]
    multipliers = analysis["multipliers"]

    if not layers:
        print(f"  Skipping plot for {method} - no layers")
        return

    fig, axes = plt.subplots(3, 1, figsize=(20, 14))
    fig.suptitle(f"Steering Results: {method.upper()} directions ({analysis['metric']})", fontsize=14)

    x = np.arange(len(layers))
    expected_sign = analysis["expected_slope_sign"]
    sign_str = "negative" if expected_sign < 0 else "positive"

    # Panel 1: Confidence slope by layer (line plot)
    ax1 = axes[0]
    intro_slopes = np.array([analysis["per_layer"][l]["introspection_slope"] for l in layers])
    ctrl_slopes = np.array([analysis["per_layer"][l]["control_slope_mean"] for l in layers])
    ctrl_stds = np.array([analysis["per_layer"][l]["control_slope_std"] for l in layers])
    p_values_pooled = [analysis["per_layer"][l]["p_value_pooled"] for l in layers]
    sign_correct = [analysis["per_layer"][l]["sign_matches_expected"] for l in layers]

    # Plot control band
    ax1.fill_between(x, ctrl_slopes - ctrl_stds, ctrl_slopes + ctrl_stds,
                     color='gray', alpha=0.2, label='Control ±1σ')
    ax1.plot(x, ctrl_slopes, '--', color='gray', linewidth=1, alpha=0.8, label='Control mean')

    # Plot introspection line
    ax1.plot(x, intro_slopes, '-', color='blue', linewidth=1.5, alpha=0.8, label=f'{method}')

    # Mark significant layers (pooled p < 0.05), colored by sign correctness
    sig_correct_x = [i for i, (p, sc) in enumerate(zip(p_values_pooled, sign_correct)) if p < 0.05 and sc]
    sig_wrong_x = [i for i, (p, sc) in enumerate(zip(p_values_pooled, sign_correct)) if p < 0.05 and not sc]

    if sig_correct_x:
        ax1.scatter(sig_correct_x, [intro_slopes[i] for i in sig_correct_x],
                   color='green', s=40, zorder=5, edgecolor='black', linewidth=0.5, label='Sig + correct sign')
    if sig_wrong_x:
        ax1.scatter(sig_wrong_x, [intro_slopes[i] for i in sig_wrong_x],
                   color='red', s=40, zorder=5, edgecolor='black', linewidth=0.5, label='Sig + wrong sign')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Confidence Slope (Δconf / Δmult)")
    ax1.set_title(f"Confidence Slope by Layer (expected slope: {sign_str})")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Confidence vs multiplier for all significant layers (sorted by effect size)
    ax2 = axes[1]

    # Get significant layers sorted by effect size (descending)
    sig_layers = [l for l in layers if analysis["per_layer"][l]["p_value_pooled"] < 0.05]
    sig_layers_sorted = sorted(sig_layers, key=lambda l: abs(analysis["per_layer"][l]["effect_size_z"]), reverse=True)

    # Use a colormap for different layers
    if sig_layers_sorted:
        cmap = plt.cm.viridis
        colors_for_layers = [cmap(i / max(1, len(sig_layers_sorted) - 1)) for i in range(len(sig_layers_sorted))]

        for idx, layer in enumerate(sig_layers_sorted):
            layer_data = analysis["per_layer"][layer]
            intro_conf_by_mult = layer_data["intro_mean_conf_by_mult"]
            intro_confs = [intro_conf_by_mult[m] for m in multipliers]

            # Mark whether sign is correct
            sign_marker = "+" if layer_data["sign_matches_expected"] else "-"
            effect_z = layer_data["effect_size_z"]

            ax2.plot(multipliers, intro_confs, 'o-',
                    label=f'L{layer} (Z={effect_z:+.1f}) {sign_marker}',
                    linewidth=1.5, color=colors_for_layers[idx], markersize=4, alpha=0.8)

        # Plot average control for reference (from best layer)
        best_layer = analysis["summary"]["best_layer"]
        ctrl_conf_by_mult = analysis["per_layer"][best_layer]["ctrl_mean_conf_by_mult"]
        ctrl_confs = [ctrl_conf_by_mult[m] for m in multipliers]
        ax2.plot(multipliers, ctrl_confs, 's--', label='Control avg', linewidth=2, color='gray', alpha=0.5, markersize=4)

        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_xlabel("Steering Multiplier")
        ax2.set_ylabel("Mean Confidence")
        ax2.set_title(f"Confidence vs Multiplier - {len(sig_layers_sorted)} Significant Layers (sorted by |Z|)")
        ax2.legend(loc='best', fontsize=8, ncol=2 if len(sig_layers_sorted) > 6 else 1)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No significant layers found", ha='center', va='center', fontsize=12)
        ax2.set_title("Confidence vs Multiplier - No Significant Layers")

    # Panel 3: Summary text
    ax3 = axes[2]
    ax3.axis('off')

    summary = analysis["summary"]
    best_layer = summary["best_layer"]
    best_stats = analysis["per_layer"][best_layer]

    summary_text = f"""
STEERING ANALYSIS: {method.upper()}

Metric: {analysis['metric']}
Expected slope sign: {sign_str} ({"−" if expected_sign < 0 else "+"}direction → {"lower" if expected_sign < 0 else "higher"} confidence)
Layers tested: {len(layers)}
Questions: {analysis['num_questions']}
Controls per layer: {analysis['num_controls']}
Pooled null size: {summary['pooled_null_size']}

Results:
  Significant layers (p<0.05 pooled): {summary['n_significant_pooled']}
  Significant layers (FDR<0.05): {summary['n_significant_fdr']}
  Sign correct (pooled + expected sign): {summary['n_sign_correct_pooled']}
  Sign correct (FDR + expected sign): {summary['n_sign_correct_fdr']}

Best layer (by |Z|): {summary['best_layer']}
  Slope: {summary['best_slope']:.4f}
  Effect size (Z): {summary['best_effect_z']:.2f}
  p-value (pooled): {best_stats['p_value_pooled']:.4f}
  p-value (FDR): {best_stats['p_value_fdr']:.4f}
  Sign correct: {"Yes" if best_stats['sign_matches_expected'] else "No"}

Interpretation:
"""
    if summary['n_sign_correct_fdr'] > 0:
        summary_text += f"""  ✓ SIGNIFICANT (FDR) with CORRECT SIGN
  {summary['n_sign_correct_fdr']} layer(s) show steering effects in
  the expected direction after FDR correction."""
    elif summary['n_sign_correct_pooled'] > 0:
        summary_text += f"""  ✓ SIGNIFICANT (pooled) with CORRECT SIGN
  {summary['n_sign_correct_pooled']} layer(s) show steering effects in
  the expected direction (nominally significant)."""
    elif summary['n_significant_fdr'] > 0:
        summary_text += f"""  ⚠ SIGNIFICANT but WRONG SIGN
  {summary['n_significant_fdr']} layer(s) show steering effects,
  but in the opposite direction from expected."""
    elif summary['n_significant_pooled'] > 0:
        summary_text += f"""  ⚠ Nominally significant, wrong sign
  {summary['n_significant_pooled']} layer(s) show effects (not FDR-corrected),
  but in the opposite direction from expected."""
    else:
        summary_text += """  ✗ No significant effect detected
  Direction may not be sufficient for steering confidence."""

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()


def plot_method_comparison(analyses: Dict[str, Dict], output_path: Path):
    """
    Create comparison plot of different direction methods.
    """
    methods = list(analyses.keys())
    if len(methods) < 2:
        print("  Skipping comparison plot - need at least 2 methods")
        return

    # Use layers from first method
    layers = analyses[methods[0]]["layers"]
    multipliers = analyses[methods[0]]["multipliers"]

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle("Method Comparison: Steering Effects", fontsize=14)

    x = np.arange(len(layers))
    colors = {'probe': 'tab:blue', 'mean_diff': 'tab:orange'}

    # Panel 1: Slope curves by layer (line plot)
    ax1 = axes[0]
    for method in methods:
        slopes = [analyses[method]["per_layer"][l]["introspection_slope"] for l in layers]
        p_values_pooled = [analyses[method]["per_layer"][l]["p_value_pooled"] for l in layers]
        color = colors.get(method, 'gray')

        # Line plot
        ax1.plot(x, slopes, '-', label=method, color=color, linewidth=1.5, alpha=0.8)

        # Mark significant layers with filled markers (pooled p < 0.05)
        sig_x = [i for i, p in enumerate(p_values_pooled) if p < 0.05]
        sig_y = [slopes[i] for i in sig_x]
        ax1.scatter(sig_x, sig_y, color=color, s=40, zorder=5, edgecolor='black', linewidth=0.5)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Confidence Slope")
    ax1.set_title("Confidence Slope by Method (filled markers = pooled p<0.05)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Summary comparison
    ax2 = axes[1]
    ax2.axis('off')

    expected_sign = analyses[methods[0]]["expected_slope_sign"]
    sign_str = "negative" if expected_sign < 0 else "positive"

    comparison_text = "METHOD COMPARISON\n" + "=" * 40 + "\n\n"
    comparison_text += f"Metric: {analyses[methods[0]]['metric']}\n"
    comparison_text += f"Expected slope sign: {sign_str}\n\n"

    for method in methods:
        summary = analyses[method]["summary"]
        comparison_text += f"{method.upper()}:\n"
        comparison_text += f"  Significant layers (pooled): {summary['n_significant_pooled']}\n"
        comparison_text += f"  Significant layers (FDR): {summary['n_significant_fdr']}\n"
        comparison_text += f"  Sign correct (pooled): {summary['n_sign_correct_pooled']}\n"
        comparison_text += f"  Sign correct (FDR): {summary['n_sign_correct_fdr']}\n"
        comparison_text += f"  Best layer: {summary['best_layer']} (slope={summary['best_slope']:.4f})\n\n"

    # Winner by sign-correct layers (pooled since FDR may be 0)
    best_method = max(methods, key=lambda m: analyses[m]["summary"]["n_sign_correct_pooled"])
    comparison_text += f"Method with most sign-correct layers (pooled): {best_method.upper()}\n"

    ax2.text(0.1, 0.9, comparison_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()


def print_summary(analyses: Dict[str, Dict]):
    """Print summary of steering results."""
    print("\n" + "=" * 70)
    print("STEERING CAUSALITY TEST RESULTS")
    print("=" * 70)

    # Print expected sign info once
    first_method = list(analyses.keys())[0]
    expected_sign = analyses[first_method]["expected_slope_sign"]
    sign_str = "negative" if expected_sign < 0 else "positive"
    print(f"\nMetric: {analyses[first_method]['metric']}")
    print(f"Expected slope sign: {sign_str}")

    for method, analysis in analyses.items():
        summary = analysis["summary"]
        print(f"\n{method.upper()} directions:")
        print(f"  Layers tested: {len(analysis['layers'])}")
        print(f"  Significant (pooled p<0.05): {summary['n_significant_pooled']}")
        print(f"  Significant (FDR p<0.05): {summary['n_significant_fdr']}")
        print(f"  Sign correct (pooled): {summary['n_sign_correct_pooled']}")
        print(f"  Sign correct (FDR): {summary['n_sign_correct_fdr']}")
        print(f"  Best layer: {summary['best_layer']} (slope={summary['best_slope']:.4f}, Z={summary['best_effect_z']:.2f})")

        if summary['sign_correct_layers_pooled']:
            print(f"  Sign-correct layers (pooled): {summary['sign_correct_layers_pooled'][:10]}{'...' if len(summary['sign_correct_layers_pooled']) > 10 else ''}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STEERING CAUSALITY TEST")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print(f"Adapter: {ADAPTER}")
    print(f"Input: {INPUT_BASE_NAME}")
    print(f"Metric: {METRIC}")
    print(f"Meta-task: {META_TASK}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Controls: {NUM_CONTROLS} (final), {NUM_CONTROLS_NONFINAL} (non-final)")
    print(f"Multipliers: {STEERING_MULTIPLIERS}")
    print(f"Positions: {PROBE_POSITIONS}")

    # Load directions
    print("\nLoading directions...")
    all_directions = load_directions(INPUT_BASE_NAME, METRIC)
    available_methods = list(all_directions.keys())
    print(f"  Found methods: {available_methods}")

    # Filter to requested methods
    if METHODS is not None:
        methods = [m for m in METHODS if m in available_methods]
        if not methods:
            raise ValueError(f"None of requested methods {METHODS} found in {available_methods}")
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
        print(f"  Using transfer test split: {len(data_items)} questions (from {n_total} total, seed={SEED})")
    else:
        # Legacy behavior: first NUM_QUESTIONS
        data_items = all_data[:NUM_QUESTIONS]
        print(f"  Using first {len(data_items)} questions (legacy mode)")

    questions = data_items
    metric_values = np.array([item[METRIC] for item in data_items])
    print(f"  Questions: {len(questions)}")
    print(f"  {METRIC}: mean={metric_values.mean():.3f}, std={metric_values.std():.3f}")

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
                    pos_layers = get_layers_from_transfer(transfer_data, METRIC, pos, TRANSFER_R2_THRESHOLD, method)
                    if pos_layers:
                        print(f"  {pos}/{method}: {len(pos_layers)} layers with {METRIC} R²≥{TRANSFER_R2_THRESHOLD}: {pos_layers}")
                    else:
                        # Try fallback to final
                        fallback_layers = get_layers_from_transfer(transfer_data, METRIC, "final", TRANSFER_R2_THRESHOLD, method)
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

    # Run steering for each position and method
    # Structure: {position: {method: analysis}}
    all_results_by_position = {}
    all_analyses_by_position = {}

    for position in PROBE_POSITIONS:
        print(f"\n{'#'*70}")
        print(f"# POSITION: {position}")
        print(f"{'#'*70}")

        # Determine number of controls for this position
        position_num_controls = NUM_CONTROLS if position == "final" else NUM_CONTROLS_NONFINAL

        all_results_by_position[position] = {}
        all_analyses_by_position[position] = {}

        for method in methods:
            print(f"\n{'='*60}")
            print(f"STEERING EXPERIMENT: {method.upper()} @ {position}")
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
                        transfer_data, METRIC, position, TRANSFER_R2_THRESHOLD, method
                    )
                    if not method_layers:
                        # Fall back to "final" position transfer data if position-specific not available
                        method_layers = get_layers_from_transfer(
                            transfer_data, METRIC, "final", TRANSFER_R2_THRESHOLD, method
                        )
                else:
                    method_layers = all_available_layers

                if not method_layers:
                    print("\n" + "!"*70)
                    print("!!! WARNING: FALLING BACK TO ALL LAYERS !!!")
                    print(f"!!! No layers meet R²≥{TRANSFER_R2_THRESHOLD} threshold for {method}/{METRIC}")
                    print(f"!!! This will test {len(all_available_layers)} layers instead of ~50")
                    print(f"!!! Check that METRIC and method match transfer results")
                    print("!"*70)
                    print("Continuing in 3 seconds (Ctrl+C to abort)...")
                    import time
                    time.sleep(3)
                    method_layers = all_available_layers

            print(f"  Layers: {len(method_layers)} (range {min(method_layers)}-{max(method_layers)})")
            print(f"  Controls: {position_num_controls}")

            results = run_steering_for_method(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                metric_values=metric_values,
                directions=all_directions[method],
                num_controls=position_num_controls,
                meta_task=META_TASK,
                multipliers=STEERING_MULTIPLIERS,
                use_chat_template=use_chat_template,
                layers=method_layers,
                position=position,
            )
            all_results_by_position[position][method] = results

            # Analyze results
            print(f"\n  Analyzing results...")
            analysis = analyze_steering_results(results, METRIC)
            all_analyses_by_position[position][method] = analysis

            summary = analysis["summary"]
            print(f"  Significant layers (FDR): {summary['n_significant_fdr']}")
            print(f"  Sign correct (pooled): {summary['n_sign_correct_pooled']}")
            print(f"  Best layer: {summary['best_layer']} (slope={summary['best_slope']:.4f})")

        # Incremental save after each position completes (crash protection)
        model_short = get_model_short_name(MODEL)
        base_output = f"{model_short}_{INPUT_BASE_NAME.split('_')[-1]}_steering_{META_TASK}_{METRIC}"
        checkpoint_path = OUTPUT_DIR / f"{base_output}_checkpoint.json"
        checkpoint_json = {
            "config": {
                "model": MODEL,
                "adapter": ADAPTER,
                "input_base_name": INPUT_BASE_NAME,
                "metric": METRIC,
                "meta_task": META_TASK,
                "num_questions": len(questions),
                "use_transfer_split": USE_TRANSFER_SPLIT,
                "multipliers": STEERING_MULTIPLIERS,
                "positions_completed": [p for p in PROBE_POSITIONS if all_analyses_by_position.get(p)],
            },
            "by_position": {},
        }
        for pos in PROBE_POSITIONS:
            if all_analyses_by_position.get(pos):
                checkpoint_json["by_position"][pos] = {}
                for m, analysis in all_analyses_by_position[pos].items():
                    checkpoint_json["by_position"][pos][m] = {
                        "per_layer": analysis["per_layer"],
                        "summary": analysis["summary"],
                    }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_json, f, indent=2)
        print(f"  Checkpoint saved: {checkpoint_path.name}")

    # Generate output filename
    model_short = get_model_short_name(MODEL)
    base_output = f"{model_short}_{INPUT_BASE_NAME.split('_')[-1]}_steering_{META_TASK}_{METRIC}"

    # Save JSON results
    print("\nSaving results...")
    results_path = OUTPUT_DIR / f"{base_output}_results.json"

    output_json = {
        "config": {
            "model": MODEL,
            "adapter": ADAPTER,
            "input_base_name": INPUT_BASE_NAME,
            "metric": METRIC,
            "meta_task": META_TASK,
            "num_questions": len(questions),
            "use_transfer_split": USE_TRANSFER_SPLIT,
            "num_controls_final": NUM_CONTROLS,
            "num_controls_nonfinal": NUM_CONTROLS_NONFINAL,
            "transfer_r2_threshold": TRANSFER_R2_THRESHOLD,
            "multipliers": STEERING_MULTIPLIERS,
            "methods_tested": methods,
            "positions_tested": PROBE_POSITIONS,
        },
    }

    # Per-position results
    for position in PROBE_POSITIONS:
        output_json[position] = {}
        for method, analysis in all_analyses_by_position[position].items():
            output_json[position][method] = {
                "per_layer": analysis["per_layer"],
                "summary": analysis["summary"],
            }

    # Backward compatibility: keep "final" results at top level (if final was tested)
    default_position = "final" if "final" in all_analyses_by_position else PROBE_POSITIONS[0]
    for method, analysis in all_analyses_by_position[default_position].items():
        output_json[method] = {
            "per_layer": analysis["per_layer"],
            "summary": analysis["summary"],
        }

    # Comparison summary (for default position, for backward compat)
    if len(methods) >= 2:
        default_analyses = all_analyses_by_position[default_position]
        output_json["comparison"] = {
            method: {
                "n_significant_pooled": default_analyses[method]["summary"]["n_significant_pooled"],
                "n_significant_fdr": default_analyses[method]["summary"]["n_significant_fdr"],
                "n_sign_correct_pooled": default_analyses[method]["summary"]["n_sign_correct_pooled"],
                "n_sign_correct_fdr": default_analyses[method]["summary"]["n_sign_correct_fdr"],
                "best_layer": default_analyses[method]["summary"]["best_layer"],
                "best_slope": default_analyses[method]["summary"]["best_slope"],
            }
            for method in methods
        }
        best_method = max(methods, key=lambda m: default_analyses[m]["summary"]["n_sign_correct_pooled"])
        output_json["comparison"]["method_with_more_sign_correct"] = best_method

    with open(results_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Saved {results_path}")

    # Generate plots for each position
    print("\nGenerating plots...")
    for position in PROBE_POSITIONS:
        for method in methods:
            plot_path = OUTPUT_DIR / f"{base_output}_{position}_{method}.png"
            plot_steering_results(all_analyses_by_position[position][method], method, plot_path)

        if len(methods) >= 2:
            comparison_path = OUTPUT_DIR / f"{base_output}_{position}_comparison.png"
            plot_method_comparison(all_analyses_by_position[position], comparison_path)

    # Print summary for each position
    for position in PROBE_POSITIONS:
        print(f"\n{'='*70}")
        print(f"POSITION: {position}")
        print_summary(all_analyses_by_position[position])

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {results_path.name}")
    for position in PROBE_POSITIONS:
        for method in methods:
            print(f"  {base_output}_{position}_{method}.png")
        if len(methods) >= 2:
            print(f"  {base_output}_{position}_comparison.png")


if __name__ == "__main__":
    main()
