"""
Compute contrastive direction vectors from introspection data.

This script loads data from run_introspection_experiment.py and computes
contrastive directions (confidence and/or calibration) without running
steering or ablation experiments.

Direction types:
- "confidence": high_conf vs low_conf within calibrated examples
- "calibration": calibrated vs uncalibrated examples

Usage:
    python compute_contrastive_directions.py
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

from core import get_model_short_name

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path if using fine-tuned model

# Lists of datasets and meta_tasks to process (will iterate through all combinations)
# Set to a single-item list for single runs, or multiple items to batch process
DATASETS = ["SimpleMC", "TriviaMC"]  # Options: "SimpleMC", "TriviaMC", "GPQA", etc.
META_TASKS = ["confidence", "delegate"]  # Options: "confidence", "delegate"

# Legacy single-value variables (used by functions that reference them)
DATASET_NAME = DATASETS[0]  # Will be updated during iteration
META_TASK = META_TASKS[0]  # Will be updated during iteration

METRIC = "entropy"  # Which metric to use for direction computation

# Direction types to compute: "confidence" and/or "calibration"
DIRECTION_TYPES = ["confidence", "calibration"]

OUTPUTS_DIR = Path("outputs")

# Metric configuration
METRIC_KEY_MAP = {
    "entropy": "direct_entropies",
    "top_prob": "direct_top_probs",
    "margin": "direct_margins",
    "logit_gap": "direct_logit_gaps",
    "top_logit": "direct_top_logits",
}

# Whether higher metric value means more confident
METRIC_HIGHER_IS_CONFIDENT = {
    "entropy": False,      # Low entropy = confident
    "top_prob": True,      # High top_prob = confident
    "margin": True,        # High margin = confident
    "logit_gap": True,     # High logit_gap = confident
    "top_logit": True,     # High top_logit = confident
}


# =============================================================================
# Data Loading
# =============================================================================

def get_introspection_prefix() -> str:
    """Get prefix for introspection data files (from run_introspection_experiment.py)."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


def get_output_prefix() -> str:
    """Get prefix for output files from this script."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_{METRIC}{task_suffix}_contrastive")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_{METRIC}{task_suffix}_contrastive")


def load_introspection_data(run_name: str = None, metric: str = None) -> dict:
    """
    Load previously collected introspection data.

    Looks for:
    - {run_name}_paired_data.json (or computed from config)
    - {run_name}_meta_activations.npz

    Args:
        run_name: Optional run name prefix for data files
        metric: Which metric to load (defaults to METRIC config)
    """
    if metric is None:
        metric = METRIC

    # Try to find data files
    if run_name:
        paired_path = Path(f"{run_name}_paired_data.json")
        acts_path = Path(f"{run_name}_meta_activations.npz")
    else:
        prefix = get_introspection_prefix()
        paired_path = Path(f"{prefix}_paired_data.json")
        acts_path = Path(f"{prefix}_meta_activations.npz")

    if not paired_path.exists():
        raise FileNotFoundError(
            f"Could not find {paired_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    if not acts_path.exists():
        raise FileNotFoundError(
            f"Could not find {acts_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    # Load paired data
    print(f"Loading paired data from {paired_path}...")
    with open(paired_path) as f:
        paired_data = json.load(f)

    # Extract arrays - handle both old format (list of dicts) and new format (dict with arrays)
    if isinstance(paired_data, list):
        # Old format: list of {"direct_entropy": ..., "stated_confidence": ...}
        metric_values = np.array([d["direct_entropy"] for d in paired_data])
        stated_confidences = np.array([d["stated_confidence"] for d in paired_data])
    else:
        # New format from run_introspection_experiment.py
        # Check for nested direct_metrics format first
        if "direct_metrics" in paired_data and isinstance(paired_data["direct_metrics"], dict):
            # Nested format: {"direct_metrics": {"entropy": [...], "top_prob": [...], ...}}
            direct_metrics = paired_data["direct_metrics"]
            if metric not in direct_metrics:
                available = list(direct_metrics.keys())
                raise ValueError(f"Metric '{metric}' not found in direct_metrics. Available: {available}")
            metric_values = np.array(direct_metrics[metric])
        else:
            # Flat format: {"direct_entropies": [...], "direct_top_probs": [...], ...}
            metric_key = METRIC_KEY_MAP.get(metric, "direct_entropies")
            if metric_key not in paired_data:
                available = [k for k in paired_data.keys() if k.startswith("direct_")]
                raise ValueError(f"Metric '{metric}' (key: {metric_key}) not found in data. Available: {available}")
            metric_values = np.array(paired_data[metric_key])

        meta_responses = paired_data["meta_responses"]
        meta_probs = paired_data.get("meta_probs")
        meta_mappings = paired_data.get("meta_mappings")
        meta_task = paired_data.get("config", {}).get("meta_task", "confidence")

        # Convert meta responses to confidence values
        if meta_task == "delegate":
            # For delegate task, confidence = P(Answer)
            stated_confidences = []
            for i, (response, probs, mapping) in enumerate(zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses),
                meta_mappings or [None] * len(meta_responses)
            )):
                if probs is not None and len(probs) >= 2 and mapping is not None:
                    if mapping.get("1") == "Answer":
                        stated_confidences.append(float(probs[0]))
                    else:
                        stated_confidences.append(float(probs[1]))
                elif probs is not None and len(probs) >= 1:
                    stated_confidences.append(float(probs[0]))
                else:
                    stated_confidences.append(1.0 if response == "1" else 0.0)
            stated_confidences = np.array(stated_confidences)
        else:
            # For confidence task, use logprob-weighted expected value
            META_RANGE_MIDPOINTS = {
                "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
                "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
            }
            options = list(META_RANGE_MIDPOINTS.keys())
            midpoints = np.array([META_RANGE_MIDPOINTS[opt] for opt in options])

            stated_confidences = []
            for i, (response, probs) in enumerate(zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses)
            )):
                if probs is not None and len(probs) == len(midpoints):
                    # Logprob-weighted expected value: sum(P(option) * midpoint)
                    probs_arr = np.array(probs)
                    expected_conf = float(np.dot(probs_arr, midpoints))
                    stated_confidences.append(expected_conf)
                else:
                    # Fallback to argmax response midpoint if probs unavailable
                    stated_confidences.append(META_RANGE_MIDPOINTS.get(response, 0.5))
            stated_confidences = np.array(stated_confidences)

    # Load meta activations
    print(f"Loading meta activations from {acts_path}...")
    acts_data = np.load(acts_path)

    # activations are stored as layer_0, layer_1, etc.
    layer_keys = sorted([k for k in acts_data.keys() if k.startswith("layer_")],
                        key=lambda x: int(x.split("_")[1]))
    num_layers = len(layer_keys)

    meta_activations = {
        int(k.split("_")[1]): acts_data[k]
        for k in layer_keys
    }

    n_examples = len(metric_values)
    print(f"Loaded {n_examples} examples with {num_layers} layers")
    print(f"{metric.capitalize()} range: [{metric_values.min():.3f}, {metric_values.max():.3f}]")
    print(f"Confidence range: [{stated_confidences.min():.3f}, {stated_confidences.max():.3f}]")

    return {
        "paired_data": paired_data,
        "metric_values": metric_values,
        "metric_name": metric,
        "stated_confidences": stated_confidences,
        "meta_activations": meta_activations,
        "num_layers": num_layers,
    }


# =============================================================================
# Direction Computation
# =============================================================================

def compute_confidence_direction(
    meta_activations: np.ndarray,
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    metric_higher_is_confident: bool = False,
) -> dict:
    """
    Compute confidence direction: high_conf vs low_conf within calibrated examples.

    Contrasts correctly high-confidence examples vs correctly low-confidence examples:
    - High confidence group: high stated confidence AND high model confidence
    - Low confidence group: low stated confidence AND low model confidence

    This captures the confidence axis within calibrated examples only.

    Args:
        meta_activations: Activation vectors for each example
        metric_values: The uncertainty metric values (e.g., entropy, top_prob)
        stated_confidences: Model's stated confidence values
        metric_higher_is_confident: If True, higher metric = more confident

    Returns direction and detailed info about selected examples.
    """
    # Z-score normalize
    metric_z = stats.zscore(metric_values)
    confidence_z = stats.zscore(stated_confidences)

    # Normalize metric direction so positive = more confident
    if not metric_higher_is_confident:
        metric_z_conf = -metric_z  # Invert so positive = more confident
    else:
        metric_z_conf = metric_z

    # Introspection score: positive when calibrated
    introspection_scores = metric_z_conf * confidence_z

    # Only consider well-calibrated examples (positive introspection score)
    calibrated_mask = introspection_scores > 0

    # Within calibrated examples, split by confidence
    high_conf_mask = calibrated_mask & (confidence_z > 0) & (metric_z_conf > 0)
    low_conf_mask = calibrated_mask & (confidence_z < 0) & (metric_z_conf < 0)

    high_conf_acts = meta_activations[high_conf_mask]
    low_conf_acts = meta_activations[low_conf_mask]

    if len(high_conf_acts) == 0 or len(low_conf_acts) == 0:
        return {
            "direction": None,
            "direction_magnitude": 0.0,
            "n_high_conf": int(high_conf_mask.sum()),
            "n_low_conf": int(low_conf_mask.sum()),
            "n_calibrated": int(calibrated_mask.sum()),
            "error": f"Not enough examples: high_conf={high_conf_mask.sum()}, low_conf={low_conf_mask.sum()}"
        }

    # Compute direction: high confidence - low confidence
    high_conf_mean = high_conf_acts.mean(axis=0)
    low_conf_mean = low_conf_acts.mean(axis=0)

    direction = high_conf_mean - low_conf_mean
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm

    return {
        "direction": direction_normalized,
        "direction_magnitude": direction_norm,
        "n_high_conf": int(high_conf_mask.sum()),
        "n_low_conf": int(low_conf_mask.sum()),
        "n_calibrated": int(calibrated_mask.sum()),
        "high_conf_metric_mean": float(metric_values[high_conf_mask].mean()),
        "high_conf_confidence_mean": float(stated_confidences[high_conf_mask].mean()),
        "low_conf_metric_mean": float(metric_values[low_conf_mask].mean()),
        "low_conf_confidence_mean": float(stated_confidences[low_conf_mask].mean()),
    }


def compute_calibration_direction(
    meta_activations: np.ndarray,
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    metric_higher_is_confident: bool = False,
) -> dict:
    """
    Compute calibration direction: calibrated vs uncalibrated examples.

    - Calibrated: introspection_score > 0 (stated confidence tracks model confidence)
    - Uncalibrated: introspection_score < 0 (stated confidence inversely tracks model confidence)

    Args:
        meta_activations: Activation vectors for each example
        metric_values: The uncertainty metric values (e.g., entropy, top_prob)
        stated_confidences: Model's stated confidence values
        metric_higher_is_confident: If True, higher metric = more confident

    Returns direction and detailed info about selected examples.
    """
    # Z-score normalize
    metric_z = stats.zscore(metric_values)
    confidence_z = stats.zscore(stated_confidences)

    # Normalize metric direction so positive = more confident
    if not metric_higher_is_confident:
        metric_z_conf = -metric_z
    else:
        metric_z_conf = metric_z

    # Introspection score: positive when calibrated
    introspection_scores = metric_z_conf * confidence_z

    # Split by calibration status
    calibrated_mask = introspection_scores > 0
    uncalibrated_mask = introspection_scores < 0

    calibrated_acts = meta_activations[calibrated_mask]
    uncalibrated_acts = meta_activations[uncalibrated_mask]

    if len(calibrated_acts) == 0 or len(uncalibrated_acts) == 0:
        return {
            "direction": None,
            "direction_magnitude": 0.0,
            "n_calibrated": int(calibrated_mask.sum()),
            "n_uncalibrated": int(uncalibrated_mask.sum()),
            "error": f"Not enough examples: calibrated={calibrated_mask.sum()}, uncalibrated={uncalibrated_mask.sum()}"
        }

    # Compute direction: calibrated - uncalibrated
    calibrated_mean = calibrated_acts.mean(axis=0)
    uncalibrated_mean = uncalibrated_acts.mean(axis=0)

    direction = calibrated_mean - uncalibrated_mean
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm

    return {
        "direction": direction_normalized,
        "direction_magnitude": direction_norm,
        "n_calibrated": int(calibrated_mask.sum()),
        "n_uncalibrated": int(uncalibrated_mask.sum()),
        "calibrated_metric_mean": float(metric_values[calibrated_mask].mean()),
        "calibrated_confidence_mean": float(stated_confidences[calibrated_mask].mean()),
        "uncalibrated_metric_mean": float(metric_values[uncalibrated_mask].mean()),
        "uncalibrated_confidence_mean": float(stated_confidences[uncalibrated_mask].mean()),
    }


def compute_projection_correlation(
    activations: np.ndarray,
    direction: np.ndarray,
    target_values: np.ndarray,
) -> dict:
    """
    Compute correlation between activation projection onto direction and target values.

    Args:
        activations: (n_examples, hidden_dim) activation matrix
        direction: (hidden_dim,) unit direction vector
        target_values: (n_examples,) target metric values

    Returns dict with correlation, R², and p-value.
    """
    # Project activations onto direction
    projections = activations @ direction

    # Compute correlation
    r, p = stats.pearsonr(projections, target_values)
    r2 = r ** 2

    return {
        "r": float(r),
        "r2": float(r2),
        "p_value": float(p),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_direction_quality(
    layer_stats: dict,
    correlations: dict,
    direction_type: str,
    output_path: str,
):
    """
    Create a 4-panel visualization of direction quality.

    Panels:
    1. Direction magnitude per layer (bar chart)
    2. Projection correlation per layer (R or R²)
    3. Sample sizes per layer
    4. Summary text with best layer info
    """
    layers = sorted(layer_stats.keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Direction magnitude
    ax1 = axes[0, 0]
    magnitudes = [layer_stats[l].get("direction_magnitude", 0) for l in layers]
    ax1.bar(layers, magnitudes, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Direction Magnitude (L2 norm)")
    ax1.set_title(f"{direction_type.capitalize()} Direction Magnitude by Layer")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Projection correlation
    ax2 = axes[0, 1]
    r_values = [correlations.get(l, {}).get("r", 0) for l in layers]
    colors = ["green" if r > 0 else "red" for r in r_values]
    ax2.bar(layers, r_values, color=colors, alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Pearson r")
    ax2.set_title(f"Projection → {METRIC.capitalize()} Correlation")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # Find best layer by absolute correlation
    best_layer = max(layers, key=lambda l: abs(correlations.get(l, {}).get("r", 0)))
    ax2.axvline(best_layer, color="gold", linestyle="--", linewidth=2, label=f"Best: L{best_layer}")
    ax2.legend()

    # Panel 3: Sample sizes
    ax3 = axes[1, 0]
    if direction_type == "confidence":
        n_high = [layer_stats[l].get("n_high_conf", 0) for l in layers]
        n_low = [layer_stats[l].get("n_low_conf", 0) for l in layers]
        width = 0.35
        x = np.arange(n_layers)
        ax3.bar(x - width/2, n_high, width, label="High conf", color="green", alpha=0.7)
        ax3.bar(x + width/2, n_low, width, label="Low conf", color="red", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(l) for l in layers])
    else:  # calibration
        n_cal = [layer_stats[l].get("n_calibrated", 0) for l in layers]
        n_uncal = [layer_stats[l].get("n_uncalibrated", 0) for l in layers]
        width = 0.35
        x = np.arange(n_layers)
        ax3.bar(x - width/2, n_cal, width, label="Calibrated", color="green", alpha=0.7)
        ax3.bar(x + width/2, n_uncal, width, label="Uncalibrated", color="orange", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(l) for l in layers])
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Sample Count")
    ax3.set_title("Sample Sizes by Layer")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis("off")

    best_r = correlations.get(best_layer, {}).get("r", 0)
    best_r2 = correlations.get(best_layer, {}).get("r2", 0)
    best_p = correlations.get(best_layer, {}).get("p_value", 1)
    best_stats = layer_stats.get(best_layer, {})

    summary = f"""
DIRECTION QUALITY SUMMARY
{'=' * 40}

Direction type: {direction_type}
Metric: {METRIC}
Dataset: {DATASET_NAME}
Task: {META_TASK}

Best layer: {best_layer}
  Correlation (r): {best_r:.4f}
  R²: {best_r2:.4f}
  p-value: {best_p:.2e}
  Magnitude: {best_stats.get('direction_magnitude', 0):.4f}
"""

    if direction_type == "confidence":
        summary += f"""
  n_high_conf: {best_stats.get('n_high_conf', 0)}
  n_low_conf: {best_stats.get('n_low_conf', 0)}
  n_calibrated: {best_stats.get('n_calibrated', 0)}
"""
    else:
        summary += f"""
  n_calibrated: {best_stats.get('n_calibrated', 0)}
  n_uncalibrated: {best_stats.get('n_uncalibrated', 0)}
"""

    # Add interpretation
    if abs(best_r) > 0.3:
        summary += f"\nInterpretation: STRONG direction found"
    elif abs(best_r) > 0.15:
        summary += f"\nInterpretation: MODERATE direction found"
    else:
        summary += f"\nInterpretation: WEAK direction"

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    model_short = get_model_short_name(BASE_MODEL_NAME)
    fig.suptitle(f"{direction_type.capitalize()} Direction Quality\n{model_short} - {DATASET_NAME} - {METRIC}",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def run_single_computation(dataset_name: str, meta_task: str):
    """Run contrastive direction computation for a single dataset/task combination."""
    global DATASET_NAME, META_TASK

    # Update global variables for this run
    DATASET_NAME = dataset_name
    META_TASK = meta_task

    print(f"\n{'=' * 70}")
    print("CONTRASTIVE DIRECTION COMPUTATION")
    print("=" * 70)
    print(f"Model: {get_model_short_name(BASE_MODEL_NAME)}")
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {METRIC}")
    print(f"Task: {meta_task}")
    print(f"Direction types: {DIRECTION_TYPES}")

    # Load data
    try:
        data = load_introspection_data()
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        return

    metric_values = data["metric_values"]
    stated_confidences = data["stated_confidences"]
    meta_activations = data["meta_activations"]
    num_layers = data["num_layers"]

    metric_higher_is_confident = METRIC_HIGHER_IS_CONFIDENT.get(METRIC, False)

    output_prefix = get_output_prefix()

    # Process each direction type
    for dir_type in DIRECTION_TYPES:
        print(f"\n{'=' * 60}")
        print(f"Computing {dir_type} direction...")
        print("=" * 60)

        layer_stats = {}
        layer_correlations = {}
        directions = {}

        for layer_idx in range(num_layers):
            acts = meta_activations[layer_idx]

            # Compute direction
            if dir_type == "confidence":
                result = compute_confidence_direction(
                    acts, metric_values, stated_confidences, metric_higher_is_confident
                )
            else:  # calibration
                result = compute_calibration_direction(
                    acts, metric_values, stated_confidences, metric_higher_is_confident
                )

            layer_stats[layer_idx] = result

            if result["direction"] is not None:
                directions[f"layer_{layer_idx}"] = result["direction"]

                # Compute projection correlation
                corr = compute_projection_correlation(
                    acts, result["direction"], metric_values
                )
                layer_correlations[layer_idx] = corr
            else:
                print(f"  Layer {layer_idx}: {result.get('error', 'Unknown error')}")

        # Find best layer
        if layer_correlations:
            best_layer = max(layer_correlations.keys(),
                            key=lambda l: abs(layer_correlations[l]["r"]))
            best_r = layer_correlations[best_layer]["r"]
            print(f"\nBest layer: {best_layer} (r={best_r:.4f})")

        # Save directions
        directions_path = f"{output_prefix}_{dir_type}_directions.npz"
        np.savez_compressed(directions_path, **directions)
        print(f"Directions saved to: {directions_path}")

        # Save stats JSON
        stats_path = f"{output_prefix}_{dir_type}_stats.json"
        # Convert numpy types for JSON serialization
        stats_serializable = {}
        for layer_idx, s in layer_stats.items():
            stats_serializable[str(layer_idx)] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in s.items()
                if k != "direction"  # Don't save direction in JSON
            }
            if layer_idx in layer_correlations:
                stats_serializable[str(layer_idx)]["correlation"] = layer_correlations[layer_idx]

        with open(stats_path, "w") as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"Stats saved to: {stats_path}")

        # Create visualization
        plot_path = f"{output_prefix}_{dir_type}_quality.png"
        plot_direction_quality(layer_stats, layer_correlations, dir_type, plot_path)

    print(f"\n✓ Complete: {dataset_name} / {meta_task}")


def main():
    print(f"\n{'=' * 70}")
    print("BATCH CONTRASTIVE DIRECTION COMPUTATION")
    print("=" * 70)
    print(f"Datasets to process: {DATASETS}")
    print(f"Meta-tasks to process: {META_TASKS}")
    print(f"Total combinations: {len(DATASETS) * len(META_TASKS)}")
    print(f"Metric: {METRIC}")
    print(f"Direction types: {DIRECTION_TYPES}")

    # Run all dataset/task combinations
    for dataset_name in DATASETS:
        for meta_task in META_TASKS:
            run_single_computation(dataset_name, meta_task)

    print(f"\n{'=' * 70}")
    print("ALL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
