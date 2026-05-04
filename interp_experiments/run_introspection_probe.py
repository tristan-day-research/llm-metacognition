"""
Train probe to find introspection direction in meta-activation space.

This script:
1. Loads meta activations and computes introspection alignment scores
2. Trains probe: meta activations → introspection_score
3. Runs permutation tests for statistical significance
4. Saves the introspection direction for steering experiments

Introspection score = -metric_z * confidence_z
- Positive when aligned (high uncertainty + low confidence, or low uncertainty + high confidence)
- Negative when misaligned

The metric can be any uncertainty measure:
- Prob-based (nonlinear): entropy, top_prob, margin
- Logit-based (linear): logit_gap, top_logit

Output files:
- {prefix}_probe_results.json: Probe metrics and significance tests
- {prefix}_probe_directions.npz: Direction vectors for steering

Usage:
    python run_introspection_probe.py --metric logit_gap   # Probe logit_gap (default)
    python run_introspection_probe.py --metric entropy     # Probe entropy
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
from dotenv import load_dotenv
import random

# Import centralized task handling from tasks.py
from prompts import (
    response_to_confidence as _response_to_confidence_impl,
    STATED_CONFIDENCE_MIDPOINTS,
)

load_dotenv()

# =============================================================================
# CONFIGURATION — edit values in experiment_config.IntrospectionProbeConfig
# =============================================================================
from experiment_config import IntrospectionProbeConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
MODEL_NAME = _C.MODEL_NAME
DATASETS = list(_C.DATASETS)
META_TASKS = list(_C.META_TASKS)
DATASET_NAME = _C.DATASET_NAME
META_TASK = _C.META_TASK
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUTS_DIR = _C.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(exist_ok=True)
AVAILABLE_METRICS = list(_C.AVAILABLE_METRICS)
METRIC = _C.METRIC


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from a model path."""
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    # Include meta task type in output prefix for clarity
    task_suffix = f"_{META_TASK}" if META_TASK != "confidence" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


def get_directions_prefix() -> str:
    """Generate output filename prefix for direction files.

    Direction files ARE task-dependent because they're trained on introspection_score,
    which is computed from stated_confidence values that differ between tasks
    (confidence vs delegate produce different confidence distributions).
    """
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = f"_{META_TASK}" if META_TASK != "confidence" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


SEED = 42

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100

# Significance testing
NUM_PERMUTATIONS = 1000

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Backward compatibility alias (now imported from tasks.py)
META_RANGE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS


def response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict = None
) -> float:
    """
    Convert a meta response to a confidence value.

    This is a thin wrapper around tasks.response_to_confidence that uses
    the global META_TASK to determine task type.

    Args:
        response: The model's response ("1", "2", or S-Z for confidence)
        probs: Probability array [P("1"), P("2")] for delegate, or [P(S)...P(Z)] for confidence
        mapping: For delegate task, the mapping {"1": "Answer"/"Delegate", "2": ...}
    """
    return _response_to_confidence_impl(response, probs, mapping, task_type=META_TASK)


# ============================================================================
# INTROSPECTION SCORE COMPUTATION
# ============================================================================

def compute_introspection_scores(
    direct_metric_values: np.ndarray,
    meta_responses: List[str],
    meta_probs: List[List[float]] = None,
    meta_mappings: List[Dict] = None,
    metric_name: str = "entropy"
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray]:
    """
    Compute introspection alignment scores.

    For true introspection: high uncertainty should correlate with LOW confidence.
    So correlation should be NEGATIVE.

    Introspection score = -metric_z * confidence_z
    - Positive when aligned (high uncertainty + low conf, or low uncertainty + high conf)
    - Negative when misaligned

    Note: For metrics where HIGH value = HIGH confidence (top_prob, margin, logit_gap, top_logit),
    we flip the sign so that the correlation interpretation remains consistent:
    negative correlation = introspective behavior.

    Args:
        direct_metric_values: Array of uncertainty metric values from direct MC questions
        meta_responses: List of model responses (S-Z for confidence, "1"/"2" for delegate)
        meta_probs: List of probability arrays for each response (optional, used for delegate task)
        meta_mappings: List of mappings for delegate task (optional)
        metric_name: Name of the metric being used (for stats reporting)
    """
    # Convert meta responses to confidence values using response_to_confidence
    # which handles both confidence and delegate tasks
    stated_confidences = np.array([
        response_to_confidence(
            r,
            np.array(p) if p else None,
            m
        )
        for r, p, m in zip(
            meta_responses,
            meta_probs or [None] * len(meta_responses),
            meta_mappings or [None] * len(meta_responses)
        )
    ])

    # For metrics where HIGH value = HIGH confidence, flip the sign
    # so that introspective behavior always shows as NEGATIVE correlation
    # - entropy: HIGH = uncertain (no flip needed)
    # - top_prob, margin, logit_gap, top_logit: HIGH = confident (flip needed)
    uncertainty_values = direct_metric_values.copy()
    if metric_name != "entropy":
        # These metrics are confidence-like (high = confident), so negate for uncertainty
        uncertainty_values = -uncertainty_values

    # Raw correlation (should be negative for introspection)
    correlation = np.corrcoef(uncertainty_values, stated_confidences)[0, 1]

    # Z-score both
    metric_z = stats.zscore(uncertainty_values)
    confidence_z = stats.zscore(stated_confidences)

    # Introspection score = negative product
    # High score = aligned (high uncertainty and low confidence, or vice versa)
    introspection_scores = -1 * metric_z * confidence_z

    stats_dict = {
        "meta_task": META_TASK,
        "metric": metric_name,
        "correlation_metric_confidence": float(correlation),
        "correlation_interpretation": "negative=introspective, positive=anti-introspective",
        "mean_metric": float(direct_metric_values.mean()),
        "std_metric": float(direct_metric_values.std()),
        "mean_confidence": float(stated_confidences.mean()),
        "std_confidence": float(stated_confidences.std()),
        "mean_introspection_score": float(introspection_scores.mean()),
        "std_introspection_score": float(introspection_scores.std()),
        "fraction_aligned": float((introspection_scores > 0).mean()),
    }

    # Backward compatibility: also include entropy-named keys
    stats_dict["correlation_entropy_confidence"] = stats_dict["correlation_metric_confidence"]
    stats_dict["mean_entropy"] = stats_dict["mean_metric"]
    stats_dict["std_entropy"] = stats_dict["std_metric"]

    # Add delegate-specific statistics
    if META_TASK == "delegate" and meta_mappings is not None:
        # Determine delegation decisions based on response and mapping
        delegated = []
        for response, mapping in zip(meta_responses, meta_mappings):
            if mapping is not None:
                decision = mapping.get(response, "Unknown")
                is_delegated = (decision == "Delegate")
                delegated.append(is_delegated)

        if delegated:
            delegation_rate = sum(delegated) / len(delegated)
            stats_dict["delegation_rate"] = float(delegation_rate)
            stats_dict["num_delegated"] = sum(delegated)
            stats_dict["num_self_answered"] = len(delegated) - sum(delegated)

    return introspection_scores, stats_dict, metric_z, confidence_z


# ============================================================================
# PROBE TRAINING WITH SIGNIFICANCE TESTING
# ============================================================================

def train_probe_with_significance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_permutations: int = 1000
) -> Dict:
    """
    Train a probe and compute significance via permutation test.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    pca = None
    if USE_PCA:
        n_components = min(PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Train probe
    probe = Ridge(alpha=PROBE_ALPHA)
    probe.fit(X_train_final, y_train)

    # Evaluate
    y_pred_train = probe.predict(X_train_final)
    y_pred_test = probe.predict(X_test_final)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Permutation test for significance
    null_r2s = []
    for _ in range(num_permutations):
        y_train_perm = np.random.permutation(y_train)
        probe_perm = Ridge(alpha=PROBE_ALPHA)
        probe_perm.fit(X_train_final, y_train_perm)
        y_pred_perm = probe_perm.predict(X_test_final)
        null_r2s.append(r2_score(y_test, y_pred_perm))

    null_r2s = np.array(null_r2s)
    p_value = (null_r2s >= test_r2).mean()

    # Extract direction
    direction = extract_direction(scaler, pca, probe)

    return {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "p_value": float(p_value),
        "null_r2_mean": float(null_r2s.mean()),
        "null_r2_std": float(null_r2s.std()),
        "null_r2_95th": float(np.percentile(null_r2s, 95)),
        "significant_p05": p_value < 0.05,
        "significant_p01": p_value < 0.01,
        "direction": direction,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()) if pca else None,
    }


def extract_direction(
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge
) -> np.ndarray:
    """Extract normalized direction from probe."""
    coef = probe.coef_

    if pca is not None:
        direction_scaled = pca.components_.T @ coef
    else:
        direction_scaled = coef

    direction_original = direction_scaled / scaler.scale_
    direction_original = direction_original / np.linalg.norm(direction_original)

    return direction_original


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_introspection_probe_analysis(
    meta_activations: Dict[int, np.ndarray],
    introspection_scores: np.ndarray,
    num_permutations: int = 1000
) -> Tuple[Dict, np.ndarray]:
    """
    Train probes to predict introspection score from meta activations.

    This finds directions in meta-activation space that correlate with
    whether the model's stated confidence aligns with its actual entropy.
    """
    print(f"\nTraining introspection probes ({num_permutations} permutations)...")

    # Split data
    n_questions = len(introspection_scores)
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)

    results = {
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "num_permutations": num_permutations,
        "layer_results": {},
    }

    for layer_idx in tqdm(sorted(meta_activations.keys()), desc="Training probes"):
        X = meta_activations[layer_idx]

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = introspection_scores[train_idx]
        y_test = introspection_scores[test_idx]

        # Train probe: meta activations → introspection score
        probe_results = train_probe_with_significance(
            X_train, y_train,
            X_test, y_test,
            num_permutations
        )

        results["layer_results"][layer_idx] = {
            "train_r2": probe_results["train_r2"],
            "test_r2": probe_results["test_r2"],
            "test_mae": probe_results["test_mae"],
            "p_value": probe_results["p_value"],
            "null_r2_mean": probe_results["null_r2_mean"],
            "null_r2_std": probe_results["null_r2_std"],
            "null_r2_95th": probe_results["null_r2_95th"],
            "significant_p05": probe_results["significant_p05"],
            "significant_p01": probe_results["significant_p01"],
            "pca_variance_explained": probe_results["pca_variance_explained"],
            "direction": probe_results["direction"].tolist(),
        }

    return results, test_idx


def find_best_layers(results: Dict) -> Dict:
    """Find best layers for introspection probe."""
    layers = sorted(results["layer_results"].keys())

    # Only consider significant results
    significant_layers = [
        l for l in layers
        if results["layer_results"][l]["significant_p05"]
    ]

    if significant_layers:
        best_layer = max(significant_layers,
                        key=lambda l: results["layer_results"][l]["test_r2"])
        return {
            "layer": best_layer,
            "test_r2": results["layer_results"][best_layer]["test_r2"],
            "p_value": results["layer_results"][best_layer]["p_value"],
        }
    else:
        # No significant results - report best anyway with warning
        best_layer = max(layers,
                        key=lambda l: results["layer_results"][l]["test_r2"])
        return {
            "layer": best_layer,
            "test_r2": results["layer_results"][best_layer]["test_r2"],
            "p_value": results["layer_results"][best_layer]["p_value"],
            "warning": "No statistically significant results"
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results: Dict, score_stats: Dict, output_prefix: str):
    """Create visualization of probe results."""
    layers = sorted(results["layer_results"].keys())

    # Extract data
    test_r2 = [results["layer_results"][l]["test_r2"] for l in layers]
    null_95th = [results["layer_results"][l]["null_r2_95th"] for l in layers]
    p_values = [results["layer_results"][l]["p_value"] for l in layers]

    best_info = find_best_layers(results)
    best_layer = best_info["layer"]
    best_r2 = best_info["test_r2"]
    sig_layers = [l for l in layers if results["layer_results"][l]["significant_p05"]]

    # Find first significant layer for shading
    first_sig = min(sig_layers) if sig_layers else None

    # Create figure with custom layout: 2 plots on top, 1 wide plot + text on bottom
    fig = plt.figure(figsize=(12, 10))

    # Top row: two equal plots
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    # Bottom row: R² across layers (wider) and summary text
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Plot 1: R² by layer with significance shading
    ax1.plot(layers, test_r2, 'o-', label='Test R²', linewidth=2, color='green', markersize=4)
    ax1.plot(layers, null_95th, '--', color='red', alpha=0.7, linewidth=1.5, label='95th pct null')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Shade significant region
    if first_sig is not None:
        ax1.axvspan(first_sig - 0.5, max(layers) + 0.5, alpha=0.1, color='green',
                    label=f'Significant (layers {first_sig}+)')

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Probe Performance by Layer')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: P-values
    p_values_clipped = [max(p, 1e-4) for p in p_values]
    ax2.semilogy(layers, p_values_clipped, 'o-', linewidth=2, color='green', markersize=4)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='p=0.05')
    ax2.axhline(y=0.01, color='orange', linestyle='--', linewidth=1.5, label='p=0.01')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('p-value (log scale)')
    ax2.set_title('Statistical Significance')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-4, 1.1)

    # Plot 3: Zoomed view of significant layers (or all if none significant)
    if sig_layers:
        zoom_layers = [l for l in layers if l >= first_sig - 5]
        zoom_r2 = [results["layer_results"][l]["test_r2"] for l in zoom_layers]
        zoom_null = [results["layer_results"][l]["null_r2_95th"] for l in zoom_layers]
    else:
        zoom_layers = layers
        zoom_r2 = test_r2
        zoom_null = null_95th

    ax3.plot(zoom_layers, zoom_r2, 'o-', label='Test R²', linewidth=2, color='green', markersize=5)
    ax3.plot(zoom_layers, zoom_null, '--', color='red', alpha=0.7, linewidth=1.5, label='95th pct null')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Mark best layer
    ax3.scatter([best_layer], [best_r2], color='red', s=100, zorder=5,
                edgecolor='black', linewidth=1.5, label=f'Best: L{best_layer}')

    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Test R²')
    title = 'Significant Layers (zoomed)' if sig_layers else 'All Layers'
    ax3.set_title(title)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary text box (constrained width)
    ax4.axis('off')

    # Build interpretation
    if "warning" in best_info:
        interpretation = f"Warning: {best_info['warning']}"
    elif best_r2 > 0.1:
        interpretation = "Strong introspection signal detected"
    elif best_r2 > 0.05:
        interpretation = "Moderate introspection signal"
    else:
        interpretation = "Weak introspection signal"

    # Format significant layers compactly
    if sig_layers:
        if len(sig_layers) > 10:
            sig_str = f"{sig_layers[0]}-{sig_layers[-1]}"
        else:
            sig_str = str(sig_layers)
    else:
        sig_str = "None"

    summary_lines = [
        "SUMMARY",
        "",
        f"Task: {score_stats.get('meta_task', 'confidence')}",
        "",
        "Behavioral:",
        f"  Entropy-Conf corr: {score_stats['correlation_entropy_confidence']:.3f}",
        f"  Fraction aligned: {score_stats['fraction_aligned']:.1%}",
    ]

    # Add delegate-specific stats if available
    if "delegation_rate" in score_stats:
        summary_lines.extend([
            f"  Delegation rate: {score_stats['delegation_rate']:.1%}",
        ])

    summary_lines.extend([
        "",
        "Probe Results:",
        f"  Best layer: {best_layer}",
        f"  Best R²: {best_r2:.4f}",
        f"  p-value: {best_info['p_value']:.4f}",
        "",
        f"Significant layers: {len(sig_layers)}/{len(layers)}",
        f"  {sig_str}",
        "",
        f"Interpretation:",
        f"  {interpretation}",
    ])
    summary = "\n".join(summary_lines)

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_probe_results.png", dpi=150, bbox_inches='tight')
    print(f"Saved {output_prefix}_probe_results.png")
    plt.close()


def print_results(results: Dict, score_stats: Dict):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("INTROSPECTION PROBE RESULTS")
    print("=" * 70)

    print(f"\n--- Behavioral Statistics ---")
    print(f"Meta-judgment task: {score_stats.get('meta_task', 'confidence')}")
    print(f"Entropy-Confidence Correlation: {score_stats['correlation_entropy_confidence']:.4f}")
    print(f"  ({score_stats['correlation_interpretation']})")
    print(f"Fraction aligned: {score_stats['fraction_aligned']:.1%}")

    # Print delegate-specific stats if available
    if "delegation_rate" in score_stats:
        print(f"\n--- Delegate Task Statistics ---")
        print(f"Delegation rate: {score_stats['delegation_rate']:.1%}")
        print(f"  Self-answered: {score_stats['num_self_answered']}")
        print(f"  Delegated: {score_stats['num_delegated']}")

    print(f"\n--- Probe Performance by Layer ---")
    print(f"{'Layer':<6} {'Test R²':<10} {'p-value':<10} {'Null 95th':<10} {'Sig?':<6}")
    print("-" * 45)

    layers = sorted(results["layer_results"].keys())
    for layer in layers:
        lr = results["layer_results"][layer]
        sig = "*" if lr["significant_p05"] else " "
        sig += "*" if lr["significant_p01"] else " "

        print(f"{layer:<6} {lr['test_r2']:<10.4f} {lr['p_value']:<10.4f} "
              f"{lr['null_r2_95th']:<10.4f} {sig}")

    print("\n* = p < 0.05, ** = p < 0.01")

    best = find_best_layers(results)
    print(f"\n--- Best Layer ---")
    warning = f" WARNING: {best['warning']}" if "warning" in best else ""
    print(f"Layer {best['layer']}: R² = {best['test_r2']:.4f}, p = {best['p_value']:.4f}{warning}")

    # List all significant layers
    sig_layers = [l for l in layers if results["layer_results"][l]["significant_p05"]]
    if sig_layers:
        print(f"\nSignificant layers for steering: {sig_layers}")
    else:
        print("\nNo significant layers found.")


# ============================================================================
# MAIN
# ============================================================================

def run_single_probe(dataset_name: str, meta_task: str, metric: str):
    """Run introspection probe for a single dataset/task combination."""
    global DATASET_NAME, META_TASK, METRIC

    # Update global variables for this run
    DATASET_NAME = dataset_name
    META_TASK = meta_task
    METRIC = metric

    print("\n" + "=" * 80)
    print(f"Running: {dataset_name} / {meta_task} / {metric}")
    print("=" * 80)

    # Generate output prefix (base prefix without metric for input files)
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Compute input/output paths from prefix
    paired_data_path = f"{output_prefix}_paired_data.json"
    meta_activations_path = f"{output_prefix}_meta_activations.npz"

    # Check if input files exist
    if not Path(paired_data_path).exists():
        print(f"  Skipping: {paired_data_path} not found")
        return
    if not Path(meta_activations_path).exists():
        print(f"  Skipping: {meta_activations_path} not found")
        return

    # Load paired data
    print(f"\nLoading paired data from {paired_data_path}...")
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)

    # Load the selected metric's values
    # New format has direct_metrics (dict of metric_name -> list of values)
    # Old format has direct_entropies (list of values)
    if "direct_metrics" in paired_data and metric in paired_data["direct_metrics"]:
        direct_metric_values = np.array(paired_data["direct_metrics"][metric])
        print(f"Using metric '{metric}' from paired data")
    elif "direct_entropies" in paired_data:
        # Backward compatibility: fall back to direct_entropies (only works for entropy)
        if metric != "entropy":
            print(f"  Skipping: Metric '{metric}' not found in paired data (old format only has entropy)")
            return
        direct_metric_values = np.array(paired_data["direct_entropies"])
        print(f"Using 'entropy' (backward compatible fallback)")
    else:
        print("  Skipping: Paired data missing both 'direct_metrics' and 'direct_entropies'")
        return

    meta_responses = paired_data["meta_responses"]
    # Load meta_probs and meta_mappings for delegate task support
    meta_probs = paired_data.get("meta_probs")
    meta_mappings = paired_data.get("meta_mappings")

    print(f"Loaded {len(direct_metric_values)} questions")
    print(f"  {metric}: range=[{direct_metric_values.min():.3f}, {direct_metric_values.max():.3f}], "
          f"mean={direct_metric_values.mean():.3f}, std={direct_metric_values.std():.3f}")

    # Compute introspection scores
    # For delegate task, confidence = P(Answer) from meta_probs
    # For confidence task, confidence = midpoint of chosen range
    print("\nComputing introspection scores...")
    introspection_scores, score_stats, metric_z, confidence_z = \
        compute_introspection_scores(
            direct_metric_values,
            meta_responses,
            meta_probs,
            meta_mappings,
            metric_name=metric
        )

    print(f"  Correlation ({metric}, confidence): {score_stats['correlation_metric_confidence']:.4f}")
    print(f"  Fraction aligned: {score_stats['fraction_aligned']:.1%}")

    # Load meta activations
    print(f"\nLoading meta activations from {meta_activations_path}...")
    meta_acts_data = np.load(meta_activations_path)
    meta_activations = {
        int(k.split("_")[1]): meta_acts_data[k]
        for k in meta_acts_data.files if k.startswith("layer_")
    }
    print(f"Loaded {len(meta_activations)} layers")

    # Run analysis
    results, test_idx = run_introspection_probe_analysis(
        meta_activations, introspection_scores, NUM_PERMUTATIONS
    )

    results["score_stats"] = score_stats
    results["best_layer"] = find_best_layers(results)

    # Save results
    results_to_save = {
        "config": {
            "metric": metric,
            "meta_task": meta_task,
            "model": BASE_MODEL_NAME,
            "dataset": dataset_name,
        },
        "score_stats": score_stats,
        "train_size": results["train_size"],
        "test_size": results["test_size"],
        "num_permutations": results["num_permutations"],
        "best_layer": results["best_layer"],
        "test_indices": test_idx.tolist(),
        "layer_results": {}
    }

    for layer_idx, layer_results in results["layer_results"].items():
        results_to_save["layer_results"][str(layer_idx)] = {
            k: v for k, v in layer_results.items() if k != "direction"
        }

    def json_serializer(obj):
        """Handle numpy types for JSON serialization."""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Include metric in output filenames
    output_results = f"{output_prefix}_{metric}_probe_results.json"
    with open(output_results, "w") as f:
        json.dump(results_to_save, f, indent=2, default=json_serializer)
    print(f"\nSaved {output_results}")

    # Save directions (task-independent filename)
    # Directions are shared across tasks since they capture the same underlying signal
    directions = {}
    for layer_idx, layer_results in results["layer_results"].items():
        directions[f"layer_{layer_idx}_introspection"] = np.array(layer_results["direction"])

    # Add metadata
    directions["_metadata_metric"] = np.array(metric)
    directions["_metadata_dataset"] = np.array(dataset_name)
    directions["_metadata_model"] = np.array(BASE_MODEL_NAME)

    directions_prefix = get_directions_prefix()
    output_directions = f"{directions_prefix}_{metric}_probe_directions.npz"
    np.savez_compressed(output_directions, **directions)
    print(f"Saved {output_directions}")

    # Print and plot results
    print_results(results, score_stats)
    plot_results(results, score_stats, f"{output_prefix}_{metric}")

    print(f"\n✓ Complete: {dataset_name} / {meta_task} / {metric}")


def main():
    parser = argparse.ArgumentParser(description="Train introspection probe on meta activations")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Uncertainty metric to use for introspection score (default: {METRIC})")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Metric: {args.metric}")
    print(f"Datasets to process: {DATASETS}")
    print(f"Meta-tasks to process: {META_TASKS}")
    print(f"Total combinations: {len(DATASETS) * len(META_TASKS)}")

    # Run all dataset/task combinations
    for dataset_name in DATASETS:
        for meta_task in META_TASKS:
            run_single_probe(dataset_name, meta_task, args.metric)

    print("\n" + "=" * 80)
    print("✓ All experiments complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
