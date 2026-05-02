"""
Test whether probes trained on direct MC task transfer to meta-tasks.

This tests the core introspection hypothesis: if the model encodes uncertainty
during direct MC answering, does the same representation appear when the model
reports its confidence?

Approach (matching run_introspection_experiment.py):
1. Create train/test split (same indices for both tasks)
2. Train probes on direct_train activations → direct metric values
3. Test D→D on direct_test (sanity check)
4. Test D→M on meta_test (the transfer test)

Loads from identify_mc_correlate.py outputs:
- {model}_{dataset}_mc_activations.npz: Direct task activations
- {model}_{dataset}_mc_dataset.json: Questions and metric values

Uses TWO scaling approaches for D→M (matching run_introspection_experiment.py):
1. Centered Scaler (Rigorous): Center meta with own mean, scale with direct's std
2. Separate Scaler (Upper Bound): Refit scaler on meta data - domain adaptation

Configuration is set at the top of the script - no CLI args needed.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import joblib

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    should_use_chat_template,
    BatchedExtractor,
    apply_probe_shared,
    apply_probe_centered,
    apply_probe_separate,
    metric_sign_for_confidence,
)
from core.directions import probe_direction
from prompts import (
    format_stated_confidence_prompt,
    format_answer_or_delegate_prompt,
    get_stated_confidence_signal,
    get_answer_or_delegate_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
    find_mc_positions,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base name for input files from identify_mc_correlate.py
# Will load: {INPUT_BASE_NAME}_mc_{metric}_probes.joblib and {INPUT_BASE_NAME}_mc_dataset.json
# The model is inferred from the dataset JSON, so no need to specify MODEL separately.
INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC"

# Which metrics to test transfer for
METRICS = ["entropy"]

# Optional adapter (must match identify step if used)
ADAPTER = None
# ADAPTER = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"

# Meta task to test: "confidence" or "delegate"
META_TASK = "delegate"

# Processing
BATCH_SIZE = 4 if META_TASK == "delegate" else 8

# Bootstrap for confidence intervals
N_BOOTSTRAP = 100

# Train/test split (must match original for fair comparison)
TRAIN_SPLIT = 0.8
SEED = 42

# Probe training parameters (should match identify_mc_correlate.py)
PROBE_ALPHA = 1000.0
PROBE_PCA_COMPONENTS = 100

# Quantization (auto-detected from model name, but can override)
LOAD_IN_4BIT = False #True if "70B" in model_name else False
LOAD_IN_8BIT = False

# Token positions to probe for transfer
# question_mark: "?" at end of embedded MC question
# question_newline: newline after "?"
# options_newline: newline after last MC option (D: ...)
# final: last token (current behavior)
PROBE_POSITIONS = ["question_mark", "question_newline", "options_newline", "final"]

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_probes(probes_path: Path) -> dict:
    """Load probe pipeline from a _probes.joblib file."""
    data = joblib.load(probes_path)
    return data


def _find_directions_npz(input_base_name: str, metric: str, output_dir: Path) -> Path:
    """Locate the *_directions.npz file produced by identify_mc_correlate.py for a metric."""
    candidates = [
        output_dir / f"{input_base_name}_mc_{metric}_directions.npz",
        output_dir / f"{input_base_name}_{metric}_directions.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: loose glob (keeps things robust to naming tweaks)
    matches = sorted(output_dir.glob(f"{input_base_name}*{metric}*directions*.npz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find directions npz for metric='{metric}'. Tried: {candidates} and glob."
    )


def load_mean_diff_directions(directions_path: Path, num_layers: int) -> dict[int, np.ndarray]:
    """Load mean-diff direction vectors from a *_directions.npz file."""
    data = np.load(directions_path)
    dirs: dict[int, np.ndarray] = {}
    for layer in range(num_layers):
        key = f"mean_diff_layer_{layer}"
        if key not in data:
            continue
        v = np.asarray(data[key], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        dirs[layer] = v
    return dirs


def _bootstrap_corr_std(a: np.ndarray, b: np.ndarray, n_bootstrap: int = 100, seed: int = 42) -> float:
    """Cheap bootstrap std for Pearson r by resampling paired examples."""
    rng = np.random.RandomState(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    vals = []
    for _ in range(int(n_bootstrap)):
        idx = rng.choice(n, n, replace=True)
        aa = a[idx]
        bb = b[idx]
        if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            continue
        r, _ = pearsonr(aa, bb)
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.std(vals)) if len(vals) > 1 else 0.0

def bootstrap_transfer_r2(
    activations: np.ndarray,
    targets: np.ndarray,
    scaler,
    pca,
    ridge,
    scaling: str,  # "centered" or "separate"
    n_bootstrap: int = 100,
    n_boot: int | None = None,
    seed: int = 42,) -> tuple:
    """
    Bootstrap uncertainty for transfer performance.

    Important: vanilla R² on bootstrap resamples can explode negative when the resampled
    `targets` have unusually low variance (SS_tot ≈ 0). That can make plots look
    like "garbage" even when the underlying predictions are fine.

    So we compute a *stable* out-of-sample R² using a fixed denominator equal to the
    variance of the *original* targets (the non-resampled test set):

        R²_stable = 1 - MSE_boot / Var(targets_original)

    This preserves the "R² scale" (<= 1, negative when worse than mean predictor),
    but avoids pathological -100/-1000 values caused purely by bootstrap variance collapse.

    Returns:
        (mean_r2, std_r2): mean and std over bootstrap replicates of R²_stable
    """
    rng = np.random.RandomState(seed)

    # Backward-compat alias: allow callers to pass n_boot
    if n_boot is not None:
        n_bootstrap = int(n_boot)

    targets = np.asarray(targets, dtype=np.float64)
    n_samples = targets.shape[0]

    var_full = float(np.var(targets))
    if not np.isfinite(var_full) or var_full < 1e-12:
        # Degenerate target variance: R² is not meaningful; fall back to MSE scale.
        var_full = 1e-12

    r2s = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        acts_boot = activations[idx]
        y_boot = targets[idx]

        if scaling == "centered":
            result = apply_probe_centered(acts_boot, y_boot, scaler, pca, ridge)
        else:
            result = apply_probe_separate(acts_boot, y_boot, pca, ridge)

        y_pred = np.asarray(result["predictions"], dtype=np.float64)

        mse = float(np.mean((y_boot - y_pred) ** 2))
        r2 = 1.0 - (mse / var_full)

        # Guard tiny numerical overshoot (should be <= 1 for this definition)
        if r2 > 1.0 and r2 < 1.0 + 1e-6:
            r2 = 1.0

        if np.isfinite(r2):
            r2s.append(r2)

    if len(r2s) == 0:
        return float("nan"), float("nan")

    r2s = np.array(r2s, dtype=np.float64)
    return float(r2s.mean()), float(r2s.std(ddof=0))


def bootstrap_r2_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 100,
    n_boot: int | None = None,
    seed: int = 42,) -> tuple:
    """
    Cheap bootstrap uncertainty for R² when the model is already trained.

    We resample *test examples only* and recompute a stable R² using a fixed
    denominator Var(y_true_full) to avoid pathological negative explosions when
    a bootstrap resample happens to have very low target variance.

        R²_stable = 1 - MSE_boot / Var(y_true_full)

    Returns:
        (mean_r2, std_r2) over bootstrap replicates.
    """
    rng = np.random.RandomState(seed)

    # Backward-compat alias: allow callers to pass n_boot
    if n_boot is not None:
        n_bootstrap = int(n_boot)

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = y_true.shape[0]
    if n == 0:
        return float("nan"), float("nan")

    var_full = float(np.var(y_true))
    if not np.isfinite(var_full) or var_full < 1e-12:
        var_full = 1e-12

    r2s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        mse = float(np.mean((yt - yp) ** 2))
        r2 = 1.0 - (mse / var_full)
        if r2 > 1.0 and r2 < 1.0 + 1e-6:
            r2 = 1.0
        if np.isfinite(r2):
            r2s.append(r2)

    if len(r2s) == 0:
        return float("nan"), float("nan")
    r2s = np.array(r2s, dtype=np.float64)
    return float(r2s.mean()), float(r2s.std(ddof=0))


def load_dataset(dataset_path: Path) -> dict:
    """Load dataset JSON with questions and metric values."""
    with open(dataset_path) as f:
        data = json.load(f)

    # Extract questions
    questions = data["data"]

    # Extract metric values as arrays
    metric_values = {}
    for item in questions:
        for key, val in item.items():
            if key in ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]:
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(val)

    # Convert to numpy
    for key in metric_values:
        metric_values[key] = np.array(metric_values[key])

    return {
        "config": data["config"],
        "stats": data["stats"],
        "questions": questions,
        "metric_values": metric_values,
    }


def get_meta_format_fn(meta_task: str):
    """Get the prompt formatting function for a meta task."""
    if meta_task == "confidence":
        return format_stated_confidence_prompt
    elif meta_task == "delegate":
        return format_answer_or_delegate_prompt
    else:
        raise ValueError(f"Unknown meta task: {meta_task}")


def get_meta_signal_fn(meta_task: str):
    """Get the signal extraction function for a meta task."""
    if meta_task == "confidence":
        return lambda probs, mapping: get_stated_confidence_signal(probs)
    elif meta_task == "delegate":
        return get_answer_or_delegate_signal
    else:
        raise ValueError(f"Unknown meta task: {meta_task}")


def get_meta_options(meta_task: str):
    """Get option tokens for a meta task."""
    if meta_task == "confidence":
        return list(STATED_CONFIDENCE_OPTIONS.keys())
    elif meta_task == "delegate":
        return ANSWER_OR_DELEGATE_OPTIONS
    else:
        raise ValueError(f"Unknown meta task: {meta_task}")


def plot_transfer_results(
    transfer_results: dict,
    direct_r2: dict,
    direct_r2_std: dict,  # CIs from training (loaded from results JSON)
    behavioral: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
    title_prefix: str = "Transfer Analysis",
):
    """
    Plot transfer R² across layers for all metrics.

    2x2 grid matching run_introspection_experiment.py:
    - Panel 1: Transferred predictions → stated confidence
    - Panel 2: Centered Scaler (Rigorous)
    - Panel 3: Pearson Correlation (Shift-Invariant)
    - Panel 4: Normalized timecourse comparison

    Each panel shows D→D (from training), D→M (transfer), and chance baseline (0).
    Shaded bands show ±1 std confidence intervals.
    """
    metrics = list(transfer_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix}: {meta_task}", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))
    colors = {'entropy': 'tab:blue', 'top_logit': 'tab:orange', 'logit_gap': 'tab:green'}

    # For display only: prevent a few extreme negative R² values from blowing out the y-axis.
    # Raw values are still saved to JSON/NPZ.
    R2_PLOT_FLOOR = -0.5
    def _clip_r2_for_plot(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, R2_PLOT_FLOOR, 1.0)

    # Panel 1: Does the transferred signal explain stated confidence?
    ax1 = axes[0, 0]
    ax1.set_title("Method 1: Transferred signal → stated confidence\nPearson r(sign·ŷ(meta), confidence)", fontsize=10)

    # We'll show three kinds of horizontal reference lines:
    #  - r=0 (gray)
    #  - behavioral metric↔confidence correlations (colored, dashed)
    # The behavioral lines are *not* "zero" baselines; they’re just an anchor for
    # how strongly the *raw metric values* relate to stated confidence.

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')

        vals = np.array(
            [transfer_results[metric][l]["centered"].get("pred_conf_pearson", np.nan) for l in layers],
            dtype=float,
        )

        # Pick best layer by absolute correlation, ignoring NaNs
        finite = np.isfinite(vals)
        if finite.any():
            best_layer = int(np.argmax(np.abs(vals[finite])))
            best_layer = int(np.array(layers)[finite][best_layer])
            best_r = float(vals[best_layer])
        else:
            best_layer = 0
            best_r = float("nan")

        ax1.plot(
            layers,
            vals,
            '-',
            label=f'{metric} (best L{best_layer}: {best_r:.3f})' if np.isfinite(best_r) else f'{metric}',
            color=color,
            linewidth=2,
        )

        # CI band (cheap bootstrap over test indices, stored in pred_conf_pearson_std)
        stds = np.array(
            [transfer_results[metric][l]["centered"].get("pred_conf_pearson_std", 0.0) for l in layers],
            dtype=float,
        )
        if np.any(stds > 0):
            ax1.fill_between(
                layers,
                vals - stds,
                vals + stds,
                color=color,
                alpha=0.15,
                linewidth=0,
            )
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Corr with confidence (r)')
    ax1.legend(loc='upper left', fontsize=7)
    ax1.grid(True, alpha=0.3)

# Panel 2: Centered Scaler (Rigorous)
    ax2 = axes[0, 1]
    ax2.set_title("Method 2: Centered Scaler (Rigorous)\n(Geometry Check)", fontsize=10)

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')

        # D→D (from training)
        if metric in direct_r2:
            d2d_r2 = _clip_r2_for_plot(np.array([direct_r2[metric].get(l, 0) for l in layers]))
            ax2.plot(layers, d2d_r2, '-', label=f'{metric} D→D',
                     color=color, linewidth=2, alpha=0.4)
            # D→D CI
            if metric in direct_r2_std:
                d2d_std = np.array([direct_r2_std[metric].get(l, 0) for l in layers])
                ax2.fill_between(layers, _clip_r2_for_plot(d2d_r2 - d2d_std), _clip_r2_for_plot(d2d_r2 + d2d_std),
                                 color=color, alpha=0.18)

        # D→M centered
        centered_r2 = np.array([transfer_results[metric][l]["centered"]["r2"] for l in layers], dtype=float)
        centered_std = np.array([transfer_results[metric][l]["centered"].get("r2_std", 0) for l in layers])
        best_layer = max(layers, key=lambda l: transfer_results[metric][l]["centered"]["r2"])
        best_r2 = transfer_results[metric][best_layer]["centered"]["r2"]
        ax2.plot(layers, centered_r2, '-', label=f'{metric} D→M Cen (best L{best_layer}: {best_r2:.3f})',
                 color=color, linewidth=2)
        # D→M CI
        if centered_std.any():
            ax2.fill_between(layers, _clip_r2_for_plot(centered_r2 - centered_std), _clip_r2_for_plot(centered_r2 + centered_std),
                             color=color, alpha=0.2)

    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('R² (out-of-sample)')
    ax2.legend(loc='upper left', fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Pearson Correlation (Shift Invariant)
    ax3 = axes[1, 0]
    ax3.set_title("Method 3: Pearson Correlation\n(Shift Invariant Signal Check)", fontsize=10)

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')

        # D→D Pearson (from training results - use correlation from fits)
        if metric in direct_r2:
            # Correlation = sqrt(R²) with sign preserved (assume positive for D→D)
            d2d_corr = [np.sqrt(max(direct_r2[metric].get(l, 0), 0.0)) for l in layers]
            ax3.plot(layers, d2d_corr, '-', label=f'{metric} D→D',
                     color=color, linewidth=2, alpha=0.4)

        # D→M Pearson
        pearson_r = [transfer_results[metric][l]["centered"]["pearson"] for l in layers]
        best_layer = max(layers, key=lambda l: abs(transfer_results[metric][l]["centered"]["pearson"]))
        best_corr = transfer_results[metric][best_layer]["centered"]["pearson"]
        ax3.plot(layers, pearson_r, '-', label=f'{metric} D→M (best L{best_layer}: {best_corr:.3f})',
                 color=color, linewidth=2)

    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Correlation (r)')
    ax3.legend(loc='upper left', fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Signal Emergence (Normalized timecourse)
    ax4 = axes[1, 1]
    ax4.set_title("Signal Emergence (Min-Max Scaled)\nCheck: Do lines rise together?", fontsize=10)

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')

        # D→D normalized (from training R²)
        if metric in direct_r2:
            d2d_r2 = _clip_r2_for_plot(np.array([direct_r2[metric].get(l, 0) for l in layers]))
            if d2d_r2.max() > d2d_r2.min():
                d2d_norm = (d2d_r2 - d2d_r2.min()) / (d2d_r2.max() - d2d_r2.min())
            else:
                d2d_norm = np.zeros_like(d2d_r2)
            ax4.plot(layers, d2d_norm, '-', label=f'{metric} D→D',
                     color=color, linewidth=2, alpha=0.4)

        # D→M normalized
        centered_r2 = np.array([transfer_results[metric][l]["centered"]["r2"] for l in layers], dtype=float)
        centered_r2_plot = _clip_r2_for_plot(centered_r2)
        # (use raw for normalization; clip only for display if needed)
        if centered_r2.max() > centered_r2.min():
            normalized = (centered_r2 - centered_r2.min()) / (centered_r2.max() - centered_r2.min())
        else:
            normalized = np.zeros_like(centered_r2)
        ax4.plot(layers, normalized, '-', label=f'{metric} D→M',
                 color=color, linewidth=2)

    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Normalized R² (0-1)')
    ax4.legend(loc='upper left', fontsize=7)
    ax4.grid(True, alpha=0.3)

    # Add behavioral correlation text box
    behav_text = "Metric ↔ Confidence (full dataset):\n"
    for metric in metrics:
        sign = metric_sign_for_confidence(metric)
        sign_str = " (inv)" if sign < 0 else ""
        behav_r = behavioral.get(metric, {}).get("pearson_r", float("nan"))
        behav_text += f"  {metric}{sign_str}: r={behav_r:.3f}\n"

    behav_text += "\nMetric ↔ Confidence (test set):\n"
    for metric in metrics:
        sign = metric_sign_for_confidence(metric)
        sign_str = " (inv)" if sign < 0 else ""
        test_r = behavioral.get(metric, {}).get("test_pearson_r", float("nan"))
        behav_text += f"  {metric}{sign_str}: r={test_r:.3f}\n"

    fig.text(0.02, 0.02, behav_text, fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_mean_diff_transfer_results(
    transfer_results: dict,
    direct_r2: dict,
    direct_r2_std: dict,
    behavioral: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
    title_prefix: str = "Mean-diff Transfer",
):
    """
    Plot mean-diff transfer R² across layers for all metrics.

    2x2 grid format (same as probe plots):
    - Panel 1: Transferred predictions → stated confidence
    - Panel 2: Centered Scaler (R² geometry check)
    - Panel 3: Pearson Correlation (Shift-Invariant)
    - Panel 4: Normalized timecourse comparison

    Mean-diff only has "centered" results (no "separate").
    """
    metrics = list(transfer_results.keys())
    if not metrics:
        print("  No metrics for mean-diff plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix}: {meta_task}", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))
    colors = {'entropy': 'tab:blue', 'top_logit': 'tab:orange', 'logit_gap': 'tab:green'}

    R2_PLOT_FLOOR = -0.5
    def _clip_r2_for_plot(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, R2_PLOT_FLOOR, 1.0)

    # Panel 1: Transferred signal → stated confidence
    ax1 = axes[0, 0]
    ax1.set_title("Method 1: Transferred signal → stated confidence\nPearson r(sign·ŷ(meta), confidence)", fontsize=10)

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')
        if metric not in transfer_results:
            continue

        vals = np.array(
            [transfer_results[metric].get(l, {}).get("centered", {}).get("pred_conf_pearson", np.nan) for l in layers],
            dtype=float,
        )

        finite = np.isfinite(vals)
        if finite.any():
            best_layer = int(np.argmax(np.abs(vals[finite])))
            best_layer = int(np.array(layers)[finite][best_layer])
            best_r = float(vals[best_layer])
        else:
            best_layer = 0
            best_r = float("nan")

        ax1.plot(layers, vals, '-',
                 label=f'{metric} (best L{best_layer}: {best_r:.3f})' if np.isfinite(best_r) else f'{metric}',
                 color=color, linewidth=2)

        stds = np.array(
            [transfer_results[metric].get(l, {}).get("centered", {}).get("pred_conf_pearson_std", 0.0) for l in layers],
            dtype=float,
        )
        if np.any(stds > 0):
            ax1.fill_between(layers, vals - stds, vals + stds, color=color, alpha=0.15, linewidth=0)

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Corr with confidence (r)')
    ax1.legend(loc='upper left', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Centered R² (Rigorous)
    ax2 = axes[0, 1]
    ax2.set_title("Method 2: Centered Scaler (Rigorous)\n(Geometry Check)", fontsize=10)

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')

        # D→D
        if metric in direct_r2:
            d2d_r2 = _clip_r2_for_plot(np.array([direct_r2[metric].get(l, 0) for l in layers]))
            ax2.plot(layers, d2d_r2, '-', label=f'{metric} D→D', color=color, linewidth=2, alpha=0.4)
            if metric in direct_r2_std:
                d2d_std = np.array([direct_r2_std[metric].get(l, 0) for l in layers])
                ax2.fill_between(layers, _clip_r2_for_plot(d2d_r2 - d2d_std), _clip_r2_for_plot(d2d_r2 + d2d_std),
                                 color=color, alpha=0.18)

        # D→M centered
        centered_r2 = np.array([transfer_results[metric].get(l, {}).get("centered", {}).get("r2", np.nan) for l in layers], dtype=float)
        centered_std = np.array([transfer_results[metric].get(l, {}).get("centered", {}).get("r2_std", 0) for l in layers])

        finite = np.isfinite(centered_r2)
        if finite.any():
            best_layer = int(np.argmax(np.where(finite, centered_r2, -np.inf)))
            best_r2 = centered_r2[best_layer]
        else:
            best_layer = 0
            best_r2 = np.nan

        ax2.plot(layers, _clip_r2_for_plot(centered_r2), '-',
                 label=f'{metric} D→M (best L{best_layer}: {best_r2:.3f})' if np.isfinite(best_r2) else f'{metric} D→M',
                 color=color, linewidth=2)
        if np.any(centered_std > 0):
            ax2.fill_between(layers, _clip_r2_for_plot(centered_r2 - centered_std), _clip_r2_for_plot(centered_r2 + centered_std),
                             color=color, alpha=0.2)

    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('R² (out-of-sample)')
    ax2.legend(loc='upper left', fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Pearson Correlation (Shift Invariant)
    ax3 = axes[1, 0]
    ax3.set_title("Method 3: Pearson Correlation\n(Shift Invariant Signal Check)", fontsize=10)

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')

        # D→D Pearson (sqrt of R²)
        if metric in direct_r2:
            d2d_corr = [np.sqrt(max(direct_r2[metric].get(l, 0), 0.0)) for l in layers]
            ax3.plot(layers, d2d_corr, '-', label=f'{metric} D→D', color=color, linewidth=2, alpha=0.4)

        # D→M Pearson
        pearson_r = np.array([transfer_results[metric].get(l, {}).get("centered", {}).get("pearson", np.nan) for l in layers], dtype=float)

        finite = np.isfinite(pearson_r)
        if finite.any():
            best_layer = int(np.argmax(np.abs(pearson_r[finite])))
            best_layer = int(np.array(layers)[finite][best_layer])
            best_corr = pearson_r[best_layer]
        else:
            best_layer = 0
            best_corr = np.nan

        ax3.plot(layers, pearson_r, '-',
                 label=f'{metric} D→M (best L{best_layer}: {best_corr:.3f})' if np.isfinite(best_corr) else f'{metric} D→M',
                 color=color, linewidth=2)

    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Correlation (r)')
    ax3.legend(loc='upper left', fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Signal Emergence (Normalized timecourse)
    ax4 = axes[1, 1]
    ax4.set_title("Signal Emergence (Min-Max Scaled)\nCheck: Do lines rise together?", fontsize=10)

    for metric in metrics:
        color = colors.get(metric, 'tab:gray')

        # D→D normalized
        if metric in direct_r2:
            d2d_r2 = _clip_r2_for_plot(np.array([direct_r2[metric].get(l, 0) for l in layers]))
            if d2d_r2.max() > d2d_r2.min():
                d2d_norm = (d2d_r2 - d2d_r2.min()) / (d2d_r2.max() - d2d_r2.min())
            else:
                d2d_norm = np.zeros_like(d2d_r2)
            ax4.plot(layers, d2d_norm, '-', label=f'{metric} D→D', color=color, linewidth=2, alpha=0.4)

        # D→M normalized
        centered_r2 = np.array([transfer_results[metric].get(l, {}).get("centered", {}).get("r2", np.nan) for l in layers], dtype=float)
        centered_r2_plot = _clip_r2_for_plot(centered_r2)
        if np.nanmax(centered_r2) > np.nanmin(centered_r2):
            normalized = (centered_r2 - np.nanmin(centered_r2)) / (np.nanmax(centered_r2) - np.nanmin(centered_r2))
        else:
            normalized = np.zeros_like(centered_r2)
        ax4.plot(layers, normalized, '-', label=f'{metric} D→M', color=color, linewidth=2)

    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Normalized R² (0-1)')
    ax4.legend(loc='upper left', fontsize=7)
    ax4.grid(True, alpha=0.3)

    # Add behavioral correlation text box
    behav_text = "Metric ↔ Confidence (full dataset):\n"
    for metric in metrics:
        sign = metric_sign_for_confidence(metric)
        sign_str = " (inv)" if sign < 0 else ""
        behav_r = behavioral.get(metric, {}).get("pearson_r", float("nan"))
        behav_text += f"  {metric}{sign_str}: r={behav_r:.3f}\n"

    behav_text += "\nMetric ↔ Confidence (test set):\n"
    for metric in metrics:
        sign = metric_sign_for_confidence(metric)
        sign_str = " (inv)" if sign < 0 else ""
        test_r = behavioral.get(metric, {}).get("test_pearson_r", float("nan"))
        behav_text += f"  {metric}{sign_str}: r={test_r:.3f}\n"

    fig.text(0.02, 0.02, behav_text, fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_position_comparison(
    transfer_results_by_pos: dict,
    mean_diff_transfer_by_pos: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
):
    """
    Plot transfer R² across layers comparing different token positions.

    Creates a 2x2 grid:
    - Top row: Probe-based transfer for each metric
    - Bottom row: Mean-diff transfer for each metric

    Each panel shows one metric with lines for each position.
    """
    # Clip extreme negative R² values for display (same floor as other plots)
    R2_PLOT_FLOOR = -0.5  # Tighter floor for position comparison since we care about positive values
    R2_PLOT_CEIL = 1.0

    def _clip_r2(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, R2_PLOT_FLOOR, R2_PLOT_CEIL)

    metrics = set()
    for pos_data in transfer_results_by_pos.values():
        metrics.update(pos_data.keys())
    metrics = sorted(metrics)

    if len(metrics) == 0:
        print("  No metrics found for position comparison plot")
        return

    positions = list(transfer_results_by_pos.keys())
    pos_colors = {
        "question_mark": "tab:blue",
        "question_newline": "tab:cyan",
        "options_newline": "tab:green",
        "final": "tab:red",
    }
    # More readable position labels
    pos_labels = {
        "question_mark": "question ?",
        "question_newline": "question \\n",
        "options_newline": "options \\n",
        "final": "final",
    }

    # Use 2 rows x N cols where N = number of metrics
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6 * n_metrics, 10), squeeze=False)
    fig.suptitle(f"Position Comparison: {meta_task}", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))

    # Top row: probe-based
    for col, metric in enumerate(metrics):
        ax = axes[0, col]
        ax.set_title(f"Probe Transfer: {metric}", fontsize=11)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        for pos in positions:
            if metric not in transfer_results_by_pos.get(pos, {}):
                continue
            color = pos_colors.get(pos, "tab:gray")
            display_name = pos_labels.get(pos, pos)

            r2_vals = []
            for l in layers:
                if l in transfer_results_by_pos[pos][metric]:
                    r2_vals.append(transfer_results_by_pos[pos][metric][l]["centered"]["r2"])
                else:
                    r2_vals.append(np.nan)
            r2_vals = np.array(r2_vals, dtype=float)

            # Find best layer BEFORE clipping (use true values)
            finite = np.isfinite(r2_vals)
            if finite.any():
                best_layer = int(np.argmax(np.where(finite, r2_vals, -np.inf)))
                best_r2 = r2_vals[best_layer]
                label = f"{display_name} (L{best_layer}: {best_r2:.3f})"
            else:
                label = display_name

            # Clip for plotting
            r2_vals_clipped = _clip_r2(r2_vals)
            ax.plot(layers, r2_vals_clipped, '-', label=label, color=color, linewidth=2)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('R²')
        ax.set_ylim(R2_PLOT_FLOOR, R2_PLOT_CEIL)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Bottom row: mean-diff
    for col, metric in enumerate(metrics):
        ax = axes[1, col]
        ax.set_title(f"Mean-Diff Transfer: {metric}", fontsize=11)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        for pos in positions:
            if metric not in mean_diff_transfer_by_pos.get(pos, {}):
                continue
            color = pos_colors.get(pos, "tab:gray")
            display_name = pos_labels.get(pos, pos)

            r2_vals = []
            for l in layers:
                if l in mean_diff_transfer_by_pos[pos][metric]:
                    r2_vals.append(mean_diff_transfer_by_pos[pos][metric][l]["centered"]["r2"])
                else:
                    r2_vals.append(np.nan)
            r2_vals = np.array(r2_vals, dtype=float)

            # Find best layer BEFORE clipping (use true values)
            finite = np.isfinite(r2_vals)
            if finite.any():
                best_layer = int(np.argmax(np.where(finite, r2_vals, -np.inf)))
                best_r2 = r2_vals[best_layer]
                label = f"{display_name} (L{best_layer}: {best_r2:.3f})"
            else:
                label = display_name

            # Clip for plotting
            r2_vals_clipped = _clip_r2(r2_vals)
            ax.plot(layers, r2_vals_clipped, '-', label=label, color=color, linewidth=2)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('R²')
        ax.set_ylim(R2_PLOT_FLOOR, R2_PLOT_CEIL)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Compute the correct base name for input files
    # If ADAPTER is specified, files from identify_mc_correlate.py include the adapter name
    if ADAPTER:
        # Parse INPUT_BASE_NAME to extract model and dataset
        # Format: {model}_{dataset}
        parts = INPUT_BASE_NAME.rsplit('_', 1)
        if len(parts) == 2:
            model_part, dataset_part = parts
            adapter_short = get_model_short_name(ADAPTER)
            input_base_with_adapter = f"{model_part}_adapter-{adapter_short}_{dataset_part}"
        else:
            # Fallback: couldn't parse, try with adapter appended
            adapter_short = get_model_short_name(ADAPTER)
            input_base_with_adapter = f"{INPUT_BASE_NAME}_adapter-{adapter_short}"
    else:
        input_base_with_adapter = INPUT_BASE_NAME
    
    # Load dataset first to get model info
    dataset_path = OUTPUT_DIR / f"{input_base_with_adapter}_mc_dataset.json"
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)

    # Infer model from dataset
    model_name = dataset['config']['base_model']
    model_short = get_model_short_name(model_name)

    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset['config']['dataset']}")
    print(f"  Questions: {len(dataset['questions'])}")
    print(f"  Metrics available: {list(dataset['metric_values'].keys())}")

    # Load direct activations (needed to train probes with proper train/test split)
    direct_activations_path = OUTPUT_DIR / f"{input_base_with_adapter}_mc_activations.npz"
    if not direct_activations_path.exists():
        raise ValueError(f"Direct activations not found: {direct_activations_path}\n"
                        f"Run identify_mc_correlate.py first.")

    print(f"\nLoading direct activations from {direct_activations_path}...")
    direct_loaded = np.load(direct_activations_path)

    # Reconstruct activations_by_layer
    layer_keys = [k for k in direct_loaded.files if k.startswith("layer_")]
    num_layers = len(layer_keys)
    direct_activations = {i: direct_loaded[f"layer_{i}"] for i in range(num_layers)}
    print(f"  Loaded {num_layers} layers, shape: {direct_activations[0].shape}")

    # Create train/test split (same split for direct and meta)
    n_questions = len(dataset['questions'])
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED
    )
    print(f"\nTrain/test split: {len(train_idx)} train, {len(test_idx)} test (seed={SEED})")

    # Determine output paths
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        base_output = f"{model_short}_adapter-{adapter_short}_{dataset['config']['dataset']}_transfer_{META_TASK}"
    else:
        base_output = f"{model_short}_{dataset['config']['dataset']}_transfer_{META_TASK}"

    activations_path = OUTPUT_DIR / f"{base_output}_activations.npz"
    results_json_path = OUTPUT_DIR / f"{base_output}_results.json"
    results_npz_path = OUTPUT_DIR / f"{base_output}_results.npz"
    plot_path = OUTPUT_DIR / f"{base_output}_results.png"
    plot_path_mean_diff = OUTPUT_DIR / f"{base_output}_mean_diff_results.png"
    plot_path_positions = OUTPUT_DIR / f"{base_output}_position_comparison.png"

    print(f"\nMeta task: {META_TASK}")
    print(f"Output base: {base_output}")

    # Check for cached activations
    if activations_path.exists():
        print(f"\nFound existing activations: {activations_path}")
        print("Loading from file (skipping model load and extraction)...")
        loaded = np.load(activations_path)

        # Detect format: multi-position (layer_N_posname) vs legacy (layer_N)
        has_positions = any("_" in k.replace("layer_", "", 1) for k in loaded.files if k.startswith("layer_"))

        if has_positions:
            # Multi-position format: {position: {layer: array}}
            meta_activations = {pos: {} for pos in PROBE_POSITIONS}
            position_valid_arrays = {}
            for key in loaded.files:
                if key.startswith("layer_"):
                    parts = key.split("_")
                    layer = int(parts[1])
                    pos_name = "_".join(parts[2:])
                    if pos_name in meta_activations:
                        meta_activations[pos_name][layer] = loaded[key]
                elif key.startswith("valid_"):
                    pos_name = key[6:]  # Remove "valid_" prefix
                    position_valid_arrays[pos_name] = loaded[key]
                elif key == "confidences":
                    confidences = loaded[key]
            # If no validity masks found (old format), assume all valid
            if not position_valid_arrays:
                n_samples = len(confidences)
                position_valid_arrays = {pos: np.ones(n_samples, dtype=bool) for pos in PROBE_POSITIONS}
            print(f"  Loaded {len(meta_activations)} positions, {len(meta_activations.get('final', {}))} layers")
        else:
            # Legacy format: {layer: array} - wrap in "final" position
            legacy_activations = {}
            for key in loaded.files:
                if key.startswith("layer_"):
                    layer = int(key.split("_")[1])
                    legacy_activations[layer] = loaded[key]
                elif key == "confidences":
                    confidences = loaded[key]
            meta_activations = {"final": legacy_activations}
            # Legacy format: only final position, all valid
            n_samples = len(confidences)
            position_valid_arrays = {"final": np.ones(n_samples, dtype=bool)}
            print(f"  Loaded legacy format with {len(legacy_activations)} layers (final position only)")
    else:
        # Load model with appropriate quantization
        load_4bit = LOAD_IN_4BIT
        if load_4bit is None:
            load_4bit = "70B" in model_name or "70b" in model_name
            if load_4bit:
                print(f"\nAuto-detected 70B model, using 4-bit quantization")

        print("\nLoading model...")
        model, tokenizer, num_layers_model = load_model_and_tokenizer(
            model_name,
            adapter_path=ADAPTER,
            load_in_4bit=load_4bit,
            load_in_8bit=LOAD_IN_8BIT,
        )
        use_chat_template = should_use_chat_template(model_name, tokenizer)

        if num_layers_model != num_layers:
            print(f"  Warning: model has {num_layers_model} layers but probes have {num_layers}")
            num_layers = min(num_layers, num_layers_model)

        # Get questions
        questions = dataset["questions"]

        # Get meta task setup
        format_fn = get_meta_format_fn(META_TASK)
        signal_fn = get_meta_signal_fn(META_TASK)
        meta_options = get_meta_options(META_TASK)
        option_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in meta_options]

        print(f"  Meta options: {meta_options}")
        print(f"  Option token IDs: {option_token_ids}")

        # Extract meta activations at multiple token positions
        print(f"\nExtracting meta activations (batch_size={BATCH_SIZE})...")
        print(f"  Probe positions: {PROBE_POSITIONS}")

        # Initialize storage: {position: {layer: [activations]}}
        all_activations = {
            pos: {layer: [] for layer in range(num_layers)}
            for pos in PROBE_POSITIONS
        }
        all_confidences = []
        all_mappings = []
        # Track which examples have valid positions for each position name
        # (when find_mc_positions fails, it returns only {"final": -1})
        position_valid = {pos: [] for pos in PROBE_POSITIONS}

        with BatchedExtractor(model, num_layers) as extractor:
            for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
                batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

                prompts = []
                batch_mappings = []
                batch_positions = []  # List of position dicts per item
                for i, q in enumerate(batch_questions):
                    trial_idx = batch_start + i
                    if META_TASK == "delegate":
                        prompt, _, mapping = format_fn(q, tokenizer, trial_index=trial_idx, use_chat_template=use_chat_template)
                        batch_mappings.append(mapping)
                    else:
                        prompt, _ = format_fn(q, tokenizer, use_chat_template=use_chat_template)
                        batch_mappings.append(None)
                    prompts.append(prompt)

                    # Find token positions for this prompt
                    positions = find_mc_positions(prompt, tokenizer, q)
                    batch_positions.append(positions)

                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,  # Prompts already have special tokens from chat template
                )
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)

                # Build token_positions dict: {pos_name: [idx_for_each_batch_item]}
                # Also track validity: a position is valid if it exists in the dict
                # (find_mc_positions returns only {"final": -1} on failure)
                token_positions = {}
                for pos_name in PROBE_POSITIONS:
                    token_positions[pos_name] = [
                        bp.get(pos_name, -1) for bp in batch_positions
                    ]
                    # Track validity for this batch
                    for bp in batch_positions:
                        position_valid[pos_name].append(pos_name in bp)

                layer_acts_by_pos, probs, _, _ = extractor.extract_batch(
                    input_ids, attention_mask, option_token_ids, token_positions
                )

                # Store activations per position
                for pos_name in PROBE_POSITIONS:
                    for item_acts in layer_acts_by_pos[pos_name]:
                        for layer, act in item_acts.items():
                            all_activations[pos_name][layer].append(act)

                for p, mapping in zip(probs, batch_mappings):
                    confidence = signal_fn(p, mapping)
                    all_confidences.append(confidence)
                    all_mappings.append(mapping)

        # Stack activations: {position: {layer: np.array}}
        print("\nStacking activations...")
        meta_activations = {
            pos: {layer: np.stack(acts) for layer, acts in pos_acts.items()}
            for pos, pos_acts in all_activations.items()
        }
        confidences = np.array(all_confidences)
        # Convert validity masks to arrays
        position_valid_arrays = {pos: np.array(valid) for pos, valid in position_valid.items()}

        # Report validity stats
        for pos in PROBE_POSITIONS:
            n_valid = position_valid_arrays[pos].sum()
            n_total = len(position_valid_arrays[pos])
            if n_valid < n_total:
                print(f"  Warning: {pos} has {n_valid}/{n_total} valid positions")

        # Save activations for future runs
        print(f"Saving activations to {activations_path}...")
        save_dict = {"confidences": confidences}
        for pos_name, pos_acts in meta_activations.items():
            for layer, acts in pos_acts.items():
                save_dict[f"layer_{layer}_{pos_name}"] = acts
        # Save validity masks
        for pos_name, valid_arr in position_valid_arrays.items():
            save_dict[f"valid_{pos_name}"] = valid_arr
        np.savez_compressed(activations_path, **save_dict)

    # Get positions available in loaded data
    positions_available = list(meta_activations.keys())
    first_pos = positions_available[0]
    first_layer = list(meta_activations[first_pos].keys())[0]
    print(f"\nActivation shape per layer: {meta_activations[first_pos][first_layer].shape}")
    print(f"Positions available: {positions_available}")
    target_name = "Stated confidence" if META_TASK == "confidence" else "P(Answer)"
    print(f"{target_name}: mean={confidences.mean():.3f}, std={confidences.std():.3f}")

    # Train probes and test transfer for each metric and position
    # Key: use same train/test split for both D→D and D→M (like run_introspection_experiment.py)
    print("\n" + "=" * 60)
    print("TRAINING PROBES AND TESTING TRANSFER")
    print("=" * 60)

    # Results structure: {position: {metric: {layer: {...}}}}
    transfer_results_by_pos = {pos: {} for pos in positions_available}
    direct_r2 = {}
    direct_r2_std = {}
    metrics_tested = [m for m in METRICS if m in dataset["metric_values"]]

    # Helper functions defined once
    def _safe_corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size == 0 or b.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan"), float("nan"), float("nan"), float("nan")
        r, p = pearsonr(a, b)
        rs, ps = spearmanr(a, b)
        return float(r), float(p), float(rs), float(ps)

    def _bootstrap_corr_std(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> float:
        rng = np.random.RandomState(seed)
        n = len(a)
        if n < 3:
            return 0.0
        vals = []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            aa = a[idx]
            bb = b[idx]
            if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
                continue
            r, _ = pearsonr(aa, bb)
            if np.isfinite(r):
                vals.append(float(r))
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    for metric in metrics_tested:
        print(f"\n--- {metric.upper()} ---")

        direct_values = dataset["metric_values"][metric]

        # Split data
        y_train = direct_values[train_idx]
        y_test = direct_values[test_idx]
        conf_test = confidences[test_idx]
        metric_sign = metric_sign_for_confidence(metric)

        direct_r2[metric] = {}
        direct_r2_std[metric] = {}

        # Initialize results for each position
        for pos in positions_available:
            transfer_results_by_pos[pos][metric] = {}

        for layer in tqdm(range(num_layers), desc=f"  {metric}"):
            X_direct_train = direct_activations[layer][train_idx]
            X_direct_test = direct_activations[layer][test_idx]

            # Train probe on direct_train (same probe for all positions)
            _, probe_info = probe_direction(
                X_direct_train, y_train,
                alpha=PROBE_ALPHA,
                pca_components=PROBE_PCA_COMPONENTS,
                bootstrap_splits=None,  # Fit on all train data
                return_probe=True,
            )

            scaler = probe_info["scaler"]
            pca = probe_info["pca"]
            ridge = probe_info["ridge"]

            # D→D: Test on direct_test (sanity check) - same for all positions
            d2d_result = apply_probe_shared(
                X_direct_test, y_test, scaler, pca, ridge
            )
            direct_r2[metric][layer] = d2d_result["r2"]
            _, d2d_std = bootstrap_r2_from_predictions(
                y_test, d2d_result["predictions"],
                n_boot=N_BOOTSTRAP, seed=SEED + layer
            )
            direct_r2_std[metric][layer] = float(d2d_std) if np.isfinite(d2d_std) else 0.0

            # D→M: Test transfer at each position
            for pos in positions_available:
                if layer not in meta_activations[pos]:
                    continue

                # Get validity mask for this position (only use examples with valid positions)
                pos_valid = position_valid_arrays.get(pos, np.ones(len(test_idx), dtype=bool))
                valid_test_mask = pos_valid[test_idx]
                n_valid = valid_test_mask.sum()

                if n_valid < 10:  # Need minimum samples for meaningful statistics
                    continue

                # Filter to valid examples only
                X_meta_test = meta_activations[pos][layer][test_idx][valid_test_mask]
                y_test_valid = y_test[valid_test_mask]
                conf_test_valid = conf_test[valid_test_mask]

                # D→M: Centered scaling (rigorous transfer test)
                centered_result = apply_probe_centered(
                    X_meta_test, y_test_valid, scaler, pca, ridge
                )

                # D→M: Separate scaling (upper bound)
                separate_result = apply_probe_separate(
                    X_meta_test, y_test_valid, pca, ridge
                )

                # Correlate probe predictions with stated confidence
                cen_r, cen_p, cen_rs, cen_ps = _safe_corr(centered_result["predictions"] * metric_sign, conf_test_valid)
                centered_result["pred_conf_pearson"] = cen_r
                centered_result["pred_conf_p"] = cen_p
                centered_result["pred_conf_spearman"] = cen_rs
                centered_result["pred_conf_spearman_p"] = cen_ps

                centered_result["pred_conf_pearson_std"] = _bootstrap_corr_std(
                    centered_result["predictions"] * metric_sign,
                    conf_test_valid,
                    n_boot=N_BOOTSTRAP,
                    seed=SEED + 10000 + layer,
                )

                sep_r, sep_p, sep_rs, sep_ps = _safe_corr(separate_result["predictions"] * metric_sign, conf_test_valid)
                separate_result["pred_conf_pearson"] = sep_r
                separate_result["pred_conf_p"] = sep_p
                separate_result["pred_conf_spearman"] = sep_rs
                separate_result["pred_conf_spearman_p"] = sep_ps

                # Bootstrap CIs for transfer R² (resample test set only)
                _, centered_std = bootstrap_transfer_r2(
                    X_meta_test, y_test_valid,
                    scaler, pca, ridge, "centered",
                    n_boot=N_BOOTSTRAP, seed=SEED + layer
                )
                centered_std = float(centered_std) if np.isfinite(centered_std) else 0.0
                _, separate_std = bootstrap_transfer_r2(
                    X_meta_test, y_test_valid,
                    scaler, pca, ridge, "separate",
                    n_boot=N_BOOTSTRAP, seed=SEED + layer
                )
                separate_std = float(separate_std) if np.isfinite(separate_std) else 0.0

                centered_result["r2_std"] = centered_std
                separate_result["r2_std"] = separate_std

                transfer_results_by_pos[pos][metric][layer] = {
                    "centered": centered_result,
                    "separate": separate_result,
                }

        # Print summary for each position
        print(f"\n  Transfer R² by position:")
        for pos in positions_available:
            if metric not in transfer_results_by_pos[pos] or not transfer_results_by_pos[pos][metric]:
                continue
            layers_available = list(transfer_results_by_pos[pos][metric].keys())
            best_layer_cen = max(layers_available, key=lambda l: transfer_results_by_pos[pos][metric][l]["centered"]["r2"])
            best_r2_cen = transfer_results_by_pos[pos][metric][best_layer_cen]["centered"]["r2"]
            print(f"    {pos}: R²={best_r2_cen:.3f} (L{best_layer_cen})")

        # Show D→D test R² (computed above)
        if metric in direct_r2 and direct_r2[metric]:
            best_d2d_layer = max(direct_r2[metric].keys(), key=lambda l: direct_r2[metric][l])
            best_d2d_r2 = direct_r2[metric][best_d2d_layer]
            print(f"  D→D (test): R²={best_d2d_r2:.3f} (L{best_d2d_layer})")

    # For backward compatibility, use "final" position for existing code
    transfer_results = transfer_results_by_pos.get("final", transfer_results_by_pos.get(positions_available[0], {}))

    # Behavioral correlation: metric vs meta-task target
    # For confidence task: correlation between metric and stated confidence
    # For delegate task: correlation between metric and P(Answer)
    meta_target_name = "stated_confidence" if META_TASK == "confidence" else "P(Answer)"
    print("\n" + "-" * 40)
    print(f"BEHAVIORAL CORRELATION (metric vs {meta_target_name})")
    print("-" * 40)

    behavioral = {"meta_target": meta_target_name}
    for metric in metrics_tested:
        direct_values = dataset["metric_values"][metric]
        sign = metric_sign_for_confidence(metric)

        corr, p_value = pearsonr(direct_values * sign, confidences)
        spearman_corr, spearman_p = spearmanr(direct_values * sign, confidences)

        # Bootstrap CI for full dataset correlation
        full_std = _bootstrap_corr_std(direct_values * sign, confidences, n_boot=N_BOOTSTRAP, seed=SEED)

        behavioral[metric] = {
            "pearson_r": float(corr),
            "pearson_p": float(p_value),
            "pearson_r_std": float(full_std),
            "spearman_r": float(spearman_corr),
            "spearman_p": float(spearman_p),
        }

        sign_str = "(inverted)" if sign < 0 else ""
        print(f"  {metric} {sign_str}: r={corr:.3f}±{full_std:.3f} (p={p_value:.2e}), ρ={spearman_corr:.3f}")


    # Test-set baseline: correlation between (signed) raw metric values and meta-task target
    # on the held-out test set (same indices used for probe evaluation).
    conf_test = confidences[test_idx]
    for metric in metrics_tested:
        direct_test = dataset["metric_values"][metric][test_idx]
        sign = metric_sign_for_confidence(metric)

        test_r, test_p = pearsonr(direct_test * sign, conf_test)
        test_rs, test_rs_p = spearmanr(direct_test * sign, conf_test)

        # Bootstrap CI for test set correlation
        test_std = _bootstrap_corr_std(direct_test * sign, conf_test, n_boot=N_BOOTSTRAP, seed=SEED)

        behavioral[metric]["test_pearson_r"] = float(test_r)
        behavioral[metric]["test_pearson_p"] = float(test_p)
        behavioral[metric]["test_pearson_r_std"] = float(test_std)
        behavioral[metric]["test_spearman_r"] = float(test_rs)
        behavioral[metric]["test_spearman_p"] = float(test_rs_p)

        sign_str = "(inverted)" if sign < 0 else ""
        print(f"  {metric} {sign_str} test-set: r={test_r:.3f}±{test_std:.3f}, ρ={test_rs:.3f}")

    # =============================================================================
    # MEAN-DIFF TRANSFER (precomputed directions)
    # =============================================================================
    print("\n" + "=" * 60)
    print("MEAN-DIFF TRANSFER ANALYSIS")
    print("=" * 60)

    # Results by position: {position: {metric: {layer: {...}}}}
    mean_diff_transfer_by_pos: dict = {pos: {} for pos in positions_available}
    mean_diff_direct_r2: dict = {}
    mean_diff_direct_r2_std: dict = {}

    conf_test = confidences[test_idx]

    for metric in metrics_tested:
        print(f"\n--- {metric.upper()} (mean-diff) ---")
        # Locate and load directions file for this metric
        try:
            directions_path = _find_directions_npz(input_base_with_adapter, metric, OUTPUT_DIR)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
        mean_dirs = load_mean_diff_directions(directions_path, num_layers)

        if len(mean_dirs) == 0:
            print(f"  Warning: no mean_diff_layer_* keys found in {directions_path.name}; skipping.")
            continue

        direct_values = dataset["metric_values"][metric]
        y_train = direct_values[train_idx]
        y_test = direct_values[test_idx]

        for pos in positions_available:
            mean_diff_transfer_by_pos[pos][metric] = {}
        mean_diff_direct_r2[metric] = {}
        mean_diff_direct_r2_std[metric] = {}

        metric_sign = metric_sign_for_confidence(metric)

        for layer in tqdm(range(num_layers), desc=f"  {metric} mean-diff"):
            if layer not in mean_dirs:
                continue
            d = mean_dirs[layer]

            X_direct_train = direct_activations[layer][train_idx]
            X_direct_test = direct_activations[layer][test_idx]

            # 1) Compute 1D scores by projection on direct data
            s_train = X_direct_train @ d
            s_test = X_direct_test @ d

            # 2) Score standardization stats from DIRECT train (stable)
            s_mu = float(np.mean(s_train))
            s_std = float(np.std(s_train))
            if not np.isfinite(s_std) or s_std < 1e-8:
                s_std = 1e-8

            z_train = (s_train - s_mu) / s_std

            # 3) Fit a 1D calibrator on DIRECT train: y ≈ a*z + b
            from sklearn.linear_model import Ridge
            cal = Ridge(alpha=1e-6, fit_intercept=True)
            cal.fit(z_train.reshape(-1, 1), y_train)

            # D→D: evaluate on DIRECT test using DIRECT stats
            z_test = (s_test - s_mu) / s_std
            yhat_test = cal.predict(z_test.reshape(-1, 1))

            from sklearn.metrics import r2_score, mean_absolute_error
            d2d_r2 = float(r2_score(y_test, yhat_test))

            mean_diff_direct_r2[metric][layer] = d2d_r2
            _, d2d_std = bootstrap_r2_from_predictions(
                y_test, yhat_test,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED + 20000 + layer,
            )
            mean_diff_direct_r2_std[metric][layer] = d2d_std

            # D→M: Test transfer at each position
            for pos in positions_available:
                if layer not in meta_activations[pos]:
                    continue

                # Get validity mask for this position
                pos_valid = position_valid_arrays.get(pos, np.ones(len(test_idx), dtype=bool))
                valid_test_mask = pos_valid[test_idx]
                n_valid = valid_test_mask.sum()

                if n_valid < 10:
                    continue

                # Filter to valid examples only
                X_meta_test = meta_activations[pos][layer][test_idx][valid_test_mask]
                y_test_valid = y_test[valid_test_mask]
                conf_test_valid = conf_test[valid_test_mask]

                s_meta = X_meta_test @ d

                # D→M Centered: center META scores with their own mean, but scale with DIRECT std
                z_meta = (s_meta - float(np.mean(s_meta))) / s_std
                yhat_meta = cal.predict(z_meta.reshape(-1, 1))

                cen_r2 = float(r2_score(y_test_valid, yhat_meta))
                cen_mae = float(mean_absolute_error(y_test_valid, yhat_meta))
                cen_pear, _ = pearsonr(y_test_valid, yhat_meta)

                centered_result = {
                    "r2": cen_r2,
                    "mae": cen_mae,
                    "pearson": float(cen_pear),
                    "predictions": yhat_meta,
                }

                # Bootstrap for centered R²
                rng = np.random.RandomState(SEED + 30000 + layer)
                n = len(y_test_valid)
                vals_r2 = []
                vals_pc = []
                for _ in range(N_BOOTSTRAP):
                    idx = rng.choice(n, n, replace=True)
                    sm = (X_meta_test[idx] @ d)
                    zm = (sm - float(np.mean(sm))) / s_std
                    yhat_b = cal.predict(zm.reshape(-1, 1))
                    r2_b = 1.0 - float(np.mean((y_test_valid[idx] - yhat_b) ** 2)) / float(np.var(y_test_valid))
                    if np.isfinite(r2_b):
                        vals_r2.append(r2_b)
                    if np.std(yhat_b) > 1e-12 and np.std(conf_test_valid[idx]) > 1e-12:
                        r_b, _ = pearsonr(yhat_b * metric_sign, conf_test_valid[idx])
                        if np.isfinite(r_b):
                            vals_pc.append(float(r_b))

                centered_result["r2_std"] = float(np.std(vals_r2)) if len(vals_r2) > 1 else 0.0

                # Prediction→confidence correlation
                pc_r, pc_p, pc_rs, pc_ps = _safe_corr(yhat_meta * metric_sign, conf_test_valid)
                centered_result["pred_conf_pearson"] = pc_r
                centered_result["pred_conf_p"] = pc_p
                centered_result["pred_conf_spearman"] = pc_rs
                centered_result["pred_conf_spearman_p"] = pc_ps
                centered_result["pred_conf_pearson_std"] = float(np.std(vals_pc)) if len(vals_pc) > 1 else 0.0

                mean_diff_transfer_by_pos[pos][metric][layer] = {"centered": centered_result}

        # Summary by position
        print(f"\n  Transfer R² by position (mean-diff):")
        for pos in positions_available:
            if metric not in mean_diff_transfer_by_pos[pos] or not mean_diff_transfer_by_pos[pos][metric]:
                continue
            layers_available = list(mean_diff_transfer_by_pos[pos][metric].keys())
            if not layers_available:
                continue
            best_layer = max(layers_available, key=lambda l: mean_diff_transfer_by_pos[pos][metric][l]["centered"]["r2"])
            best_r2 = mean_diff_transfer_by_pos[pos][metric][best_layer]["centered"]["r2"]
            print(f"    {pos}: R²={best_r2:.3f} (L{best_layer})")

    # For backward compatibility, use "final" position for legacy plots
    mean_diff_transfer_results = mean_diff_transfer_by_pos.get("final", mean_diff_transfer_by_pos.get(positions_available[0], {}))

    # Plot probe transfer results - one 4-panel plot per position
    print(f"\nPlotting probe transfer results (per position)...")
    for pos in positions_available:
        if not transfer_results_by_pos[pos]:
            print(f"  Skipping {pos} (no data)")
            continue
        pos_plot_path = plot_path.with_name(f"{plot_path.stem}_{pos}.png")
        plot_transfer_results(
            transfer_results=transfer_results_by_pos[pos],
            direct_r2=direct_r2,
            direct_r2_std=direct_r2_std,
            behavioral=behavioral,
            num_layers=num_layers,
            output_path=pos_plot_path,
            meta_task=META_TASK,
            title_prefix=f"Probe Transfer ({pos})",
        )

    # Plot mean-diff transfer results - one 4-panel plot per position
    print(f"\nPlotting mean-diff transfer results (per position)...")
    for pos in positions_available:
        if not mean_diff_transfer_by_pos[pos]:
            print(f"  Skipping mean-diff {pos} (no data)")
            continue
        pos_plot_path = plot_path_mean_diff.with_name(f"{plot_path_mean_diff.stem}_{pos}.png")
        plot_mean_diff_transfer_results(
            transfer_results=mean_diff_transfer_by_pos[pos],
            direct_r2=mean_diff_direct_r2,
            direct_r2_std=mean_diff_direct_r2_std,
            behavioral=behavioral,
            num_layers=num_layers,
            output_path=pos_plot_path,
            meta_task=META_TASK,
            title_prefix=f"Mean-diff Transfer ({pos})",
        )

    # Plot position comparison (overlay view)
    print(f"\nPlotting position comparison...")
    plot_position_comparison(
        transfer_results_by_pos,
        mean_diff_transfer_by_pos,
        num_layers,
        plot_path_positions,
        META_TASK,
    )

    # Save JSON results
    print(f"Saving results to {results_json_path}...")
    results_json = {
        "config": {
            "model": model_name,
            "dataset": dataset['config']['dataset'],
            "meta_task": META_TASK,
            "num_questions": len(dataset['questions']),
            "num_layers": num_layers,
            "input_base_name": INPUT_BASE_NAME,
            "train_split": TRAIN_SPLIT,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "seed": SEED,
            "probe_alpha": PROBE_ALPHA,
            "probe_pca_components": PROBE_PCA_COMPONENTS,
            "probe_positions": positions_available,
        },
        "meta_target_stats": {
            "name": "stated_confidence" if META_TASK == "confidence" else "P(Answer)",
            "mean": float(confidences.mean()),
            "std": float(confidences.std()),
            "min": float(confidences.min()),
            "max": float(confidences.max()),
        },
        "transfer": {},
        "behavioral": behavioral,
    }

    for metric in metrics_tested:
        layers_available = list(transfer_results[metric].keys())
        best_layer_cen = max(layers_available, key=lambda l: transfer_results[metric][l]["centered"]["r2"])
        best_layer_sep = max(layers_available, key=lambda l: transfer_results[metric][l]["separate"]["r2"])

        results_json["transfer"][metric] = {
            "d2m_centered": {
                "best_layer": best_layer_cen,
                "best_r2": transfer_results[metric][best_layer_cen]["centered"]["r2"],
                "best_r2_std": transfer_results[metric][best_layer_cen]["centered"]["r2_std"],
                "best_pearson": transfer_results[metric][best_layer_cen]["centered"]["pearson"],
            },
            "d2m_separate": {
                "best_layer": best_layer_sep,
                "best_r2": transfer_results[metric][best_layer_sep]["separate"]["r2"],
                "best_r2_std": transfer_results[metric][best_layer_sep]["separate"]["r2_std"],
                "best_pearson": transfer_results[metric][best_layer_sep]["separate"]["pearson"],
            },
            "per_layer": {
                l: {
                    "d2m_centered_r2": transfer_results[metric][l]["centered"]["r2"],
                    "d2m_centered_r2_std": transfer_results[metric][l]["centered"]["r2_std"],
                    "d2m_centered_pearson": transfer_results[metric][l]["centered"]["pearson"],
                    "d2m_centered_pred_conf_pearson": transfer_results[metric][l]["centered"].get("pred_conf_pearson"),
                    "d2m_separate_r2": transfer_results[metric][l]["separate"]["r2"],
                    "d2m_separate_r2_std": transfer_results[metric][l]["separate"]["r2_std"],
                    "d2m_separate_pred_conf_pearson": transfer_results[metric][l]["separate"].get("pred_conf_pearson"),
                    "d2m_separate_pearson": transfer_results[metric][l]["separate"]["pearson"],
                    "d2m_separate_pred_conf_pearson": transfer_results[metric][l]["separate"].get("pred_conf_pearson"),
                }
                for l in layers_available
            }
        }

        # Add D→D results if available (from training)
        if metric in direct_r2 and direct_r2[metric]:
            best_d2d = max(direct_r2[metric].keys(), key=lambda l: direct_r2[metric][l])
            results_json["transfer"][metric]["d2d"] = {
                "best_layer": best_d2d,
                "best_r2": direct_r2[metric][best_d2d],
            }
            for l in direct_r2[metric].keys():
                if l in results_json["transfer"][metric]["per_layer"]:
                    results_json["transfer"][metric]["per_layer"][l]["d2d_r2"] = direct_r2[metric][l]

    # Add position-level summary (best R² per position)
    results_json["transfer_by_position"] = {}
    for pos in positions_available:
        results_json["transfer_by_position"][pos] = {}
        for metric in metrics_tested:
            if metric not in transfer_results_by_pos[pos] or not transfer_results_by_pos[pos][metric]:
                continue
            layers_available = list(transfer_results_by_pos[pos][metric].keys())
            if not layers_available:
                continue
            best_layer = max(layers_available, key=lambda l: transfer_results_by_pos[pos][metric][l]["centered"]["r2"])
            results_json["transfer_by_position"][pos][metric] = {
                "best_layer": best_layer,
                "best_r2": transfer_results_by_pos[pos][metric][best_layer]["centered"]["r2"],
                "per_layer": {
                    l: {
                        "centered_r2": transfer_results_by_pos[pos][metric][l]["centered"]["r2"],
                        "centered_r2_std": transfer_results_by_pos[pos][metric][l]["centered"].get("r2_std", 0.0),
                        "centered_pearson": transfer_results_by_pos[pos][metric][l]["centered"]["pearson"],
                        "centered_pred_conf_pearson": transfer_results_by_pos[pos][metric][l]["centered"].get("pred_conf_pearson"),
                        "centered_pred_conf_pearson_std": transfer_results_by_pos[pos][metric][l]["centered"].get("pred_conf_pearson_std", 0.0),
                        "separate_r2": transfer_results_by_pos[pos][metric][l]["separate"]["r2"],
                        "separate_r2_std": transfer_results_by_pos[pos][metric][l]["separate"].get("r2_std", 0.0),
                        "separate_pearson": transfer_results_by_pos[pos][metric][l]["separate"]["pearson"],
                    }
                    for l in layers_available
                },
            }

    # Add mean-diff position-level summary with full per-layer data
    results_json["mean_diff_by_position"] = {}
    for pos in positions_available:
        results_json["mean_diff_by_position"][pos] = {}
        for metric in metrics_tested:
            if metric not in mean_diff_transfer_by_pos[pos] or not mean_diff_transfer_by_pos[pos][metric]:
                continue
            layers_available = list(mean_diff_transfer_by_pos[pos][metric].keys())
            if not layers_available:
                continue
            best_layer = max(layers_available, key=lambda l: mean_diff_transfer_by_pos[pos][metric][l]["centered"]["r2"])
            results_json["mean_diff_by_position"][pos][metric] = {
                "best_layer": best_layer,
                "best_r2": mean_diff_transfer_by_pos[pos][metric][best_layer]["centered"]["r2"],
                "per_layer": {
                    l: {
                        "centered_r2": mean_diff_transfer_by_pos[pos][metric][l]["centered"]["r2"],
                        "centered_r2_std": mean_diff_transfer_by_pos[pos][metric][l]["centered"].get("r2_std", 0.0),
                        "centered_pearson": mean_diff_transfer_by_pos[pos][metric][l]["centered"]["pearson"],
                        "centered_pred_conf_pearson": mean_diff_transfer_by_pos[pos][metric][l]["centered"].get("pred_conf_pearson"),
                        "centered_pred_conf_pearson_std": mean_diff_transfer_by_pos[pos][metric][l]["centered"].get("pred_conf_pearson_std", 0.0),
                    }
                    for l in layers_available
                },
            }

    # Add per-question paired data for easy verification of correlations
    # This mirrors what run_introspection_experiment.py saves in _paired_data.json
    results_json["per_question"] = []
    for i, q in enumerate(dataset["questions"]):
        item = {
            "question": q.get("question", ""),
            "correct_answer": q.get("correct_answer", ""),
            "stated_confidence": float(confidences[i]),
        }
        # Add all metric values from the MC dataset
        for metric in metrics_tested:
            if metric in dataset["metric_values"]:
                item[metric] = float(dataset["metric_values"][metric][i])
        results_json["per_question"].append(item)

    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # Save NPZ
    print(f"Saving to {results_npz_path}...")
    save_dict = {
        "model": model_name,
        "dataset": dataset['config']['dataset'],
        "meta_task": META_TASK,
        "metrics": np.array(metrics_tested),
        "num_questions": len(dataset['questions']),
        "num_layers": num_layers,
        "confidences": confidences,
    }

    for metric in metrics_tested:
        for layer in transfer_results[metric].keys():
            save_dict[f"transfer_{metric}_layer{layer}_centered_r2"] = transfer_results[metric][layer]["centered"]["r2"]
            save_dict[f"transfer_{metric}_layer{layer}_centered_r2_std"] = transfer_results[metric][layer]["centered"]["r2_std"]
            save_dict[f"transfer_{metric}_layer{layer}_centered_pred_conf_pearson"] = transfer_results[metric][layer]["centered"].get("pred_conf_pearson", np.nan)
            save_dict[f"transfer_{metric}_layer{layer}_separate_pred_conf_pearson"] = transfer_results[metric][layer]["separate"].get("pred_conf_pearson", np.nan)
            save_dict[f"transfer_{metric}_layer{layer}_separate_r2"] = transfer_results[metric][layer]["separate"]["r2"]
            save_dict[f"transfer_{metric}_layer{layer}_separate_r2_std"] = transfer_results[metric][layer]["separate"]["r2_std"]
            # Add D→D from training if available
            if metric in direct_r2 and layer in direct_r2[metric]:
                save_dict[f"d2d_{metric}_layer{layer}_r2"] = direct_r2[metric][layer]
            if metric in direct_r2_std and layer in direct_r2_std[metric]:
                save_dict[f"d2d_{metric}_layer{layer}_r2_std"] = direct_r2_std[metric][layer]

        save_dict[f"behavioral_{metric}_pearson_r"] = behavioral[metric]["pearson_r"]
        save_dict[f"behavioral_{metric}_spearman_r"] = behavioral[metric]["spearman_r"]

        if "test_pearson_r" in behavioral[metric]:
            save_dict[f"behavioral_{metric}_test_pearson_r"] = behavioral[metric]["test_pearson_r"]
        if "test_spearman_r" in behavioral[metric]:
            save_dict[f"behavioral_{metric}_test_spearman_r"] = behavioral[metric]["test_spearman_r"]

    np.savez(results_npz_path, **save_dict)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for metric in metrics_tested:
        print(f"\n{metric}:")
        layers_available = list(transfer_results[metric].keys())

        # D→D (test set)
        if metric in direct_r2 and direct_r2[metric]:
            best_d2d = max(direct_r2[metric].keys(), key=lambda l: direct_r2[metric][l])
            d2d_r2_val = direct_r2[metric][best_d2d]
            d2d_std_val = direct_r2_std.get(metric, {}).get(best_d2d, 0)
            print(f"  D→D (test): R²={d2d_r2_val:.3f}±{d2d_std_val:.3f} (L{best_d2d})")

        # D→M transfer
        best_layer_cen = max(layers_available, key=lambda l: transfer_results[metric][l]["centered"]["r2"])
        centered_r2 = transfer_results[metric][best_layer_cen]["centered"]["r2"]
        centered_std = transfer_results[metric][best_layer_cen]["centered"]["r2_std"]
        centered_pearson = transfer_results[metric][best_layer_cen]["centered"]["pearson"]

        best_layer_sep = max(layers_available, key=lambda l: transfer_results[metric][l]["separate"]["r2"])
        separate_r2 = transfer_results[metric][best_layer_sep]["separate"]["r2"]
        separate_std = transfer_results[metric][best_layer_sep]["separate"]["r2_std"]

        print(f"  D→M Centered: R²={centered_r2:.3f}±{centered_std:.3f}, r={centered_pearson:.3f} (L{best_layer_cen})")
        print(f"  D→M Separate: R²={separate_r2:.3f}±{separate_std:.3f} (L{best_layer_sep})")

        sign_str = "(inv)" if metric_sign_for_confidence(metric) < 0 else ""
        print(f"  Behavioral{sign_str}: r={behavioral[metric]['pearson_r']:.3f}")

    print("\nOutput files:")
    print(f"  {activations_path.name}")
    print(f"  {results_json_path.name}")
    print(f"  {results_npz_path.name}")
    print(f"  {plot_path.name}")
    print(f"  {plot_path_mean_diff.name}")
    print(f"  {plot_path_positions.name}")


if __name__ == "__main__":
    main()
