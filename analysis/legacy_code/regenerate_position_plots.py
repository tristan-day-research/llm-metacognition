#!/usr/bin/env python3
"""Regenerate transfer plots from saved JSON data.

Generates the original 4-panel format for each position:
- Panel 1: Transferred predictions → stated confidence
- Panel 2: Centered Scaler R² (Geometry Check)
- Panel 3: Pearson Correlation (Shift-Invariant)
- Panel 4: Signal Emergence (Normalized timecourse)
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")

# Clip extreme negative R² values for display
R2_PLOT_FLOOR = -0.5
R2_PLOT_CEIL = 1.0

POS_COLORS = {
    "question_mark": "tab:blue",
    "question_newline": "tab:cyan",
    "options_newline": "tab:green",
    "final": "tab:red",
}

POS_LABELS = {
    "question_mark": "question ?",
    "question_newline": "question \\n",
    "options_newline": "options \\n",
    "final": "final",
}

METRIC_COLORS = {
    "entropy": "tab:blue",
    "top_logit": "tab:orange",
    "logit_gap": "tab:green",
}

# Metrics where higher value = lower confidence (need sign flip)
INVERTED_METRICS = {"entropy"}


def _clip_r2(arr):
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return np.clip(arr, R2_PLOT_FLOOR, R2_PLOT_CEIL)


def metric_sign_for_confidence(metric: str) -> int:
    """Return -1 for metrics that are inversely related to confidence."""
    return -1 if metric in INVERTED_METRICS else 1


def plot_transfer_results_from_json(
    pos_data: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
    behavioral: dict,
    direct_r2: dict = None,
    title_prefix: str = "Transfer Analysis",
):
    """
    Plot transfer R² across layers for all metrics (4-panel format).

    Args:
        pos_data: Dict of {metric: {"per_layer": {layer_str: {...}}, ...}}
        num_layers: Number of layers
        output_path: Where to save the plot
        meta_task: Task name for title
        behavioral: Behavioral correlation data
        direct_r2: Optional D→D baseline from legacy transfer data {metric: {layer_str: r2}}
        title_prefix: Title prefix
    """
    metrics = list(pos_data.keys())
    if not metrics:
        print(f"  No metrics for {output_path.name}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix}: {meta_task}", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))

    # Panel 1: Transferred signal → stated confidence
    ax1 = axes[0, 0]
    ax1.set_title("Method 1: Transferred signal → stated confidence\nPearson r(sign·ŷ(meta), confidence)", fontsize=10)

    for metric in metrics:
        color = METRIC_COLORS.get(metric, 'tab:gray')
        per_layer = pos_data[metric].get("per_layer", {})

        vals = []
        stds = []
        for l in layers:
            l_str = str(l)
            if l_str in per_layer:
                vals.append(per_layer[l_str].get("centered_pred_conf_pearson", np.nan))
                stds.append(per_layer[l_str].get("centered_pred_conf_pearson_std", 0.0))
            else:
                vals.append(np.nan)
                stds.append(0.0)
        vals = np.array(vals, dtype=float)
        stds = np.array(stds, dtype=float)

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

        # CI band
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
        color = METRIC_COLORS.get(metric, 'tab:gray')
        per_layer = pos_data[metric].get("per_layer", {})

        # D→D baseline (same for all positions)
        if direct_r2 and metric in direct_r2:
            d2d_vals = []
            for l in layers:
                l_str = str(l)
                if l_str in direct_r2[metric]:
                    d2d_vals.append(direct_r2[metric][l_str])
                else:
                    d2d_vals.append(np.nan)
            d2d_vals = np.array(d2d_vals, dtype=float)
            ax2.plot(layers, _clip_r2(d2d_vals), '-', label=f'{metric} D→D',
                     color=color, linewidth=2, alpha=0.4)

        centered_r2 = []
        centered_std = []
        for l in layers:
            l_str = str(l)
            if l_str in per_layer:
                centered_r2.append(per_layer[l_str].get("centered_r2", np.nan))
                centered_std.append(per_layer[l_str].get("centered_r2_std", 0))
            else:
                centered_r2.append(np.nan)
                centered_std.append(0)
        centered_r2 = np.array(centered_r2, dtype=float)
        centered_std = np.array(centered_std, dtype=float)

        finite = np.isfinite(centered_r2)
        if finite.any():
            best_layer = int(np.argmax(np.where(finite, centered_r2, -np.inf)))
            best_r2 = centered_r2[best_layer]
        else:
            best_layer = 0
            best_r2 = np.nan

        ax2.plot(layers, _clip_r2(centered_r2), '-',
                 label=f'{metric} D→M (best L{best_layer}: {best_r2:.3f})' if np.isfinite(best_r2) else f'{metric} D→M',
                 color=color, linewidth=2)
        if np.any(centered_std > 0):
            ax2.fill_between(layers, _clip_r2(centered_r2 - centered_std), _clip_r2(centered_r2 + centered_std),
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
        color = METRIC_COLORS.get(metric, 'tab:gray')
        per_layer = pos_data[metric].get("per_layer", {})

        # D→D Pearson (sqrt of R²)
        if direct_r2 and metric in direct_r2:
            d2d_corr = []
            for l in layers:
                l_str = str(l)
                if l_str in direct_r2[metric]:
                    d2d_corr.append(np.sqrt(max(direct_r2[metric][l_str], 0.0)))
                else:
                    d2d_corr.append(np.nan)
            ax3.plot(layers, d2d_corr, '-', label=f'{metric} D→D',
                     color=color, linewidth=2, alpha=0.4)

        pearson_r = []
        for l in layers:
            l_str = str(l)
            if l_str in per_layer:
                pearson_r.append(per_layer[l_str].get("centered_pearson", np.nan))
            else:
                pearson_r.append(np.nan)
        pearson_r = np.array(pearson_r, dtype=float)

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
        color = METRIC_COLORS.get(metric, 'tab:gray')
        per_layer = pos_data[metric].get("per_layer", {})

        # D→D normalized
        if direct_r2 and metric in direct_r2:
            d2d_vals = []
            for l in layers:
                l_str = str(l)
                if l_str in direct_r2[metric]:
                    d2d_vals.append(direct_r2[metric][l_str])
                else:
                    d2d_vals.append(np.nan)
            d2d_vals = np.array(d2d_vals, dtype=float)
            if np.nanmax(d2d_vals) > np.nanmin(d2d_vals):
                d2d_norm = (d2d_vals - np.nanmin(d2d_vals)) / (np.nanmax(d2d_vals) - np.nanmin(d2d_vals))
            else:
                d2d_norm = np.zeros_like(d2d_vals)
            ax4.plot(layers, d2d_norm, '-', label=f'{metric} D→D',
                     color=color, linewidth=2, alpha=0.4)

        centered_r2 = []
        for l in layers:
            l_str = str(l)
            if l_str in per_layer:
                centered_r2.append(per_layer[l_str].get("centered_r2", np.nan))
            else:
                centered_r2.append(np.nan)
        centered_r2 = np.array(centered_r2, dtype=float)

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
    transfer_by_pos: dict,
    mean_diff_by_pos: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
):
    """Plot transfer R² across layers comparing different token positions."""
    R2_FLOOR = -0.5
    R2_CEIL = 1.0

    def clip_r2(arr):
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, R2_FLOOR, R2_CEIL)

    metrics = set()
    for pos_data in transfer_by_pos.values():
        metrics.update(pos_data.keys())
    metrics = sorted(metrics)

    if len(metrics) == 0:
        print(f"  No metrics found for {meta_task}")
        return

    positions = list(transfer_by_pos.keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6 * n_metrics, 10), squeeze=False)
    fig.suptitle(f"Position Comparison: {meta_task}", fontsize=14, fontweight="bold")

    layers = list(range(num_layers))

    # Top row: probe-based
    for col, metric in enumerate(metrics):
        ax = axes[0, col]
        ax.set_title(f"Probe Transfer: {metric}", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        for pos in positions:
            if metric not in transfer_by_pos.get(pos, {}):
                continue
            color = POS_COLORS.get(pos, "tab:gray")
            display_name = POS_LABELS.get(pos, pos)

            per_layer = transfer_by_pos[pos][metric].get("per_layer", {})
            r2_vals = []
            for l in layers:
                l_str = str(l)
                if l_str in per_layer:
                    r2_vals.append(per_layer[l_str].get("centered_r2", np.nan))
                else:
                    r2_vals.append(np.nan)
            r2_vals = np.array(r2_vals, dtype=float)

            finite = np.isfinite(r2_vals)
            if finite.any():
                best_layer = int(np.argmax(np.where(finite, r2_vals, -np.inf)))
                best_r2 = r2_vals[best_layer]
                label = f"{display_name} (L{best_layer}: {best_r2:.3f})"
            else:
                label = display_name

            ax.plot(layers, clip_r2(r2_vals), "-", label=label, color=color, linewidth=2)

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("R²")
        ax.set_ylim(R2_FLOOR, R2_CEIL)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Bottom row: mean-diff
    for col, metric in enumerate(metrics):
        ax = axes[1, col]
        ax.set_title(f"Mean-Diff Transfer: {metric}", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        for pos in positions:
            if metric not in mean_diff_by_pos.get(pos, {}):
                continue
            color = POS_COLORS.get(pos, "tab:gray")
            display_name = POS_LABELS.get(pos, pos)

            per_layer = mean_diff_by_pos[pos][metric].get("per_layer", {})
            r2_vals = []
            for l in layers:
                l_str = str(l)
                if l_str in per_layer:
                    r2_vals.append(per_layer[l_str].get("centered_r2", np.nan))
                else:
                    r2_vals.append(np.nan)
            r2_vals = np.array(r2_vals, dtype=float)

            finite = np.isfinite(r2_vals)
            if finite.any():
                best_layer = int(np.argmax(np.where(finite, r2_vals, -np.inf)))
                best_r2 = r2_vals[best_layer]
                label = f"{display_name} (L{best_layer}: {best_r2:.3f})"
            else:
                label = display_name

            ax.plot(layers, clip_r2(r2_vals), "-", label=label, color=color, linewidth=2)

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("R²")
        ax.set_ylim(R2_FLOOR, R2_CEIL)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    # Find all transfer results JSON files
    transfer_files = list(OUTPUT_DIR.glob("*_transfer_*_results.json"))

    for json_path in transfer_files:
        with open(json_path) as f:
            data = json.load(f)

        if "transfer_by_position" not in data or "mean_diff_by_position" not in data:
            print(f"Skipping {json_path.name} (no position data)")
            continue

        # Extract task name from filename
        parts = json_path.stem.split("_transfer_")
        if len(parts) != 2:
            continue
        meta_task = parts[1].replace("_results", "")

        base_name = parts[0]
        num_layers = data["config"]["num_layers"]
        behavioral = data.get("behavioral", {})
        positions = list(data["transfer_by_position"].keys())

        # Extract D→D baseline from legacy transfer section (same for all positions)
        # Format: {metric: {layer_str: r2_value}}
        probe_direct_r2 = {}
        if "transfer" in data:
            for metric, metric_data in data["transfer"].items():
                probe_direct_r2[metric] = {}
                per_layer = metric_data.get("per_layer", {})
                for layer_str, layer_data in per_layer.items():
                    if "d2d_r2" in layer_data:
                        probe_direct_r2[metric][layer_str] = layer_data["d2d_r2"]

        print(f"\nProcessing {json_path.name}...")
        print(f"  Positions: {positions}")

        # 1. Generate 4-panel probe transfer plot for each position
        for pos in positions:
            pos_data = data["transfer_by_position"].get(pos, {})
            if not pos_data:
                print(f"  Skipping probe {pos} (no data)")
                continue

            probe_path = json_path.parent / f"{base_name}_transfer_{meta_task}_results_{pos}.png"
            print(f"  Generating probe transfer plot for {pos}...")
            plot_transfer_results_from_json(
                pos_data=pos_data,
                num_layers=num_layers,
                output_path=probe_path,
                meta_task=meta_task,
                behavioral=behavioral,
                direct_r2=probe_direct_r2,
                title_prefix=f"Probe Transfer ({pos})",
            )

        # 2. Generate 4-panel mean-diff transfer plot for each position
        # Note: Mean-diff D→D not stored in JSON, so no direct_r2 for these plots
        for pos in positions:
            pos_data = data["mean_diff_by_position"].get(pos, {})
            if not pos_data:
                print(f"  Skipping mean-diff {pos} (no data)")
                continue

            mean_diff_path = json_path.parent / f"{base_name}_transfer_{meta_task}_mean_diff_results_{pos}.png"
            print(f"  Generating mean-diff transfer plot for {pos}...")
            plot_transfer_results_from_json(
                pos_data=pos_data,
                num_layers=num_layers,
                output_path=mean_diff_path,
                meta_task=meta_task,
                behavioral=behavioral,
                direct_r2=None,  # Mean-diff D→D not in JSON
                title_prefix=f"Mean-diff Transfer ({pos})",
            )

        # 3. Position comparison plot (overlay view)
        comparison_path = json_path.parent / f"{base_name}_transfer_{meta_task}_position_comparison.png"
        print(f"  Generating position comparison plot...")
        plot_position_comparison(
            transfer_by_pos=data["transfer_by_position"],
            mean_diff_by_pos=data["mean_diff_by_position"],
            num_layers=num_layers,
            output_path=comparison_path,
            meta_task=meta_task,
        )


if __name__ == "__main__":
    main()
