#!/usr/bin/env python3
"""
Synthesize causal test results across ablation and steering experiments.

This script loads results from all 4 causal tests (ablation × 2 tasks + steering × 2 tasks)
and determines which layers pass all tests for a given metric and method.

Outputs:
- JSON with full per-layer data and intersection analysis
- Visualization showing pass/fail status for each layer × test
- Text summary with exact criteria and results
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PREFIX = "Llama-3.3-70B-Instruct"
DATASET = "TriviaMC"
METRIC = "top_logit"  # Which metric's directions to analyze
METHOD = "mean_diff"  # Which direction method to analyze
P_THRESHOLD = 0.05    # Significance threshold for all tests

OUTPUT_DIR = Path("outputs")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_causal_test_results(
    model_prefix: str,
    dataset: str,
    metric: str,
    method: str,
) -> Dict[str, Dict]:
    """
    Load all 4 causal test result files and extract per-layer data.

    Returns:
        {
            "ablation_confidence": {layer: {"p_value": float, "effect_size_z": float, ...}},
            "ablation_delegate": {...},
            "steering_confidence": {...},
            "steering_delegate": {...},
        }
    """
    tests = {
        "ablation_confidence": f"{model_prefix}_{dataset}_ablation_confidence_{metric}_results.json",
        "ablation_delegate": f"{model_prefix}_{dataset}_ablation_delegate_{metric}_results.json",
        "steering_confidence": f"{model_prefix}_{dataset}_steering_confidence_{metric}_results.json",
        "steering_delegate": f"{model_prefix}_{dataset}_steering_delegate_{metric}_results.json",
    }

    results = {}
    files_loaded = {}

    for test_name, filename in tests.items():
        filepath = OUTPUT_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required file: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        files_loaded[test_name] = str(filepath)

        # Extract per-layer data for the specified method
        if method not in data:
            raise KeyError(f"Method '{method}' not found in {filepath}. Available: {list(data.keys())}")

        method_data = data[method]
        per_layer = method_data["per_layer"]

        results[test_name] = {}
        for layer_str, layer_data in per_layer.items():
            layer = int(layer_str)
            results[test_name][layer] = {
                "p_value_pooled": layer_data["p_value_pooled"],
                "p_value_fdr": layer_data.get("p_value_fdr"),
                "effect_size_z": layer_data["effect_size_z"],
            }

    return results, files_loaded


def load_transfer_results(
    model_prefix: str,
    dataset: str,
    metric: str,
) -> Optional[Dict[str, Dict]]:
    """
    Load transfer results for context (R² values).

    Returns:
        {
            "confidence": {layer: {"d2m_r2": float}},
            "delegate": {layer: {"d2m_r2": float}},
        }
    """
    transfer_results = {}

    for task in ["confidence", "delegate"]:
        filename = f"{model_prefix}_{dataset}_transfer_{task}_results.json"
        filepath = OUTPUT_DIR / filename

        if not filepath.exists():
            print(f"  Warning: Transfer file not found: {filepath}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        # Extract per-layer R² for the specified metric
        if "transfer" not in data or metric not in data["transfer"]:
            print(f"  Warning: Metric '{metric}' not found in transfer results for {task}")
            continue

        per_layer = data["transfer"][metric]["per_layer"]
        transfer_results[task] = {}

        for layer_str, layer_data in per_layer.items():
            layer = int(layer_str)
            # Use d2m_separate_r2 as the primary transfer metric
            transfer_results[task][layer] = {
                "d2m_r2": layer_data.get("d2m_separate_r2", layer_data.get("d2m_centered_r2")),
            }

    return transfer_results if transfer_results else None


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_layer_status(
    results: Dict[str, Dict],
    p_threshold: float = 0.05,
) -> Dict[int, Dict]:
    """
    For each layer, determine pass/fail status on each test.

    Returns:
        {
            layer: {
                "ablation_confidence": {"pass": bool, "p_value": float, "z": float},
                "ablation_delegate": {...},
                "steering_confidence": {...},
                "steering_delegate": {...},
                "passes_all": bool,
                "num_passed": int,
            }
        }
    """
    # Get all layers (should be same across all tests)
    all_layers = set()
    for test_data in results.values():
        all_layers.update(test_data.keys())

    layer_status = {}

    for layer in sorted(all_layers):
        layer_status[layer] = {}
        passes = []

        for test_name, test_data in results.items():
            if layer not in test_data:
                layer_status[layer][test_name] = {
                    "pass": False,
                    "p_value": None,
                    "z": None,
                    "missing": True,
                }
                passes.append(False)
            else:
                p_val = test_data[layer]["p_value_pooled"]
                z = test_data[layer]["effect_size_z"]
                passed = p_val < p_threshold

                layer_status[layer][test_name] = {
                    "pass": passed,
                    "p_value": p_val,
                    "z": z,
                }
                passes.append(passed)

        layer_status[layer]["passes_all"] = all(passes)
        layer_status[layer]["num_passed"] = sum(passes)

    return layer_status


def get_passing_layers(
    layer_status: Dict[int, Dict],
    min_layer: int = 0,
    max_layer: int = 999,
) -> List[int]:
    """Get layers that pass all 4 tests, optionally filtered by layer range."""
    return [
        layer for layer, status in layer_status.items()
        if status["passes_all"] and min_layer <= layer <= max_layer
    ]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_synthesis_results(
    layer_status: Dict[int, Dict],
    transfer_results: Optional[Dict],
    output_path: Path,
    metric: str,
    method: str,
    p_threshold: float,
    focus_range: Tuple[int, int] = (15, 55),
):
    """
    Create visualization of causal test results.

    - Heatmap showing pass/fail for each layer × test (focused on mid-layers)
    - Summary panel with counts and passing layers
    """
    test_names = ["ablation_confidence", "ablation_delegate",
                  "steering_confidence", "steering_delegate"]
    test_labels = ["Abl. Conf", "Abl. Del", "Steer. Conf", "Steer. Del"]

    # Focus on mid-layers for the heatmap
    min_layer, max_layer = focus_range
    focus_layers = [l for l in sorted(layer_status.keys()) if min_layer <= l <= max_layer]

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # Layout: heatmap on left (large), summary on right (smaller)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1],
                          hspace=0.3, wspace=0.3)

    # Panel 1: Heatmap of pass/fail status
    ax1 = fig.add_subplot(gs[:, 0])

    # Build matrix: rows=layers, cols=tests
    matrix = np.zeros((len(focus_layers), len(test_names)))
    for i, layer in enumerate(focus_layers):
        for j, test in enumerate(test_names):
            if layer_status[layer][test]["pass"]:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = 0.0

    # Custom colormap: red for fail, green for pass
    cmap = plt.cm.colors.ListedColormap(['#ff6b6b', '#51cf66'])

    im = ax1.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add p-value annotations
    for i, layer in enumerate(focus_layers):
        for j, test in enumerate(test_names):
            p_val = layer_status[layer][test]["p_value"]
            if p_val is not None:
                # Show p-value, bold if significant
                text_color = 'white' if matrix[i, j] == 1 else 'black'
                fontweight = 'bold' if matrix[i, j] == 1 else 'normal'
                ax1.text(j, i, f'{p_val:.3f}', ha='center', va='center',
                        fontsize=7, color=text_color, fontweight=fontweight)

    # Highlight layers that pass all 4
    passing_all = [l for l in focus_layers if layer_status[l]["passes_all"]]
    for layer in passing_all:
        i = focus_layers.index(layer)
        rect = plt.Rectangle((-0.5, i-0.5), len(test_names), 1,
                             fill=False, edgecolor='gold', linewidth=3)
        ax1.add_patch(rect)

    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_labels, fontsize=10)
    ax1.set_yticks(range(len(focus_layers)))
    ax1.set_yticklabels(focus_layers, fontsize=8)
    ax1.set_xlabel("Test", fontsize=12)
    ax1.set_ylabel("Layer", fontsize=12)
    ax1.set_title(f"Causal Test Results: {metric} {method}\n(p < {p_threshold}, gold border = passes all 4)",
                  fontsize=12)

    # Legend
    pass_patch = mpatches.Patch(color='#51cf66', label='Pass (p < 0.05)')
    fail_patch = mpatches.Patch(color='#ff6b6b', label='Fail (p >= 0.05)')
    ax1.legend(handles=[pass_patch, fail_patch], loc='upper right', fontsize=9)

    # Panel 2: Summary statistics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Count layers passing each test
    all_layers = sorted(layer_status.keys())
    counts = {}
    for test in test_names:
        counts[test] = sum(1 for l in all_layers if layer_status[l][test]["pass"])

    # Layers passing all
    all_passing = get_passing_layers(layer_status)
    mid_passing = get_passing_layers(layer_status, min_layer=10)  # Exclude early layers

    summary_text = f"""SUMMARY
{'='*40}

Metric: {metric}
Method: {method}
Threshold: p < {p_threshold}
Total layers: {len(all_layers)}

LAYERS PASSING EACH TEST:
  Ablation confidence:  {counts['ablation_confidence']:3d}
  Ablation delegate:    {counts['ablation_delegate']:3d}
  Steering confidence:  {counts['steering_confidence']:3d}
  Steering delegate:    {counts['steering_delegate']:3d}

LAYERS PASSING ALL 4 TESTS:
  All layers:    {all_passing if all_passing else 'None'}
  Mid-layers:    {mid_passing if mid_passing else 'None'}
  (>10)
"""

    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    # Panel 3: Details for passing layers
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    if mid_passing:
        detail_text = "DETAILS FOR PASSING MID-LAYERS:\n" + "="*40 + "\n\n"
        for layer in mid_passing[:5]:  # Show up to 5
            detail_text += f"Layer {layer}:\n"
            for test, label in zip(test_names, test_labels):
                p = layer_status[layer][test]["p_value"]
                z = layer_status[layer][test]["z"]
                detail_text += f"  {label:12s}: p={p:.4f}, Z={z:+.2f}\n"

            # Add transfer R² if available
            if transfer_results:
                for task in ["confidence", "delegate"]:
                    if task in transfer_results and layer in transfer_results[task]:
                        r2 = transfer_results[task][layer]["d2m_r2"]
                        detail_text += f"  Transfer {task[:4]:4s}:  R²={r2:.3f}\n"
            detail_text += "\n"
    else:
        detail_text = "NO MID-LAYERS PASS ALL 4 TESTS\n\n"

        # Show layers that pass 3/4
        almost = [(l, layer_status[l]["num_passed"])
                  for l in all_layers if layer_status[l]["num_passed"] == 3 and l > 10]
        if almost:
            detail_text += "Layers passing 3/4 tests (mid-layers):\n"
            for layer, _ in almost[:10]:
                failing = [t for t in test_names if not layer_status[layer][t]["pass"]]
                detail_text += f"  Layer {layer}: fails {failing[0]}\n"

    ax3.text(0.05, 0.95, detail_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# TEXT SUMMARY
# =============================================================================

def write_summary(
    layer_status: Dict[int, Dict],
    transfer_results: Optional[Dict],
    files_loaded: Dict[str, str],
    output_path: Path,
    metric: str,
    method: str,
    p_threshold: float,
):
    """Write detailed text summary."""

    test_names = ["ablation_confidence", "ablation_delegate",
                  "steering_confidence", "steering_delegate"]

    all_layers = sorted(layer_status.keys())
    all_passing = get_passing_layers(layer_status)
    mid_passing = get_passing_layers(layer_status, min_layer=10)

    lines = []
    lines.append("CAUSAL TEST SYNTHESIS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Metric: {metric}")
    lines.append(f"Method: {method}")
    lines.append(f"Significance threshold: p < {p_threshold} (pooled)")
    lines.append(f"Total layers analyzed: {len(all_layers)}")
    lines.append("")

    lines.append("FILES ANALYZED:")
    for test_name, filepath in files_loaded.items():
        lines.append(f"  {test_name}: {filepath}")
    lines.append("")

    lines.append("LAYERS PASSING EACH TEST (p < 0.05):")
    for test in test_names:
        passing = sorted([l for l in all_layers if layer_status[l][test]["pass"]])
        lines.append(f"  {test}:")
        lines.append(f"    Count: {len(passing)}")
        lines.append(f"    Layers: {passing[:20]}{'...' if len(passing) > 20 else ''}")
    lines.append("")

    lines.append("LAYERS PASSING ALL 4 TESTS:")
    lines.append(f"  All layers: {all_passing if all_passing else 'None'}")
    lines.append(f"  Mid-layers (>10): {mid_passing if mid_passing else 'None'}")
    lines.append("")

    if mid_passing:
        lines.append("DETAILED STATS FOR PASSING MID-LAYERS:")
        lines.append("-" * 60)
        for layer in mid_passing:
            lines.append(f"\nLayer {layer}:")
            for test in test_names:
                p = layer_status[layer][test]["p_value"]
                z = layer_status[layer][test]["z"]
                lines.append(f"  {test:25s}: p={p:.6f}, Z={z:+.3f}")

            if transfer_results:
                for task in ["confidence", "delegate"]:
                    if task in transfer_results and layer in transfer_results[task]:
                        r2 = transfer_results[task][layer]["d2m_r2"]
                        lines.append(f"  transfer_{task:10s} R²:       {r2:.4f}")
    else:
        lines.append("NO MID-LAYERS PASS ALL 4 TESTS")
        lines.append("")
        lines.append("Layers passing 3/4 tests (showing which test they fail):")
        almost = [(l, layer_status[l]["num_passed"])
                  for l in all_layers if layer_status[l]["num_passed"] == 3 and l > 10]
        for layer, _ in sorted(almost, key=lambda x: x[0]):
            failing = [t for t in test_names if not layer_status[layer][t]["pass"]]
            p = layer_status[layer][failing[0]]["p_value"]
            lines.append(f"  Layer {layer:2d}: fails {failing[0]} (p={p:.4f})")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CAUSAL TEST SYNTHESIS")
    print("=" * 60)
    print(f"Metric: {METRIC}")
    print(f"Method: {METHOD}")
    print(f"Threshold: p < {P_THRESHOLD}")
    print()

    # Load causal test results
    print("Loading causal test results...")
    try:
        results, files_loaded = load_causal_test_results(
            MODEL_PREFIX, DATASET, METRIC, METHOD
        )
        for test_name, filepath in files_loaded.items():
            print(f"  {test_name}: {filepath}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Load transfer results (optional)
    print("\nLoading transfer results (for context)...")
    transfer_results = load_transfer_results(MODEL_PREFIX, DATASET, METRIC)

    # Compute layer status
    print("\nAnalyzing layer status...")
    layer_status = compute_layer_status(results, P_THRESHOLD)

    # Report results
    all_passing = get_passing_layers(layer_status)
    mid_passing = get_passing_layers(layer_status, min_layer=10)

    print(f"\nRESULTS:")
    print(f"  Layers passing all 4 tests: {all_passing if all_passing else 'None'}")
    print(f"  Mid-layers (>10) passing all 4: {mid_passing if mid_passing else 'None'}")

    # Generate outputs
    base_output = f"{MODEL_PREFIX}_{DATASET}_causal_synthesis_{METRIC}_{METHOD}"

    # JSON output
    json_path = OUTPUT_DIR / f"{base_output}.json"
    output_json = {
        "config": {
            "model_prefix": MODEL_PREFIX,
            "dataset": DATASET,
            "metric": METRIC,
            "method": METHOD,
            "p_threshold": P_THRESHOLD,
        },
        "files_analyzed": files_loaded,
        "layer_status": {
            str(layer): status for layer, status in layer_status.items()
        },
        "passing_all_layers": all_passing,
        "passing_mid_layers": mid_passing,
    }
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Visualization
    plot_path = OUTPUT_DIR / f"{base_output}.png"
    plot_synthesis_results(
        layer_status, transfer_results, plot_path,
        METRIC, METHOD, P_THRESHOLD
    )

    # Text summary
    summary_path = OUTPUT_DIR / f"{base_output}_summary.txt"
    write_summary(
        layer_status, transfer_results, files_loaded, summary_path,
        METRIC, METHOD, P_THRESHOLD
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
