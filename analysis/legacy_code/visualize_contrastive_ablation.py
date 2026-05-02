"""
Visualize contrastive direction steering and ablation results.

Reads the steering_results.json and ablation_analysis.json and creates visualizations:

Steering:
1. Confidence change vs multiplier (steering curve)
2. Alignment change vs multiplier
3. Contrastive vs control comparison

Ablation:
1. Alignment change by layer (contrastive vs control)
2. Confidence change by layer
3. Summary statistics
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

# Configuration - match the model/dataset/metric from run_contrastive_direction.py
BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = BASE_MODEL_NAME
DATASET_NAME = "TriviaMC"
METRIC = "top_logit"
OUTPUTS_DIR = Path("outputs")


def get_model_short_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def get_output_prefix(task: str = "confidence") -> str:
    """Generate output prefix for a specific task."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if task == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_{METRIC}{task_suffix}_contrastive")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_{METRIC}{task_suffix}_contrastive")


def find_available_tasks() -> list:
    """Find which tasks have results available."""
    available = []
    for task in ["confidence", "delegate"]:
        prefix = get_output_prefix(task)
        # Check if either steering or ablation results exist
        if Path(f"{prefix}_steering_results.json").exists() or Path(f"{prefix}_ablation_analysis.json").exists():
            available.append(task)
    return available


def analyze_steering_results(steering_data: dict) -> dict:
    """
    Extract summary statistics from the large steering results JSON.

    Returns dict with:
        layers: list of layer indices
        multipliers: list of multiplier values
        contrastive: {layer -> {mult -> {mean_alignment, mean_confidence}}}
        controls_mean: {layer -> {mult -> {mean_alignment, mean_confidence}}}
        baseline: {layer -> {mean_alignment, mean_confidence}}
    """
    layers = list(steering_data.get("layer_results", {}).keys())
    # Convert to int and sort
    layers = sorted([int(l) if isinstance(l, str) else l for l in layers])

    multipliers = steering_data.get("multipliers", [])

    analysis = {
        "layers": layers,
        "multipliers": multipliers,
        "contrastive": {},
        "controls_mean": {},
        "baseline": {},
    }

    for layer in layers:
        layer_key = str(layer) if str(layer) in steering_data["layer_results"] else layer
        lr = steering_data["layer_results"][layer_key]

        # Baseline
        baseline_results = lr.get("baseline", [])
        if baseline_results:
            analysis["baseline"][layer] = {
                "mean_alignment": np.mean([r["alignment"] for r in baseline_results]),
                "mean_confidence": np.mean([r["confidence"] for r in baseline_results]),
            }

        # Contrastive by multiplier
        analysis["contrastive"][layer] = {}
        contrastive_data = lr.get("contrastive", {})
        for mult in multipliers:
            mult_key = str(mult) if str(mult) in contrastive_data else mult
            if mult_key in contrastive_data:
                results = contrastive_data[mult_key]
                analysis["contrastive"][layer][mult] = {
                    "mean_alignment": np.mean([r["alignment"] for r in results]),
                    "mean_confidence": np.mean([r["confidence"] for r in results]),
                }

        # Controls mean by multiplier
        analysis["controls_mean"][layer] = {}
        controls_data = lr.get("controls", {})
        for mult in multipliers:
            mult_key = str(mult) if str(mult) in list(controls_data.values())[0] else mult
            ctrl_alignments = []
            ctrl_confidences = []
            for ctrl_key, ctrl_results_by_mult in controls_data.items():
                if mult_key in ctrl_results_by_mult:
                    results = ctrl_results_by_mult[mult_key]
                    ctrl_alignments.extend([r["alignment"] for r in results])
                    ctrl_confidences.extend([r["confidence"] for r in results])
            if ctrl_alignments:
                analysis["controls_mean"][layer][mult] = {
                    "mean_alignment": np.mean(ctrl_alignments),
                    "mean_confidence": np.mean(ctrl_confidences),
                }

    return analysis


def plot_steering_results(steering_analysis: dict, output_path: str, task: str = "confidence"):
    """Create steering visualization."""
    layers = steering_analysis["layers"]
    multipliers = steering_analysis["multipliers"]
    task_label = "Delegate" if task == "delegate" else "Confidence"

    # First, compute slopes for all layers to find the best one
    pos_mults = [m for m in multipliers if m > 0]
    slopes_contr = []
    slopes_ctrl = []

    for layer in layers:
        baseline_conf = steering_analysis["baseline"][layer]["mean_confidence"]

        # Contrastive slope
        confs = []
        for mult in pos_mults:
            if mult in steering_analysis["contrastive"][layer]:
                confs.append((mult, steering_analysis["contrastive"][layer][mult]["mean_confidence"] - baseline_conf))
        if len(confs) >= 2:
            x = np.array([c[0] for c in confs])
            y = np.array([c[1] for c in confs])
            slope = np.polyfit(x, y, 1)[0]
            slopes_contr.append(slope)
        else:
            slopes_contr.append(0)

        # Control slope
        confs = []
        for mult in pos_mults:
            if mult in steering_analysis["controls_mean"][layer]:
                confs.append((mult, steering_analysis["controls_mean"][layer][mult]["mean_confidence"] - baseline_conf))
        if len(confs) >= 2:
            x = np.array([c[0] for c in confs])
            y = np.array([c[1] for c in confs])
            slope = np.polyfit(x, y, 1)[0]
            slopes_ctrl.append(slope)
        else:
            slopes_ctrl.append(0)

    # Find best layer by steering slope
    best_layer = layers[np.argmax(np.abs(slopes_contr))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Confidence change vs multiplier (for BEST layer)
    ax = axes[0, 0]
    baseline_conf = steering_analysis["baseline"][best_layer]["mean_confidence"]

    contr_confs = []
    ctrl_confs = []
    for mult in multipliers:
        if mult in steering_analysis["contrastive"][best_layer]:
            contr_confs.append(steering_analysis["contrastive"][best_layer][mult]["mean_confidence"] - baseline_conf)
        else:
            contr_confs.append(0)
        if mult in steering_analysis["controls_mean"][best_layer]:
            ctrl_confs.append(steering_analysis["controls_mean"][best_layer][mult]["mean_confidence"] - baseline_conf)
        else:
            ctrl_confs.append(0)

    ax.plot(multipliers, contr_confs, 'b-o', label='Contrastive', linewidth=2, markersize=6)
    ax.plot(multipliers, ctrl_confs, 'gray', marker='s', alpha=0.6, label='Control (mean)', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Steering Multiplier')
    ax.set_ylabel('Confidence Change')
    ax.set_title(f'Confidence Change vs Steering Strength\n(Best Layer: {best_layer})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Alignment change vs multiplier (for BEST layer)
    ax = axes[0, 1]
    baseline_align = steering_analysis["baseline"][best_layer]["mean_alignment"]

    contr_aligns = []
    ctrl_aligns = []
    for mult in multipliers:
        if mult in steering_analysis["contrastive"][best_layer]:
            contr_aligns.append(steering_analysis["contrastive"][best_layer][mult]["mean_alignment"] - baseline_align)
        else:
            contr_aligns.append(0)
        if mult in steering_analysis["controls_mean"][best_layer]:
            ctrl_aligns.append(steering_analysis["controls_mean"][best_layer][mult]["mean_alignment"] - baseline_align)
        else:
            ctrl_aligns.append(0)

    ax.plot(multipliers, contr_aligns, 'b-o', label='Contrastive', linewidth=2, markersize=6)
    ax.plot(multipliers, ctrl_aligns, 'gray', marker='s', alpha=0.6, label='Control (mean)', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Steering Multiplier')
    ax.set_ylabel('Alignment Change')
    ax.set_title(f'Alignment Change vs Steering Strength\n(Best Layer: {best_layer})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Steering slope by layer (conf change per unit multiplier)
    ax = axes[1, 0]
    # (slopes already computed above)
    ax.plot(layers, slopes_contr, 'b-o', label='Contrastive', markersize=3, linewidth=1)
    ax.plot(layers, slopes_ctrl, 'gray', alpha=0.5, label='Control', linewidth=1)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Confidence Slope (per unit multiplier)')
    ax.set_title('Steering Effect (Slope) by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    mean_contr_slope = np.mean(slopes_contr)
    mean_ctrl_slope = np.mean(slopes_ctrl)
    best_layer_idx = layers.index(best_layer)
    best_contr_slope = slopes_contr[best_layer_idx]
    best_ctrl_slope = slopes_ctrl[best_layer_idx]

    summary = f"""
STEERING EXPERIMENT SUMMARY

Layers tested: {layers[0]} to {layers[-1]} ({len(layers)} total)
Multipliers: {multipliers}

Mean confidence slope:
  Contrastive: {mean_contr_slope:.4f}
  Control:     {mean_ctrl_slope:.4f}

Best layer: {best_layer}
  Contrastive slope: {best_contr_slope:.4f}
  Control slope:     {best_ctrl_slope:.4f}
  Ratio: {abs(best_contr_slope) / max(abs(best_ctrl_slope), 1e-6):.1f}x

Interpretation:
"""
    # Compare best layer's contrastive vs control slope
    slope_ratio = abs(best_contr_slope) / max(abs(best_ctrl_slope), 1e-6)
    if slope_ratio > 2.0 and abs(best_contr_slope) > 0.002:
        summary += f"""  STEERING WORKS: Contrastive direction
  has {slope_ratio:.1f}x stronger effect than
  random directions at layer {best_layer}."""
    elif abs(best_contr_slope) > 0.002:
        summary += f"""  WEAK EFFECT: Contrastive direction
  shows some steering effect but not much
  stronger than random directions."""
    else:
        summary += f"""  NO EFFECT: Steering has minimal
  effect on confidence."""

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'Contrastive Direction Steering Results ({task_label} Task)\n{get_model_short_name(BASE_MODEL_NAME)} - {DATASET_NAME}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Steering visualization saved to: {output_path}")


def plot_ablation_results(ablation_data: dict, output_path: str, task: str = "confidence"):
    """Create ablation visualization."""
    task_label = "Delegate" if task == "delegate" else "Confidence"

    # Handle both old flat format and new nested format with "effects" key
    if "effects" in ablation_data:
        effects = ablation_data["effects"]
    else:
        effects = ablation_data

    layers = sorted([int(k) for k in effects.keys() if k.isdigit()])

    baseline_alignments = []
    contrastive_alignment_changes = []
    control_alignment_changes = []
    contrastive_confidence_changes = []

    for layer in layers:
        data = effects[str(layer)]
        # Handle nested structure: baseline, contrastive_ablated, control_ablated
        baseline = data.get("baseline", data)
        contrastive = data.get("contrastive_ablated", data)
        control = data.get("control_ablated", data)

        baseline_alignments.append(baseline.get("mean_alignment", baseline.get("baseline_mean_alignment", 0)))
        contrastive_alignment_changes.append(contrastive.get("alignment_change", data.get("contrastive_alignment_change", 0)))
        control_alignment_changes.append(control.get("alignment_change", data.get("controls_alignment_change", 0)))

        baseline_conf = baseline.get("mean_confidence", data.get("baseline_mean_confidence", 0))
        contrastive_conf = contrastive.get("mean_confidence", data.get("contrastive_ablated_mean_confidence", 0))
        contrastive_confidence_changes.append(contrastive_conf - baseline_conf)

    baseline_alignments = np.array(baseline_alignments)
    contrastive_alignment_changes = np.array(contrastive_alignment_changes)
    control_alignment_changes = np.array(control_alignment_changes)
    contrastive_confidence_changes = np.array(contrastive_confidence_changes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Alignment change by layer
    ax = axes[0, 0]
    ax.plot(layers, contrastive_alignment_changes, 'b-o', label='Contrastive', markersize=3, linewidth=1)
    ax.plot(layers, control_alignment_changes, 'gray', alpha=0.5, label='Control (random)', linewidth=1)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.fill_between(layers, contrastive_alignment_changes, 0,
                    where=contrastive_alignment_changes < 0, alpha=0.3, color='red')
    ax.fill_between(layers, contrastive_alignment_changes, 0,
                    where=contrastive_alignment_changes > 0, alpha=0.3, color='green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Alignment Change')
    ax.set_title('Alignment Change from Ablation by Layer\n(negative = calibration worsened)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Confidence change by layer
    ax = axes[0, 1]
    ax.plot(layers, contrastive_confidence_changes, 'purple', marker='o', markersize=3, linewidth=1)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Confidence Change')
    ax.set_title('Confidence Change from Ablation by Layer')
    ax.grid(True, alpha=0.3)

    # 3. Contrastive vs Control difference
    ax = axes[1, 0]
    diff = contrastive_alignment_changes - control_alignment_changes
    ax.bar(layers, diff, color=['red' if d < 0 else 'green' for d in diff], alpha=0.7, width=0.8)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Contrastive - Control')
    ax.set_title('Alignment Change: Contrastive vs Control\n(negative = contrastive had larger effect)')
    ax.grid(True, alpha=0.3)

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    n_negative = (contrastive_alignment_changes < 0).sum()
    n_positive = (contrastive_alignment_changes > 0).sum()
    mean_effect = contrastive_alignment_changes.mean()
    control_mean = control_alignment_changes.mean()

    worst_layer = layers[np.argmin(contrastive_alignment_changes)]
    worst_change = contrastive_alignment_changes.min()

    summary = f"""
ABLATION EXPERIMENT SUMMARY

Layers tested: {layers[0]} to {layers[-1]} ({len(layers)} total)
Baseline alignment: {baseline_alignments[0]:.4f}

Alignment change from ablation:
  Contrastive mean: {mean_effect:.4f}
  Control mean:     {control_mean:.4f}

  Range: [{contrastive_alignment_changes.min():.4f},
          {contrastive_alignment_changes.max():.4f}]

Layers worsened: {n_negative}/{len(layers)}
Layers improved: {n_positive}/{len(layers)}

Strongest effect: Layer {worst_layer}
  Alignment change: {worst_change:.4f}

Interpretation:
"""
    if mean_effect < -0.01 and mean_effect < control_mean - 0.01:
        summary += """  CAUSAL EVIDENCE: Ablating direction
  disrupts calibration more than
  random directions."""
    elif mean_effect > 0.01 and mean_effect > control_mean + 0.01:
        summary += """  UNEXPECTED: Ablating direction
  improves calibration."""
    else:
        summary += """  NO CLEAR CAUSAL EFFECT:
  Ablation has minimal/inconsistent
  effect on calibration."""

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Contrastive Direction Ablation Results ({task_label} Task)\n{get_model_short_name(BASE_MODEL_NAME)} - {DATASET_NAME}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Ablation visualization saved to: {output_path}")

    return {
        "layers": layers,
        "contrastive_alignment_changes": contrastive_alignment_changes,
        "control_alignment_changes": control_alignment_changes,
        "mean_effect": mean_effect,
        "control_mean": control_mean,
    }


def process_task(task: str):
    """Process steering and ablation results for a single task."""
    output_prefix = get_output_prefix(task)
    task_label = "Delegate" if task == "delegate" else "Confidence"

    print(f"\n{'='*70}")
    print(f"PROCESSING: {task_label} Task")
    print(f"{'='*70}")
    print(f"Output prefix: {output_prefix}")

    ablation_summary = None

    # Try to load steering results
    steering_path = f"{output_prefix}_steering_results.json"
    if Path(steering_path).exists():
        print(f"\nLoading steering results from {steering_path}...")
        print("(This may take a moment for large files)")
        with open(steering_path) as f:
            steering_data = json.load(f)

        print("Analyzing steering results...")
        steering_analysis = analyze_steering_results(steering_data)

        steering_output = f"{output_prefix}_steering_visualization.png"
        plot_steering_results(steering_analysis, steering_output, task)
    else:
        print(f"\nSteering results not found: {steering_path}")

    # Load ablation analysis
    ablation_path = f"{output_prefix}_ablation_analysis.json"
    if Path(ablation_path).exists():
        print(f"\nLoading ablation analysis from {ablation_path}...")
        with open(ablation_path) as f:
            ablation_data = json.load(f)

        ablation_output = f"{output_prefix}_ablation_visualization.png"
        ablation_summary = plot_ablation_results(ablation_data, ablation_output, task)

        # Print summary
        print(f"\nAblation results ({task_label}):")
        print(f"  Mean alignment change (contrastive): {ablation_summary['mean_effect']:.4f}")
        print(f"  Mean alignment change (control):     {ablation_summary['control_mean']:.4f}")

        if ablation_summary['mean_effect'] < -0.01:
            print("\n  --> Ablation WORSENS calibration (expected if direction is causal)")
        elif ablation_summary['mean_effect'] > 0.01:
            print("\n  --> Ablation IMPROVES calibration (unexpected)")
        else:
            print("\n  --> Ablation has minimal effect")
    else:
        print(f"\nAblation analysis not found: {ablation_path}")

    return ablation_summary


def main():
    print(f"\n{'='*70}")
    print("CONTRASTIVE DIRECTION VISUALIZATION")
    print(f"{'='*70}")
    print(f"Model: {get_model_short_name(BASE_MODEL_NAME)}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Metric: {METRIC}")

    # Find all available tasks
    available_tasks = find_available_tasks()

    if not available_tasks:
        print(f"\nNo results found for any task. Expected files like:")
        print(f"  {get_output_prefix('confidence')}_steering_results.json")
        print(f"  {get_output_prefix('delegate')}_steering_results.json")
        return

    print(f"\nFound results for tasks: {available_tasks}")

    # Process each task
    for task in available_tasks:
        process_task(task)

    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
