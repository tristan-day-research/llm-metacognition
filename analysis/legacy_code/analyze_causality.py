"""
Analyze steering causality results and plot individual multiplier effects.

Reads JSON output from run_steering_causality.py and creates enhanced visualizations
showing the effect of each individual multiplier (not just the regression slope).
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
from typing import Dict, List, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing the steering results JSON files
# RESULTS_DIR = "outputs/v3_8b_FT_entropy_no_quantization"
RESULTS_DIR = "outputs/v3_8b_base_entropy_no_quantization"

# Which methods to plot (None = all available)
METHODS = None  # e.g., ["probe", "mean_diff"] or None for all

# Which position to plot (uses "final" if available, otherwise first position)
POSITION = None  # e.g., "final", "question_mark", etc. None = auto

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_steering_results(results_path: Path) -> Optional[Dict]:
    """Load steering results JSON."""
    if not results_path.exists():
        return None
    
    with open(results_path, "r") as f:
        return json.load(f)


def get_model_description(config: Dict) -> str:
    """Generate a description of the model (base vs finetuned)."""
    model = config.get("model", "Unknown")
    adapter = config.get("adapter", None)
    
    # Extract model short name
    if "/" in model:
        model_name = model.split("/")[-1]
    else:
        model_name = model
    
    if adapter and adapter.strip():
        # Finetuned model
        if "/" in adapter:
            adapter_name = adapter.split("/")[-1]
        else:
            adapter_name = adapter
        return f"{model_name} (finetuned: {adapter_name})"
    else:
        # Base model
        return f"{model_name} (base instruct)"


def plot_multiplier_effects(
    results: Dict,
    method: str,
    meta_task: str,
    output_path: Path,
    position: str = "final"
):
    """
    Create enhanced steering visualization with individual multiplier effects.
    
    Adds a new panel showing raw multiplier effects (no regression).
    """
    # Get config info
    config = results.get("config", {})
    model_desc = get_model_description(config)
    metric = config.get("metric", "unknown")
    
    # Get position-specific data if available
    position_key = position if position in results else method
    if position in results and method in results[position]:
        analysis = results[position][method]
    elif method in results:
        analysis = results[method]
    else:
        print(f"  Warning: {method} not found in results")
        return
    
    # Extract layers from per_layer keys (they're stored as strings)
    per_layer = analysis.get("per_layer", {})
    if not per_layer:
        print(f"  Warning: No per_layer data found for {method}")
        return
    
    layers = sorted([int(k) for k in per_layer.keys()])
    if not layers:
        print(f"  Warning: No layers found for {method}")
        return
    
    multipliers = results.get("config", {}).get("multipliers", [-3, -2, -1, 0, 1, 2, 3])
    
    # Extract data for new panel: mean confidence at each multiplier for each layer
    multiplier_data = {mult: [] for mult in multipliers}
    baseline_data = []
    
    for layer in layers:
        layer_data = per_layer.get(str(layer), {})
        intro_conf = layer_data.get("intro_mean_conf_by_mult", {})
        baseline_conf = layer_data.get("baseline_confidence_mean", 0.0)
        baseline_data.append(baseline_conf)
        
        for mult in multipliers:
            conf = intro_conf.get(str(float(mult)), None)
            multiplier_data[mult].append(conf if conf is not None else np.nan)
    
    # Create figure with 4 panels instead of 3
    fig, axes = plt.subplots(4, 1, figsize=(20, 18))
    fig.suptitle(f"Steering Results: {method.upper()} @ {position}\n{model_desc} - {metric}", 
                 fontsize=14, fontweight='bold')
    
    x = np.arange(len(layers))
    
    # =========================================================================
    # NEW PANEL 0: Individual Multiplier Effects (Raw Confidence)
    # =========================================================================
    ax0 = axes[0]
    
    # Use a colormap
    cmap = plt.cm.coolwarm
    norm_mults = np.array(multipliers)
    if len(multipliers) > 1:
        mult_min, mult_max = min(multipliers), max(multipliers)
        colors = [cmap((m - mult_min) / (mult_max - mult_min)) for m in multipliers]
    else:
        colors = ['blue'] * len(multipliers)
    
    # Plot each multiplier
    for mult, color in zip(multipliers, colors):
        confs = multiplier_data[mult]
        label = f'mult={mult:+.1f}' if mult != 0 else 'baseline (0.0)'
        linestyle = '--' if mult == 0 else '-'
        linewidth = 2.5 if mult == 0 else 1.5
        alpha = 0.9 if mult == 0 else 0.7
        ax0.plot(x, confs, linestyle, label=label, color=color if mult != 0 else 'black',
                linewidth=linewidth, alpha=alpha, marker='o', markersize=2)
    
    ax0.set_xticks(x)
    ax0.set_xticklabels(layers)
    ax0.set_xlabel("Layer")
    ax0.set_ylabel("Mean Confidence")
    ax0.set_title("Raw Multiplier Effects (no regression)")
    ax0.legend(loc='best', fontsize=9, ncol=2)
    ax0.grid(True, alpha=0.3)
    
    # =========================================================================
    # NEW PANEL 0b: Delta from Baseline (Alternative view)
    # =========================================================================
    ax0b = axes[1]
    
    # Plot delta from baseline for each multiplier
    for mult, color in zip(multipliers, colors):
        if mult == 0:
            continue  # Skip baseline (delta = 0 by definition)
        confs = np.array(multiplier_data[mult])
        baseline = np.array(multiplier_data[0.0])
        delta = confs - baseline
        ax0b.plot(x, delta, '-', label=f'mult={mult:+.1f}', color=color,
                 linewidth=1.5, alpha=0.7, marker='o', markersize=2)
    
    ax0b.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='baseline')
    ax0b.set_xticks(x)
    ax0b.set_xticklabels(layers)
    ax0b.set_xlabel("Layer")
    ax0b.set_ylabel("Δ Confidence (from baseline)")
    ax0b.set_title("Multiplier Effects: Change from Baseline")
    ax0b.legend(loc='best', fontsize=9, ncol=2)
    ax0b.grid(True, alpha=0.3)
    
    # =========================================================================
    # PANEL 1: Confidence Slope by Layer (original panel 1)
    # =========================================================================
    ax1 = axes[2]
    
    expected_sign = analysis.get("expected_slope_sign", -1)
    sign_str = "negative" if expected_sign < 0 else "positive"
    
    intro_slopes = np.array([per_layer[str(l)]["introspection_slope"] for l in layers])
    ctrl_slopes = np.array([per_layer[str(l)]["control_slope_mean"] for l in layers])
    ctrl_stds = np.array([per_layer[str(l)]["control_slope_std"] for l in layers])
    p_values_pooled = [per_layer[str(l)]["p_value_pooled"] for l in layers]
    sign_correct = [per_layer[str(l)]["sign_matches_expected"] for l in layers]
    
    # Plot control band
    ax1.fill_between(x, ctrl_slopes - ctrl_stds, ctrl_slopes + ctrl_stds,
                     color='gray', alpha=0.2, label='Control ±1σ')
    ax1.plot(x, ctrl_slopes, '--', color='gray', linewidth=1, alpha=0.8, label='Control mean')
    
    # Plot introspection line
    ax1.plot(x, intro_slopes, '-', color='blue', linewidth=1.5, alpha=0.8, label=f'{method}')
    
    # Mark significant layers
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
    ax1.set_title(f"Linear Regression Slope by Layer (expected: {sign_str})")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # PANEL 2: Summary text
    # =========================================================================
    ax2 = axes[3]
    ax2.axis('off')
    
    summary = analysis.get("summary", {})
    best_layer = summary.get("best_layer", layers[0])
    best_stats = per_layer.get(str(best_layer), {})
    
    summary_text = f"""
STEERING ANALYSIS: {method.upper()} @ {position}

Model: {model_desc}
Metric: {metric}
Expected slope sign: {sign_str} ({"−" if expected_sign < 0 else "+"}direction → {"lower" if expected_sign < 0 else "higher"} confidence)
Layers tested: {len(layers)}
Questions: {analysis.get('num_questions', 'N/A')}
Controls per layer: {analysis.get('num_controls', 'N/A')}
Multipliers tested: {multipliers}

Results:
  Significant layers (p<0.05 pooled): {summary.get('n_significant_pooled', 0)}
  Significant layers (FDR<0.05): {summary.get('n_significant_fdr', 0)}
  Sign correct (pooled + expected sign): {summary.get('n_sign_correct_pooled', 0)}
  Sign correct (FDR + expected sign): {summary.get('n_sign_correct_fdr', 0)}

Best layer (by |Z|): {best_layer}
  Slope: {summary.get('best_slope', 0):.4f}
  Effect size (Z): {summary.get('best_effect_z', 0):.2f}
  p-value (pooled): {best_stats.get('p_value_pooled', 1.0):.4f}
  p-value (FDR): {best_stats.get('p_value_fdr', 1.0):.4f}
  Sign correct: {"Yes" if best_stats.get('sign_matches_expected', False) else "No"}

Note: Panel 0 shows raw mean confidence at each multiplier (no averaging/regression)
      Panel 1 shows delta from baseline (multiplier=0)
      Panel 2 shows the linear regression slope (single summary statistic)
"""
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()


def main():
    results_dir = Path(RESULTS_DIR)
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return
    
    print(f"Analyzing steering results in: {results_dir}")
    print("=" * 70)
    
    # Find steering results files
    meta_tasks = ["confidence", "delegate"]
    
    for meta_task in meta_tasks:
        # Look for results file
        pattern = f"*_steering_{meta_task}_*_results.json"
        json_files = list(results_dir.glob(pattern))
        
        if not json_files:
            print(f"\nNo {meta_task} results found (pattern: {pattern})")
            continue
        
        for json_file in json_files:
            print(f"\nProcessing: {json_file.name}")
            
            results = load_steering_results(json_file)
            if results is None:
                print(f"  Error loading file")
                continue
            
            # Get available methods
            config = results.get("config", {})
            available_methods = config.get("methods_tested", [])
            
            # If methods not in config, try to infer from keys
            if not available_methods:
                available_methods = [k for k in results.keys() 
                                   if k not in ["config", "comparison", "final", "question_mark", 
                                               "question_newline", "options_newline"]]
            
            if METHODS is not None:
                methods_to_plot = [m for m in METHODS if m in available_methods]
            else:
                methods_to_plot = available_methods
            
            if not methods_to_plot:
                print(f"  No methods to plot (available: {available_methods})")
                continue
            
            # Determine position to use
            if POSITION is not None:
                position = POSITION
            elif "final" in results:
                position = "final"
            else:
                # Use first available position
                positions = config.get("positions_tested", ["final"])
                position = positions[0] if positions else "final"
            
            print(f"  Methods: {methods_to_plot}")
            print(f"  Position: {position}")
            
            # Create plots for each method
            for method in methods_to_plot:
                output_name = f"ANALYSIS_{json_file.stem}_{method}_{position}.png"
                output_path = results_dir / output_name
                
                print(f"  Creating plot for {method}...")
                plot_multiplier_effects(results, method, meta_task, output_path, position)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
