"""
Deep analysis of contrast JSON files to extract additional insights.

Usage:
    python analyze_contrast_insights.py [path/to/contrast/directory]
    
Example:
    python analyze_contrast_insights.py outputs/v3_8b_FT_entropy
    python analyze_contrast_insights.py outputs/v3_8b_base_entropy_no_quantization/confidence_contrasts

If no path is provided, uses DEFAULT_CONTRAST_DIR below.

Explores:
1. Layer-wise progression and evolution patterns
2. Best layer comparison between entropy and confidence
3. Correlation strength patterns (not just R²)
4. Statistical significance across layers
5. Magnitude comparisons (signal strength)
6. Group separation quality
7. Cross-layer stability
8. Detailed token contribution analysis
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
from scipy.stats import spearmanr
import sys
import argparse

# =============================================================================
# CONFIGURATION - Edit this path to your contrast directory
# =============================================================================
DEFAULT_CONTRAST_DIR = Path("outputs/8b_instruct_entropy/confidence_contrast")
# Files will be auto-discovered within this directory
# =============================================================================


def find_contrast_files(search_dir: Path):
    """
    Auto-discover contrast JSON files in a directory.
    
    Returns:
        tuple: (confidence_json, entropy_json, cosine_json) or (None, None, None) if not found
    """
    search_dir = Path(search_dir)
    
    print(f"Searching for contrast files in: {search_dir}")
    print(f"  Directory exists: {search_dir.exists()}")
    
    if not search_dir.exists():
        return None, None, None
    
    # Search patterns
    confidence_patterns = ["*confidence_contrast.json", "*confidence*.json"]
    entropy_patterns = ["*entropy_contrast.json", "*entropy*.json"]
    cosine_patterns = ["*cosine_similarity*.json", "*cosine*.json"]
    
    confidence_file = None
    entropy_file = None
    cosine_file = None
    
    # Find confidence contrast
    for pattern in confidence_patterns:
        matches = list(search_dir.glob(pattern))
        if matches:
            confidence_file = matches[0]
            print(f"  Found confidence: {confidence_file.name}")
            break
    
    # Find entropy contrast
    for pattern in entropy_patterns:
        matches = list(search_dir.glob(pattern))
        if matches:
            entropy_file = matches[0]
            print(f"  Found entropy: {entropy_file.name}")
            break
    
    # Find cosine similarity
    for pattern in cosine_patterns:
        matches = list(search_dir.glob(pattern))
        if matches:
            cosine_file = matches[0]
            print(f"  Found cosine: {cosine_file.name}")
            break
    
    return confidence_file, entropy_file, cosine_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze contrast JSON files to extract deep insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_contrast_insights.py outputs/v3_8b_FT_entropy
  python analyze_contrast_insights.py outputs/v3_8b_base_entropy_no_quantization/confidence_contrasts
  python analyze_contrast_insights.py  # searches common locations
        """
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default=None,
        help='Directory containing contrast JSON files (default: auto-search)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine search directory
    search_dir = None  # Initialize
    
    if args.directory:
        search_dir = Path(args.directory)
        # Find the files in the specified directory
        confidence_file, entropy_file, cosine_file = find_contrast_files(search_dir)
    else:
        # Use default directory from configuration
        search_dir = DEFAULT_CONTRAST_DIR
        print(f"No directory specified, using default: {search_dir}")
        confidence_file, entropy_file, cosine_file = find_contrast_files(search_dir)
        
        if not confidence_file or not entropy_file or not cosine_file:
            print("\nError: Could not find contrast files in default location.")
            print(f"Default: {DEFAULT_CONTRAST_DIR.absolute()}")
            print("\nEither:")
            print(f"  1. Edit DEFAULT_CONTRAST_DIR at the top of this script")
            print(f"  2. Or run: python analyze_contrast_insights.py path/to/contrast/directory")
            sys.exit(1)
    
    if not confidence_file or not entropy_file or not cosine_file:
        print("\nError: Could not find all required files in the directory.")
        print("\nRequired files:")
        print("  - *confidence_contrast.json")
        print("  - *entropy_contrast.json")
        print("  - *cosine_similarity*.json")
        print(f"\nSearched in: {search_dir.absolute()}")
        sys.exit(1)
    print("\n" + "=" * 80)
    print("DEEP CONTRAST ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing files from: {search_dir.absolute()}")
    
    # Load data
    print("\nLoading data...")
    with open(confidence_file) as f:
        conf_data = json.load(f)
    print(f"  ✓ Confidence contrast loaded")
    
    with open(entropy_file) as f:
        entropy_data = json.load(f)
    print(f"  ✓ Entropy contrast loaded")
    
    with open(cosine_file) as f:
        cosine_data = json.load(f)
    print(f"  ✓ Cosine similarity loaded")
    
    num_layers = conf_data["num_layers"]
    
    # =============================================================================
    # 1. LAYER-WISE BEST PERFORMANCE
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("1. BEST LAYERS COMPARISON")
    print("=" * 80)
    
    # Extract R² values
    entropy_r2 = np.array([layer["r2"] for layer in entropy_data["per_layer"]])
    conf_r2 = np.array([layer["r2"] for layer in conf_data["per_layer"]])
    
    # Find best layers
    best_entropy_layer = int(np.argmax(entropy_r2))
    best_conf_layer = int(np.argmax(conf_r2))
    
    print(f"\nBest Entropy Layer: {best_entropy_layer}")
    print(f"  R²: {entropy_r2[best_entropy_layer]:.4f}")
    print(f"  Corr: {entropy_data['per_layer'][best_entropy_layer]['corr']:.4f}")
    print(f"  p-value: {entropy_data['per_layer'][best_entropy_layer]['corr_pvalue']:.2e}")
    
    print(f"\nBest Confidence Layer: {best_conf_layer}")
    print(f"  R²: {conf_r2[best_conf_layer]:.4f}")
    print(f"  Corr: {conf_data['per_layer'][best_conf_layer]['corr']:.4f}")
    print(f"  p-value: {conf_data['per_layer'][best_conf_layer]['corr_pvalue']:.2e}")
    
    # Check if they align
    layer_diff = abs(best_entropy_layer - best_conf_layer)
    print(f"\nLayer alignment: {layer_diff} layers apart")
    if layer_diff <= 2:
        print("  ✓ Very close! Both peak at similar layers.")
    elif layer_diff <= 5:
        print("  ~ Somewhat close - within a few layers.")
    else:
        print("  ✗ Far apart - different processing depths.")
    
    # =============================================================================
    # 2. LAYER-WISE PROGRESSION PATTERNS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("2. LAYER-WISE PROGRESSION PATTERNS")
    print("=" * 80)
    
    # Compute derivatives (rate of change)
    entropy_deriv = np.diff(entropy_r2)
    conf_deriv = np.diff(conf_r2)
    
    # Find key transition points
    entropy_max_growth_layer = int(np.argmax(entropy_deriv))
    conf_max_growth_layer = int(np.argmax(conf_deriv))
    
    print(f"\nEntropy contrast evolution:")
    print(f"  Early layers (0-7): R² = {entropy_r2[:8].mean():.4f}")
    print(f"  Middle layers (8-23): R² = {entropy_r2[8:24].mean():.4f}")
    print(f"  Late layers (24-31): R² = {entropy_r2[24:].mean():.4f}")
    print(f"  Fastest growth at layer: {entropy_max_growth_layer} (ΔR²={entropy_deriv[entropy_max_growth_layer]:.4f})")
    
    print(f"\nConfidence contrast evolution:")
    print(f"  Early layers (0-7): R² = {conf_r2[:8].mean():.4f}")
    print(f"  Middle layers (8-23): R² = {conf_r2[8:24].mean():.4f}")
    print(f"  Late layers (24-31): R² = {conf_r2[24:].mean():.4f}")
    print(f"  Fastest growth at layer: {conf_max_growth_layer} (ΔR²={conf_deriv[conf_max_growth_layer]:.4f})")
    
    # Check if progression patterns are similar
    prog_corr, _ = spearmanr(entropy_r2, conf_r2)
    print(f"\nProgression similarity (Spearman ρ): {prog_corr:.4f}")
    if prog_corr > 0.7:
        print("  ✓ Strong similarity - both evolve together across layers")
    elif prog_corr > 0.4:
        print("  ~ Moderate similarity - some shared progression")
    else:
        print("  ✗ Weak similarity - different evolution patterns")
    
    # =============================================================================
    # 3. CORRELATION STRENGTH ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("3. CORRELATION STRENGTH PATTERNS")
    print("=" * 80)
    
    entropy_corr = np.array([layer["corr"] for layer in entropy_data["per_layer"]])
    conf_corr = np.array([layer["corr"] for layer in conf_data["per_layer"]])
    
    print(f"\nEntropy correlations:")
    print(f"  Mean: {entropy_corr.mean():.4f}")
    print(f"  Max: {entropy_corr.max():.4f} (layer {int(np.argmax(entropy_corr))})")
    print(f"  Std: {entropy_corr.std():.4f}")
    print(f"  Layers with r > 0.5: {np.sum(entropy_corr > 0.5)}/{num_layers}")
    
    print(f"\nConfidence correlations:")
    print(f"  Mean: {conf_corr.mean():.4f}")
    print(f"  Max: {conf_corr.max():.4f} (layer {int(np.argmax(conf_corr))})")
    print(f"  Std: {conf_corr.std():.4f}")
    print(f"  Layers with r > 0.5: {np.sum(conf_corr > 0.5)}/{num_layers}")
    
    # =============================================================================
    # 4. STATISTICAL SIGNIFICANCE
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("4. STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    entropy_pvals = np.array([layer["corr_pvalue"] for layer in entropy_data["per_layer"]])
    conf_pvals = np.array([layer["corr_pvalue"] for layer in conf_data["per_layer"]])
    
    # Count significant layers (p < 0.001 is very strong, p < 0.05 is standard)
    entropy_sig_001 = np.sum(entropy_pvals < 0.001)
    entropy_sig_005 = np.sum(entropy_pvals < 0.05)
    conf_sig_001 = np.sum(conf_pvals < 0.001)
    conf_sig_005 = np.sum(conf_pvals < 0.05)
    
    print(f"\nEntropy significance:")
    print(f"  p < 0.001: {entropy_sig_001}/{num_layers} layers ({100*entropy_sig_001/num_layers:.1f}%)")
    print(f"  p < 0.05:  {entropy_sig_005}/{num_layers} layers ({100*entropy_sig_005/num_layers:.1f}%)")
    
    print(f"\nConfidence significance:")
    print(f"  p < 0.001: {conf_sig_001}/{num_layers} layers ({100*conf_sig_001/num_layers:.1f}%)")
    print(f"  p < 0.05:  {conf_sig_005}/{num_layers} layers ({100*conf_sig_005/num_layers:.1f}%)")
    
    # =============================================================================
    # 5. DIRECTION MAGNITUDE ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("5. DIRECTION MAGNITUDE (SIGNAL STRENGTH)")
    print("=" * 80)
    
    entropy_mag = np.array([layer["magnitude"] for layer in entropy_data["per_layer"]])
    conf_mag = np.array([layer["magnitude"] for layer in conf_data["per_layer"]])
    
    print(f"\nEntropy direction magnitudes:")
    print(f"  Mean: {entropy_mag.mean():.4f}")
    print(f"  Max: {entropy_mag.max():.4f} (layer {int(np.argmax(entropy_mag))})")
    print(f"  Min: {entropy_mag.min():.4f} (layer {int(np.argmin(entropy_mag))})")
    print(f"  Std: {entropy_mag.std():.4f}")
    
    print(f"\nConfidence direction magnitudes:")
    print(f"  Mean: {conf_mag.mean():.4f}")
    print(f"  Max: {conf_mag.max():.4f} (layer {int(np.argmax(conf_mag))})")
    print(f"  Min: {conf_mag.min():.4f} (layer {int(np.argmin(conf_mag))})")
    print(f"  Std: {conf_mag.std():.4f}")
    
    print(f"\nMagnitude ratio (entropy/confidence): {entropy_mag.mean()/conf_mag.mean():.2f}x")
    
    # =============================================================================
    # 6. GROUP SEPARATION QUALITY
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("6. GROUP SEPARATION QUALITY")
    print("=" * 80)
    
    # Entropy group separation
    entropy_low = entropy_data["per_layer"][0]["entropy_mean_low"]
    entropy_high = entropy_data["per_layer"][0]["entropy_mean_high"]
    entropy_gap = entropy_high - entropy_low
    entropy_total_range = entropy_data["entropy_stats"]["max"] - entropy_data["entropy_stats"]["min"]
    
    print(f"\nEntropy groups:")
    print(f"  Low group mean: {entropy_low:.4f}")
    print(f"  High group mean: {entropy_high:.4f}")
    print(f"  Gap: {entropy_gap:.4f}")
    print(f"  Gap as % of total range: {100*entropy_gap/entropy_total_range:.1f}%")
    
    # Confidence group separation
    conf_low = conf_data["per_layer"][0]["confidence_mean_low"]
    conf_high = conf_data["per_layer"][0]["confidence_mean_high"]
    conf_gap = conf_high - conf_low
    conf_total_range = conf_data["confidence_stats"]["max"] - conf_data["confidence_stats"]["min"]
    
    print(f"\nConfidence groups:")
    print(f"  Low group mean: {conf_low:.4f}")
    print(f"  High group mean: {conf_high:.4f}")
    print(f"  Gap: {conf_gap:.4f}")
    print(f"  Gap as % of total range: {100*conf_gap/conf_total_range:.1f}%")
    
    # =============================================================================
    # 7. COSINE SIMILARITY PATTERNS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("7. COSINE SIMILARITY PATTERNS")
    print("=" * 80)
    
    cosines = np.array(cosine_data["cosine"])
    
    # Analyze by layer region
    early_cos = cosines[:8].mean()
    middle_cos = cosines[8:24].mean()
    late_cos = cosines[24:].mean()
    
    print(f"\nCosine similarity by region:")
    print(f"  Early layers (0-7): {early_cos:.4f}")
    print(f"  Middle layers (8-23): {middle_cos:.4f}")
    print(f"  Late layers (24-31): {late_cos:.4f}")
    
    # Find layers with strongest alignment
    abs_cosines = np.abs(cosines)
    strong_align = np.where(abs_cosines > 0.1)[0]
    
    if len(strong_align) > 0:
        print(f"\nLayers with |cosine| > 0.1: {list(strong_align)}")
        for layer in strong_align:
            print(f"  Layer {layer}: {cosines[layer]:.4f}")
    else:
        print(f"\nNo layers with |cosine| > 0.1 (all are nearly orthogonal)")
    
    # Check if alignment changes across layers
    cos_trend_corr = np.corrcoef(np.arange(num_layers), cosines)[0, 1]
    print(f"\nCosine trend across layers: {cos_trend_corr:.4f}")
    if abs(cos_trend_corr) > 0.3:
        if cos_trend_corr > 0:
            print("  → Directions become MORE aligned in later layers")
        else:
            print("  → Directions become LESS aligned in later layers")
    else:
        print("  → No clear trend - relatively stable across layers")
    
    # =============================================================================
    # 8. TOKEN DISTRIBUTION INSIGHTS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("8. TOKEN DISTRIBUTION INSIGHTS")
    print("=" * 80)
    
    conf_tokens = conf_data["token_distribution"]
    total = conf_data["num_samples"]
    
    print(f"\nConfidence token diversity:")
    non_zero = sum(1 for count in conf_tokens.values() if count > 0)
    print(f"  Tokens used: {non_zero}/8")
    print(f"  Effective diversity: {non_zero} levels")
    
    # Compute effective number of classes (entropy-based)
    probs = np.array([count/total for count in conf_tokens.values() if count > 0])
    token_entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(non_zero)
    diversity_ratio = token_entropy / max_entropy if max_entropy > 0 else 0
    
    print(f"  Token entropy: {token_entropy:.3f} bits")
    print(f"  Max possible: {max_entropy:.3f} bits")
    print(f"  Diversity ratio: {diversity_ratio:.3f} (1.0 = perfectly balanced)")
    
    if diversity_ratio < 0.3:
        print(f"  ⚠️  Very low diversity - severely imbalanced!")
    elif diversity_ratio < 0.6:
        print(f"  ⚠️  Low diversity - moderately imbalanced")
    else:
        print(f"  ✓ Good diversity")
    
    # =============================================================================
    # 9. VISUALIZATION
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("9. GENERATING INSIGHT PLOTS")
    print("=" * 80)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    layers = np.arange(num_layers)
    
    # Plot 1: R² comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(layers, entropy_r2, 'o-', label='Entropy (MC Questions)', color='tab:green', linewidth=2)
    ax1.plot(layers, conf_r2, 'o-', label='Stated Confidence', color='tab:purple', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('R²')
    ax1.set_title('R² Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(layers, entropy_corr, 'o-', label='Entropy (MC Questions)', color='tab:green', linewidth=2)
    ax2.plot(layers, conf_corr, 'o-', label='Stated Confidence', color='tab:purple', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Pearson r')
    ax2.set_title('Correlation Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cosine similarity
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(layers, cosines, 'o-', color='tab:blue', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.1, color='gray', linestyle=':', alpha=0.3)
    ax3.axhline(y=-0.1, color='gray', linestyle=':', alpha=0.3)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Direction Alignment')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Magnitude comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(layers, entropy_mag, 'o-', label='Entropy (MC Questions)', color='tab:green', linewidth=2)
    ax4.plot(layers, conf_mag, 'o-', label='Stated Confidence', color='tab:purple', linewidth=2)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Direction Magnitude')
    ax4.set_title('Signal Strength')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Statistical significance (-log10 p-value)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(layers, -np.log10(entropy_pvals + 1e-300), 'o-', label='Entropy (MC Questions)', color='tab:green', linewidth=2)
    ax5.plot(layers, -np.log10(conf_pvals + 1e-300), 'o-', label='Stated Confidence', color='tab:purple', linewidth=2)
    ax5.axhline(y=-np.log10(0.05), color='orange', linestyle='--', alpha=0.5, label='p=0.05')
    ax5.axhline(y=-np.log10(0.001), color='red', linestyle='--', alpha=0.5, label='p=0.001')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('-log10(p-value)')
    ax5.set_title('Statistical Significance')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: R² growth rate
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(layers[:-1], entropy_deriv, 'o-', label='Entropy (MC Questions)', color='tab:green', linewidth=2)
    ax6.plot(layers[:-1], conf_deriv, 'o-', label='Stated Confidence', color='tab:purple', linewidth=2)
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('ΔR² (layer to layer)')
    ax6.set_title('Signal Growth Rate')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: R² vs R² scatter (layer similarity)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(entropy_r2, conf_r2, c=layers, cmap='viridis', s=50, alpha=0.7)
    ax7.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Perfect correlation')
    ax7.set_xlabel('Entropy (MC Questions) R²')
    ax7.set_ylabel('Stated Confidence R²')
    ax7.set_title(f'Layer Similarity (ρ={prog_corr:.3f})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax7.collections[0], ax=ax7)
    cbar.set_label('Layer')
    
    # Plot 8: Correlation vs Cosine (relationship)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(entropy_corr, cosines, label='Entropy (MC) corr', alpha=0.6, s=30)
    ax8.scatter(conf_corr, cosines, label='Stated Conf corr', alpha=0.6, s=30)
    ax8.set_xlabel('Correlation strength (r)')
    ax8.set_ylabel('Cosine similarity')
    ax8.set_title('Signal Strength vs Alignment')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax8.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Plot 9: Summary stats
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""SUMMARY STATISTICS
    
Entropy (MC Questions):
  Best R²: {entropy_r2.max():.4f} (L{best_entropy_layer})
  Mean corr: {entropy_corr.mean():.4f}
  Sig layers: {entropy_sig_001}/{num_layers}
  
Stated Confidence:
  Best R²: {conf_r2.max():.4f} (L{best_conf_layer})
  Mean corr: {conf_corr.mean():.4f}
  Sig layers: {conf_sig_001}/{num_layers}
  
Alignment:
  Mean cosine: {cosines.mean():.4f}
  Max |cosine|: {abs_cosines.max():.4f}
  Progression ρ: {prog_corr:.4f}
  
Data Quality:
  Token diversity: {diversity_ratio:.3f}
  Entropy gap: {entropy_gap:.3f}
  Conf gap: {conf_gap:.3f}
"""
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Deep Contrast Analysis: Entropy (MC Questions) vs Stated Confidence', 
                 fontsize=14, fontweight='bold')
    
    output_path = search_dir / "ANALYSIS_deep_insights.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path.absolute()}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
