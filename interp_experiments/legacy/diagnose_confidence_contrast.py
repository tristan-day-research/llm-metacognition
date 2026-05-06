"""
Diagnostic script to evaluate confidence contrast quality.

Usage:
    python diagnose_confidence_contrast.py [path/to/contrast/directory]
    
Example:
    python diagnose_confidence_contrast.py outputs/v3_8b_FT_entropy
    python diagnose_confidence_contrast.py outputs/v3_8b_base_entropy_no_quantization/confidence_contrasts

If no path is provided, uses DEFAULT_CONTRAST_DIR below.

Analyzes the confidence contrast JSON to identify potential issues:
- Overfitting (high train R², negative val R²)
- Data imbalance
- Poor generalization
- Comparison with entropy contrast strength
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# =============================================================================
# CONFIGURATION - Edit this path to your contrast directory
# =============================================================================
DEFAULT_CONTRAST_DIR = Path("outputs/8b_FT_entropy/confidence_contrast")
# Files will be auto-discovered within this directory
# =============================================================================


def find_contrast_files(search_dir: Path):
    """Auto-discover contrast JSON files in a directory."""
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return None, None
    
    # Search patterns
    confidence_patterns = ["*confidence_contrast.json", "*confidence*.json"]
    entropy_patterns = ["*entropy_contrast.json", "*entropy*.json"]
    
    confidence_file = None
    entropy_file = None
    
    # Find confidence contrast
    for pattern in confidence_patterns:
        matches = list(search_dir.glob(pattern))
        if matches:
            confidence_file = matches[0]
            break
    
    # Find entropy contrast
    for pattern in entropy_patterns:
        matches = list(search_dir.glob(pattern))
        if matches:
            entropy_file = matches[0]
            break
    
    return confidence_file, entropy_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose confidence contrast quality and identify issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    if args.directory:
        search_dir = Path(args.directory)
    else:
        # Use default directory from configuration
        search_dir = DEFAULT_CONTRAST_DIR
        print(f"No directory specified, using default: {search_dir}")
    
    # Find the files
    confidence_file, entropy_file = find_contrast_files(search_dir)
    
    if not confidence_file or not entropy_file:
        print(f"\nError: Could not find required files in {search_dir.absolute()}")
        print("\nRequired:")
        print("  - *confidence_contrast.json")
        print("  - *entropy_contrast.json")
        print("\nEither:")
        print(f"  1. Edit DEFAULT_CONTRAST_DIR at the top of this script")
        print(f"  2. Or run: python diagnose_confidence_contrast.py path/to/contrast/directory")
        sys.exit(1)
    
    print(f"\nFound files in: {search_dir.absolute()}")
    print(f"  Confidence: {confidence_file.name}")
    print(f"  Entropy: {entropy_file.name}\n")
    print("=" * 80)
    print("CONFIDENCE CONTRAST DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Load data
    with open(confidence_file) as f:
        conf_data = json.load(f)
    
    with open(entropy_file) as f:
        entropy_data = json.load(f)
    
    # =============================================================================
    # 1. DATA DISTRIBUTION ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("1. DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    token_dist = conf_data["token_distribution"]
    total_samples = conf_data["num_samples"]
    
    print(f"\nTotal samples: {total_samples}")
    print(f"\nToken distribution:")
    for token, count in sorted(token_dist.items()):
        pct = 100 * count / total_samples
        midpoint = conf_data["token_midpoints"][token]
        print(f"  {token} ({midpoint:.3f}): {count:3d} ({pct:5.1f}%)")
    
    # Check for severe imbalance
    max_token = max(token_dist.items(), key=lambda x: x[1])
    imbalance_ratio = max_token[1] / total_samples
    
    print(f"\n⚠️  IMBALANCE CHECK:")
    print(f"  Most frequent token: {max_token[0]} ({max_token[1]}/{total_samples} = {imbalance_ratio:.1%})")
    
    if imbalance_ratio > 0.8:
        print(f"  ❌ SEVERE IMBALANCE: {imbalance_ratio:.1%} of samples are '{max_token[0]}'")
        print(f"     This makes it very hard to learn a good predictor!")
        print(f"     The model can achieve high accuracy by just predicting '{max_token[0]}' always.")
    elif imbalance_ratio > 0.5:
        print(f"  ⚠️  MODERATE IMBALANCE: {imbalance_ratio:.1%} of samples")
    else:
        print(f"  ✓ Acceptable balance")
    
    # Check confidence statistics
    conf_stats = conf_data["confidence_stats"]
    print(f"\nConfidence value statistics:")
    print(f"  Mean: {conf_stats['mean']:.3f}")
    print(f"  Std:  {conf_stats['std']:.3f}")
    print(f"  Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")
    print(f"  Median: {conf_stats['median']:.3f}")
    
    if conf_stats['std'] < 0.2:
        print(f"  ⚠️  LOW VARIANCE: Std={conf_stats['std']:.3f} means limited signal to learn from")
    
    # =============================================================================
    # 2. OVERFITTING ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("2. OVERFITTING ANALYSIS")
    print("=" * 80)
    
    layers = []
    train_r2s = []
    val_r2s = []
    
    for layer_info in conf_data["per_layer"]:
        layers.append(layer_info["layer"])
        train_r2s.append(layer_info["train_r2"])
        val_r2s.append(layer_info["val_r2"])
    
    train_r2s = np.array(train_r2s)
    val_r2s = np.array(val_r2s)
    
    # Count problematic layers
    negative_val = np.sum(val_r2s < 0)
    overfitting_layers = np.sum((train_r2s > 0.95) & (val_r2s < 0.5))
    
    print(f"\nTrain R² statistics:")
    print(f"  Mean: {train_r2s.mean():.4f}")
    print(f"  Min:  {train_r2s.min():.4f}")
    print(f"  Max:  {train_r2s.max():.4f}")
    
    print(f"\nValidation R² statistics:")
    print(f"  Mean: {val_r2s.mean():.4f}")
    print(f"  Min:  {val_r2s.min():.4f}")
    print(f"  Max:  {val_r2s.max():.4f}")
    
    print(f"\n⚠️  OVERFITTING CHECK:")
    print(f"  Layers with negative val R²: {negative_val}/{len(layers)}")
    print(f"  Layers with severe overfitting (train>0.95, val<0.5): {overfitting_layers}/{len(layers)}")
    
    if negative_val > len(layers) * 0.3:
        print(f"\n  ❌ SEVERE OVERFITTING DETECTED!")
        print(f"     {negative_val} layers have negative validation R²")
        print(f"     This means the model does WORSE than predicting the mean!")
        print(f"\n  Likely causes:")
        print(f"     1. Data imbalance (88.8% are 'Z' tokens)")
        print(f"     2. Insufficient regularization (alpha={conf_data['per_layer'][0]['alpha']})")
        print(f"     3. Too few samples ({total_samples}) for high-dim space")
    
    # Find best validation layer
    best_val_layer = int(np.argmax(val_r2s))
    best_val_r2 = val_r2s[best_val_layer]
    
    print(f"\n  Best validation layer: {best_val_layer} (R²={best_val_r2:.4f})")
    
    # =============================================================================
    # 3. COMPARISON WITH ENTROPY CONTRAST
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("3. COMPARISON WITH ENTROPY CONTRAST")
    print("=" * 80)
    
    entropy_r2s = np.array([layer["r2"] for layer in entropy_data["per_layer"]])
    
    print(f"\nEntropy contrast R² statistics:")
    print(f"  Mean: {entropy_r2s.mean():.4f}")
    print(f"  Min:  {entropy_r2s.min():.4f}")
    print(f"  Max:  {entropy_r2s.max():.4f}")
    print(f"  Best layer: {int(np.argmax(entropy_r2s))} (R²={entropy_r2s.max():.4f})")
    
    print(f"\nEntropy statistics:")
    ent_stats = entropy_data["entropy_stats"]
    print(f"  Mean: {ent_stats['mean']:.3f}")
    print(f"  Std:  {ent_stats['std']:.3f}")
    print(f"  Range: [{ent_stats['min']:.3f}, {ent_stats['max']:.3f}]")
    
    if entropy_r2s.max() < 0.15:
        print(f"\n  ⚠️  WEAK ENTROPY SIGNAL: Best R²={entropy_r2s.max():.4f}")
        print(f"     The entropy contrast is also weak, which explains why")
        print(f"     the cosine similarity is near zero (orthogonal directions).")
    
    # =============================================================================
    # 4. RECOMMENDATIONS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("4. RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nTo improve the confidence contrast:")
    
    if imbalance_ratio > 0.8:
        print("\n1. ADDRESS DATA IMBALANCE:")
        print(f"   - Current: {imbalance_ratio:.1%} of samples are '{max_token[0]}'")
        print(f"   - Use balanced sampling: equal numbers of each confidence level")
        print(f"   - Or use weighted loss in ridge regression")
        print(f"   - Or collect more diverse questions (mix easy/hard)")
    
    if overfitting_layers > len(layers) * 0.3:
        print("\n2. REDUCE OVERFITTING:")
        print(f"   - Try stronger regularization (current alpha={conf_data['per_layer'][0]['alpha']})")
        print(f"   - Try alpha candidates: [1.0, 10.0, 100.0, 1000.0]")
        print(f"   - Or use fewer PCA components")
        print(f"   - Or collect more samples (current: {total_samples})")
    
    if conf_stats['std'] < 0.2:
        print("\n3. INCREASE SIGNAL VARIANCE:")
        print(f"   - Current std: {conf_stats['std']:.3f}")
        print(f"   - Use more diverse questions (avoid ceiling effect)")
        print(f"   - Make sure model is uncertain on some questions")
    
    print("\n4. ALTERNATIVE APPROACHES:")
    print("   - Use quantile-based mean-diff (like entropy) instead of regression")
    print("   - This is more robust to imbalance and outliers")
    print("   - Try: top 25% confident vs bottom 25% confident")
    
    # =============================================================================
    # 5. VISUALIZATIONS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("5. GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Train vs Val R²
    ax1 = axes[0, 0]
    ax1.plot(layers, train_r2s, 'o-', label='Train R²', color='tab:blue')
    ax1.plot(layers, val_r2s, 'o-', label='Val R²', color='tab:orange')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('R²')
    ax1.set_title('Confidence Contrast: Train vs Validation R²')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Token distribution
    ax2 = axes[0, 1]
    tokens = sorted(token_dist.keys())
    counts = [token_dist[t] for t in tokens]
    colors = ['red' if token_dist[t] > total_samples * 0.8 else 'tab:blue' for t in tokens]
    ax2.bar(tokens, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence Token')
    ax2.set_ylabel('Count')
    ax2.set_title('Token Distribution (Red = Severe Imbalance)')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (t, c) in enumerate(zip(tokens, counts)):
        ax2.text(i, c + 5, str(c), ha='center', fontsize=9)
    
    # Plot 3: Overfitting gap
    ax3 = axes[1, 0]
    gap = train_r2s - val_r2s
    colors_gap = ['red' if g > 1.0 else 'orange' if g > 0.5 else 'green' for g in gap]
    ax3.bar(layers, gap, color=colors_gap, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Train R² - Val R²')
    ax3.set_title('Overfitting Gap (Red = Severe)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Confidence vs Entropy R²
    ax4 = axes[1, 1]
    ax4.plot(layers, val_r2s, 'o-', label='Confidence (val R²)', color='tab:orange')
    ax4.plot(layers, entropy_r2s, 'o-', label='Entropy (R²)', color='tab:green')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('R²')
    ax4.set_title('Confidence vs Entropy Contrast Strength')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = search_dir / "DIAGNOSTIC_confidence_contrast.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path.absolute()}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey findings:")
    print(f"  - Data imbalance: {imbalance_ratio:.1%} are '{max_token[0]}'")
    print(f"  - Overfitting: {negative_val}/{len(layers)} layers have negative val R²")
    print(f"  - Best val layer: {best_val_layer} (R²={best_val_r2:.4f})")
    print(f"  - Entropy contrast is also weak (max R²={entropy_r2s.max():.4f})")
    print(f"\nSee recommendations above for how to improve the contrast.")


if __name__ == "__main__":
    main()
