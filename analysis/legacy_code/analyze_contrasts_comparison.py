"""
Compare contrast directions between base and finetuned models.

Usage:
    python analyze_contrasts_comparison.py

Edit the directories below to point to your base and finetuned contrast directories.

Note: Signal strength (magnitude) measures between-cluster distance, NOT within-cluster tightness.
Cluster tightness measures how compact each cluster is (mean distance to centroid).
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
import sys

# =============================================================================
# CONFIGURATION - Edit these paths
# =============================================================================
BASE_CONTRAST_DIR = Path("outputs/8b_instruct_entropy/confidence_contrast")
FT_CONTRAST_DIR = Path("outputs/8b_FT_entropy/confidence_contrast")
# =============================================================================


def find_contrast_files(search_dir: Path):
    """
    Auto-discover contrast JSON files in a directory.
    
    Returns:
        tuple: (confidence_json, entropy_json, cosine_json) or (None, None, None) if not found
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        print(f"Directory not found: {search_dir}")
        return None, None, None
    
    # Search patterns
    confidence_patterns = ["*confidence_contrast.json", "*confidence*.json"]
    entropy_patterns = ["*entropy_contrast.json", "*entropy*.json"]
    cosine_patterns = ["*cosine_similarity*.json", "*cosine*.json"]
    
    confidence_file = None
    entropy_file = None
    cosine_file = None
    
    # Find confidence contrast (exclude cosine files)
    for pattern in confidence_patterns:
        matches = [f for f in search_dir.glob(pattern) if "cosine" not in f.name.lower()]
        if matches:
            confidence_file = matches[0]
            break
    
    # Find entropy contrast (exclude cosine files)
    for pattern in entropy_patterns:
        matches = [f for f in search_dir.glob(pattern) if "cosine" not in f.name.lower() and "confidence" not in f.name.lower()]
        if matches:
            entropy_file = matches[0]
            break
    
    # Find cosine similarity
    for pattern in cosine_patterns:
        matches = list(search_dir.glob(pattern))
        if matches:
            cosine_file = matches[0]
            break
    
    return confidence_file, entropy_file, cosine_file


def load_contrast_data(contrast_dir: Path):
    """Load all contrast files from a directory including cosine similarity."""
    contrast_dir = Path(contrast_dir)
    conf_file, entropy_file, cosine_file = find_contrast_files(contrast_dir)
    
    if not conf_file or not entropy_file:
        raise FileNotFoundError(f"Could not find contrast files in {contrast_dir}")
    
    with open(conf_file) as f:
        conf_data = json.load(f)
    
    with open(entropy_file) as f:
        entropy_data = json.load(f)
    
    # Load cosine similarity JSON (entropy vs confidence alignment within this model)
    cosine_data = None
    if cosine_file and cosine_file.exists():
        with open(cosine_file) as f:
            cosine_data = json.load(f)
    
    return conf_data, entropy_data, cosine_data


def create_comparison_plots(base_conf, base_entropy, ft_conf, ft_entropy, 
                            base_cosine, ft_cosine, output_path):
    """
    Create 4 comparison plots:
    1. R² comparison
    2. Statistical significance comparison
    3. Direction alignment (base vs FT)
    4. Cluster tightness (within-cluster distances)
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Extract data for base model
    base_conf_layers = [layer['layer'] for layer in base_conf['per_layer']]
    base_conf_r2 = [layer['r2'] for layer in base_conf['per_layer']]
    base_conf_pval = [layer.get('p_value', layer.get('corr_pvalue', 1.0)) for layer in base_conf['per_layer']]
    base_conf_tight_low = [layer.get('cluster_tightness_low', 0) for layer in base_conf['per_layer']]
    base_conf_tight_high = [layer.get('cluster_tightness_high', 0) for layer in base_conf['per_layer']]
    
    base_entropy_layers = [layer['layer'] for layer in base_entropy['per_layer']]
    base_entropy_r2 = [layer['r2'] for layer in base_entropy['per_layer']]
    base_entropy_pval = [layer.get('p_value', layer.get('corr_pvalue', 1.0)) for layer in base_entropy['per_layer']]
    base_entropy_tight_low = [layer.get('cluster_tightness_low', 0) for layer in base_entropy['per_layer']]
    base_entropy_tight_high = [layer.get('cluster_tightness_high', 0) for layer in base_entropy['per_layer']]
    
    # Extract data for FT model
    ft_conf_layers = [layer['layer'] for layer in ft_conf['per_layer']]
    ft_conf_r2 = [layer['r2'] for layer in ft_conf['per_layer']]
    ft_conf_pval = [layer.get('p_value', layer.get('corr_pvalue', 1.0)) for layer in ft_conf['per_layer']]
    ft_conf_tight_low = [layer.get('cluster_tightness_low', 0) for layer in ft_conf['per_layer']]
    ft_conf_tight_high = [layer.get('cluster_tightness_high', 0) for layer in ft_conf['per_layer']]
    
    ft_entropy_layers = [layer['layer'] for layer in ft_entropy['per_layer']]
    ft_entropy_r2 = [layer['r2'] for layer in ft_entropy['per_layer']]
    ft_entropy_pval = [layer.get('p_value', layer.get('corr_pvalue', 1.0)) for layer in ft_entropy['per_layer']]
    ft_entropy_tight_low = [layer.get('cluster_tightness_low', 0) for layer in ft_entropy['per_layer']]
    ft_entropy_tight_high = [layer.get('cluster_tightness_high', 0) for layer in ft_entropy['per_layer']]
    
    # Colors: entropy=red, confidence=blue, base=lighter, FT=darker
    # Direction alignment=orange
    base_entropy_color = '#ff9999'  # light red
    ft_entropy_color = '#cc0000'    # dark red
    base_conf_color = '#99ccff'     # light blue
    ft_conf_color = '#0066cc'       # dark blue
    base_align_color = '#ffcc99'    # light orange
    ft_align_color = '#cc6600'      # dark orange
    
    # Plot 1: R² Comparison
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(base_entropy_layers, base_entropy_r2, '-', label='Base: Entropy (MC)', 
             color=base_entropy_color, linewidth=2)
    ax1.plot(base_conf_layers, base_conf_r2, '-', label='Base: Stated Conf', 
             color=base_conf_color, linewidth=2)
    ax1.plot(ft_entropy_layers, ft_entropy_r2, '-', label='FT: Entropy (MC)', 
             color=ft_entropy_color, linewidth=2)
    ax1.plot(ft_conf_layers, ft_conf_r2, '-', label='FT: Stated Conf', 
             color=ft_conf_color, linewidth=2)
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('R²', fontsize=11)
    ax1.set_title('R² Comparison: Base vs Finetuned', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Plot 2: Statistical Significance
    ax2 = fig.add_subplot(2, 2, 2)
    base_entropy_log_pval = -np.log10(np.array(base_entropy_pval) + 1e-300)
    base_conf_log_pval = -np.log10(np.array(base_conf_pval) + 1e-300)
    ft_entropy_log_pval = -np.log10(np.array(ft_entropy_pval) + 1e-300)
    ft_conf_log_pval = -np.log10(np.array(ft_conf_pval) + 1e-300)
    
    ax2.plot(base_entropy_layers, base_entropy_log_pval, '-', label='Base: Entropy (MC)', 
             color=base_entropy_color, linewidth=2)
    ax2.plot(base_conf_layers, base_conf_log_pval, '-', label='Base: Stated Conf', 
             color=base_conf_color, linewidth=2)
    ax2.plot(ft_entropy_layers, ft_entropy_log_pval, '-', label='FT: Entropy (MC)', 
             color=ft_entropy_color, linewidth=2)
    ax2.plot(ft_conf_layers, ft_conf_log_pval, '-', label='FT: Stated Conf', 
             color=ft_conf_color, linewidth=2)
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('-log10(p-value)', fontsize=11)
    ax2.set_title('Statistical Significance', fontsize=12, fontweight='bold')
    ax2.axhline(y=-np.log10(0.001), color='gray', linestyle='--', alpha=0.5, label='p=0.001')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Direction Alignment (Entropy vs Confidence within each model)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Load cosine similarity from cosine JSONs (entropy vs confidence alignment)
    if base_cosine and 'cosine' in base_cosine:
        base_cosines = base_cosine['cosine']
        base_cos_layers = list(range(len(base_cosines)))
        ax3.plot(base_cos_layers, base_cosines, '-', label='Base Model', 
                 color=base_align_color, linewidth=2)
    
    if ft_cosine and 'cosine' in ft_cosine:
        ft_cosines = ft_cosine['cosine']
        ft_cos_layers = list(range(len(ft_cosines)))
        ax3.plot(ft_cos_layers, ft_cosines, '-', label='Finetuned Model', 
                 color=ft_align_color, linewidth=2)
    
    ax3.set_xlabel('Layer', fontsize=11)
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_title('Direction Alignment: Entropy vs Stated Confidence', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    if (base_cosine and 'cosine' in base_cosine) or (ft_cosine and 'cosine' in ft_cosine):
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Cosine similarity data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.05, 1.05)
    
    # Plot 4: Cluster Tightness (within-cluster distances)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Check if data exists (non-zero)
    has_tightness = (max(base_entropy_tight_high) > 0 or max(ft_entropy_tight_high) > 0)
    
    if has_tightness:
        ax4.plot(base_entropy_layers, base_entropy_tight_high, '-', 
                 label='Base: Entropy High', color=base_entropy_color, linewidth=2)
        ax4.plot(base_entropy_layers, base_entropy_tight_low, '--', 
                 label='Base: Entropy Low', color=base_entropy_color, linewidth=2, alpha=0.6)
        ax4.plot(base_conf_layers, base_conf_tight_high, '-', 
                 label='Base: Conf High', color=base_conf_color, linewidth=2)
        ax4.plot(base_conf_layers, base_conf_tight_low, '--', 
                 label='Base: Conf Low', color=base_conf_color, linewidth=2, alpha=0.6)
        
        ax4.plot(ft_entropy_layers, ft_entropy_tight_high, '-', 
                 label='FT: Entropy High', color=ft_entropy_color, linewidth=2)
        ax4.plot(ft_entropy_layers, ft_entropy_tight_low, '--', 
                 label='FT: Entropy Low', color=ft_entropy_color, linewidth=2, alpha=0.6)
        ax4.plot(ft_conf_layers, ft_conf_tight_high, '-', 
                 label='FT: Conf High', color=ft_conf_color, linewidth=2)
        ax4.plot(ft_conf_layers, ft_conf_tight_low, '--', 
                 label='FT: Conf Low', color=ft_conf_color, linewidth=2, alpha=0.6)
        ax4.legend(fontsize=7, ncol=2)
    else:
        ax4.text(0.5, 0.5, 'Cluster tightness data not available.\nRerun confidence_contrast.py to generate.', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
    
    ax4.set_xlabel('Layer', fontsize=11)
    ax4.set_ylabel('Mean Distance to Centroid', fontsize=11)
    ax4.set_title('Cluster Tightness: Within-Cluster Compactness', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nR² Performance:")
    print(f"  Base Entropy (MC):     {np.mean(base_entropy_r2):.4f} ± {np.std(base_entropy_r2):.4f}")
    print(f"  Base Stated Conf:      {np.mean(base_conf_r2):.4f} ± {np.std(base_conf_r2):.4f}")
    print(f"  FT Entropy (MC):       {np.mean(ft_entropy_r2):.4f} ± {np.std(ft_entropy_r2):.4f}")
    print(f"  FT Stated Conf:        {np.mean(ft_conf_r2):.4f} ± {np.std(ft_conf_r2):.4f}")
    
    print("\nDirection Alignment (Entropy vs Confidence within model):")
    if base_cosine and 'cosine' in base_cosine:
        print(f"  Base Model:            {np.mean(base_cosine['cosine']):.4f}")
    else:
        print(f"  Base Model:            N/A (cosine data not available)")
    if ft_cosine and 'cosine' in ft_cosine:
        print(f"  Finetuned Model:       {np.mean(ft_cosine['cosine']):.4f}")
    else:
        print(f"  Finetuned Model:       N/A (cosine data not available)")
    
    print("\nSignificant Layers (p < 0.001):")
    print(f"  Base Entropy:          {sum(p < 0.001 for p in base_entropy_pval)}/{len(base_entropy_pval)}")
    print(f"  Base Stated Conf:      {sum(p < 0.001 for p in base_conf_pval)}/{len(base_conf_pval)}")
    print(f"  FT Entropy:            {sum(p < 0.001 for p in ft_entropy_pval)}/{len(ft_entropy_pval)}")
    print(f"  FT Stated Conf:        {sum(p < 0.001 for p in ft_conf_pval)}/{len(ft_conf_pval)}")
    
    print("\nCluster Tightness (mean distance to centroid):")
    print(f"  Base Entropy High:     {np.mean(base_entropy_tight_high):.2f} ± {np.std(base_entropy_tight_high):.2f}")
    print(f"  Base Entropy Low:      {np.mean(base_entropy_tight_low):.2f} ± {np.std(base_entropy_tight_low):.2f}")
    print(f"  Base Conf High:        {np.mean(base_conf_tight_high):.2f} ± {np.std(base_conf_tight_high):.2f}")
    print(f"  Base Conf Low:         {np.mean(base_conf_tight_low):.2f} ± {np.std(base_conf_tight_low):.2f}")
    print(f"  FT Entropy High:       {np.mean(ft_entropy_tight_high):.2f} ± {np.std(ft_entropy_tight_high):.2f}")
    print(f"  FT Entropy Low:        {np.mean(ft_entropy_tight_low):.2f} ± {np.std(ft_entropy_tight_low):.2f}")
    print(f"  FT Conf High:          {np.mean(ft_conf_tight_high):.2f} ± {np.std(ft_conf_tight_high):.2f}")
    print(f"  FT Conf Low:           {np.mean(ft_conf_tight_low):.2f} ± {np.std(ft_conf_tight_low):.2f}")
    print("="*80)


def main():
    print("="*80)
    print("BASE vs FINETUNED CONTRAST COMPARISON")
    print("="*80)
    
    print(f"\nBase directory:      {BASE_CONTRAST_DIR}")
    print(f"Finetuned directory: {FT_CONTRAST_DIR}")
    
    # Load base model data
    print("\n[1/2] Loading base model contrasts...")
    try:
        base_conf, base_entropy, base_cosine = load_contrast_data(BASE_CONTRAST_DIR)
        print("  ✓ Base model loaded")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    
    # Load finetuned model data
    print("\n[2/2] Loading finetuned model contrasts...")
    try:
        ft_conf, ft_entropy, ft_cosine = load_contrast_data(FT_CONTRAST_DIR)
        print("  ✓ Finetuned model loaded")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    
    # Create output path
    output_path = FT_CONTRAST_DIR / "COMPARISON_base_vs_ft.png"
    
    # Generate comparison plots
    print("\n[3/3] Generating comparison plots...")
    create_comparison_plots(base_conf, base_entropy, ft_conf, ft_entropy, 
                            base_cosine, ft_cosine, output_path)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
