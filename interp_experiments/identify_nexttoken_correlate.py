"""
Identify internal correlates of uncertainty in next-token prediction.

Tests whether model activations encode uncertainty metrics (entropy, logit_gap, etc.)
on diverse text, using both probe and mean_diff direction-finding methods.

Requires a stratified dataset from build_nexttoken_dataset.py.

Outputs:
- {model}_nexttoken_activations.npz: Reusable activations and metrics
- {model}_nexttoken_dataset.json: Full sample metadata
- {model}_nexttoken_entropy_distribution.png: Entropy distribution plot
- {model}_nexttoken_{metric}_directions.npz: Direction vectors per metric
- {model}_nexttoken_{metric}_results.json: Statistics per metric
- {model}_nexttoken_{metric}_results.png: R² curves per metric

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

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    compute_nexttoken_metrics,
    find_directions,
    METRIC_INFO,
    DEVICE,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: path to PEFT/LoRA adapter
DATASET_PATH = None  # Auto-detect based on model, or set explicitly
METRICS = ["entropy", "logit_gap"]  # Which metrics to analyze
SEED = 42
BATCH_SIZE = 8
MAX_PROMPT_LENGTH = 500

# Direction-finding parameters
PROBE_ALPHA = 1000.0
PROBE_PCA_COMPONENTS = 100
PROBE_N_BOOTSTRAP = 100  # Bootstrap iterations for confidence intervals
PROBE_TRAIN_SPLIT = 0.8
MEAN_DIFF_QUANTILE = 0.25
N_JOBS = -1  # Parallel jobs: -1 = all cores, 1 = sequential with progress bar

# Checkpointing (for large datasets)
CHECKPOINT_INTERVAL = 200

# Quantization (for large models)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_entropy_distribution(
    entropies: np.ndarray,
    output_path: Path
):
    """Plot entropy distribution for next-token prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Overall entropy histogram (percentage)
    ax1 = axes[0]
    ax1.hist(entropies, bins=30, edgecolor='black', alpha=0.7,
             weights=np.ones(len(entropies)) / len(entropies) * 100)
    ax1.axvline(entropies.mean(), color='red', linestyle='--',
                label=f'Mean: {entropies.mean():.3f}')
    ax1.axvline(np.median(entropies), color='orange', linestyle='--',
                label=f'Median: {np.median(entropies):.3f}')
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Percentage')
    ax1.set_title(f'Next-Token Entropy Distribution (n={len(entropies)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. CDF
    ax2 = axes[1]
    sorted_ent = np.sort(entropies)
    cdf = np.arange(1, len(sorted_ent) + 1) / len(sorted_ent)
    ax2.plot(sorted_ent, cdf, linewidth=2)
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Entropy CDF')
    ax2.grid(True, alpha=0.3)
    # Mark quartiles
    for q, label in [(0.25, 'Q1'), (0.5, 'Median'), (0.75, 'Q3')]:
        val = np.percentile(entropies, q * 100)
        ax2.axhline(q, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(val, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_results(
    all_results: dict,
    metric: str,
    output_path: Path
):
    """Plot R² across layers for both methods with confidence intervals."""
    colors = {'probe': 'tab:blue', 'mean_diff': 'tab:orange'}

    results = all_results[metric]
    layers = sorted(results["fits"]["probe"].keys())

    # Single panel - just the R² curves
    fig, ax = plt.subplots(figsize=(10, 5))

    for method in ["probe", "mean_diff"]:
        fits = results["fits"][method]
        r2_values = [fits[l]["r2"] for l in layers]

        # Check for std (bootstrap)
        has_std = "r2_std" in fits[layers[0]]
        if has_std:
            r2_std = [fits[l]["r2_std"] for l in layers]
            ax.fill_between(
                layers,
                np.array(r2_values) - np.array(r2_std),
                np.array(r2_values) + np.array(r2_std),
                alpha=0.2, color=colors[method]
            )

        # Find best layer for this method
        best_layer = max(layers, key=lambda l: fits[l]["r2"])
        best_r2 = fits[best_layer]["r2"]
        label = f'{method} (best: L{best_layer}, R²={best_r2:.3f})'

        ax.plot(layers, r2_values, 'o-', label=label, color=colors[method], markersize=4)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('R² Score')
    ax.set_title(f'{metric} Predictability by Layer (Next-Token)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_diagnostic_summary(metrics_dict: dict, all_results: dict, num_layers: int):
    """Print diagnostic summary."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    # Distribution stats for each metric
    for metric_name, values in metrics_dict.items():
        variance = values.var()
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        print(f"\n{metric_name.upper()} Distribution:")
        print(f"  Mean: {values.mean():.3f}, Std: {values.std():.3f}, Variance: {variance:.4f}")
        print(f"  Median: {np.median(values):.3f}, IQR: {iqr:.3f}")
        print(f"  Range: [{values.min():.3f}, {values.max():.3f}]")

    # Early vs late layer comparison
    print(f"\nLayer R² Comparison (early vs late):")
    for metric_name, results in all_results.items():
        layers = sorted(results["fits"]["probe"].keys())
        n_layers = len(layers)
        early_layers = layers[:n_layers // 4]
        late_layers = layers[3 * n_layers // 4:]

        for method in ["probe", "mean_diff"]:
            fits = results["fits"][method]
            early_r2 = np.mean([fits[l]["r2"] for l in early_layers])
            late_r2 = np.mean([fits[l]["r2"] for l in late_layers])
            r2_increase = late_r2 - early_r2

            print(f"  {metric_name}/{method:<10}: early={early_r2:.3f}, late={late_r2:.3f}, Δ={r2_increase:+.3f}")

    print("=" * 80)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_dataset_path(model_name: str) -> Path:
    """Find the stratified dataset file."""
    if DATASET_PATH:
        return Path(DATASET_PATH)

    model_short = get_model_short_name(model_name)

    # Try model-specific path
    model_specific = OUTPUT_DIR / f"{model_short}_nexttoken_entropy_dataset.json"
    if model_specific.exists():
        return model_specific

    # Fall back to generic
    generic = OUTPUT_DIR / "entropy_dataset.json"
    if generic.exists():
        return generic

    raise FileNotFoundError(
        f"Could not find dataset. Tried: {model_specific}, {generic}. "
        "Run build_nexttoken_dataset.py first."
    )


def load_dataset(path: Path) -> list:
    """Load the stratified entropy dataset."""
    print(f"Loading dataset from {path}...")
    with open(path) as f:
        raw = json.load(f)

    # Handle both formats
    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
        config = raw.get("config")
        if config:
            print(f"  Config: {config}")
    else:
        data = raw

    print(f"  Loaded {len(data)} samples")
    return data


def extract_activations_and_metrics(
    dataset: list,
    model,
    tokenizer,
    num_layers: int,
    checkpoint_path: Path
) -> tuple:
    """
    Extract activations and compute metrics for all samples.

    Returns:
        activations_by_layer: {layer: (n_samples, hidden_dim)}
        metrics_dict: {metric_name: (n_samples,)}
        predicted_tokens: (n_samples,)
    """
    # Check for checkpoint
    start_idx = 0
    all_activations = {i: [] for i in range(num_layers)}
    all_metrics = {m: [] for m in ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]}
    all_predicted_tokens = []

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        if "processed_count" in ckpt.files:
            start_idx = int(ckpt["processed_count"])
            for i in range(num_layers):
                key = f"layer_{i}"
                if key in ckpt.files:
                    all_activations[i] = list(ckpt[key])
            for m in all_metrics:
                if m in ckpt.files:
                    all_metrics[m] = list(ckpt[m])
            if "predicted_tokens" in ckpt.files:
                all_predicted_tokens = list(ckpt["predicted_tokens"])
            print(f"  Resuming from sample {start_idx}")

    # Set up hooks
    activations_cache = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations_cache[layer_idx] = hidden[:, -1, :].detach()
        return hook

    if hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        layers = model.model.layers

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    total = len(dataset)

    try:
        for batch_start in tqdm(range(start_idx, total, BATCH_SIZE)):
            batch = dataset[batch_start:batch_start + BATCH_SIZE]
            texts = [item["text"] for item in batch]

            # Tokenize
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_LENGTH,
                add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            activations_cache.clear()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # Get logits at last position
            batch_logits = outputs.logits[:, -1, :].cpu().numpy()

            for b in range(len(batch)):
                logits = batch_logits[b]

                # Compute metrics
                metrics = compute_nexttoken_metrics(logits)
                for m, v in metrics.items():
                    all_metrics[m].append(v)

                all_predicted_tokens.append(int(np.argmax(logits)))

            # Store activations
            for layer_idx, acts in activations_cache.items():
                all_activations[layer_idx].extend(acts.cpu().numpy())

            # Checkpoint
            processed = batch_start + len(batch)
            if processed % CHECKPOINT_INTERVAL < BATCH_SIZE and processed < total:
                save_dict = {f"layer_{i}": np.array(a) for i, a in all_activations.items()}
                for m, v in all_metrics.items():
                    save_dict[m] = np.array(v)
                save_dict["predicted_tokens"] = np.array(all_predicted_tokens)
                save_dict["processed_count"] = processed
                np.savez_compressed(checkpoint_path, **save_dict)
                print(f"  Checkpoint: {processed}/{total}")

            # Memory cleanup
            del encoded, input_ids, attention_mask, outputs
            if batch_start % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    finally:
        for h in hooks:
            h.remove()

    # Convert to arrays
    activations_by_layer = {i: np.array(a) for i, a in all_activations.items()}
    metrics_dict = {m: np.array(v) for m, v in all_metrics.items()}
    predicted_tokens = np.array(all_predicted_tokens)

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  Removed checkpoint")

    return activations_by_layer, metrics_dict, predicted_tokens


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_short = get_model_short_name(MODEL)
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        base_name = f"{model_short}_adapter-{adapter_short}_nexttoken"
    else:
        base_name = f"{model_short}_nexttoken"

    checkpoint_path = OUTPUT_DIR / f"{base_name}_checkpoint.npz"

    print(f"Model: {MODEL}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Metrics: {METRICS}")
    print(f"Bootstrap iterations: {PROBE_N_BOOTSTRAP}")
    print(f"Output base: {base_name}")
    print()

    # Find dataset
    dataset_path = find_dataset_path(MODEL)

    # Load dataset
    dataset = load_dataset(dataset_path)

    # Check if activations already exist (resume from crash)
    activations_path = OUTPUT_DIR / f"{base_name}_activations.npz"
    if activations_path.exists():
        print(f"\nFound existing activations: {activations_path}")
        print("Loading from file (skipping model load and extraction)...")
        loaded = np.load(activations_path)

        # Reconstruct activations_by_layer
        layer_keys = [k for k in loaded.files if k.startswith("layer_")]
        num_layers = len(layer_keys)
        activations_by_layer = {i: loaded[f"layer_{i}"] for i in range(num_layers)}

        # Reconstruct metrics_dict
        metrics_dict = {}
        for m in METRICS:
            if m in loaded.files:
                metrics_dict[m] = loaded[m]

        # Get predicted tokens
        predicted_tokens = loaded["predicted_tokens"] if "predicted_tokens" in loaded.files else None

        print(f"  Loaded {num_layers} layers, {len(metrics_dict)} metrics")
    else:
        # Load model
        print("\nLoading model...")
        model, tokenizer, num_layers = load_model_and_tokenizer(
            MODEL,
            adapter_path=ADAPTER,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT
        )
        print(f"  Layers: {num_layers}")
        print(f"  Device: {DEVICE}")

        # Extract activations and metrics
        print(f"\nExtracting activations (batch_size={BATCH_SIZE})...")
        activations_by_layer, metrics_dict, predicted_tokens = extract_activations_and_metrics(
            dataset, model, tokenizer, num_layers, checkpoint_path
        )

        print(f"\nActivations shape: {activations_by_layer[0].shape}")
        print("\nMetric statistics:")
        for m, v in metrics_dict.items():
            print(f"  {m}: mean={v.mean():.3f}, std={v.std():.3f}, range=[{v.min():.3f}, {v.max():.3f}]")

        # Save activations file
        print(f"\nSaving activations to {activations_path}...")
        act_save = {f"layer_{i}": activations_by_layer[i] for i in range(num_layers)}
        for m_name, m_values in metrics_dict.items():
            act_save[m_name] = m_values
        act_save["predicted_tokens"] = predicted_tokens
        np.savez_compressed(activations_path, **act_save)

    # Save dataset JSON
    dataset_output_path = OUTPUT_DIR / f"{base_name}_dataset.json"
    print(f"Saving dataset to {dataset_output_path}...")
    metadata = []
    for i, item in enumerate(dataset):
        meta_item = {
            "text": item["text"],
            "predicted_token": int(predicted_tokens[i]),
        }
        for m_name, m_values in metrics_dict.items():
            meta_item[m_name] = float(m_values[i])
        metadata.append(meta_item)

    dataset_json = {
        "config": {
            "dataset_path": str(dataset_path),
            "num_samples": len(dataset),
            "base_model": MODEL,
            "adapter": ADAPTER,
            "seed": SEED,
        },
        "stats": {},
        "data": metadata,
    }
    # Add metric stats
    for m_name, m_values in metrics_dict.items():
        dataset_json["stats"][f"{m_name}_mean"] = float(m_values.mean())
        dataset_json["stats"][f"{m_name}_std"] = float(m_values.std())

    with open(dataset_output_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    # Plot entropy distribution
    if "entropy" in metrics_dict:
        entropy_plot_path = OUTPUT_DIR / f"{base_name}_entropy_distribution.png"
        print(f"\nPlotting entropy distribution...")
        plot_entropy_distribution(metrics_dict["entropy"], entropy_plot_path)

    # Find directions for each metric
    print("\n" + "=" * 60)
    print("FINDING DIRECTIONS")
    print("=" * 60)

    metrics_to_analyze = {m: metrics_dict[m] for m in METRICS}
    all_results = {}

    for metric in METRICS:
        print(f"\n--- {metric.upper()} ({PROBE_N_BOOTSTRAP} bootstrap iterations) ---")
        target_values = metrics_to_analyze[metric]

        results = find_directions(
            activations_by_layer,
            target_values,
            methods=["probe", "mean_diff"],
            probe_alpha=PROBE_ALPHA,
            probe_pca_components=PROBE_PCA_COMPONENTS,
            probe_n_bootstrap=PROBE_N_BOOTSTRAP,
            probe_train_split=PROBE_TRAIN_SPLIT,
            mean_diff_quantile=MEAN_DIFF_QUANTILE,
            seed=SEED,
            n_jobs=N_JOBS,
            return_scaler=True,  # Save scaler info for transfer tests
        )

        all_results[metric] = results

        # Print summary per method
        for method in ["probe", "mean_diff"]:
            fits = results["fits"][method]
            best_layer = max(fits.keys(), key=lambda l: fits[l]["r2"])
            best_r2 = fits[best_layer]["r2"]
            best_corr = fits[best_layer]["corr"]

            r2_std_str = ""
            if "r2_std" in fits[best_layer]:
                r2_std_str = f" ± {fits[best_layer]['r2_std']:.3f}"

            avg_r2 = np.mean([f["r2"] for f in fits.values()])

            print(f"  {method:12s}: best layer={best_layer:2d} (R²={best_r2:.3f}{r2_std_str}, r={best_corr:.3f}), avg R²={avg_r2:.3f}")

        # Method comparison
        if results["comparison"]:
            mid_layer = num_layers // 2
            cos_sim = results["comparison"][mid_layer]["cosine_sim"]
            print(f"  probe vs mean_diff cosine similarity (layer {mid_layer}): {cos_sim:.3f}")

        # Save directions file for this metric
        directions_path = OUTPUT_DIR / f"{base_name}_{metric}_directions.npz"
        dir_save = {
            "_metadata_dataset": str(dataset_path),
            "_metadata_model": MODEL,
            "_metadata_metric": metric,
        }
        for method in ["probe", "mean_diff"]:
            for layer in range(num_layers):
                dir_save[f"{method}_layer_{layer}"] = results["directions"][method][layer]
                # Save scaler info for probe method (for centered transfer)
                if method == "probe" and "scaler_scale" in results["fits"][method][layer]:
                    dir_save[f"{method}_scaler_scale_{layer}"] = results["fits"][method][layer]["scaler_scale"]
                    dir_save[f"{method}_scaler_mean_{layer}"] = results["fits"][method][layer]["scaler_mean"]
        np.savez(directions_path, **dir_save)
        print(f"  Saved directions: {directions_path}")

        # Save results JSON for this metric
        results_path = OUTPUT_DIR / f"{base_name}_{metric}_results.json"
        results_json = {
            "config": {
                "train_split": PROBE_TRAIN_SPLIT,
                "probe_alpha": PROBE_ALPHA,
                "use_pca": True,
                "pca_components": PROBE_PCA_COMPONENTS,
                "n_bootstrap": PROBE_N_BOOTSTRAP,
                "mean_diff_quantile": MEAN_DIFF_QUANTILE,
                "seed": SEED,
            },
            "metric_stats": {
                "mean": float(target_values.mean()),
                "std": float(target_values.std()),
                "min": float(target_values.min()),
                "max": float(target_values.max()),
                "variance": float(target_values.var()),
                "median": float(np.median(target_values)),
                "iqr": float(np.percentile(target_values, 75) - np.percentile(target_values, 25)),
            },
            "results": {},
        }
        for method in ["probe", "mean_diff"]:
            results_json["results"][method] = {}
            for layer in range(num_layers):
                layer_info = {}
                for k, v in results["fits"][method][layer].items():
                    # Skip numpy arrays (scaler_scale, scaler_mean) - those go in .npz
                    if isinstance(v, np.ndarray):
                        continue
                    # Convert numpy scalars to Python types
                    if isinstance(v, np.floating):
                        layer_info[k] = float(v)
                    elif isinstance(v, np.integer):
                        layer_info[k] = int(v)
                    else:
                        layer_info[k] = v
                results_json["results"][method][layer] = layer_info

        with open(results_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"  Saved results: {results_path}")

        # Plot results for this metric
        plot_path = OUTPUT_DIR / f"{base_name}_{metric}_results.png"
        plot_results(all_results, metric, plot_path)

    # Diagnostic summary
    print_diagnostic_summary(metrics_to_analyze, all_results, num_layers)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for metric in METRICS:
        print(f"\n{metric}:")
        info = METRIC_INFO[metric]
        print(f"  (higher = {info['higher_means']}, linear={info['linear']})")

        for method in ["probe", "mean_diff"]:
            fits = all_results[metric]["fits"][method]
            best_layer = max(fits.keys(), key=lambda l: fits[l]["r2"])
            best_r2 = fits[best_layer]["r2"]
            mae_str = ""
            if "mae" in fits[best_layer]:
                mae_str = f", MAE={fits[best_layer]['mae']:.3f}"
            print(f"  {method}: best R²={best_r2:.3f}{mae_str} at layer {best_layer}")

    print("\nOutput files:")
    print(f"  {base_name}_activations.npz")
    print(f"  {base_name}_dataset.json")
    if "entropy" in metrics_dict:
        print(f"  {base_name}_entropy_distribution.png")
    for metric in METRICS:
        print(f"  {base_name}_{metric}_directions.npz")
        print(f"  {base_name}_{metric}_results.json")
        print(f"  {base_name}_{metric}_results.png")


if __name__ == "__main__":
    main()
