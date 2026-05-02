"""
Identify internal correlates of uncertainty in MC question answering.

Tests whether model activations encode uncertainty metrics (entropy, logit_gap, etc.)
using both probe and mean_diff direction-finding methods.

Outputs:
- {model}_{dataset}_mc_activations.npz: Reusable activations and metrics
- {model}_{dataset}_mc_dataset.json: Full question metadata
- {model}_{dataset}_mc_entropy_distribution.png: Entropy distribution plot
- {model}_{dataset}_mc_{metric}_directions.npz: Direction vectors per metric
- {model}_{dataset}_mc_{metric}_results.json: Statistics per metric
- {model}_{dataset}_mc_{metric}_results.png: R² curves per metric

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
import joblib

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    should_use_chat_template,
    BatchedExtractor,
    compute_mc_metrics,
    find_directions,
    METRIC_INFO,
)
from core.directions import probe_direction  # For saving probe objects
from core.questions import load_questions
from prompts import format_direct_prompt

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# ADAPTER = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"
ADAPTER = None
DATASET = "TriviaMC"
METRICS = ["entropy"]  # Which metrics to analyze
NUM_QUESTIONS = 500
SEED = 42
BATCH_SIZE = 8

# Quantization (for large models like 70B)
LOAD_IN_4BIT = False  # Set True for 70B+ models
LOAD_IN_8BIT = False

# Direction-finding parameters
PROBE_ALPHA = 1000.0
PROBE_PCA_COMPONENTS = 100
PROBE_N_BOOTSTRAP = 100  # Bootstrap iterations for confidence intervals
PROBE_TRAIN_SPLIT = 0.8
MEAN_DIFF_QUANTILE = 0.25

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_entropy_distribution(
    entropies: np.ndarray,
    metadata: list,
    output_path: Path
):
    """Plot entropy distribution with accuracy breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
    ax1.set_title(f'MC Entropy Distribution (n={len(entropies)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Entropy by correctness (percentage within each group)
    ax2 = axes[1]
    correct_entropies = [m["entropy"] for m in metadata if m["is_correct"]]
    incorrect_entropies = [m["entropy"] for m in metadata if not m["is_correct"]]

    if correct_entropies:
        ax2.hist(correct_entropies, bins=20, alpha=0.6,
                 label=f'Correct (n={len(correct_entropies)})',
                 color='green',
                 weights=np.ones(len(correct_entropies)) / len(correct_entropies) * 100)
    if incorrect_entropies:
        ax2.hist(incorrect_entropies, bins=20, alpha=0.6,
                 label=f'Incorrect (n={len(incorrect_entropies)})',
                 color='red',
                 weights=np.ones(len(incorrect_entropies)) / len(incorrect_entropies) * 100)
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Entropy by Correctness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy vs entropy bins
    ax3 = axes[2]
    n_bins = 10
    entropy_bins = np.linspace(entropies.min(), entropies.max(), n_bins + 1)
    bin_accuracies = []
    bin_centers = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (entropies >= entropy_bins[i]) & (entropies < entropy_bins[i + 1])
        if i == n_bins - 1:
            bin_mask = (entropies >= entropy_bins[i]) & (entropies <= entropy_bins[i + 1])

        bin_items = [m for j, m in enumerate(metadata) if bin_mask[j]]
        if len(bin_items) > 0:
            acc = sum(1 for m in bin_items if m["is_correct"]) / len(bin_items)
            bin_accuracies.append(acc)
            bin_centers.append((entropy_bins[i] + entropy_bins[i + 1]) / 2)
            bin_counts.append(len(bin_items))

    ax3.bar(bin_centers, bin_accuracies, width=(entropy_bins[1] - entropy_bins[0]) * 0.8,
            alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Entropy')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy vs Entropy')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    for x, y, c in zip(bin_centers, bin_accuracies, bin_counts):
        ax3.text(x, y + 0.02, f'n={c}', ha='center', va='bottom', fontsize=8)

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
    ax.set_title(f'{metric} Predictability by Layer')
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
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_short = get_model_short_name(MODEL)
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        base_name = f"{model_short}_adapter-{adapter_short}_{DATASET}"
    else:
        base_name = f"{model_short}_{DATASET}"

    print(f"Model: {MODEL}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Dataset: {DATASET}")
    print(f"Metrics: {METRICS}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Bootstrap iterations: {PROBE_N_BOOTSTRAP}")
    print(f"Output base: {base_name}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    print(f"  Layers: {num_layers}")
    print(f"  Chat template: {use_chat_template}")

    # Load questions
    print(f"\nLoading questions from {DATASET}...")
    questions = load_questions(DATASET, num_questions=NUM_QUESTIONS, seed=SEED)
    print(f"  Loaded {len(questions)} questions")

    # Get option token IDs
    option_keys = list(questions[0]["options"].keys())
    option_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in option_keys]
    print(f"  Option tokens: {dict(zip(option_keys, option_token_ids))}")

    # Extract activations and probabilities
    print(f"\nExtracting activations (batch_size={BATCH_SIZE})...")

    all_activations = {layer: [] for layer in range(num_layers)}
    all_probs = []
    all_logits = []
    all_predicted = []

    with BatchedExtractor(model, num_layers) as extractor:
        for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
            batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

            prompts = []
            for q in batch_questions:
                prompt, _ = format_direct_prompt(q, tokenizer, use_chat_template)
                prompts.append(prompt)

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            layer_acts_by_pos, probs, logits, _ = extractor.extract_batch(input_ids, attention_mask, option_token_ids)

            # extract_batch returns {pos_name: [per-item dicts]}; we only need "final"
            for item_acts in layer_acts_by_pos["final"]:
                for layer, act in item_acts.items():
                    all_activations[layer].append(act)

            for p, l in zip(probs, logits):
                all_probs.append(p)
                all_logits.append(l)
                all_predicted.append(option_keys[np.argmax(p)])

    # Stack activations
    print("\nStacking activations...")
    activations_by_layer = {
        layer: np.stack(acts) for layer, acts in all_activations.items()
    }
    print(f"  Shape per layer: {activations_by_layer[0].shape}")

    # Compute ALL metrics (not just requested ones, for dataset file)
    print("\nComputing all metrics...")
    all_probs_arr = np.array(all_probs)
    all_logits_arr = np.array(all_logits)
    all_metrics = compute_mc_metrics(all_probs_arr, all_logits_arr, metrics=None)  # All metrics

    # Build metadata for each question
    metadata = []
    correct_count = 0
    for i, q in enumerate(questions):
        predicted = all_predicted[i]
        is_correct = predicted == q["correct_answer"]
        if is_correct:
            correct_count += 1

        item = {
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "options": q["options"],
            "probabilities": all_probs[i].tolist(),
            "logits": all_logits[i].tolist(),
        }
        # Add all metric values
        for m_name, m_values in all_metrics.items():
            item[m_name] = float(m_values[i])
        metadata.append(item)

    accuracy = correct_count / len(questions)
    print(f"\nAccuracy: {accuracy:.1%} ({correct_count}/{len(questions)})")

    for metric, values in all_metrics.items():
        print(f"  {metric}: mean={values.mean():.3f}, std={values.std():.3f}")

    # Save activations file
    activations_path = OUTPUT_DIR / f"{base_name}_mc_activations.npz"
    print(f"\nSaving activations to {activations_path}...")
    act_save = {f"layer_{i}": activations_by_layer[i] for i in range(num_layers)}
    for m_name, m_values in all_metrics.items():
        act_save[m_name] = m_values
    np.savez_compressed(activations_path, **act_save)

    # Save dataset JSON
    dataset_path = OUTPUT_DIR / f"{base_name}_mc_dataset.json"
    print(f"Saving dataset to {dataset_path}...")
    dataset_json = {
        "config": {
            "dataset": DATASET,
            "num_questions": len(questions),
            "base_model": MODEL,
            "adapter": ADAPTER,
            "seed": SEED,
        },
        "stats": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(questions),
        },
        "data": metadata,
    }
    # Add metric stats
    for m_name, m_values in all_metrics.items():
        dataset_json["stats"][f"{m_name}_mean"] = float(m_values.mean())
        dataset_json["stats"][f"{m_name}_std"] = float(m_values.std())

    with open(dataset_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    # Plot entropy distribution
    if "entropy" in all_metrics:
        entropy_plot_path = OUTPUT_DIR / f"{base_name}_mc_entropy_distribution.png"
        print(f"\nPlotting entropy distribution...")
        plot_entropy_distribution(all_metrics["entropy"], metadata, entropy_plot_path)

    # Find directions for each metric
    print("\n" + "=" * 60)
    print("FINDING DIRECTIONS")
    print("=" * 60)

    metrics_to_analyze = {m: all_metrics[m] for m in METRICS}
    all_results = {}

    for metric in METRICS:
        print(f"\n--- {metric.upper()} ({PROBE_N_BOOTSTRAP} bootstrap iterations) ---")
        target_values = metrics_to_analyze[metric]

        # Step 1: Run parallel direction finding WITH bootstrap for R² confidence intervals
        # This is fast because it uses all CPU cores
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
            return_scaler=True,  # Save scaler info (fast, parallelizable)
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

        # Step 2: Fit probes separately (no bootstrap) to save objects for transfer tests
        # This is fast because it's just one fit per layer, no bootstrap iterations
        print(f"  Fitting probe objects for transfer tests...")
        probe_objects = {}
        for layer in tqdm(range(num_layers), desc="Fitting probes", leave=False):
            X = activations_by_layer[layer]
            _, info = probe_direction(
                X, target_values,
                alpha=PROBE_ALPHA,
                pca_components=PROBE_PCA_COMPONENTS,
                bootstrap_splits=None,  # No bootstrap - just fit once
                return_probe=True,  # Get the actual objects
            )
            probe_objects[layer] = {
                "scaler": info["scaler"],
                "pca": info["pca"],
                "ridge": info["ridge"],
            }

        # Save directions file for this metric (npz for directions, joblib for probes)
        directions_path = OUTPUT_DIR / f"{base_name}_mc_{metric}_directions.npz"
        probes_path = OUTPUT_DIR / f"{base_name}_mc_{metric}_probes.joblib"

        dir_save = {
            "_metadata_dataset": DATASET,
            "_metadata_model": MODEL,
            "_metadata_metric": metric,
        }
        probe_save = {
            "metadata": {"dataset": DATASET, "model": MODEL, "metric": metric},
            "probes": probe_objects,  # Use the separately-fit probe objects
        }

        for method in ["probe", "mean_diff"]:
            for layer in range(num_layers):
                dir_save[f"{method}_layer_{layer}"] = results["directions"][method][layer]
                # Save scaler info for probe method (from parallel results)
                if method == "probe" and "scaler_scale" in results["fits"][method][layer]:
                    dir_save[f"{method}_scaler_scale_{layer}"] = results["fits"][method][layer]["scaler_scale"]
                    dir_save[f"{method}_scaler_mean_{layer}"] = results["fits"][method][layer]["scaler_mean"]

        np.savez(directions_path, **dir_save)
        joblib.dump(probe_save, probes_path)
        print(f"  Saved directions: {directions_path}")
        print(f"  Saved probes: {probes_path}")

        # Save results JSON for this metric
        results_path = OUTPUT_DIR / f"{base_name}_mc_{metric}_results.json"
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
        plot_path = OUTPUT_DIR / f"{base_name}_mc_{metric}_results.png"
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
    print(f"  {base_name}_mc_activations.npz")
    print(f"  {base_name}_mc_dataset.json")
    if "entropy" in all_metrics:
        print(f"  {base_name}_mc_entropy_distribution.png")
    for metric in METRICS:
        print(f"  {base_name}_mc_{metric}_directions.npz")
        print(f"  {base_name}_mc_{metric}_probes.joblib")
        print(f"  {base_name}_mc_{metric}_results.json")
        print(f"  {base_name}_mc_{metric}_results.png")


if __name__ == "__main__":
    main()
