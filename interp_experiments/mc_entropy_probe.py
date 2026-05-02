"""
Run uncertainty prediction experiment on multiple-choice questions.

This script:
1. Loads MC questions from dataset file
2. Formats them and runs through the model to get logits/probs over answer tokens
3. Computes multiple uncertainty metrics (all saved, one probed per run):

   Prob-based (nonlinear - may be harder for linear probes):
   - entropy: Shannon entropy -sum(p * log(p))
   - top_prob: P(argmax) - probability of most likely answer
   - margin: P(top) - P(second) - prob gap between top two

   Logit-based (linear - better aligned with linear probes):
   - logit_gap: z(top) - z(second) - logit gap between top two
   - top_logit: z(top) - mean(z) - centered top logit

4. Extracts activations from all layers
5. Trains linear probes to predict the selected metric from each layer
6. Saves directions: {prefix}_directions.npz

Usage:
    python mc_entropy_probe.py --metric logit_gap   # Probe logit_gap (default)
    python mc_entropy_probe.py --metric entropy     # Probe entropy
    python mc_entropy_probe.py --plot-only          # Retrain from saved activations
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_model_short_name,
    LinearProbe,
    compute_entropy_from_probs,
)
from prompts import MC_SETUP_PROMPT, format_direct_prompt

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"###
DATASET_NAME = "SimpleMC"
NUM_QUESTIONS = 447 if DATASET_NAME.startswith("GP") else 500
SEED = 42

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100
N_BOOTSTRAP = 100  # Number of bootstrap iterations for confidence intervals

np.random.seed(SEED)
torch.manual_seed(SEED)

# MC_SETUP_PROMPT imported from tasks.py

# Available uncertainty metrics:
# Prob-based (nonlinear targets - may be harder for linear probes):
#   entropy   - Shannon entropy -sum(p * log(p))
#   top_prob  - P(argmax) - probability of most likely answer
#   margin    - P(top) - P(second) - prob gap between top two
# Logit-based (linear targets - better aligned with linear probes):
#   logit_gap - z(top) - z(second) - logit gap between top two
#   top_logit - z(top) - mean(z) - centered top logit
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
METRIC = "logit_gap"  # Which metric to probe (set via --metric flag)


def compute_uncertainty_metrics(probs: np.ndarray, logits: np.ndarray = None) -> Dict[str, float]:
    """
    Compute multiple uncertainty metrics from probability and logit distributions.

    Args:
        probs: Probability distribution over answer options (sums to 1)
        logits: Raw logits for answer options (before softmax). If None, logit-based
                metrics will be computed from log(probs) as an approximation.

    Returns:
        Dict with keys: entropy, top_prob, margin, logit_gap, top_logit
    """
    # === Prob-based metrics (nonlinear) ===

    # Entropy: -sum(p * log(p))
    entropy = compute_entropy_from_probs(probs)

    # Top probability: P(argmax)
    top_prob = float(np.max(probs))

    # Margin: P(top) - P(second)
    sorted_probs = np.sort(probs)[::-1]  # Descending
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])

    # === Logit-based metrics (linear - better for linear probes) ===

    # If logits not provided, approximate from log-probs
    # (This loses the constant offset but preserves gaps)
    if logits is None:
        logits = np.log(probs + 1e-10)

    # Sort logits descending
    sorted_logits = np.sort(logits)[::-1]

    # Logit gap: z(top) - z(second)
    # This is the cleanest linear target - invariant to temperature/scale shifts
    logit_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else float(sorted_logits[0])

    # Top logit (centered): z(top) - mean(z)
    # Subtracting mean makes it invariant to adding a constant to all logits
    top_logit = float(sorted_logits[0] - np.mean(logits))

    return {
        "entropy": entropy,
        "top_prob": top_prob,
        "margin": margin,
        "logit_gap": logit_gap,
        "top_logit": top_logit,
    }


def get_base_output_prefix() -> str:
    """Generate base output filename prefix (without metric) for shared files like activations."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_mc")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_mc")


def get_output_prefix(metric: str) -> str:
    """Generate output filename prefix for metric-specific files."""
    return f"{get_base_output_prefix()}_{metric}"


def load_questions(dataset_name: str, num_questions: int = None) -> List[Dict]:
    """Load MC questions using load_and_format_dataset."""
    from core.datasets import load_and_format_dataset

    questions = load_and_format_dataset(dataset_name, num_questions_needed=num_questions)

    if questions is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    return questions


def format_mc_prompt(question: Dict, tokenizer) -> Tuple[str, List[str], List[int]]:
    """
    Format a multiple-choice question and get option token IDs.

    Uses centralized format_direct_prompt from tasks.py.

    Returns:
        Tuple of (full_prompt, option_keys, option_token_ids)
    """
    # Use centralized prompt formatting
    full_prompt, options = format_direct_prompt(question, tokenizer, use_chat_template=True)

    # Get token IDs for answer options
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]

    return full_prompt, options, option_token_ids


def extract_mc_activations_and_metrics(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    batch_size: int = 8
) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray], List[Dict]]:
    """
    Extract activations and compute uncertainty metrics in a single forward pass per question.

    Args:
        questions: List of question dicts
        model: The model
        tokenizer: The tokenizer
        num_layers: Number of layers
        batch_size: Batch size for forward passes (default 8, reduce for large models)

    Returns:
        activations: Dict mapping layer_idx -> array of shape (num_questions, hidden_dim)
        metrics: Dict mapping metric_name -> array of shape (num_questions,)
        metadata: List of dicts with question info, probabilities, etc.
    """
    print(f"Processing {len(questions)} questions with batch_size={batch_size}...")

    # Initialize storage
    all_layer_activations = {i: [] for i in range(num_layers)}
    all_metrics = {metric: [] for metric in AVAILABLE_METRICS}
    metadata = []

    # Set up hooks for activation extraction
    # Key optimization: store only last-token activations per batch item
    activations_cache = {}
    current_last_indices = None  # Will be set per batch
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            nonlocal current_last_indices
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Extract only last-token activations for each batch item
            # current_last_indices: (batch_size,) tensor of last token positions
            batch_size_actual = hidden_states.shape[0]
            # Use advanced indexing to get last token for each batch item
            last_token_acts = hidden_states[
                torch.arange(batch_size_actual, device=hidden_states.device),
                current_last_indices[:batch_size_actual]
            ]  # Shape: (batch_size, hidden_dim)
            activations_cache[layer_idx] = last_token_acts.detach()
        return hook

    # Get layers
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
        layers = base.model.layers
    else:
        layers = model.model.layers

    # Register hooks
    for i, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook(i))
        hooks.append(handle)

    model.eval()

    # Pre-format all prompts
    print("Formatting prompts...")
    formatted_data = []
    for question in questions:
        full_prompt, options, option_token_ids = format_mc_prompt(question, tokenizer)
        formatted_data.append({
            "prompt": full_prompt,
            "options": options,
            "option_token_ids": option_token_ids,
            "question": question
        })

    try:
        # Process in batches
        for batch_start in tqdm(range(0, len(formatted_data), batch_size)):
            batch_end = min(batch_start + batch_size, len(formatted_data))
            batch_items = formatted_data[batch_start:batch_end]
            batch_prompts = [item["prompt"] for item in batch_items]

            # Tokenize batch with padding (no truncation to match other scripts)
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True  # Pad to longest in batch
            )
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            # Compute last token indices
            # With left-padding, the last real token is always at the end of the sequence
            # So the last token index is simply seq_len - 1 for all items in the batch
            seq_len = input_ids.shape[1]
            current_last_indices = torch.full(
                (input_ids.shape[0],), seq_len - 1, device=input_ids.device, dtype=torch.long
            )

            # Clear cache
            activations_cache.clear()

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # Single CPU transfer: stack all layers and transfer at once
            # activations_cache[layer_idx] is (batch_size, hidden_dim)
            stacked = torch.stack([activations_cache[i] for i in range(num_layers)], dim=0)
            # stacked shape: (num_layers, batch_size, hidden_dim)
            stacked_cpu = stacked.cpu().numpy()

            # Distribute to per-layer storage
            for layer_idx in range(num_layers):
                # stacked_cpu[layer_idx] is (batch_size, hidden_dim)
                for batch_item_idx in range(len(batch_items)):
                    all_layer_activations[layer_idx].append(stacked_cpu[layer_idx, batch_item_idx])

            # Compute metrics for each item in batch
            for batch_item_idx, item in enumerate(batch_items):
                option_token_ids = item["option_token_ids"]
                options = item["options"]
                question = item["question"]

                # Get logits at last token position for this batch item
                last_idx = current_last_indices[batch_item_idx].item()
                final_logits = outputs.logits[batch_item_idx, last_idx, :]
                option_logits = final_logits[option_token_ids]
                option_logits_np = option_logits.cpu().numpy()
                probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

                # Compute all uncertainty metrics (pass logits for logit-based metrics)
                item_metrics = compute_uncertainty_metrics(probs, option_logits_np)

                # Compute accuracy
                predicted_idx = np.argmax(probs)
                predicted_answer = options[predicted_idx]
                correct_answer = question.get("correct_answer", "")
                is_correct = predicted_answer == correct_answer

                # Store metrics
                for metric_name, metric_value in item_metrics.items():
                    all_metrics[metric_name].append(metric_value)

                metadata.append({
                    "question_id": batch_start + batch_item_idx,
                    "question": question.get("question", ""),
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "options": options,
                    "probabilities": probs.tolist(),
                    "logits": option_logits_np.tolist(),  # Store logits too
                    **item_metrics,  # Include all metrics
                })

            # Clear memory
            del inputs, input_ids, attention_mask, outputs, stacked, stacked_cpu
            if (batch_start + batch_size) % 100 == 0:
                torch.cuda.empty_cache()

    finally:
        # Remove hooks
        for handle in hooks:
            handle.remove()

    # Convert to numpy arrays
    activations = {
        layer_idx: np.array(acts)
        for layer_idx, acts in all_layer_activations.items()
    }
    metrics = {
        metric_name: np.array(values)
        for metric_name, values in all_metrics.items()
    }

    # Compute accuracy statistics
    correct_count = sum(1 for m in metadata if m["is_correct"])
    accuracy = correct_count / len(metadata)

    print(f"Extracted activations shape (per layer): {activations[0].shape}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(metadata)})")
    print(f"\nUncertainty metrics:")
    for metric_name, values in metrics.items():
        print(f"  {metric_name}: range=[{values.min():.3f}, {values.max():.3f}], "
              f"mean={values.mean():.3f}, std={values.std():.3f}")

    return activations, metrics, metadata


def _train_probe_for_layer(
    layer_idx: int,
    X: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int,
    train_split: float,
    seed: int,
    use_pca: bool,
    pca_components: int,
    alpha: float
) -> Tuple[int, Dict, np.ndarray]:
    """Train probe for a single layer with bootstrap. Used for parallel execution.

    Returns (layer_idx, results_dict, direction_vector).
    Direction is extracted from a final probe trained on full training set.
    """
    rng = np.random.RandomState(seed + layer_idx)
    n = len(targets)

    test_r2s = []
    test_maes = []

    # Bootstrap for confidence intervals
    for _ in range(n_bootstrap):
        # Random split
        indices = np.arange(n)
        rng.shuffle(indices)
        split_idx = int(n * train_split)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = targets[train_idx]
        y_test = targets[test_idx]

        # Train probe
        probe = LinearProbe(
            alpha=alpha,
            use_pca=use_pca,
            pca_components=pca_components
        )
        probe.fit(X_train, y_train)

        # Evaluate
        test_eval = probe.evaluate(X_test, y_test)
        test_r2s.append(test_eval["r2"])
        test_maes.append(test_eval["mae"])

    # Train final probe on canonical split for direction extraction
    rng_final = np.random.RandomState(seed)  # Same seed across layers for consistent split
    indices = np.arange(n)
    rng_final.shuffle(indices)
    split_idx = int(n * train_split)
    train_idx = indices[:split_idx]

    final_probe = LinearProbe(
        alpha=alpha,
        use_pca=use_pca,
        pca_components=pca_components
    )
    final_probe.fit(X[train_idx], targets[train_idx])
    direction = final_probe.get_direction()  # Always in original space

    return layer_idx, {
        "test_r2_mean": float(np.mean(test_r2s)),
        "test_r2_std": float(np.std(test_r2s)),
        "test_mae_mean": float(np.mean(test_maes)),
        "test_mae_std": float(np.std(test_maes)),
    }, direction


def run_all_probes(
    activations: Dict[int, np.ndarray],
    metrics: Dict[str, np.ndarray],
    n_jobs: int = -1,
    use_pca: bool = USE_PCA,
    pca_components: int = PCA_COMPONENTS,
    alpha: float = PROBE_ALPHA
) -> Tuple[Dict[str, Dict[int, Dict]], Dict[str, Dict[int, np.ndarray]]]:
    """Train probes for all layers and all metrics with bootstrap confidence intervals.

    Returns:
        all_results: Dict mapping metric_name -> {layer_idx -> result dict with R², MAE stats}
        all_directions: Dict mapping metric_name -> {layer_idx -> normalized direction vector}
    """
    pca_str = f"PCA={pca_components}" if use_pca else "no PCA"
    layer_indices = sorted(activations.keys())

    all_results = {}
    all_directions = {}

    for metric_name, targets in metrics.items():
        print(f"\nTraining probes for metric '{metric_name}' across {len(activations)} layers "
              f"({N_BOOTSTRAP} bootstrap iterations, {pca_str})...")

        # Run in parallel across layers
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_train_probe_for_layer)(
                layer_idx,
                activations[layer_idx],
                targets,
                N_BOOTSTRAP,
                TRAIN_SPLIT,
                SEED,
                use_pca,
                pca_components,
                alpha
            )
            for layer_idx in layer_indices
        )

        # Convert list of (layer_idx, result, direction) tuples to dicts
        all_results[metric_name] = {layer_idx: result for layer_idx, result, _ in results_list}
        all_directions[metric_name] = {layer_idx: direction for layer_idx, _, direction in results_list}

    return all_results, all_directions


def print_results(all_results: Dict[str, Dict[int, Dict]]):
    """Print summary of results for all metrics."""
    for metric_name, results in all_results.items():
        print("\n" + "="*80)
        print(f"RESULTS SUMMARY: {metric_name.upper()}")
        print("="*80)
        print(f"{'Layer':<8} {'Test R²':<20} {'Test MAE':<20}")
        print("-"*80)

        for layer_idx in sorted(results.keys()):
            res = results[layer_idx]
            r2_str = f"{res['test_r2_mean']:.4f} ± {res['test_r2_std']:.4f}"
            mae_str = f"{res['test_mae_mean']:.4f} ± {res['test_mae_std']:.4f}"
            print(f"{layer_idx:<8} {r2_str:<20} {mae_str:<20}")

        print("="*80)

        # Find best layer
        best_layer = max(results.keys(), key=lambda l: results[l]["test_r2_mean"])
        best_r2 = results[best_layer]["test_r2_mean"]
        best_std = results[best_layer]["test_r2_std"]
        print(f"Best layer: {best_layer} (Test R² = {best_r2:.4f} ± {best_std:.4f})")

    # Print comparison across metrics
    print("\n" + "="*80)
    print("METRIC COMPARISON (Best R² per metric)")
    print("="*80)
    for metric_name, results in all_results.items():
        best_layer = max(results.keys(), key=lambda l: results[l]["test_r2_mean"])
        best_r2 = results[best_layer]["test_r2_mean"]
        print(f"  {metric_name:<12}: layer {best_layer:<3} R² = {best_r2:.4f}")


def plot_entropy_distribution(
    entropies: np.ndarray,
    metadata: List[Dict],
    output_path: Path
):
    """Plot entropy distribution with accuracy breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Overall entropy histogram (percentage)
    ax1 = axes[0]
    ax1.hist(entropies, bins=30, edgecolor='black', alpha=0.7, weights=np.ones(len(entropies)) / len(entropies) * 100)
    ax1.axvline(entropies.mean(), color='red', linestyle='--', label=f'Mean: {entropies.mean():.3f}')
    ax1.axvline(np.median(entropies), color='orange', linestyle='--', label=f'Median: {np.median(entropies):.3f}')
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
        ax2.hist(correct_entropies, bins=20, alpha=0.6, label=f'Correct (n={len(correct_entropies)})',
                 color='green', weights=np.ones(len(correct_entropies)) / len(correct_entropies) * 100)
    if incorrect_entropies:
        ax2.hist(incorrect_entropies, bins=20, alpha=0.6, label=f'Incorrect (n={len(incorrect_entropies)})',
                 color='red', weights=np.ones(len(incorrect_entropies)) / len(incorrect_entropies) * 100)
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
        if i == n_bins - 1:  # Include right edge in last bin
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

    # Add count labels on bars
    for x, y, c in zip(bin_centers, bin_accuracies, bin_counts):
        ax3.text(x, y + 0.02, f'n={c}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy distribution plot saved to {output_path}")


def plot_results(all_results: Dict[str, Dict[int, Dict]], output_path: Path):
    """Plot R² across layers for all metrics with confidence intervals."""
    metric_names = list(all_results.keys())
    n_metrics = len(metric_names)

    # Colors for different metrics
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get layers from first metric (all should have same layers)
    layers = sorted(all_results[metric_names[0]].keys())

    # R² plot - all metrics on same axes
    ax1 = axes[0]
    for i, metric_name in enumerate(metric_names):
        results = all_results[metric_name]
        test_r2_mean = [results[l]["test_r2_mean"] for l in layers]
        test_r2_std = [results[l]["test_r2_std"] for l in layers]

        color = colors[i % len(colors)]
        ax1.plot(layers, test_r2_mean, 'o-', label=metric_name, color=color, markersize=4)
        ax1.fill_between(
            layers,
            np.array(test_r2_mean) - np.array(test_r2_std),
            np.array(test_r2_mean) + np.array(test_r2_std),
            alpha=0.2, color=color
        )

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Uncertainty Metric Predictability by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Best R² comparison bar chart
    ax2 = axes[1]
    best_r2s = []
    best_layers = []
    for metric_name in metric_names:
        results = all_results[metric_name]
        best_layer = max(results.keys(), key=lambda l: results[l]["test_r2_mean"])
        best_r2s.append(results[best_layer]["test_r2_mean"])
        best_layers.append(best_layer)

    bars = ax2.bar(metric_names, best_r2s, color=[colors[i % len(colors)] for i in range(n_metrics)])
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Best R² Score')
    ax2.set_title('Best R² by Metric')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add layer labels on bars
    for bar, layer in zip(bars, best_layers):
        height = bar.get_height()
        ax2.annotate(f'L{layer}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def print_diagnostic_summary(metrics: Dict[str, np.ndarray], all_results: Dict[str, Dict[int, Dict]]):
    """Print diagnostic summary to help identify anomalous patterns."""
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    # Print distribution stats for each metric
    for metric_name, values in metrics.items():
        variance = values.var()
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        print(f"\n{metric_name.upper()} Distribution:")
        print(f"  Mean: {values.mean():.3f}, Std: {values.std():.3f}, Variance: {variance:.4f}")
        print(f"  Median: {np.median(values):.3f}, IQR: {iqr:.3f}")
        print(f"  Range: [{values.min():.3f}, {values.max():.3f}]")

    # Early vs late layer comparison for each metric
    print(f"\nLayer R² Comparison (early vs late):")
    for metric_name, results in all_results.items():
        layers = sorted(results.keys())
        n_layers = len(layers)
        early_layers = layers[:n_layers // 4]
        late_layers = layers[3 * n_layers // 4:]

        early_r2 = np.mean([results[l]["test_r2_mean"] for l in early_layers])
        late_r2 = np.mean([results[l]["test_r2_mean"] for l in late_layers])
        r2_increase = late_r2 - early_r2

        print(f"  {metric_name:<12}: early={early_r2:.3f}, late={late_r2:.3f}, Δ={r2_increase:+.3f}")

    print("="*80)


def load_activations(activations_path: Path) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray]]:
    """Load activations and metrics from saved file."""
    print(f"Loading activations from {activations_path}...")
    data = np.load(activations_path)

    activations = {
        int(k.split("_")[1]): data[k]
        for k in data.files if k.startswith("layer_")
    }

    # Load all available metrics
    metrics = {}
    for metric_name in AVAILABLE_METRICS:
        if metric_name in data.files:
            metrics[metric_name] = data[metric_name]
        elif metric_name == "entropy" and "entropies" in data.files:
            # Backward compatibility: old files have "entropies" key
            metrics["entropy"] = data["entropies"]

    if not metrics:
        raise ValueError(f"No metrics found in {activations_path}. Re-run without --plot-only.")

    print(f"Loaded {len(activations)} layers, {len(list(metrics.values())[0])} samples, {len(metrics)} metrics")
    return activations, metrics


def main():
    global METRIC

    parser = argparse.ArgumentParser(description="Train uncertainty probes on MC questions")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Metric to probe (default: {METRIC})")
    parser.add_argument("--all-metrics", action="store_true",
                        help="Train probes for all metrics (takes ~5x longer than single metric)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip extraction, load saved activations and retrain probes")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for forward passes (default 8, reduce for large models)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (recommended for 70B+ models)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    args = parser.parse_args()

    METRIC = args.metric
    run_all_metrics = args.all_metrics
    print(f"Device: {DEVICE}")
    if run_all_metrics:
        print(f"Metrics: ALL ({', '.join(AVAILABLE_METRICS)})")
    else:
        print(f"Metric: {METRIC}")

    # Generate output prefixes
    base_prefix = get_base_output_prefix()  # For shared files (activations, dataset)
    print(f"Base output prefix: {base_prefix}")

    # Define output paths
    # Shared files (metric-agnostic): activations and dataset
    activations_path = Path(f"{base_prefix}_activations.npz")
    dataset_path = Path(f"{base_prefix}_dataset.json")
    # Metric-specific files are generated later in the loop

    metadata = None  # Will be loaded or computed

    if args.plot_only:
        # Load existing activations
        if not activations_path.exists():
            raise FileNotFoundError(
                f"Activations file not found: {activations_path}. "
                "Run without --plot-only first."
            )
        activations, all_metrics = load_activations(activations_path)

        # Get just the metric we care about
        if METRIC not in all_metrics:
            raise ValueError(f"Metric '{METRIC}' not found in saved file. Available: {list(all_metrics.keys())}")
        target = all_metrics[METRIC]

        # Load metadata
        if dataset_path.exists():
            with open(dataset_path) as f:
                dataset_data = json.load(f)
                metadata = dataset_data.get("data", [])
    else:
        # Full run: load model and extract activations
        model, tokenizer, num_layers = load_model_and_tokenizer(
            BASE_MODEL_NAME,
            adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
        )

        # Load questions
        questions = load_questions(DATASET_NAME, NUM_QUESTIONS)

        # Shuffle with fixed seed for reproducibility
        random.seed(SEED)
        random.shuffle(questions)

        # Extract activations and compute all uncertainty metrics in single pass
        activations, all_metrics, metadata = extract_mc_activations_and_metrics(
            questions, model, tokenizer, num_layers, batch_size=args.batch_size
        )
        target = all_metrics[METRIC]

        # Save activations and all metrics (so we can reuse for different metrics)
        print("\nSaving activations and metrics...")
        np.savez_compressed(
            activations_path,
            **{f"layer_{i}": acts for i, acts in activations.items()},
            **all_metrics  # Save all metrics for reuse
        )
        print(f"Saved to {activations_path}")

        # Compute accuracy stats for saving
        correct_count = sum(1 for m in metadata if m["is_correct"])
        accuracy = correct_count / len(metadata)

        # Save dataset with metadata
        output_data = {
            "config": {
                "dataset": DATASET_NAME,
                "num_questions": NUM_QUESTIONS,
                "base_model": BASE_MODEL_NAME,
                "seed": SEED,
            },
            "stats": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": len(metadata),
                **{f"{name}_mean": float(values.mean()) for name, values in all_metrics.items()},
                **{f"{name}_std": float(values.std()) for name, values in all_metrics.items()},
            },
            "data": metadata
        }
        with open(dataset_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved dataset to {dataset_path}")

    # Determine which metrics to train probes for
    if run_all_metrics:
        metrics_to_train = all_metrics  # All 5 metrics
    else:
        metrics_to_train = {METRIC: target}  # Just the selected one

    # Train probes for all requested metrics
    results, directions = run_all_probes(activations, metrics_to_train)

    # Compute accuracy from metadata if available (shared across all metrics)
    accuracy_stats = None
    if metadata:
        correct_count = sum(1 for m in metadata if m.get("is_correct", False))
        accuracy_stats = {
            "accuracy": correct_count / len(metadata),
            "correct_count": correct_count,
            "total_count": len(metadata),
        }

    # Save results and directions for each metric
    for metric_name in metrics_to_train.keys():
        metric_target = metrics_to_train[metric_name]
        layer_results = results[metric_name]
        layer_directions = directions[metric_name]

        # Generate metric-specific output paths
        metric_output_prefix = get_output_prefix(metric_name)
        metric_directions_path = Path(f"{metric_output_prefix}_directions.npz")
        metric_results_path = Path(f"{metric_output_prefix}_results.json")
        metric_plot_path = Path(f"{metric_output_prefix}_results.png")

        # Save directions
        directions_data = {
            f"layer_{layer_idx}": direction
            for layer_idx, direction in layer_directions.items()
        }
        directions_data["_metadata_dataset"] = np.array(DATASET_NAME)
        directions_data["_metadata_model"] = np.array(BASE_MODEL_NAME)
        directions_data["_metadata_metric"] = np.array(metric_name)
        np.savez_compressed(metric_directions_path, **directions_data)
        print(f"Saved {metric_name} directions to {metric_directions_path}")

        # Save results
        output_data = {
            "config": {
                "base_model": BASE_MODEL_NAME,
                "dataset": DATASET_NAME,
                "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
                "metric": metric_name,
                "train_split": TRAIN_SPLIT,
                "probe_alpha": PROBE_ALPHA,
                "use_pca": USE_PCA,
                "pca_components": PCA_COMPONENTS,
                "n_bootstrap": N_BOOTSTRAP,
                "seed": SEED,
            },
            "metric_stats": {
                "mean": float(metric_target.mean()),
                "std": float(metric_target.std()),
                "min": float(metric_target.min()),
                "max": float(metric_target.max()),
                "variance": float(metric_target.var()),
                "median": float(np.median(metric_target)),
                "iqr": float(np.percentile(metric_target, 75) - np.percentile(metric_target, 25)),
            },
            "results": {str(k): v for k, v in layer_results.items()}
        }

        if accuracy_stats:
            output_data["accuracy"] = accuracy_stats

        with open(metric_results_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {metric_name} results to {metric_results_path}")

        # Plot results for this metric
        plot_results({metric_name: layer_results}, metric_plot_path)

    # Print results summary (all metrics together)
    print_results(results)

    # Print diagnostic summary
    print_diagnostic_summary(metrics_to_train, results)


if __name__ == "__main__":
    main()
