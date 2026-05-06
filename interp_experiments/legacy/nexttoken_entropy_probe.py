"""
Run uncertainty prediction experiment on diverse text.

This script:
1. Loads the stratified dataset from build_nexttoken_dataset.py
2. Extracts activations from all layers of the model
3. Computes multiple uncertainty metrics (all saved, one or all probed per run):

   Prob-based (nonlinear - may be harder for linear probes):
   - entropy: Shannon entropy -sum(p * log(p))
   - top_prob: P(argmax) - probability of most likely token
   - margin: P(top) - P(second) - prob gap between top two

   Logit-based (linear - better aligned with linear probes):
   - logit_gap: z(top) - z(second) - logit gap between top two
   - top_logit: z(top) - mean(z) - centered top logit

4. Trains linear probes to predict the selected metric(s) from each layer
5. Saves directions: {prefix}_{metric}_directions.npz

Supports resuming from checkpoint if interrupted during extraction.

Usage:
    python nexttoken_entropy_probe.py                     # Probe entropy (default)
    python nexttoken_entropy_probe.py --metric logit_gap  # Probe logit_gap
    python nexttoken_entropy_probe.py --all-metrics       # Probe all metrics
    python nexttoken_entropy_probe.py --plot-only         # Load saved activations, retrain probes
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_model_short_name,
    LinearProbe,
    compute_entropy_from_probs,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
DATASET_PATH = None  # Auto-detect based on model name, or set explicitly
MAX_PROMPT_LENGTH = 500
SEED = 42
CHECKPOINT_INTERVAL = 200  # Save checkpoint every N prompts
BATCH_SIZE = 8  # Batch size for extraction

# Quantization (for large models like 70B)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100
N_BOOTSTRAP = 100  # Number of bootstrap iterations for confidence intervals

# Available uncertainty metrics:
# Prob-based (nonlinear targets - may be harder for linear probes):
#   entropy   - Shannon entropy -sum(p * log(p))
#   top_prob  - P(argmax) - probability of most likely answer
#   margin    - P(top) - P(second) - prob gap between top two
# Logit-based (linear targets - better aligned with linear probes):
#   logit_gap - z(top) - z(second) - logit gap between top two
#   top_logit - z(top) - mean(z) - centered top logit
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
METRIC = "entropy"  # Which metric to probe (set via --metric flag)

np.random.seed(SEED)
torch.manual_seed(SEED)


def compute_uncertainty_metrics(probs: np.ndarray, logits: np.ndarray = None) -> Dict[str, float]:
    """
    Compute multiple uncertainty metrics from probability and logit distributions.

    Args:
        probs: Probability distribution over next tokens (after softmax)
        logits: Raw logits (before softmax). If None, logit-based metrics
                will be computed from log(probs) as an approximation.

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


def find_dataset_path() -> Path:
    """Find the dataset file, auto-detecting based on model name."""
    if DATASET_PATH:
        return Path(DATASET_PATH)

    # Try model-specific path in outputs directory first
    model_short = get_model_short_name(BASE_MODEL_NAME)
    model_specific = OUTPUTS_DIR / f"{model_short}_nexttoken_entropy_dataset.json"
    if model_specific.exists():
        return model_specific

    # Fall back to generic name in outputs
    generic = OUTPUTS_DIR / "entropy_dataset.json"
    if generic.exists():
        return generic

    raise FileNotFoundError(
        f"Could not find dataset. Tried: {model_specific}, {generic}. "
        "Run build_nexttoken_dataset.py first."
    )


def load_dataset(path: Path) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Load the entropy dataset.

    Returns (data, config) where config may be None for old-format files.
    """
    print(f"Loading dataset from {path}...")
    with open(path) as f:
        raw = json.load(f)

    # Handle both old format (list) and new format (dict with config)
    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
        config = raw.get("config")
    else:
        data = raw
        config = None

    print(f"Loaded {len(data)} prompts")
    if config:
        print(f"  Dataset config: {config}")

    return data, config


def extract_all_activations_and_metrics(
    dataset: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    checkpoint_path: Path
) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """
    Extract activations from all layers for all prompts and compute uncertainty metrics.
    Supports resuming from checkpoint.
    Batched execution for speed.

    Returns:
        activations: Dict mapping layer_idx -> array of shape (num_samples, hidden_dim)
        metrics: Dict mapping metric_name -> array of shape (num_samples,)
        predicted_tokens: array of shape (num_samples,) with argmax token IDs
    """
    # Ensure pad token is set for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for existing checkpoint
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = np.load(checkpoint_path, allow_pickle=True)

        if "processed_count" in checkpoint.files:
            start_idx = int(checkpoint["processed_count"])
            all_layer_activations = {
                int(k.split("_")[1]): list(checkpoint[k])
                for k in checkpoint.files if k.startswith("layer_")
            }
            all_metrics = {metric: [] for metric in AVAILABLE_METRICS}
            for metric in AVAILABLE_METRICS:
                if metric in checkpoint.files:
                    all_metrics[metric] = list(checkpoint[metric])
                elif metric == "entropy" and "entropies" in checkpoint.files:
                    all_metrics["entropy"] = list(checkpoint["entropies"])
            
            if "predicted_tokens" in checkpoint.files:
                all_predicted_tokens = list(checkpoint["predicted_tokens"])
            else:
                all_predicted_tokens = []
            print(f"Resuming from prompt {start_idx}/{len(dataset)}")
        else:
            print("Warning: Old checkpoint format detected, starting fresh extraction")
            checkpoint_path.unlink()
            start_idx = 0
            all_layer_activations = {i: [] for i in range(num_layers)}
            all_metrics = {metric: [] for metric in AVAILABLE_METRICS}
            all_predicted_tokens = []
    else:
        start_idx = 0
        all_layer_activations = {i: [] for i in range(num_layers)}
        all_metrics = {metric: [] for metric in AVAILABLE_METRICS}
        all_predicted_tokens = []

    print(f"Extracting activations from {num_layers} layers (Batch Size: {BATCH_SIZE})...")
    model.eval()

    # Set up hooks for activation extraction
    activations_cache = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store batch of last token activations: (B, Hidden)
            activations_cache[layer_idx] = hidden_states[:, -1, :].detach()
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

    total_samples = len(dataset)
    
    try:
        # Process in batches
        for i in tqdm(range(start_idx, total_samples, BATCH_SIZE)):
            batch_items = dataset[i : min(i + BATCH_SIZE, total_samples)]
            texts = [item["text"] for item in batch_items]
            current_batch_size = len(texts)

            # Tokenize batch
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_LENGTH
            )
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            # Clear cache
            activations_cache.clear()

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # Process outputs for each item in batch
            # Get logits at last position for each sample
            # Attention mask handling: we need the position of the last real token
            # But the script assumes next-token prediction on the provided text, so last position is correct
            # if padding is on the right. Standard tokenizers pad right by default or config.
            # Assuming standard behavior: last column is the prediction for next token.
            
            # Extract metrics for the batch
            batch_logits = outputs.logits[:, -1, :].cpu().numpy()
            batch_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1).cpu().numpy()
            
            for b in range(current_batch_size):
                final_logits = batch_logits[b]
                probs = batch_probs[b]
                
                # Metrics
                item_metrics = compute_uncertainty_metrics(probs, final_logits)
                predicted_token = int(np.argmax(final_logits))
                
                all_predicted_tokens.append(predicted_token)
                for metric_name, metric_value in item_metrics.items():
                    all_metrics[metric_name].append(metric_value)

            # Extract activations
            for layer_idx, batch_acts in activations_cache.items():
                # batch_acts is (B, Hidden) on GPU (from hook)
                # Move to CPU and extend list
                all_layer_activations[layer_idx].extend(batch_acts.cpu().numpy())

            # Checkpoint
            if (i + current_batch_size) % CHECKPOINT_INTERVAL < BATCH_SIZE and (i + current_batch_size) < total_samples:
                save_checkpoint_with_metrics(
                    all_layer_activations, all_metrics, all_predicted_tokens, i + current_batch_size, checkpoint_path
                )

            # Clear memory
            del inputs, input_ids, attention_mask, outputs
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

    finally:
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

    predicted_tokens = np.array(all_predicted_tokens)

    print(f"Extracted activations shape (per layer): {activations[0].shape}")
    print(f"\nUncertainty metrics:")
    for metric_name, values in metrics.items():
        print(f"  {metric_name}: range=[{values.min():.3f}, {values.max():.3f}], "
              f"mean={values.mean():.3f}, std={values.std():.3f}")
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file")

    return activations, metrics, predicted_tokens


def save_checkpoint_with_metrics(
    layer_activations: Dict[int, List],
    metrics: Dict[str, List[float]],
    predicted_tokens: List[int],
    processed_count: int,
    checkpoint_path: Path
):
    """Save extraction checkpoint with all metrics and predicted tokens."""
    save_dict = {
        f"layer_{i}": np.array(acts)
        for i, acts in layer_activations.items()
    }
    for metric_name, values in metrics.items():
        save_dict[metric_name] = np.array(values)
    save_dict["predicted_tokens"] = np.array(predicted_tokens)
    save_dict["processed_count"] = np.array(processed_count)

    np.savez_compressed(checkpoint_path, **save_dict)
    print(f"  Checkpoint saved: {processed_count} prompts")


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
    """Train probe for a single layer with bootstrap.
    
    OPTIMIZATION: If use_pca is True, we fit PCA once on the full dataset 
    and transform X before the bootstrap loop. This avoids re-fitting PCA 100 times.
    """
    rng = np.random.RandomState(seed + layer_idx)
    n = len(targets)
    
    # Pre-compute PCA to speed up bootstrap
    X_for_bootstrap = X
    bootstrap_use_pca_flag = use_pca
    
    # Store the pre-computed PCA object if we use it, to reconstruct direction later (optional)
    # But for direction, we use the original slow method (fit on final split) to ensure exact
    # compatibility with core.LinearProbe's get_direction().
    if use_pca:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_comp = min(pca_components, X.shape[0], X.shape[1])
        pca_pre = PCA(n_components=n_comp)
        X_pca = pca_pre.fit_transform(X_scaled)
        
        # Use projected data for the repeated stats calculation
        X_for_bootstrap = X_pca
        bootstrap_use_pca_flag = False

    test_r2s = []
    test_maes = []

    # Bootstrap for confidence intervals (fast loop)
    for _ in range(n_bootstrap):
        # Random split
        indices = np.arange(n)
        rng.shuffle(indices)
        split_idx = int(n * train_split)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train = X_for_bootstrap[train_idx]
        X_test = X_for_bootstrap[test_idx]
        y_train = targets[train_idx]
        y_test = targets[test_idx]

        # Train probe (using pre-computed features if optimization was applied)
        probe = LinearProbe(
            alpha=alpha,
            use_pca=bootstrap_use_pca_flag,
            pca_components=pca_components
        )
        probe.fit(X_train, y_train)

        # Evaluate
        test_eval = probe.evaluate(X_test, y_test)
        test_r2s.append(test_eval["r2"])
        test_maes.append(test_eval["mae"])

    # Train final probe on canonical split for direction extraction
    # We use the ORIGINAL method (slow, potentially with internal PCA) here 
    # to ensure the direction vector is in the correct space and format expected by downstream tools.
    rng_final = np.random.RandomState(seed)
    indices = np.arange(n)
    rng_final.shuffle(indices)
    split_idx = int(n * train_split)
    train_idx = indices[:split_idx]

    final_probe = LinearProbe(
        alpha=alpha,
        use_pca=use_pca, # Use original setting
        pca_components=pca_components
    )
    final_probe.fit(X[train_idx], targets[train_idx])
    direction = final_probe.get_direction()

    return layer_idx, {
        "test_r2_mean": float(np.mean(test_r2s)),
        "test_r2_std": float(np.std(test_r2s)),
        "test_mae_mean": float(np.mean(test_maes)),
        "test_mae_std": float(np.std(test_maes)),
    }, direction


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int, classes: np.ndarray) -> float:
    """Compute top-k accuracy."""
    top_k_indices = np.argsort(proba, axis=1)[:, -k:]
    top_k_classes = classes[top_k_indices]
    correct = np.any(top_k_classes == y_true[:, None], axis=1)
    return float(np.mean(correct))


def _train_token_probe_for_layer(
    layer_idx: int,
    X: np.ndarray,
    tokens: np.ndarray,
    train_split: float,
    seed: int,
    use_pca: bool,
    pca_components: int,
) -> Tuple[int, Dict]:
    """
    Train logistic regression probe to predict next token.
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)

    indices = np.arange(n)
    rng.shuffle(indices)
    split_idx = int(n * train_split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = tokens[train_idx], tokens[test_idx]

    # OPTIMIZATION: Manual Scaling and PCA to ensure efficiency
    # (Previously this was standard, but explicit is good)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if use_pca:
        n_components = min(pca_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
        # Use randomized solver for speed on large dims
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
    else:
        X_train_pca = X_train_scaled
        X_test_pca = X_test_scaled

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
    clf.fit(X_train_pca, y_train)

    top1_acc = clf.score(X_test_pca, y_test)
    proba = clf.predict_proba(X_test_pca)
    top5_acc = top_k_accuracy(proba, y_test, k=5, classes=clf.classes_)
    top10_acc = top_k_accuracy(proba, y_test, k=10, classes=clf.classes_)

    return layer_idx, {
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "top10_accuracy": float(top10_acc),
        "n_classes": len(clf.classes_),
    }


def run_token_prediction_probe(
    activations: Dict[int, np.ndarray],
    predicted_tokens: np.ndarray,
    n_jobs: int = -1,
    use_pca: bool = True,
    pca_components: int = PCA_COMPONENTS,
) -> Dict[int, Dict]:
    """Train logistic regression probes to predict next token."""
    layer_indices = sorted(activations.keys())
    n_unique = len(np.unique(predicted_tokens))
    print(f"\nTraining token prediction probes across {len(activations)} layers...")
    print(f"  Target: {len(predicted_tokens)} tokens, {n_unique} unique classes")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_train_token_probe_for_layer)(
            layer_idx,
            activations[layer_idx],
            predicted_tokens,
            TRAIN_SPLIT,
            SEED,
            use_pca,
            pca_components,
        )
        for layer_idx in layer_indices
    )
    return {layer_idx: result for layer_idx, result in results_list}


def run_all_probes(
    activations: Dict[int, np.ndarray],
    metrics: Dict[str, np.ndarray],
    n_jobs: int = -1,
    use_pca: bool = USE_PCA,
    pca_components: int = PCA_COMPONENTS,
    alpha: float = PROBE_ALPHA
) -> Tuple[Dict[str, Dict[int, Dict]], Dict[str, Dict[int, np.ndarray]]]:
    """Train probes for all layers and all metrics with bootstrap."""
    pca_str = f"PCA={pca_components}" if use_pca else "no PCA"
    layer_indices = sorted(activations.keys())

    all_results = {}
    all_directions = {}

    for metric_name, targets in metrics.items():
        print(f"\nTraining probes for metric '{metric_name}' across {len(activations)} layers "
              f"({N_BOOTSTRAP} bootstrap iterations, {pca_str})...")

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
        best_layer = max(results.keys(), key=lambda l: results[l]["test_r2_mean"])
        best_r2 = results[best_layer]["test_r2_mean"]
        best_std = results[best_layer]["test_r2_std"]
        print(f"Best layer: {best_layer} (Test R² = {best_r2:.4f} ± {best_std:.4f})")

    print("\n" + "="*80)
    print("METRIC COMPARISON (Best R² per metric)")
    print("="*80)
    for metric_name, results in all_results.items():
        best_layer = max(results.keys(), key=lambda l: results[l]["test_r2_mean"])
        best_r2 = results[best_layer]["test_r2_mean"]
        print(f"  {metric_name:<12}: layer {best_layer:<3} R² = {best_r2:.4f}")


def plot_results(
    all_results: Dict[str, Dict[int, Dict]],
    output_path: Path,
    token_results: Optional[Dict[int, Dict]] = None
):
    """Plot R² and token accuracy across layers."""
    metric_names = list(all_results.keys())
    n_metrics = len(metric_names)
    layers = sorted(all_results[metric_names[0]].keys())
    colors_r2 = ['tab:blue', 'tab:purple', 'tab:cyan', 'tab:brown', 'tab:pink']

    if n_metrics == 1:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1 = axes[0]

    for i, metric_name in enumerate(metric_names):
        results = all_results[metric_name]
        test_r2_mean = [results[l]["test_r2_mean"] for l in layers]
        test_r2_std = [results[l]["test_r2_std"] for l in layers]
        color = colors_r2[i % len(colors_r2)]
        ax1.plot(layers, test_r2_mean, 'o-', label=f'{metric_name} R²',
                 color=color, markersize=4, linewidth=2)
        ax1.fill_between(layers,
                         np.array(test_r2_mean) - np.array(test_r2_std),
                         np.array(test_r2_mean) + np.array(test_r2_std),
                         alpha=0.2, color=color)

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.grid(True, alpha=0.3)

    if token_results is not None:
        ax2 = ax1.twinx()
        top1 = [token_results[l]["top1_accuracy"] for l in layers]
        top5 = [token_results[l]["top5_accuracy"] for l in layers]
        top10 = [token_results[l]["top10_accuracy"] for l in layers]

        ax2.plot(layers, top1, 'd-', label='Token Top-1', color='tab:green', markersize=4, linewidth=2)
        ax2.plot(layers, top5, 's-', label='Token Top-5', color='tab:orange', markersize=4, linewidth=2)
        ax2.plot(layers, top10, '^-', label='Token Top-10', color='tab:red', markersize=4, linewidth=2)
        ax2.set_ylabel('Accuracy')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax1.legend(loc='best')

    ax1.set_title('Probe Performance by Layer')

    if n_metrics > 1:
        ax_bar = axes[1]
        best_r2s = []
        best_layers = []
        for metric_name in metric_names:
            results = all_results[metric_name]
            best_layer = max(results.keys(), key=lambda l: results[l]["test_r2_mean"])
            best_r2s.append(results[best_layer]["test_r2_mean"])
            best_layers.append(best_layer)

        bars = ax_bar.bar(metric_names, best_r2s,
                          color=[colors_r2[i % len(colors_r2)] for i in range(n_metrics)])
        ax_bar.set_xlabel('Metric')
        ax_bar.set_ylabel('Best R² Score')
        ax_bar.set_title('Best R² by Metric')
        ax_bar.grid(True, alpha=0.3, axis='y')

        for bar, layer in zip(bars, best_layers):
            height = bar.get_height()
            ax_bar.annotate(f'L{layer}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


def plot_entropy_distribution(entropies: np.ndarray, output_path: Path):
    """Plot entropy distribution histogram."""
    _, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropies, bins=30, edgecolor='black', alpha=0.7,
            weights=np.ones(len(entropies)) / len(entropies) * 100)
    ax.axvline(entropies.mean(), color='red', linestyle='--',
               label=f'Mean: {entropies.mean():.3f}')
    ax.axvline(np.median(entropies), color='orange', linestyle='--',
               label=f'Median: {np.median(entropies):.3f}')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Next-Token Entropy Distribution (n={len(entropies)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy distribution plot saved to {output_path}")


def load_activations(activations_path: Path) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Load activations, metrics, and predicted tokens from saved file."""
    print(f"Loading activations from {activations_path}...")
    data = np.load(activations_path)

    activations = {
        int(k.split("_")[1]): data[k]
        for k in data.files if k.startswith("layer_")
    }

    metrics = {}
    for metric_name in AVAILABLE_METRICS:
        if metric_name in data.files:
            metrics[metric_name] = data[metric_name]
        elif metric_name == "entropy" and "entropies" in data.files:
            metrics["entropy"] = data["entropies"]

    if not metrics:
        raise ValueError(f"No metrics found in {activations_path}. Re-run without --plot-only.")

    predicted_tokens = data["predicted_tokens"] if "predicted_tokens" in data.files else None
    print(f"Loaded {len(activations)} layers, {len(list(metrics.values())[0])} samples, {len(metrics)} metrics")
    return activations, metrics, predicted_tokens


def get_base_output_prefix() -> str:
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_nexttoken")
    return str(OUTPUTS_DIR / f"{model_short}_nexttoken")


def get_metric_output_prefix(metric: str) -> str:
    return f"{get_base_output_prefix()}_{metric}"


def main():
    global METRIC

    parser = argparse.ArgumentParser(description="Train uncertainty probes on next-token prediction")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Metric to probe (default: {METRIC})")
    parser.add_argument("--all-metrics", action="store_true",
                        help="Train probes for all metrics")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip extraction, load saved activations and retrain probes")
    parser.add_argument("--no-token-probe", action="store_true",
                        help="Skip token prediction probe")
    args = parser.parse_args()

    METRIC = args.metric
    run_all_metrics = args.all_metrics
    print(f"Device: {DEVICE}")
    if run_all_metrics:
        print(f"Metrics: ALL ({', '.join(AVAILABLE_METRICS)})")
    else:
        print(f"Metric: {METRIC}")

    base_prefix = get_base_output_prefix()
    print(f"Base output prefix: {base_prefix}")

    activations_path = Path(f"{base_prefix}_activations.npz")
    entropy_dist_path = Path(f"{base_prefix}_entropy_distribution.png")
    checkpoint_path = Path(f"{base_prefix}_checkpoint.npz")

    if args.plot_only:
        if not activations_path.exists():
            raise FileNotFoundError(f"Activations file not found: {activations_path}")
        activations, all_metrics, predicted_tokens = load_activations(activations_path)
    else:
        model, tokenizer, num_layers = load_model_and_tokenizer(
            BASE_MODEL_NAME,
            adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        )
        dataset_path = find_dataset_path()
        dataset, dataset_config = load_dataset(dataset_path)

        activations, all_metrics, predicted_tokens = extract_all_activations_and_metrics(
            dataset, model, tokenizer, num_layers, checkpoint_path
        )

        print("\nSaving activations and metrics...")
        np.savez_compressed(
            activations_path,
            **{f"layer_{i}": acts for i, acts in activations.items()},
            **all_metrics,
            predicted_tokens=predicted_tokens,
        )
        print(f"Saved to {activations_path}")

    if run_all_metrics:
        metrics_to_train = all_metrics
    else:
        if METRIC not in all_metrics:
            raise ValueError(f"Metric '{METRIC}' not found in saved file.")
        metrics_to_train = {METRIC: all_metrics[METRIC]}

    results, directions = run_all_probes(activations, metrics_to_train)

    token_results = None
    if not args.no_token_probe and predicted_tokens is not None:
        print("\n" + "=" * 60)
        print("RUNNING NEXT-TOKEN PREDICTION PROBE")
        print("=" * 60)

        token_results = run_token_prediction_probe(
            activations, predicted_tokens,
            n_jobs=-1, use_pca=USE_PCA, pca_components=PCA_COMPONENTS
        )

        token_results_path = Path(f"{base_prefix}_token_prediction_results.json")
        token_output_data = {
            "config": {
                "base_model": BASE_MODEL_NAME,
                "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
                "use_pca": USE_PCA,
                "pca_components": PCA_COMPONENTS,
                "seed": SEED,
            },
            "n_unique_tokens": int(len(np.unique(predicted_tokens))),
            "results": {str(k): v for k, v in token_results.items()}
        }
        with open(token_results_path, "w") as f:
            json.dump(token_output_data, f, indent=2)
        print(f"Saved token prediction results to {token_results_path}")

    for metric_name in metrics_to_train.keys():
        metric_target = metrics_to_train[metric_name]
        layer_results = results[metric_name]
        layer_directions = directions[metric_name]
        metric_prefix = get_metric_output_prefix(metric_name)
        
        # Save directions
        np.savez_compressed(
            f"{metric_prefix}_directions.npz",
            **{f"layer_{i}": d for i, d in layer_directions.items()},
            _metadata_model=np.array(BASE_MODEL_NAME),
            _metadata_metric=np.array(metric_name)
        )

        # Save results
        output_data = {
            "config": {"metric": metric_name, "train_split": TRAIN_SPLIT, "seed": SEED},
            "metric_stats": {
                "mean": float(metric_target.mean()),
                "std": float(metric_target.std()),
            },
            "results": {str(k): v for k, v in layer_results.items()}
        }
        with open(f"{metric_prefix}_results.json", "w") as f:
            json.dump(output_data, f, indent=2)

        plot_results({metric_name: layer_results}, Path(f"{metric_prefix}_results.png"), token_results=token_results)

    print_results(results)

    if "entropy" in all_metrics:
        plot_entropy_distribution(all_metrics["entropy"], entropy_dist_path)


if __name__ == "__main__":
    main()