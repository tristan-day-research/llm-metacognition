"""
Create contrast directions for uncertainty analysis.

This module computes two types of contrast vectors in residual-stream space:
1. Entropy-based mean-diff contrast: High vs low output entropy (existing method)
2. Confidence-based ordinal contrast: Predicts stated confidence using ridge regression

Both contrasts are computed per-layer and saved with full metadata.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from scipy.stats import pearsonr

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    should_use_chat_template,
    BatchedExtractor,
    compute_mc_metrics,
)
from core.questions import load_questions
from prompts import (
    format_direct_prompt,
    format_stated_confidence_prompt,
    format_direct_prompt_base,
    format_stated_confidence_prompt_base,
    STATED_CONFIDENCE_MIDPOINTS,
    STATED_CONFIDENCE_OPTIONS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Which model type to run: "base", "instruct", or "finetuned"
MODEL_TYPE = "base"

MODEL_CONFIGS = {
    "base": {
        "model": "meta-llama/Llama-3.1-8B",
        "adapter": None,
    },
    "instruct": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter": None,
    },
    "finetuned": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter": "Tristan-Day/ect_20251222_215412_v0uei7y1_2000",
    },
}

# Few-shot mode for base model (ignored for instruct/finetuned):
# "fixed" performed best on ECT metrics (ECE=0.076, AUROC=0.706)
FEW_SHOT_MODE = "fixed"

# Path to JSON with predictions for random/balanced/deceptive_examples modes
RANDOM_FEW_SHOT_SOURCE = "outputs/ECT/Llama-3.1-8B-Instruct_adapter-ect_20251222_215412_v0uei7y1_2000_TriviaMC_ect_results.json"

DATASET = "TriviaMC"
NUM_QUESTIONS = 500
SEED = 42
BATCH_SIZE = 8

# Quantization
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

# Contrast parameters
ENTROPY_QUANTILE = 0.25  # For mean-diff: use top/bottom 25%
# Both contrasts now use the same quantile-based mean-diff approach

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"

# =============================================================================
# ENTROPY CONTRAST (MEAN-DIFF)
# =============================================================================

def compute_entropy_mean_diff_contrast(
    activations_by_layer: Dict[int, np.ndarray],
    entropy_values: np.ndarray,
    quantile: float = 0.25,
) -> Tuple[Dict[int, np.ndarray], Dict]:
    """
    Compute mean-diff contrast for entropy: mean(high_entropy) - mean(low_entropy).
    
    This is the existing method from identify_mc_correlate.py.
    
    Args:
        activations_by_layer: {layer: (n_samples, hidden_dim)}
        entropy_values: (n_samples,) entropy values
        quantile: Fraction for top/bottom groups
        
    Returns:
        directions: {layer: direction_vector}
        metadata: Dict with statistics
    """
    num_layers = len(activations_by_layer)
    directions = {}
    layer_metadata = []
    
    n = len(entropy_values)
    n_group = max(1, int(n * quantile))
    
    # Get top and bottom quantile indices
    sorted_idx = np.argsort(entropy_values)
    low_idx = sorted_idx[:n_group]
    high_idx = sorted_idx[-n_group:]
    
    print(f"\nComputing entropy mean-diff contrast:")
    print(f"  Total samples: {n}")
    print(f"  Quantile: {quantile} (n={n_group} per group)")
    print(f"  Low entropy: mean={entropy_values[low_idx].mean():.3f}, std={entropy_values[low_idx].std():.3f}")
    print(f"  High entropy: mean={entropy_values[high_idx].mean():.3f}, std={entropy_values[high_idx].std():.3f}")
    
    for layer in tqdm(range(num_layers), desc="  Layers"):
        X = activations_by_layer[layer]
        
        # Compute mean difference
        mean_low = X[low_idx].mean(axis=0)
        mean_high = X[high_idx].mean(axis=0)
        
        direction = mean_high - mean_low
        magnitude = np.linalg.norm(direction)
        direction_norm = direction / (magnitude + 1e-10)
        
        # Compute within-cluster tightness (mean distance to centroid)
        # Note: This is within-cluster distance, NOT between-cluster distance
        distances_low = np.linalg.norm(X[low_idx] - mean_low, axis=1)
        distances_high = np.linalg.norm(X[high_idx] - mean_high, axis=1)
        tightness_low = float(distances_low.mean())
        tightness_high = float(distances_high.mean())
        
        # Evaluate: project all samples and correlate with entropy
        projections = X @ direction_norm
        corr, p_value = pearsonr(entropy_values, projections)
        r2 = float(corr ** 2)
        
        directions[layer] = direction_norm
        
        layer_metadata.append({
            "layer": layer,
            "magnitude": float(magnitude),
            "signal_strength": float(magnitude),  # between-cluster distance
            "norm": float(np.linalg.norm(direction_norm)),
            "r2": r2,
            "corr": float(corr),
            "corr_pvalue": float(p_value),
            "p_value": float(p_value),
            "n_low": int(n_group),
            "n_high": int(n_group),
            "entropy_mean_low": float(entropy_values[low_idx].mean()),
            "entropy_mean_high": float(entropy_values[high_idx].mean()),
            "cluster_tightness_low": tightness_low,  # within-cluster distance
            "cluster_tightness_high": tightness_high,  # within-cluster distance
        })
    
    metadata = {
        "method": "mean_diff",
        "target": "entropy",
        "quantile": quantile,
        "num_samples": n,
        "num_layers": num_layers,
        "entropy_stats": {
            "mean": float(entropy_values.mean()),
            "std": float(entropy_values.std()),
            "min": float(entropy_values.min()),
            "max": float(entropy_values.max()),
            "median": float(np.median(entropy_values)),
        },
        "per_layer": layer_metadata,
    }
    
    return directions, metadata


# =============================================================================
# CONFIDENCE CONTRAST (MEAN-DIFF)
# =============================================================================

def compute_confidence_mean_diff_contrast(
    activations_by_layer: Dict[int, np.ndarray],
    confidence_tokens: list,  # List of token strings (S, T, U, etc.)
    quantile: float = 0.25,
) -> Tuple[Dict[int, np.ndarray], Dict]:
    """
    Compute mean-diff contrast for stated confidence.
    
    More robust than ridge regression for imbalanced data.
    Uses quantile-based mean difference (same approach as entropy).
    
    Direction = mean(high_confidence) - mean(low_confidence)
    
    Args:
        activations_by_layer: {layer: (n_samples, hidden_dim)}
        confidence_tokens: List of confidence tokens (e.g., ["Z", "X", "S", ...])
        quantile: Fraction for top/bottom groups (default: 0.25)
        
    Returns:
        directions: {layer: direction_vector}
        metadata: Dict with statistics
    """
    num_layers = len(activations_by_layer)
    directions = {}
    layer_metadata = []
    
    # Convert tokens to scalar values using midpoint mapping
    confidence_values = np.array([
        STATED_CONFIDENCE_MIDPOINTS.get(token, 0.5)  # Default to 0.5 if unknown
        for token in confidence_tokens
    ])
    
    n = len(confidence_values)
    n_group = max(1, int(n * quantile))
    
    # Get top and bottom quantile indices (sorted by confidence VALUE, not token)
    sorted_idx = np.argsort(confidence_values)
    low_idx = sorted_idx[:n_group]
    high_idx = sorted_idx[-n_group:]
    
    # Count token distribution
    token_counts = {}
    for token in STATED_CONFIDENCE_OPTIONS.keys():
        count = sum(1 for t in confidence_tokens if t == token)
        token_counts[token] = count
    
    print(f"\nComputing confidence mean-diff contrast:")
    print(f"  Total samples: {n}")
    print(f"  Quantile: {quantile} (n={n_group} per group)")
    print(f"  Confidence stats: mean={confidence_values.mean():.3f}, std={confidence_values.std():.3f}")
    print(f"  Range: [{confidence_values.min():.3f}, {confidence_values.max():.3f}]")
    print(f"  Low confidence: mean={confidence_values[low_idx].mean():.3f}, std={confidence_values[low_idx].std():.3f}")
    print(f"  High confidence: mean={confidence_values[high_idx].mean():.3f}, std={confidence_values[high_idx].std():.3f}")
    print(f"  Token distribution: {token_counts}")
    
    for layer in tqdm(range(num_layers), desc="  Layers"):
        X = activations_by_layer[layer]
        
        # Compute mean difference
        mean_low = X[low_idx].mean(axis=0)
        mean_high = X[high_idx].mean(axis=0)
        
        direction = mean_high - mean_low
        magnitude = np.linalg.norm(direction)
        direction_norm = direction / (magnitude + 1e-10)
        
        # Compute within-cluster tightness (mean distance to centroid)
        # Note: This is within-cluster distance, NOT between-cluster distance
        distances_low = np.linalg.norm(X[low_idx] - mean_low, axis=1)
        distances_high = np.linalg.norm(X[high_idx] - mean_high, axis=1)
        tightness_low = float(distances_low.mean())
        tightness_high = float(distances_high.mean())
        
        # Evaluate: project all samples and correlate with confidence
        projections = X @ direction_norm
        corr, p_value = pearsonr(confidence_values, projections)
        r2 = float(corr ** 2)
        
        directions[layer] = direction_norm
        
        layer_metadata.append({
            "layer": layer,
            "magnitude": float(magnitude),
            "signal_strength": float(magnitude),  # between-cluster distance
            "norm": float(np.linalg.norm(direction_norm)),
            "r2": r2,
            "corr": float(corr),
            "corr_pvalue": float(p_value),
            "p_value": float(p_value),
            "n_low": int(n_group),
            "n_high": int(n_group),
            "confidence_mean_low": float(confidence_values[low_idx].mean()),
            "confidence_mean_high": float(confidence_values[high_idx].mean()),
            "cluster_tightness_low": tightness_low,  # within-cluster distance
            "cluster_tightness_high": tightness_high,  # within-cluster distance
        })
    
    metadata = {
        "method": "mean_diff",
        "target": "stated_confidence",
        "quantile": quantile,
        "num_samples": n,
        "num_layers": num_layers,
        "token_midpoints": STATED_CONFIDENCE_MIDPOINTS,
        "token_distribution": token_counts,
        "confidence_stats": {
            "mean": float(confidence_values.mean()),
            "std": float(confidence_values.std()),
            "min": float(confidence_values.min()),
            "max": float(confidence_values.max()),
            "median": float(np.median(confidence_values)),
        },
        "per_layer": layer_metadata,
    }
    
    return directions, metadata


# =============================================================================
# COSINE SIMILARITY ANALYSIS
# =============================================================================

def compute_cosine_similarity_analysis(
    entropy_directions: Dict[int, np.ndarray],
    confidence_directions: Dict[int, np.ndarray],
    entropy_metadata: Dict,
    confidence_metadata: Dict,
) -> Tuple[Dict, plt.Figure]:
    """
    Compute cosine similarity between entropy and confidence directions per layer.
    
    Args:
        entropy_directions: {layer: direction_vector} for entropy
        confidence_directions: {layer: direction_vector} for confidence
        entropy_metadata: Metadata dict for entropy contrast
        confidence_metadata: Metadata dict for confidence contrast
        
    Returns:
        analysis_dict: Dict with cosine similarities and statistics
        fig: Matplotlib figure with plot
    """
    # Find common layers
    entropy_layers = set(entropy_directions.keys())
    conf_layers = set(confidence_directions.keys())
    common_layers = sorted(entropy_layers & conf_layers)
    
    missing_entropy = sorted(conf_layers - entropy_layers)
    missing_conf = sorted(entropy_layers - conf_layers)
    
    if missing_entropy or missing_conf:
        print(f"\n  Warning: Layer mismatch detected")
        if missing_entropy:
            print(f"    Missing from entropy: {missing_entropy}")
        if missing_conf:
            print(f"    Missing from confidence: {missing_conf}")
    
    print(f"\n  Computing cosine similarity for {len(common_layers)} layers...")
    
    cosines = []
    abs_cosines = []
    
    for layer in common_layers:
        d_entropy = entropy_directions[layer]
        d_conf = confidence_directions[layer]
        
        # Verify shapes match
        assert d_entropy.shape == d_conf.shape, \
            f"Layer {layer}: Shape mismatch {d_entropy.shape} vs {d_conf.shape}"
        
        # Verify unit norms (and normalize if needed)
        norm_entropy = np.linalg.norm(d_entropy)
        norm_conf = np.linalg.norm(d_conf)
        
        if abs(norm_entropy - 1.0) > 1e-3:
            print(f"    Warning: Layer {layer} entropy norm={norm_entropy:.6f}, normalizing")
            d_entropy = d_entropy / norm_entropy
        
        if abs(norm_conf - 1.0) > 1e-3:
            print(f"    Warning: Layer {layer} confidence norm={norm_conf:.6f}, normalizing")
            d_conf = d_conf / norm_conf
        
        # Compute cosine similarity
        cos_sim = float(np.dot(d_entropy, d_conf))
        cosines.append(cos_sim)
        abs_cosines.append(abs(cos_sim))
    
    # Convert to arrays for analysis
    cosines = np.array(cosines)
    abs_cosines = np.array(abs_cosines)
    
    # Find extrema
    max_abs_idx = int(np.argmax(abs_cosines))
    max_idx = int(np.argmax(cosines))
    min_idx = int(np.argmin(cosines))
    
    # Build analysis dict
    analysis = {
        "analysis": "cosine_similarity_entropy_vs_confidence",
        "entropy_direction_source": "entropy_mean_diff",
        "confidence_direction_source": "confidence_mean_diff",
        "layer_count": len(common_layers),
        "layers": common_layers,
        "cosine": cosines.tolist(),
        "abs_cosine": abs_cosines.tolist(),
        "max_abs_cosine": {
            "layer": common_layers[max_abs_idx],
            "value": float(abs_cosines[max_abs_idx]),
        },
        "max_cosine": {
            "layer": common_layers[max_idx],
            "value": float(cosines[max_idx]),
        },
        "min_cosine": {
            "layer": common_layers[min_idx],
            "value": float(cosines[min_idx]),
        },
        "statistics": {
            "mean_cosine": float(np.mean(cosines)),
            "std_cosine": float(np.std(cosines)),
            "mean_abs_cosine": float(np.mean(abs_cosines)),
            "median_cosine": float(np.median(cosines)),
        },
        "notes": {
            "entropy_method": entropy_metadata.get("method", "unknown"),
            "confidence_method": confidence_metadata.get("method", "unknown"),
            "intersection_only": len(missing_entropy) > 0 or len(missing_conf) > 0,
            "missing_entropy_layers": missing_entropy,
            "missing_conf_layers": missing_conf,
        }
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(common_layers, cosines, 'o-', linewidth=2, markersize=4, color='tab:blue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=-0.2, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Cosine Similarity: Entropy (MC Questions) vs Stated Confidence Direction', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (
        f"Mean: {analysis['statistics']['mean_cosine']:.3f}\n"
        f"Std: {analysis['statistics']['std_cosine']:.3f}\n"
        f"Max: {analysis['max_cosine']['value']:.3f} (L{analysis['max_cosine']['layer']})\n"
        f"Min: {analysis['min_cosine']['value']:.3f} (L{analysis['min_cosine']['layer']})"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return analysis, fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Resolve configuration ---
    config = MODEL_CONFIGS[MODEL_TYPE]
    MODEL = config["model"]
    ADAPTER = config["adapter"]
    is_base = MODEL_TYPE == "base"

    model_short = get_model_short_name(MODEL)
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        base_name = f"{model_short}_adapter-{adapter_short}_{DATASET}"
    else:
        base_name = f"{model_short}_{DATASET}"

    print(f"Model type: {MODEL_TYPE}")
    print(f"Model: {MODEL}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    if is_base:
        print(f"Few-shot mode: {FEW_SHOT_MODE}")
    print(f"Dataset: {DATASET}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Output base: {base_name}")
    print()

    # --- Configure few-shot pool for base model ---
    few_shot_pool = None
    if is_base and FEW_SHOT_MODE in ["random", "balanced", "deceptive_examples"]:
        print(f"Loading few-shot examples from: {RANDOM_FEW_SHOT_SOURCE}")
        with open(RANDOM_FEW_SHOT_SOURCE, "r") as f:
            source_data = json.load(f)
        few_shot_pool = []
        for item in source_data["data"]:
            few_shot_pool.append({
                "question": item["question"],
                "options": item["options"],
                "mc_answer": item["predicted_answer"],
                "confidence": item["stated_confidence_response"],
            })
        print(f"  Loaded {len(few_shot_pool)} examples for sampling")
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
    # Base model prompts are raw text and need BOS prepended during tokenization.
    # Instruct/finetuned prompts go through chat template which already includes BOS.
    add_bos = is_base
    print(f"  Layers: {num_layers}")
    print(f"  Chat template: {use_chat_template}")
    print(f"  Base model mode: {is_base} (add_bos={add_bos})")

    # Load questions
    print(f"\nLoading questions from {DATASET}...")
    questions = load_questions(DATASET, num_questions=NUM_QUESTIONS, seed=SEED)
    print(f"  Loaded {len(questions)} questions")
    
    # =============================================================================
    # PART 1: EXTRACT DIRECT MC ACTIVATIONS AND COMPUTE ENTROPY
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("PART 1: EXTRACTING DIRECT MC ACTIVATIONS")
    print("=" * 80)
    
    # Get option token IDs
    option_keys = list(questions[0]["options"].keys())
    option_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in option_keys]
    print(f"  Option tokens: {dict(zip(option_keys, option_token_ids))}")
    
    # Extract activations and compute entropy
    print(f"\nExtracting activations (batch_size={BATCH_SIZE})...")
    
    direct_activations = {layer: [] for layer in range(num_layers)}
    all_probs = []
    all_logits = []
    
    with BatchedExtractor(model, num_layers) as extractor:
        for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
            batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

            prompts = []
            for q in batch_questions:
                if is_base:
                    prompt, _ = format_direct_prompt_base(q, FEW_SHOT_MODE, few_shot_pool)
                else:
                    prompt, _ = format_direct_prompt(q, tokenizer, use_chat_template)
                prompts.append(prompt)

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=add_bos,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            layer_acts_by_pos, probs, logits, _ = extractor.extract_batch(
                input_ids, attention_mask, option_token_ids
            )

            # Extract final position activations
            for item_acts in layer_acts_by_pos["final"]:
                for layer, act in item_acts.items():
                    direct_activations[layer].append(act)

            for p, l in zip(probs, logits):
                all_probs.append(p)
                all_logits.append(l)
    
    # Stack activations
    print("\nStacking activations...")
    direct_activations_by_layer = {
        layer: np.stack(acts) for layer, acts in direct_activations.items()
    }
    print(f"  Shape per layer: {direct_activations_by_layer[0].shape}")
    
    # Compute entropy
    print("\nComputing entropy...")
    all_probs_arr = np.array(all_probs)
    all_logits_arr = np.array(all_logits)
    metrics = compute_mc_metrics(all_probs_arr, all_logits_arr, metrics=["entropy"])
    entropy_values = metrics["entropy"]
    print(f"  Entropy: mean={entropy_values.mean():.3f}, std={entropy_values.std():.3f}")
    
    # =============================================================================
    # PART 2: EXTRACT CONFIDENCE TASK ACTIVATIONS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("PART 2: EXTRACTING CONFIDENCE TASK ACTIVATIONS")
    print("=" * 80)
    
    # Get confidence option token IDs
    conf_options = list(STATED_CONFIDENCE_OPTIONS.keys())
    conf_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in conf_options]
    print(f"  Confidence tokens: {dict(zip(conf_options, conf_token_ids))}")
    
    # Extract confidence activations
    print(f"\nExtracting confidence activations (batch_size={BATCH_SIZE})...")
    
    confidence_activations = {layer: [] for layer in range(num_layers)}
    all_conf_probs = []
    confidence_tokens = []
    
    with BatchedExtractor(model, num_layers) as extractor:
        for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
            batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

            prompts = []
            for q in batch_questions:
                if is_base:
                    prompt, _ = format_stated_confidence_prompt_base(q, FEW_SHOT_MODE, few_shot_pool)
                else:
                    prompt, _ = format_stated_confidence_prompt(q, tokenizer, use_chat_template)
                prompts.append(prompt)

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=add_bos,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            layer_acts_by_pos, probs, _, _ = extractor.extract_batch(
                input_ids, attention_mask, conf_token_ids
            )

            # Extract final position activations
            for item_acts in layer_acts_by_pos["final"]:
                for layer, act in item_acts.items():
                    confidence_activations[layer].append(act)

            for p in probs:
                all_conf_probs.append(p)
                # Get predicted token
                predicted_idx = np.argmax(p)
                confidence_tokens.append(conf_options[predicted_idx])
    
    # Stack activations
    print("\nStacking confidence activations...")
    confidence_activations_by_layer = {
        layer: np.stack(acts) for layer, acts in confidence_activations.items()
    }
    print(f"  Shape per layer: {confidence_activations_by_layer[0].shape}")
    
    # Compute confidence statistics
    confidence_values = np.array([
        STATED_CONFIDENCE_MIDPOINTS.get(token, 0.5)
        for token in confidence_tokens
    ])
    print(f"  Confidence: mean={confidence_values.mean():.3f}, std={confidence_values.std():.3f}")
    
    # =============================================================================
    # PART 3: COMPUTE CONTRASTS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("PART 3: COMPUTING CONTRASTS")
    print("=" * 80)
    
    # Entropy mean-diff contrast
    entropy_directions, entropy_metadata = compute_entropy_mean_diff_contrast(
        direct_activations_by_layer,
        entropy_values,
        quantile=ENTROPY_QUANTILE,
    )
    
    # Find best layer for entropy
    best_entropy_layer = max(
        range(num_layers),
        key=lambda l: entropy_metadata["per_layer"][l]["r2"]
    )
    print(f"\n  Best entropy layer: {best_entropy_layer} "
          f"(R²={entropy_metadata['per_layer'][best_entropy_layer]['r2']:.3f})")
    
    # Confidence mean-diff contrast
    confidence_directions, confidence_metadata = compute_confidence_mean_diff_contrast(
        confidence_activations_by_layer,
        confidence_tokens,
        quantile=ENTROPY_QUANTILE,  # Use same quantile as entropy for consistency
    )
    
    # Find best layer for confidence
    best_conf_layer = max(
        range(num_layers),
        key=lambda l: confidence_metadata["per_layer"][l]["r2"]
    )
    print(f"\n  Best confidence layer: {best_conf_layer} "
          f"(R²={confidence_metadata['per_layer'][best_conf_layer]['r2']:.3f})")
    
    # Cosine similarity analysis
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY ANALYSIS")
    print("=" * 80)
    
    cosine_analysis, cosine_fig = compute_cosine_similarity_analysis(
        entropy_directions,
        confidence_directions,
        entropy_metadata,
        confidence_metadata,
    )
    
    print(f"\n  Cosine similarity statistics:")
    print(f"    Mean: {cosine_analysis['statistics']['mean_cosine']:.3f}")
    print(f"    Std: {cosine_analysis['statistics']['std_cosine']:.3f}")
    print(f"    Range: [{cosine_analysis['min_cosine']['value']:.3f}, "
          f"{cosine_analysis['max_cosine']['value']:.3f}]")
    print(f"    Max abs: {cosine_analysis['max_abs_cosine']['value']:.3f} "
          f"(layer {cosine_analysis['max_abs_cosine']['layer']})")
    
    # =============================================================================
    # PART 4: SAVE RESULTS
    # =============================================================================

    print("\n" + "=" * 80)
    print("PART 4: SAVING RESULTS")
    print("=" * 80)

    # Use subdirectory matching existing convention:
    #   8b_instruct_entropy/confidence_contrast/
    #   8b_FT_entropy/confidence_contrast/
    #   8b_base_entropy/confidence_contrast/
    subdir_map = {
        "base": "8b_base_entropy",
        "instruct": "8b_instruct_entropy",
        "finetuned": "8b_FT_entropy",
    }
    save_dir = OUTPUT_DIR / subdir_map[MODEL_TYPE] / "confidence_contrast"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save entropy contrast
    entropy_json_path = save_dir / f"{base_name}_entropy_contrast.json"
    entropy_pt_path = save_dir / f"{base_name}_entropy_contrast_directions.pt"
    
    print(f"\nSaving entropy contrast...")
    with open(entropy_json_path, "w") as f:
        json.dump(entropy_metadata, f, indent=2)
    print(f"  Saved: {entropy_json_path}")
    
    # Convert to torch tensor and save
    entropy_tensor = torch.stack([
        torch.from_numpy(entropy_directions[l].astype(np.float32))
        for l in range(num_layers)
    ])
    torch.save(entropy_tensor, entropy_pt_path)
    print(f"  Saved: {entropy_pt_path}")
    print(f"  Shape: {entropy_tensor.shape}")
    
    # Save confidence contrast
    conf_json_path = save_dir / f"{base_name}_confidence_contrast.json"
    conf_pt_path = save_dir / f"{base_name}_confidence_contrast_directions.pt"
    
    print(f"\nSaving confidence contrast...")
    with open(conf_json_path, "w") as f:
        json.dump(confidence_metadata, f, indent=2)
    print(f"  Saved: {conf_json_path}")
    
    # Convert to torch tensor and save
    confidence_tensor = torch.stack([
        torch.from_numpy(confidence_directions[l].astype(np.float32))
        for l in range(num_layers)
    ])
    torch.save(confidence_tensor, conf_pt_path)
    print(f"  Saved: {conf_pt_path}")
    print(f"  Shape: {confidence_tensor.shape}")
    
    # Save cosine similarity analysis
    cosine_json_path = save_dir / f"{base_name}_cosine_similarity_entropy_vs_confidence.json"
    cosine_plot_path = save_dir / f"{base_name}_cosine_similarity_entropy_vs_confidence.png"
    
    print(f"\nSaving cosine similarity analysis...")
    with open(cosine_json_path, "w") as f:
        json.dump(cosine_analysis, f, indent=2)
    print(f"  Saved: {cosine_json_path}")
    
    cosine_fig.savefig(cosine_plot_path, dpi=300, bbox_inches='tight')
    plt.close(cosine_fig)
    print(f"  Saved: {cosine_plot_path}")
    
    # =============================================================================
    # PART 5: VALIDATION AND ASSERTIONS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("PART 5: VALIDATION")
    print("=" * 80)
    
    print("\nValidating entropy contrast:")
    for layer in range(num_layers):
        d = entropy_directions[layer]
        assert np.all(np.isfinite(d)), f"Layer {layer}: Non-finite values"
        norm = np.linalg.norm(d)
        assert np.abs(norm - 1.0) < 1e-3, f"Layer {layer}: Not unit norm (||d||={norm:.6f})"
        assert d.shape[0] == direct_activations_by_layer[0].shape[1], f"Layer {layer}: Wrong shape"
    print(f"  ✓ All {num_layers} layers valid (finite, unit norm, correct shape)")
    
    print("\nValidating confidence contrast:")
    for layer in range(num_layers):
        d = confidence_directions[layer]
        assert np.all(np.isfinite(d)), f"Layer {layer}: Non-finite values"
        norm = np.linalg.norm(d)
        assert np.abs(norm - 1.0) < 1e-3, f"Layer {layer}: Not unit norm (||d||={norm:.6f})"
        assert d.shape[0] == confidence_activations_by_layer[0].shape[1], f"Layer {layer}: Wrong shape"
    print(f"  ✓ All {num_layers} layers valid (finite, unit norm, correct shape)")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nEntropy contrast (mean-diff):")
    print(f"  Method: quantile mean difference (top/bottom {ENTROPY_QUANTILE})")
    print(f"  Best layer: {best_entropy_layer}")
    print(f"  Best R²: {entropy_metadata['per_layer'][best_entropy_layer]['r2']:.3f}")
    print(f"  Best corr: {entropy_metadata['per_layer'][best_entropy_layer]['corr']:.3f}")
    
    print(f"\nConfidence contrast (mean-diff):")
    print(f"  Method: quantile mean difference (top/bottom {ENTROPY_QUANTILE})")
    print(f"  Best layer: {best_conf_layer}")
    print(f"  Best R²: {confidence_metadata['per_layer'][best_conf_layer]['r2']:.3f}")
    print(f"  Best corr: {confidence_metadata['per_layer'][best_conf_layer]['corr']:.3f}")
    
    print(f"\nCosine similarity (entropy vs confidence):")
    print(f"  Mean: {cosine_analysis['statistics']['mean_cosine']:.3f} ± "
          f"{cosine_analysis['statistics']['std_cosine']:.3f}")
    print(f"  Max: {cosine_analysis['max_cosine']['value']:.3f} "
          f"(layer {cosine_analysis['max_cosine']['layer']})")
    print(f"  Min: {cosine_analysis['min_cosine']['value']:.3f} "
          f"(layer {cosine_analysis['min_cosine']['layer']})")
    
    print(f"\nOutput directory: {save_dir}")
    print("Output files:")
    print(f"  {entropy_json_path.name}")
    print(f"  {entropy_pt_path.name}")
    print(f"  {conf_json_path.name}")
    print(f"  {conf_pt_path.name}")
    print(f"  {cosine_json_path.name}")
    print(f"  {cosine_plot_path.name}")


if __name__ == "__main__":
    main()
