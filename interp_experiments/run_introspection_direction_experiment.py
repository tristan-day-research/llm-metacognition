"""
Find the "true introspection" direction in the model.

The idea: The model doesn't always introspect. Sometimes its stated confidence
aligns with its actual MC entropy (true introspection), and sometimes it doesn't
(surface heuristics, confabulation, etc.).

This script:
1. Computes an "introspection alignment score" for each question:
   - High when low MC entropy + high stated confidence (correctly confident)
   - High when high MC entropy + low stated confidence (correctly uncertain)
   - Low when these are misaligned (confident but should be uncertain, or vice versa)

2. Trains a probe to predict this alignment score from meta activations
   - This finds the "introspection direction" - active when model is truly introspecting

3. Analyzes the relationship between this direction and the entropy direction

4. Steers along this direction to see if it makes reports more aligned with entropy
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
from dotenv import load_dotenv
import random
import seaborn as sns

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from prompts import (
    # Confidence task
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    STATED_CONFIDENCE_SETUP,
    STATED_CONFIDENCE_QUESTION,
    format_stated_confidence_prompt,
)

# Configuration — edit values in experiment_config.IntrospectionDirectionConfig
from experiment_config import IntrospectionDirectionConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
MODEL_NAME = _C.MODEL_NAME
DATASET_NAME = _C.DATASET_NAME
OUTPUTS_DIR = _C.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(exist_ok=True)


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from a model path."""
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection")

# Experiment config
NUM_QUESTIONS_TO_TEST = None  # None = use all
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100

# Steering config
RUN_STEERING = True
STEERING_LAYERS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
STEERING_MULTIPLIERS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
NUM_STEERING_QUESTIONS = 100  # Subset for steering (it's slow)
NUM_CONTROL_DIRECTIONS = 3

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Backward compatibility aliases (now imported from tasks.py)
META_OPTION_DICT = STATED_CONFIDENCE_OPTIONS
META_RANGE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())
META_SETUP_PROMPT = STATED_CONFIDENCE_SETUP
META_QUESTION_PROMPT = STATED_CONFIDENCE_QUESTION

# Import utility functions from core
from core.model_utils import is_base_model, has_chat_template


# ============================================================================
# INTROSPECTION SCORE COMPUTATION
# ============================================================================

def compute_introspection_scores(
    direct_entropies: np.ndarray,
    meta_responses: List[str],
    method: str = "continuous"
) -> Tuple[np.ndarray, Dict]:
    """
    Compute introspection alignment scores for each question.

    High score = model's stated confidence aligns with its actual entropy
    Low score = model's stated confidence misaligns with its actual entropy

    Args:
        direct_entropies: Array of MC entropies
        meta_responses: List of confidence responses (S-Z)
        method: "continuous" for z-score product, "binary" for aligned/misaligned

    Returns:
        scores: Array of introspection scores
        stats: Dictionary of statistics
    """
    # Convert meta responses to confidence values
    stated_confidences = np.array([
        META_RANGE_MIDPOINTS.get(r, 0.5) for r in meta_responses
    ])

    # Z-score both
    entropy_z = stats.zscore(direct_entropies)
    confidence_z = stats.zscore(stated_confidences)

    if method == "continuous":
        # Introspection score = negative product of z-scores
        # When entropy is high (positive z) and confidence is low (negative z): product is negative, score is positive
        # When entropy is low (negative z) and confidence is high (positive z): product is negative, score is positive
        # When they're aligned in the wrong way: product is positive, score is negative
        introspection_scores = -1 * entropy_z * confidence_z

    elif method == "binary":
        # Binary: are they on opposite sides of median?
        entropy_high = direct_entropies > np.median(direct_entropies)
        confidence_high = stated_confidences > np.median(stated_confidences)
        # Aligned = opposite signs (high entropy + low confidence, or low entropy + high confidence)
        introspection_scores = (entropy_high != confidence_high).astype(float)

    elif method == "residual":
        # How much does confidence deviate from what entropy would predict?
        # Fit a line, get residuals - large positive residual = overconfident
        slope, intercept = np.polyfit(direct_entropies, stated_confidences, 1)
        predicted_confidence = slope * direct_entropies + intercept
        residuals = stated_confidences - predicted_confidence
        # Introspection = small absolute residual (close to the line)
        introspection_scores = -np.abs(residuals)  # Negative so higher = better alignment

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute statistics
    correlation = np.corrcoef(direct_entropies, stated_confidences)[0, 1]

    stats_dict = {
        "method": method,
        "correlation_entropy_confidence": float(correlation),
        "mean_entropy": float(direct_entropies.mean()),
        "std_entropy": float(direct_entropies.std()),
        "mean_confidence": float(stated_confidences.mean()),
        "std_confidence": float(stated_confidences.std()),
        "mean_introspection_score": float(introspection_scores.mean()),
        "std_introspection_score": float(introspection_scores.std()),
        "fraction_aligned": float((introspection_scores > 0).mean()) if method == "continuous" else float(introspection_scores.mean()),
    }

    return introspection_scores, stats_dict, entropy_z, confidence_z


# ============================================================================
# PROBE TRAINING
# ============================================================================

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    return_components: bool = False,
    classification: bool = False
) -> Dict:
    """
    Train a linear probe to predict target from activations.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA if enabled
    pca = None
    if USE_PCA:
        n_components = min(PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    if classification:
        probe = LogisticRegression(C=1/PROBE_ALPHA, max_iter=1000)
        probe.fit(X_train_final, y_train)

        y_pred_train = probe.predict(X_train_final)
        y_pred_test = probe.predict(X_test_final)
        y_prob_test = probe.predict_proba(X_test_final)[:, 1]

        result = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_auc": roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else 0.5,
            "predictions": y_pred_test,
            "probabilities": y_prob_test,
        }
    else:
        probe = Ridge(alpha=PROBE_ALPHA)
        probe.fit(X_train_final, y_train)

        y_pred_train = probe.predict(X_train_final)
        y_pred_test = probe.predict(X_test_final)

        result = {
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "predictions": y_pred_test,
        }

    if USE_PCA:
        result["pca_variance_explained"] = pca.explained_variance_ratio_.sum()

    if return_components:
        result["scaler"] = scaler
        result["pca"] = pca
        result["probe"] = probe

    return result


def extract_probe_direction(
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe,
    classification: bool = False
) -> np.ndarray:
    """
    Extract the direction in activation space that the probe uses.
    """
    if classification:
        coef = probe.coef_[0]  # Shape: (n_pca_components,)
    else:
        coef = probe.coef_  # Shape: (n_pca_components,)

    if pca is not None:
        # Project back to scaled space
        direction_scaled = pca.components_.T @ coef
    else:
        direction_scaled = coef

    # Unscale
    direction_original = direction_scaled / scaler.scale_

    # Normalize
    direction_original = direction_original / np.linalg.norm(direction_original)

    return direction_original


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def _present_nested_question(question_data: Dict, outer_question: str, outer_options: Dict) -> str:
    """Format a nested/meta question for display."""
    formatted_question = ""
    formatted_question += "-" * 30 + "\n"
    formatted_question += outer_question + "\n"
    formatted_question += "-" * 10 + "\n"
    formatted_question += question_data["question"] + "\n"
    if "options" in question_data:
        for key, value in question_data["options"].items():
            formatted_question += f"  {key}: {value}\n"
    formatted_question += "-" * 10 + "\n"
    if outer_options:
        for key, value in outer_options.items():
            formatted_question += f"  {key}: {value}\n"
    formatted_question += "-" * 30
    return formatted_question


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> str:
    """Format a meta/confidence question using centralized tasks.py logic."""
    full_prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
    return full_prompt


# ============================================================================
# STEERING INFRASTRUCTURE
# ============================================================================

class SteeringHook:
    """Hook that adds a steering vector to activations at a specific layer."""

    def __init__(self, steering_vector: torch.Tensor, multiplier: float):
        # Ensure normalized so multiplier has consistent meaning across directions
        self.steering_vector = steering_vector / steering_vector.norm()
        self.multiplier = multiplier
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            steered = hidden_states + self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return (steered,) + output[1:]
        else:
            return output + self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def generate_orthogonal_directions(steering_direction: np.ndarray, num_directions: int) -> List[np.ndarray]:
    """Generate random directions orthogonal to the steering direction."""
    hidden_dim = len(steering_direction)
    orthogonal_directions = []

    for _ in range(num_directions):
        random_vec = np.random.randn(hidden_dim)
        random_vec = random_vec - np.dot(random_vec, steering_direction) * steering_direction
        for prev_dir in orthogonal_directions:
            random_vec = random_vec - np.dot(random_vec, prev_dir) * prev_dir
        random_vec = random_vec / np.linalg.norm(random_vec)
        orthogonal_directions.append(random_vec)

    return orthogonal_directions


def get_steered_confidence(
    model,
    tokenizer,
    question: Dict,
    layer_idx: int,
    steering_vector: np.ndarray,
    multiplier: float,
    use_chat_template: bool = True
) -> Tuple[str, float, np.ndarray]:
    """
    Run model with steering and get confidence response.

    Returns:
        response: Letter S-Z
        confidence_value: Midpoint of chosen range
        probs: Probability distribution over options
    """
    prompt = format_meta_prompt(question, tokenizer, use_chat_template)

    steering_tensor = torch.tensor(
        steering_vector,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
        layer_module = base.model.layers[layer_idx]
    else:
        layer_module = model.model.layers[layer_idx]

    hook = SteeringHook(steering_tensor, multiplier)
    hook.register(layer_module)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        final_logits = outputs.logits[0, -1, :]
        option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in META_OPTIONS]
        option_logits = final_logits[option_token_ids]
        option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

        response_idx = np.argmax(option_probs)
        response = META_OPTIONS[response_idx]
        confidence_value = META_RANGE_MIDPOINTS[response]

    finally:
        hook.remove()

    return response, confidence_value, option_probs


def get_baseline_confidence(
    model,
    tokenizer,
    question: Dict,
    use_chat_template: bool = True
) -> Tuple[str, float, np.ndarray]:
    """Get confidence response without steering."""
    prompt = format_meta_prompt(question, tokenizer, use_chat_template)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    final_logits = outputs.logits[0, -1, :]
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in META_OPTIONS]
    option_logits = final_logits[option_token_ids]
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    response_idx = np.argmax(option_probs)
    response = META_OPTIONS[response_idx]
    confidence_value = META_RANGE_MIDPOINTS[response]

    return response, confidence_value, option_probs


# ============================================================================
# STEERING EXPERIMENT
# ============================================================================

def run_introspection_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    introspection_directions: Dict[int, np.ndarray],
    entropy_directions: Dict[int, np.ndarray],
    use_chat_template: bool = True
) -> Dict:
    """
    Steer along the introspection direction and measure effect on alignment.

    The key question: Does steering toward "high introspection" make the model's
    stated confidence more aligned with its actual entropy?

    We also compare to steering along the entropy direction and control directions.
    """
    print(f"\nRunning introspection steering experiment...")
    print(f"  Layers: {STEERING_LAYERS}")
    print(f"  Multipliers: {STEERING_MULTIPLIERS}")
    print(f"  Questions: {len(questions)}")

    results = {
        "layers": STEERING_LAYERS,
        "multipliers": STEERING_MULTIPLIERS,
        "num_questions": len(questions),
        "layer_results": {},
    }

    for layer_idx in tqdm(STEERING_LAYERS, desc="Steering layers"):
        introspection_dir = introspection_directions[layer_idx]
        entropy_dir = entropy_directions[layer_idx]
        control_dirs = generate_orthogonal_directions(introspection_dir, NUM_CONTROL_DIRECTIONS)

        layer_results = {
            "baseline": [],
            "introspection_steering": {m: [] for m in STEERING_MULTIPLIERS},
            "entropy_steering": {m: [] for m in STEERING_MULTIPLIERS},
            "control_steering": {
                f"control_{i}": {m: [] for m in STEERING_MULTIPLIERS}
                for i in range(NUM_CONTROL_DIRECTIONS)
            },
        }

        # Baseline (no steering)
        for q_idx, question in enumerate(tqdm(questions, desc="Baseline", leave=False)):
            response, confidence, probs = get_baseline_confidence(model, tokenizer, question, use_chat_template)

            # Compute alignment with this question's entropy
            entropy = direct_entropies[q_idx]
            entropy_z = (entropy - direct_entropies.mean()) / direct_entropies.std()
            confidence_z = (confidence - 0.5) / 0.25  # Rough z-score for confidence
            alignment = -entropy_z * confidence_z  # Positive = aligned

            layer_results["baseline"].append({
                "question_idx": q_idx,
                "response": response,
                "confidence": confidence,
                "entropy": float(entropy),
                "alignment": float(alignment),
            })

        # Introspection steering
        for multiplier in tqdm(STEERING_MULTIPLIERS, desc="Introspection steering", leave=False):
            if multiplier == 0.0:
                layer_results["introspection_steering"][multiplier] = layer_results["baseline"]
                continue

            for q_idx, question in enumerate(questions):
                response, confidence, probs = get_steered_confidence(
                    model, tokenizer, question, layer_idx, introspection_dir, multiplier, use_chat_template
                )

                entropy = direct_entropies[q_idx]
                entropy_z = (entropy - direct_entropies.mean()) / direct_entropies.std()
                confidence_z = (confidence - 0.5) / 0.25
                alignment = -entropy_z * confidence_z

                layer_results["introspection_steering"][multiplier].append({
                    "question_idx": q_idx,
                    "response": response,
                    "confidence": confidence,
                    "alignment": float(alignment),
                })

        # Entropy steering (for comparison)
        for multiplier in tqdm(STEERING_MULTIPLIERS, desc="Entropy steering", leave=False):
            if multiplier == 0.0:
                layer_results["entropy_steering"][multiplier] = layer_results["baseline"]
                continue

            for q_idx, question in enumerate(questions):
                response, confidence, probs = get_steered_confidence(
                    model, tokenizer, question, layer_idx, entropy_dir, multiplier, use_chat_template
                )

                entropy = direct_entropies[q_idx]
                entropy_z = (entropy - direct_entropies.mean()) / direct_entropies.std()
                confidence_z = (confidence - 0.5) / 0.25
                alignment = -entropy_z * confidence_z

                layer_results["entropy_steering"][multiplier].append({
                    "question_idx": q_idx,
                    "response": response,
                    "confidence": confidence,
                    "alignment": float(alignment),
                })

        # Control steering
        for ctrl_idx, ctrl_dir in enumerate(control_dirs):
            for multiplier in tqdm(STEERING_MULTIPLIERS, desc=f"Control {ctrl_idx}", leave=False):
                if multiplier == 0.0:
                    layer_results["control_steering"][f"control_{ctrl_idx}"][multiplier] = layer_results["baseline"]
                    continue

                for q_idx, question in enumerate(questions):
                    response, confidence, probs = get_steered_confidence(
                        model, tokenizer, question, layer_idx, ctrl_dir, multiplier, use_chat_template
                    )

                    entropy = direct_entropies[q_idx]
                    entropy_z = (entropy - direct_entropies.mean()) / direct_entropies.std()
                    confidence_z = (confidence - 0.5) / 0.25
                    alignment = -entropy_z * confidence_z

                    layer_results["control_steering"][f"control_{ctrl_idx}"][multiplier].append({
                        "question_idx": q_idx,
                        "response": response,
                        "confidence": confidence,
                        "alignment": float(alignment),
                    })

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


def analyze_steering_results(steering_results: Dict) -> Dict:
    """Compute summary statistics for steering effects."""
    analysis = {
        "layers": steering_results["layers"],
        "multipliers": steering_results["multipliers"],
        "introspection_effects": {},
        "entropy_effects": {},
        "control_effects": {},
    }

    for layer_idx in steering_results["layers"]:
        layer_results = steering_results["layer_results"][layer_idx]

        # Baseline statistics
        baseline_alignments = np.array([r["alignment"] for r in layer_results["baseline"]])
        baseline_confidences = np.array([r["confidence"] for r in layer_results["baseline"]])
        baseline_mean_alignment = baseline_alignments.mean()
        baseline_mean_confidence = baseline_confidences.mean()

        # Introspection steering effects
        intro_effects = {}
        for mult in steering_results["multipliers"]:
            results = layer_results["introspection_steering"][mult]
            alignments = np.array([r["alignment"] for r in results])
            confidences = np.array([r["confidence"] for r in results])

            intro_effects[mult] = {
                "mean_alignment": float(alignments.mean()),
                "alignment_change": float(alignments.mean() - baseline_mean_alignment),
                "mean_confidence": float(confidences.mean()),
                "confidence_change": float(confidences.mean() - baseline_mean_confidence),
            }
        analysis["introspection_effects"][layer_idx] = intro_effects

        # Entropy steering effects
        entropy_effects = {}
        for mult in steering_results["multipliers"]:
            results = layer_results["entropy_steering"][mult]
            alignments = np.array([r["alignment"] for r in results])
            confidences = np.array([r["confidence"] for r in results])

            entropy_effects[mult] = {
                "mean_alignment": float(alignments.mean()),
                "alignment_change": float(alignments.mean() - baseline_mean_alignment),
                "mean_confidence": float(confidences.mean()),
                "confidence_change": float(confidences.mean() - baseline_mean_confidence),
            }
        analysis["entropy_effects"][layer_idx] = entropy_effects

        # Control effects (average across control directions)
        ctrl_effects = {}
        for mult in steering_results["multipliers"]:
            ctrl_alignments = []
            ctrl_confidences = []
            for ctrl_key in layer_results["control_steering"]:
                results = layer_results["control_steering"][ctrl_key][mult]
                ctrl_alignments.extend([r["alignment"] for r in results])
                ctrl_confidences.extend([r["confidence"] for r in results])

            ctrl_effects[mult] = {
                "mean_alignment": float(np.mean(ctrl_alignments)),
                "alignment_change": float(np.mean(ctrl_alignments) - baseline_mean_alignment),
                "mean_confidence": float(np.mean(ctrl_confidences)),
                "confidence_change": float(np.mean(ctrl_confidences) - baseline_mean_confidence),
            }
        analysis["control_effects"][layer_idx] = ctrl_effects

    return analysis


def plot_steering_results(
    steering_analysis: Dict,
    output_prefix: str = "introspection_direction"
):
    """Create visualizations for steering results."""

    layers = steering_analysis["layers"]
    multipliers = steering_analysis["multipliers"]

    # ========================================================================
    # Plot 1: Alignment change heatmaps
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, title) in zip(axes, [
        ("introspection_effects", "Introspection Direction"),
        ("entropy_effects", "Entropy Direction"),
        ("control_effects", "Control Directions (avg)")
    ]):
        matrix = np.zeros((len(layers), len(multipliers)))
        for i, layer in enumerate(layers):
            for j, mult in enumerate(multipliers):
                matrix[i, j] = steering_analysis[key][layer][mult]["alignment_change"]

        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                       vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())
        ax.set_xticks(range(len(multipliers)))
        ax.set_xticklabels([f"{m:.1f}" for m in multipliers], rotation=45)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("Steering Multiplier")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title}\nΔ Alignment")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_steering_alignment_heatmaps.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_steering_alignment_heatmaps.png")
    plt.close()

    # ========================================================================
    # Plot 2: Best layer comparison - alignment
    # ========================================================================
    # Find layer with strongest introspection steering effect
    best_layer = max(layers, key=lambda l:
        abs(steering_analysis["introspection_effects"][l][max(multipliers)]["alignment_change"] -
            steering_analysis["introspection_effects"][l][min(multipliers)]["alignment_change"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    intro_align = [steering_analysis["introspection_effects"][best_layer][m]["alignment_change"] for m in multipliers]
    entropy_align = [steering_analysis["entropy_effects"][best_layer][m]["alignment_change"] for m in multipliers]
    ctrl_align = [steering_analysis["control_effects"][best_layer][m]["alignment_change"] for m in multipliers]

    ax1.plot(multipliers, intro_align, 'o-', label='Introspection dir', linewidth=2, color='green')
    ax1.plot(multipliers, entropy_align, 's-', label='Entropy dir', linewidth=2, color='blue')
    ax1.plot(multipliers, ctrl_align, '^--', label='Control dirs', linewidth=2, color='gray', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel("Steering Multiplier")
    ax1.set_ylabel("Δ Alignment")
    ax1.set_title(f"Alignment Change (Layer {best_layer})\nPositive = more aligned with entropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confidence change
    ax2 = axes[1]
    intro_conf = [steering_analysis["introspection_effects"][best_layer][m]["confidence_change"] for m in multipliers]
    entropy_conf = [steering_analysis["entropy_effects"][best_layer][m]["confidence_change"] for m in multipliers]
    ctrl_conf = [steering_analysis["control_effects"][best_layer][m]["confidence_change"] for m in multipliers]

    ax2.plot(multipliers, intro_conf, 'o-', label='Introspection dir', linewidth=2, color='green')
    ax2.plot(multipliers, entropy_conf, 's-', label='Entropy dir', linewidth=2, color='blue')
    ax2.plot(multipliers, ctrl_conf, '^--', label='Control dirs', linewidth=2, color='gray', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Steering Multiplier")
    ax2.set_ylabel("Δ Confidence")
    ax2.set_title(f"Confidence Change (Layer {best_layer})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_steering_best_layer.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_steering_best_layer.png")
    plt.close()

    # ========================================================================
    # Plot 3: Slope comparison across layers
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Alignment slopes
    ax1 = axes[0]
    intro_slopes = []
    entropy_slopes = []
    ctrl_slopes = []

    for layer in layers:
        intro_align = [steering_analysis["introspection_effects"][layer][m]["alignment_change"] for m in multipliers]
        entropy_align = [steering_analysis["entropy_effects"][layer][m]["alignment_change"] for m in multipliers]
        ctrl_align = [steering_analysis["control_effects"][layer][m]["alignment_change"] for m in multipliers]

        intro_slopes.append(np.polyfit(multipliers, intro_align, 1)[0])
        entropy_slopes.append(np.polyfit(multipliers, entropy_align, 1)[0])
        ctrl_slopes.append(np.polyfit(multipliers, ctrl_align, 1)[0])

    x = np.arange(len(layers))
    width = 0.25

    ax1.bar(x - width, intro_slopes, width, label='Introspection', color='green', alpha=0.7)
    ax1.bar(x, entropy_slopes, width, label='Entropy', color='blue', alpha=0.7)
    ax1.bar(x + width, ctrl_slopes, width, label='Control', color='gray', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Alignment Slope")
    ax1.set_title("Alignment Effect Slope by Layer\n(Positive = steering up increases alignment)")
    ax1.legend()

    # Confidence slopes
    ax2 = axes[1]
    intro_conf_slopes = []
    entropy_conf_slopes = []
    ctrl_conf_slopes = []

    for layer in layers:
        intro_conf = [steering_analysis["introspection_effects"][layer][m]["confidence_change"] for m in multipliers]
        entropy_conf = [steering_analysis["entropy_effects"][layer][m]["confidence_change"] for m in multipliers]
        ctrl_conf = [steering_analysis["control_effects"][layer][m]["confidence_change"] for m in multipliers]

        intro_conf_slopes.append(np.polyfit(multipliers, intro_conf, 1)[0])
        entropy_conf_slopes.append(np.polyfit(multipliers, entropy_conf, 1)[0])
        ctrl_conf_slopes.append(np.polyfit(multipliers, ctrl_conf, 1)[0])

    ax2.bar(x - width, intro_conf_slopes, width, label='Introspection', color='green', alpha=0.7)
    ax2.bar(x, entropy_conf_slopes, width, label='Entropy', color='blue', alpha=0.7)
    ax2.bar(x + width, ctrl_conf_slopes, width, label='Control', color='gray', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Confidence Slope")
    ax2.set_title("Confidence Effect Slope by Layer")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_steering_slopes.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_steering_slopes.png")
    plt.close()


def print_steering_summary(steering_analysis: Dict):
    """Print summary of steering results."""
    print("\n" + "=" * 80)
    print("INTROSPECTION STEERING RESULTS")
    print("=" * 80)

    layers = steering_analysis["layers"]
    multipliers = steering_analysis["multipliers"]

    print("\n--- Alignment Effect by Layer (slope: Δalignment / Δmultiplier) ---")
    print(f"{'Layer':<8} {'Introspection':<15} {'Entropy':<15} {'Control':<15}")
    print("-" * 60)

    for layer in layers:
        intro_align = [steering_analysis["introspection_effects"][layer][m]["alignment_change"] for m in multipliers]
        entropy_align = [steering_analysis["entropy_effects"][layer][m]["alignment_change"] for m in multipliers]
        ctrl_align = [steering_analysis["control_effects"][layer][m]["alignment_change"] for m in multipliers]

        intro_slope = np.polyfit(multipliers, intro_align, 1)[0]
        entropy_slope = np.polyfit(multipliers, entropy_align, 1)[0]
        ctrl_slope = np.polyfit(multipliers, ctrl_align, 1)[0]

        print(f"{layer:<8} {intro_slope:<15.4f} {entropy_slope:<15.4f} {ctrl_slope:<15.4f}")

    # Find best layer
    best_layer = max(layers, key=lambda l:
        np.polyfit(multipliers,
                   [steering_analysis["introspection_effects"][l][m]["alignment_change"] for m in multipliers],
                   1)[0])

    print(f"\nBest layer for introspection steering: {best_layer}")

    intro_slope = np.polyfit(multipliers,
        [steering_analysis["introspection_effects"][best_layer][m]["alignment_change"] for m in multipliers], 1)[0]
    ctrl_slope = np.polyfit(multipliers,
        [steering_analysis["control_effects"][best_layer][m]["alignment_change"] for m in multipliers], 1)[0]

    print(f"\n--- Interpretation ---")
    if intro_slope > 0 and intro_slope > ctrl_slope:
        print("✓ Steering in 'high introspection' direction INCREASES alignment")
        print("  (Model's confidence becomes more correlated with its actual entropy)")
        print("  → Evidence that introspection direction causally affects alignment!")
    elif intro_slope < 0:
        print("✗ Steering in 'high introspection' direction DECREASES alignment")
        print("  (Unexpected - may indicate probe found wrong direction)")
    else:
        print("⚠ Introspection steering effect not stronger than controls")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_introspection_direction_analysis(
    meta_activations: Dict[int, np.ndarray],
    direct_entropies: np.ndarray,
    introspection_scores: np.ndarray,
    entropy_z: np.ndarray,
    confidence_z: np.ndarray
) -> Dict:
    """
    Train probes to find:
    1. Introspection direction (predicts alignment score)
    2. Entropy direction (predicts MC entropy) - for comparison
    3. Confidence direction (predicts stated confidence) - for comparison

    Analyze relationships between these directions.
    """
    print(f"\nRunning introspection direction analysis across {len(meta_activations)} layers...")

    # Split data
    n_questions = len(introspection_scores)
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)

    results = {
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "layer_results": {},
    }

    for layer_idx in tqdm(sorted(meta_activations.keys()), desc="Training probes"):
        X = meta_activations[layer_idx]

        X_train = X[train_idx]
        X_test = X[test_idx]

        # 1. Introspection probe (predicts alignment score)
        introspection_results = train_probe(
            X_train, introspection_scores[train_idx],
            X_test, introspection_scores[test_idx],
            return_components=True
        )
        introspection_direction = extract_probe_direction(
            introspection_results["scaler"],
            introspection_results["pca"],
            introspection_results["probe"]
        )

        # 2. Entropy probe (predicts direct entropy from meta activations)
        entropy_results = train_probe(
            X_train, direct_entropies[train_idx],
            X_test, direct_entropies[test_idx],
            return_components=True
        )
        entropy_direction = extract_probe_direction(
            entropy_results["scaler"],
            entropy_results["pca"],
            entropy_results["probe"]
        )

        # 3. Confidence probe (predicts stated confidence from meta activations)
        # Use confidence_z since we have it
        confidence_results = train_probe(
            X_train, confidence_z[train_idx],
            X_test, confidence_z[test_idx],
            return_components=True
        )
        confidence_direction = extract_probe_direction(
            confidence_results["scaler"],
            confidence_results["pca"],
            confidence_results["probe"]
        )

        # 4. Binary introspection probe (aligned vs misaligned)
        binary_labels = (introspection_scores > 0).astype(int)
        binary_results = train_probe(
            X_train, binary_labels[train_idx],
            X_test, binary_labels[test_idx],
            return_components=True,
            classification=True
        )

        # Compute direction similarities
        cosine_intro_entropy = np.dot(introspection_direction, entropy_direction)
        cosine_intro_confidence = np.dot(introspection_direction, confidence_direction)
        cosine_entropy_confidence = np.dot(entropy_direction, confidence_direction)

        results["layer_results"][layer_idx] = {
            "introspection_probe": {
                "train_r2": introspection_results["train_r2"],
                "test_r2": introspection_results["test_r2"],
                "train_mae": introspection_results["train_mae"],
                "test_mae": introspection_results["test_mae"],
            },
            "entropy_probe": {
                "train_r2": entropy_results["train_r2"],
                "test_r2": entropy_results["test_r2"],
            },
            "confidence_probe": {
                "train_r2": confidence_results["train_r2"],
                "test_r2": confidence_results["test_r2"],
            },
            "binary_introspection_probe": {
                "train_accuracy": binary_results["train_accuracy"],
                "test_accuracy": binary_results["test_accuracy"],
                "test_auc": binary_results["test_auc"],
            },
            "direction_similarities": {
                "introspection_entropy_cosine": float(cosine_intro_entropy),
                "introspection_confidence_cosine": float(cosine_intro_confidence),
                "entropy_confidence_cosine": float(cosine_entropy_confidence),
            },
            "directions": {
                "introspection": introspection_direction.tolist(),
                "entropy": entropy_direction.tolist(),
                "confidence": confidence_direction.tolist(),
            }
        }

    return results, test_idx


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_introspection_analysis(
    results: Dict,
    introspection_scores: np.ndarray,
    direct_entropies: np.ndarray,
    confidence_z: np.ndarray,
    meta_responses: List[str],
    output_prefix: str = "introspection_direction"
):
    """Create visualizations of the introspection direction analysis."""

    layers = sorted(results["layer_results"].keys())

    # ========================================================================
    # Plot 1: Probe performance comparison across layers
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract metrics
    intro_r2 = [results["layer_results"][l]["introspection_probe"]["test_r2"] for l in layers]
    entropy_r2 = [results["layer_results"][l]["entropy_probe"]["test_r2"] for l in layers]
    confidence_r2 = [results["layer_results"][l]["confidence_probe"]["test_r2"] for l in layers]
    binary_auc = [results["layer_results"][l]["binary_introspection_probe"]["test_auc"] for l in layers]

    ax1 = axes[0, 0]
    ax1.plot(layers, intro_r2, 'o-', label='Introspection Score', linewidth=2)
    ax1.plot(layers, entropy_r2, 's-', label='MC Entropy', linewidth=2)
    ax1.plot(layers, confidence_r2, '^-', label='Stated Confidence', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Probe Performance by Layer\n(What can we predict from meta activations?)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Binary classification performance
    ax2 = axes[0, 1]
    binary_acc = [results["layer_results"][l]["binary_introspection_probe"]["test_accuracy"] for l in layers]
    ax2.plot(layers, binary_acc, 'o-', label='Accuracy', linewidth=2, color='green')
    ax2.plot(layers, binary_auc, 's-', label='AUC', linewidth=2, color='purple')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='Chance')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Score')
    ax2.set_title('Binary Introspection Classification\n(Aligned vs Misaligned)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.0)

    # Direction similarities
    ax3 = axes[1, 0]
    intro_entropy_cos = [results["layer_results"][l]["direction_similarities"]["introspection_entropy_cosine"] for l in layers]
    intro_conf_cos = [results["layer_results"][l]["direction_similarities"]["introspection_confidence_cosine"] for l in layers]
    entropy_conf_cos = [results["layer_results"][l]["direction_similarities"]["entropy_confidence_cosine"] for l in layers]

    ax3.plot(layers, intro_entropy_cos, 'o-', label='Introspection ↔ Entropy', linewidth=2)
    ax3.plot(layers, intro_conf_cos, 's-', label='Introspection ↔ Confidence', linewidth=2)
    ax3.plot(layers, entropy_conf_cos, '^-', label='Entropy ↔ Confidence', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Direction Similarities\n(Are these directions related?)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1, 1)

    # Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    best_intro_layer = max(layers, key=lambda l: results["layer_results"][l]["introspection_probe"]["test_r2"])
    best_intro_r2 = results["layer_results"][best_intro_layer]["introspection_probe"]["test_r2"]
    best_binary_layer = max(layers, key=lambda l: results["layer_results"][l]["binary_introspection_probe"]["test_auc"])
    best_binary_auc = results["layer_results"][best_binary_layer]["binary_introspection_probe"]["test_auc"]

    summary_text = f"""
INTROSPECTION DIRECTION ANALYSIS

Best Introspection Probe:
  Layer {best_intro_layer}: R² = {best_intro_r2:.4f}

Best Binary Classification:
  Layer {best_binary_layer}: AUC = {best_binary_auc:.4f}

Direction Relationships (Layer {best_intro_layer}):
  Introspection ↔ Entropy: {results["layer_results"][best_intro_layer]["direction_similarities"]["introspection_entropy_cosine"]:.3f}
  Introspection ↔ Confidence: {results["layer_results"][best_intro_layer]["direction_similarities"]["introspection_confidence_cosine"]:.3f}
  Entropy ↔ Confidence: {results["layer_results"][best_intro_layer]["direction_similarities"]["entropy_confidence_cosine"]:.3f}

Interpretation:
  - If Intro↔Entropy is high: introspection uses entropy signal
  - If Intro↔Confidence is high: introspection tracks stated confidence
  - If both are moderate: introspection is a distinct signal
    capturing when these align
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_probe_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_probe_comparison.png")
    plt.close()

    # ========================================================================
    # Plot 2: Introspection score distribution and examples
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distribution of introspection scores
    ax1 = axes[0, 0]
    ax1.hist(introspection_scores, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', label='Threshold (aligned vs misaligned)')
    ax1.set_xlabel('Introspection Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Introspection Scores\n(Positive = aligned, Negative = misaligned)')
    ax1.legend()

    # Scatter: entropy vs confidence, colored by introspection score
    ax2 = axes[0, 1]
    scatter = ax2.scatter(direct_entropies, confidence_z, c=introspection_scores,
                          cmap='RdBu', alpha=0.6, s=30)
    ax2.set_xlabel('MC Entropy')
    ax2.set_ylabel('Stated Confidence (z-scored)')
    ax2.set_title('Entropy vs Confidence\n(Color = introspection score)')
    plt.colorbar(scatter, ax=ax2, label='Introspection Score')

    # Add regression line
    slope, intercept = np.polyfit(direct_entropies, confidence_z, 1)
    x_line = np.linspace(direct_entropies.min(), direct_entropies.max(), 100)
    ax2.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, label=f'Fit (r={np.corrcoef(direct_entropies, confidence_z)[0,1]:.2f})')
    ax2.legend()

    # High vs low introspection examples
    ax3 = axes[1, 0]
    high_intro_idx = np.argsort(introspection_scores)[-20:]  # Top 20
    low_intro_idx = np.argsort(introspection_scores)[:20]    # Bottom 20

    ax3.scatter(direct_entropies[high_intro_idx], confidence_z[high_intro_idx],
                c='green', s=100, alpha=0.7, label='High introspection', marker='o')
    ax3.scatter(direct_entropies[low_intro_idx], confidence_z[low_intro_idx],
                c='red', s=100, alpha=0.7, label='Low introspection', marker='x')
    ax3.set_xlabel('MC Entropy')
    ax3.set_ylabel('Stated Confidence (z-scored)')
    ax3.set_title('Extreme Cases\n(Green = true introspection, Red = misaligned)')
    ax3.legend()

    # Response distribution by introspection score
    ax4 = axes[1, 1]
    aligned = introspection_scores > 0

    # Count responses for aligned vs misaligned
    aligned_responses = [meta_responses[i] for i in range(len(meta_responses)) if aligned[i]]
    misaligned_responses = [meta_responses[i] for i in range(len(meta_responses)) if not aligned[i]]

    all_options = list(META_RANGE_MIDPOINTS.keys())
    aligned_counts = [aligned_responses.count(opt) for opt in all_options]
    misaligned_counts = [misaligned_responses.count(opt) for opt in all_options]

    x = np.arange(len(all_options))
    width = 0.35
    ax4.bar(x - width/2, aligned_counts, width, label='Aligned', color='green', alpha=0.7)
    ax4.bar(x + width/2, misaligned_counts, width, label='Misaligned', color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{opt}\n({META_RANGE_MIDPOINTS[opt]:.0%})" for opt in all_options], fontsize=8)
    ax4.set_xlabel('Confidence Response')
    ax4.set_ylabel('Count')
    ax4.set_title('Response Distribution by Alignment')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_score_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_score_distribution.png")
    plt.close()

    # ========================================================================
    # Plot 3: Layer-by-layer direction analysis
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Heatmap of direction similarities across layers
    similarity_matrix = np.zeros((len(layers), 3))
    for i, layer in enumerate(layers):
        similarity_matrix[i, 0] = results["layer_results"][layer]["direction_similarities"]["introspection_entropy_cosine"]
        similarity_matrix[i, 1] = results["layer_results"][layer]["direction_similarities"]["introspection_confidence_cosine"]
        similarity_matrix[i, 2] = results["layer_results"][layer]["direction_similarities"]["entropy_confidence_cosine"]

    ax1 = axes[0]
    im = ax1.imshow(similarity_matrix, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['Intro↔Ent', 'Intro↔Conf', 'Ent↔Conf'], rotation=45, ha='right')
    ax1.set_ylabel('Layer')
    ax1.set_title('Direction Cosine Similarities')
    plt.colorbar(im, ax=ax1)

    # R² comparison
    ax2 = axes[1]
    width = 0.25
    x = np.arange(len(layers))
    ax2.bar(x - width, intro_r2, width, label='Introspection', alpha=0.7)
    ax2.bar(x, entropy_r2, width, label='Entropy', alpha=0.7)
    ax2.bar(x + width, confidence_r2, width, label='Confidence', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Test R²')
    ax2.set_title('Probe R² by Layer')
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Are directions becoming more similar at certain layers?
    ax3 = axes[2]
    # Compute "distinctiveness" of introspection direction
    # = 1 - max(abs(cosine with entropy), abs(cosine with confidence))
    distinctiveness = []
    for layer in layers:
        cos_e = abs(results["layer_results"][layer]["direction_similarities"]["introspection_entropy_cosine"])
        cos_c = abs(results["layer_results"][layer]["direction_similarities"]["introspection_confidence_cosine"])
        distinctiveness.append(1 - max(cos_e, cos_c))

    ax3.plot(layers, distinctiveness, 'o-', linewidth=2, color='purple')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Distinctiveness')
    ax3.set_title('Introspection Direction Distinctiveness\n(1 - max overlap with entropy/confidence)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_direction_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_direction_analysis.png")
    plt.close()


def print_results_summary(results: Dict, score_stats: Dict):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("INTROSPECTION DIRECTION ANALYSIS RESULTS")
    print("=" * 80)

    print("\n--- Introspection Score Statistics ---")
    print(f"Method: {score_stats['method']}")
    print(f"Entropy-Confidence Correlation: {score_stats['correlation_entropy_confidence']:.4f}")
    print(f"Fraction Aligned (score > 0): {score_stats['fraction_aligned']:.2%}")

    print("\n--- Probe Performance by Layer ---")
    print(f"{'Layer':<8} {'Intro R²':<12} {'Entropy R²':<12} {'Conf R²':<12} {'Binary AUC':<12}")
    print("-" * 60)

    layers = sorted(results["layer_results"].keys())
    for layer in layers:
        lr = results["layer_results"][layer]
        print(f"{layer:<8} {lr['introspection_probe']['test_r2']:<12.4f} "
              f"{lr['entropy_probe']['test_r2']:<12.4f} "
              f"{lr['confidence_probe']['test_r2']:<12.4f} "
              f"{lr['binary_introspection_probe']['test_auc']:<12.4f}")

    # Best layers
    best_intro = max(layers, key=lambda l: results["layer_results"][l]["introspection_probe"]["test_r2"])
    best_entropy = max(layers, key=lambda l: results["layer_results"][l]["entropy_probe"]["test_r2"])
    best_conf = max(layers, key=lambda l: results["layer_results"][l]["confidence_probe"]["test_r2"])
    best_binary = max(layers, key=lambda l: results["layer_results"][l]["binary_introspection_probe"]["test_auc"])

    print("\n--- Best Layers ---")
    print(f"Introspection: Layer {best_intro} (R² = {results['layer_results'][best_intro]['introspection_probe']['test_r2']:.4f})")
    print(f"Entropy: Layer {best_entropy} (R² = {results['layer_results'][best_entropy]['entropy_probe']['test_r2']:.4f})")
    print(f"Confidence: Layer {best_conf} (R² = {results['layer_results'][best_conf]['confidence_probe']['test_r2']:.4f})")
    print(f"Binary: Layer {best_binary} (AUC = {results['layer_results'][best_binary]['binary_introspection_probe']['test_auc']:.4f})")

    print("\n--- Direction Similarities (Best Introspection Layer) ---")
    ds = results["layer_results"][best_intro]["direction_similarities"]
    print(f"Introspection ↔ Entropy: {ds['introspection_entropy_cosine']:.4f}")
    print(f"Introspection ↔ Confidence: {ds['introspection_confidence_cosine']:.4f}")
    print(f"Entropy ↔ Confidence: {ds['entropy_confidence_cosine']:.4f}")

    print("\n--- Interpretation ---")
    intro_ent_cos = ds['introspection_entropy_cosine']
    intro_conf_cos = ds['introspection_confidence_cosine']

    if abs(intro_ent_cos) > 0.7 and abs(intro_conf_cos) > 0.7:
        print("→ Introspection direction is highly aligned with both entropy and confidence")
        print("  (May just be capturing their shared variance)")
    elif abs(intro_ent_cos) > 0.5 or abs(intro_conf_cos) > 0.5:
        print("→ Introspection direction partially overlaps with entropy/confidence")
        print("  (Captures alignment but has some unique component)")
    else:
        print("→ Introspection direction is relatively distinct from both")
        print("  (May be capturing a genuine 'alignment detection' signal)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Device: {DEVICE}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Compute input/output paths from prefix
    paired_data_path = f"{output_prefix}_paired_data.json"
    meta_activations_path = f"{output_prefix}_meta_activations.npz"

    # Load paired data
    print(f"\nLoading paired data from {paired_data_path}...")
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)

    questions = paired_data["questions"]
    direct_entropies = np.array(paired_data["direct_entropies"])
    meta_responses = paired_data["meta_responses"]

    # Subset if requested
    if NUM_QUESTIONS_TO_TEST is not None and NUM_QUESTIONS_TO_TEST < len(questions):
        print(f"Using subset of {NUM_QUESTIONS_TO_TEST} questions")
        indices = list(range(NUM_QUESTIONS_TO_TEST))
        questions = [questions[i] for i in indices]
        direct_entropies = direct_entropies[indices]
        meta_responses = [meta_responses[i] for i in indices]

    print(f"Loaded {len(questions)} questions")

    # Compute introspection scores
    print("\nComputing introspection scores...")
    introspection_scores, score_stats, entropy_z, confidence_z = compute_introspection_scores(
        direct_entropies, meta_responses, method="continuous"
    )

    print(f"  Correlation (entropy, confidence): {score_stats['correlation_entropy_confidence']:.4f}")
    print(f"  Fraction aligned: {score_stats['fraction_aligned']:.2%}")

    # Load activations
    print(f"\nLoading activations...")
    meta_acts_data = np.load(meta_activations_path)

    if NUM_QUESTIONS_TO_TEST is not None:
        meta_activations = {
            int(k.split("_")[1]): meta_acts_data[k][:NUM_QUESTIONS_TO_TEST]
            for k in meta_acts_data.files if k.startswith("layer_")
        }
    else:
        meta_activations = {
            int(k.split("_")[1]): meta_acts_data[k]
            for k in meta_acts_data.files if k.startswith("layer_")
        }

    print(f"Loaded {len(meta_activations)} layers")

    # Run analysis
    results, test_idx = run_introspection_direction_analysis(
        meta_activations, direct_entropies, introspection_scores, entropy_z, confidence_z
    )

    # Add score stats to results
    results["score_stats"] = score_stats

    # Save results
    print("\nSaving results...")

    # Save full results (without directions to keep file size reasonable)
    results_to_save = {
        "score_stats": score_stats,
        "train_size": results["train_size"],
        "test_size": results["test_size"],
        "test_indices": test_idx.tolist(),
        "layer_results": {}
    }

    for layer_idx, layer_results in results["layer_results"].items():
        results_to_save["layer_results"][layer_idx] = {
            k: v for k, v in layer_results.items() if k != "directions"
        }

    with open(f"{output_prefix}_direction_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved {output_prefix}_direction_results.json")

    # Save directions separately (they're large)
    directions_to_save = {
        layer_idx: results["layer_results"][layer_idx]["directions"]
        for layer_idx in results["layer_results"]
    }
    np.savez_compressed(
        f"{output_prefix}_direction_vectors.npz",
        **{f"layer_{l}_{d}": np.array(directions_to_save[l][d])
           for l in directions_to_save for d in ["introspection", "entropy", "confidence"]}
    )
    print(f"Saved {output_prefix}_direction_vectors.npz")

    # Save introspection scores for further analysis
    scores_data = {
        "introspection_scores": introspection_scores.tolist(),
        "entropy_z": entropy_z.tolist(),
        "confidence_z": confidence_z.tolist(),
        "direct_entropies": direct_entropies.tolist(),
        "meta_responses": meta_responses,
        "score_stats": score_stats,
    }
    with open(f"{output_prefix}_scores_data.json", "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"Saved {output_prefix}_scores_data.json")

    # Print summary
    print_results_summary(results, score_stats)

    # Create visualizations
    # Convert confidence_z from stated_confidences
    stated_confidences = np.array([META_RANGE_MIDPOINTS.get(r, 0.5) for r in meta_responses])
    confidence_z_plot = stats.zscore(stated_confidences)

    plot_introspection_analysis(
        results, introspection_scores, direct_entropies, confidence_z_plot, meta_responses,
        output_prefix=output_prefix
    )

    print("\n✓ Introspection direction analysis complete!")

    # ========================================================================
    # STEERING EXPERIMENT
    # ========================================================================
    if RUN_STEERING:
        print("\n" + "=" * 80)
        print("RUNNING STEERING EXPERIMENT")
        print("=" * 80)

        # Load model for steering
        print(f"\nLoading model: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Left-pad for proper batched generation

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            token=HF_TOKEN
        )

        if MODEL_NAME != BASE_MODEL_NAME:
            try:
                from peft import PeftModel
                print(f"Loading fine-tuned model: {MODEL_NAME}")
                model = PeftModel.from_pretrained(model, MODEL_NAME)
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}")
                return

        # Determine whether to use chat template
        use_chat_template = has_chat_template(tokenizer) and not is_base_model(BASE_MODEL_NAME)
        print(f"Using chat template: {use_chat_template}")

        # Get directions for steering layers
        introspection_directions = {}
        entropy_directions = {}
        for layer_idx in STEERING_LAYERS:
            if layer_idx in directions_to_save:
                introspection_directions[layer_idx] = np.array(directions_to_save[layer_idx]["introspection"])
                entropy_directions[layer_idx] = np.array(directions_to_save[layer_idx]["entropy"])

        # Subset questions for steering
        steering_questions = questions[:NUM_STEERING_QUESTIONS]
        steering_entropies = direct_entropies[:NUM_STEERING_QUESTIONS]

        # Run steering experiment
        steering_results = run_introspection_steering_experiment(
            model, tokenizer, steering_questions, steering_entropies,
            introspection_directions, entropy_directions, use_chat_template
        )

        # Analyze steering results
        steering_analysis = analyze_steering_results(steering_results)

        # Save steering results
        with open(f"{output_prefix}_steering_results.json", "w") as f:
            json.dump(steering_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print(f"Saved {output_prefix}_steering_results.json")

        with open(f"{output_prefix}_steering_analysis.json", "w") as f:
            json.dump(steering_analysis, f, indent=2)
        print(f"Saved {output_prefix}_steering_analysis.json")

        # Print and plot steering results
        print_steering_summary(steering_analysis)
        plot_steering_results(steering_analysis, output_prefix=output_prefix)

        print("\n✓ Steering experiment complete!")


if __name__ == "__main__":
    main()
