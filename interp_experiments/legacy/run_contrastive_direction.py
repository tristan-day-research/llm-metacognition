"""
Find introspection mapping direction using contrastive approach.

Instead of regression, this script:
1. Loads introspection data (direct entropies, stated confidences, meta activations)
2. Selects only calibrated examples (where confidence correctly tracks entropy)
3. Within calibrated examples, contrasts:
   - High confidence + low entropy (correctly confident)
   - Low confidence + high entropy (correctly uncertain)
4. Computes direction = mean(high_conf_low_ent) - mean(low_conf_high_ent)
5. Evaluates direction quality (how well it predicts introspection score)
6. Runs steering/ablation experiments to test causality

This captures the confidence axis within calibrated examples - steering along this
direction should shift confidence while maintaining calibration.
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
import matplotlib.pyplot as plt
from scipy import stats
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from core import (
    DEVICE,
    load_model_and_tokenizer,
    compute_introspection_scores,
    generate_orthogonal_directions,
)
from core.steering import SteeringHook, AblationHook
from core.steering_experiments import (
    SteeringExperimentConfig,
    run_steering_experiment as shared_run_steering_experiment,
    run_ablation_experiment as shared_run_ablation_experiment,
    analyze_steering_results as shared_analyze_steering_results,
    analyze_ablation_results as shared_analyze_ablation_results,
    print_steering_summary,
    print_ablation_summary,
    precompute_direction_tensors,
)
from core.probes import (
    compute_cluster_centroids,
    compute_cluster_directions,
    compute_caa_direction,
    compare_directions,
)
from prompts import (
    # Confidence task
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    # Delegate task
    ANSWER_OR_DELEGATE_SETUP,
    ANSWER_OR_DELEGATE_SYSPROMPT,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    get_delegate_mapping,
    # Unified conversion
    response_to_confidence as tasks_response_to_confidence,
)

# Configuration — edit values in experiment_config.ContrastiveDirectionConfig
from experiment_config import ContrastiveDirectionConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
MODEL_NAME = _C.MODEL_NAME
DATASET_NAME = _C.DATASET_NAME
SEED = _C.SEED
VALID_METRICS = list(_C.VALID_METRICS)
METRIC_HIGHER_IS_CONFIDENT = dict(_C.METRIC_HIGHER_IS_CONFIDENT)
METRIC_KEY_MAP = dict(_C.METRIC_KEY_MAP)
METRIC = _C.METRIC
DIRECTION_TYPES = list(_C.DIRECTION_TYPES)
OUTPUTS_DIR = _C.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(exist_ok=True)


def compute_alignment(metric_z: float, confidence_z: float, metric_name: str) -> float:
    """
    Compute alignment score between metric and stated confidence.

    Alignment is positive when confidence matches uncertainty:
    - High confidence + low uncertainty (confident and should be)
    - Low confidence + high uncertainty (uncertain and knows it)

    Args:
        metric_z: Z-scored metric value
        confidence_z: Z-scored confidence value
        metric_name: Name of the metric to determine polarity

    Returns:
        Alignment score (positive = well-calibrated)
    """
    # For entropy: high value = uncertain, so negate to get "confidence-like" direction
    # For other metrics: high value = confident, use as-is
    if METRIC_HIGHER_IS_CONFIDENT.get(metric_name, False):
        # High metric = confident, so positive correlation with confidence is good
        return metric_z * confidence_z
    else:
        # High metric = uncertain (entropy), so negative correlation is good
        return -metric_z * confidence_z


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from a model path."""
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""

    # Determine direction type suffix
    if DIRECTION_TYPES is None or len(DIRECTION_TYPES) == 0:
        dir_suffix = "all_directions"
    elif len(DIRECTION_TYPES) == 1:
        dir_suffix = DIRECTION_TYPES[0]
    else:
        dir_suffix = "multi_directions"

    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_{METRIC}{task_suffix}_{dir_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_{METRIC}{task_suffix}_{dir_suffix}")

# Contrastive selection thresholds
# Use top/bottom quantiles of introspection score
TOP_QUANTILE = 0.25  # Top 25% = well-calibrated
BOTTOM_QUANTILE = 0.25  # Bottom 25% = miscalibrated

# Which layer to use for direction extraction
TARGET_LAYER = None  # Will be set to best layer from probe results, or middle layer

# Steering/ablation configuration
RUN_STEERING = True  # Set to False to skip steering experiments
META_TASK = "confidence"  # "confidence" or "delegate" - must match run_introspection_experiment.py
STEERING_LAYERS = None  # None = auto-select based on projection correlation
STEERING_MULTIPLIERS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
NUM_STEERING_QUESTIONS = 100
# Number of control directions per layer. Set to None for dynamic scaling.
# Dynamic scaling ensures enough power for FDR correction:
#   - For FDR at α=0.05 with N layers, need min p-value < 0.05/N
#   - min p-value ≈ 1/(pooled_samples), so need pooled_samples > 20*N
#   - We use FDR_SAFETY_FACTOR * N for margin, giving controls_per_layer = FDR_SAFETY_FACTOR
NUM_CONTROL_DIRECTIONS = None  # None = dynamic based on num_layers, or set explicit value
FDR_ALPHA = 0.05              # Target FDR significance level
FDR_SAFETY_FACTOR = 25        # Multiplier: pooled_samples = FDR_SAFETY_FACTOR * num_layers
MIN_CONTROLS_PER_LAYER = 10   # Minimum controls even for few layers
# Layer selection for steering: use principled criteria
# - Require statistical significance (p < MIN_PROJECTION_PVAL)
# - Require minimum effect size (R² > MIN_PROJECTION_R2, i.e., |r| > sqrt(MIN_PROJECTION_R2))
MIN_PROJECTION_PVAL = 0.001  # Require p < 0.001 for layer to be included
MIN_PROJECTION_R2 = 0.05     # Require R² > 0.05 (|r| > 0.224) - direction explains 5%+ of variance
MAX_STEERING_LAYERS = 10     # Maximum number of layers to steer (selects top N by correlation)
STEERING_BATCH_SIZE = 8  # Batch size for steering/ablation (matches run_introspection_steering.py)

# Quantization options for large models (70B+)
LOAD_IN_4BIT = False  # Recommended for 70B models on consumer GPUs
LOAD_IN_8BIT = False  # Alternative to 4-bit, slightly better quality

# Batch optimization settings (for efficient steering/ablation)
# When sweeping k multipliers simultaneously, we expand each base batch by k.
# This sets the TARGET total expanded batch size (base_batch * k_mult).
EXPANDED_BATCH_TARGET = 48  # With k_mult=6 and base batch=8, expanded=48
INTERVENTION_POSITION = "last"  # "last" = only modify final token (enables KV cache), "all" = modify all tokens

# Direction comparison mode
COMPARE_DIRECTIONS = True  # Set to True to compare different direction types
N_CLUSTERS = 3  # Number of clusters for cluster-based directions (low/mid/high)
CLUSTER_METHOD = "quantile"  # "quantile" (group by metric percentiles) or "kmeans"

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Backward compatibility aliases (now imported from tasks.py)
META_OPTION_DICT = STATED_CONFIDENCE_OPTIONS
META_RANGE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())

DELEGATE_SETUP_PROMPT = ANSWER_OR_DELEGATE_SETUP
DELEGATE_SYSPROMPT = ANSWER_OR_DELEGATE_SYSPROMPT
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS

# Cached token IDs - populated once at startup to avoid repeated tokenization
_CACHED_TOKEN_IDS = {
    "meta_options": None,
    "delegate_options": None,
}


def initialize_token_cache(tokenizer):
    """Precompute option token IDs once to avoid repeated tokenization."""
    _CACHED_TOKEN_IDS["meta_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in META_OPTIONS
    ]
    _CACHED_TOKEN_IDS["delegate_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in DELEGATE_OPTIONS
    ]
    print(f"  Cached token IDs: meta={_CACHED_TOKEN_IDS['meta_options']}, delegate={_CACHED_TOKEN_IDS['delegate_options']}")


# ============================================================================
# KV CACHE HELPERS (for efficient steering/ablation)
# ============================================================================

# Import DynamicCache safely
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


def extract_cache_tensors(past_key_values):
    """Extract raw tensors from past_key_values (tuple or DynamicCache)."""
    keys = []
    values = []
    try:
        num_layers = len(past_key_values)
    except TypeError:
        if hasattr(past_key_values, "to_legacy_cache"):
            return extract_cache_tensors(past_key_values.to_legacy_cache())
        raise ValueError(f"Cannot determine length of cache: {type(past_key_values)}")
    for i in range(num_layers):
        k, v = past_key_values[i]
        keys.append(k)
        values.append(v)
    return keys, values


def create_fresh_cache(key_tensors, value_tensors, expand_size=1):
    """Create a fresh DynamicCache (or tuple) from tensors."""
    if DynamicCache is not None:
        cache = DynamicCache()
        for i, (k, v) in enumerate(zip(key_tensors, value_tensors)):
            if expand_size > 1:
                k = k.repeat_interleave(expand_size, dim=0)
                v = v.repeat_interleave(expand_size, dim=0)
            cache.update(k, v, i)
        return cache
    else:
        layers = []
        for k, v in zip(key_tensors, value_tensors):
            if expand_size > 1:
                k = k.repeat_interleave(expand_size, dim=0)
                v = v.repeat_interleave(expand_size, dim=0)
            layers.append((k, v))
        return tuple(layers)


def get_kv_cache(model, batch_inputs):
    """Run the prefix to generate KV cache tensors."""
    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]

    # Run Prefix (Tokens 0 to T-1)
    prefix_ids = input_ids[:, :-1]
    prefix_mask = attention_mask[:, :-1]

    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
        )

    # Extract Immutable Snapshot
    keys, values = extract_cache_tensors(outputs.past_key_values)

    # Prepare next step inputs
    last_ids = input_ids[:, -1:]

    result = {
        "input_ids": last_ids,
        "attention_mask": attention_mask,
        "past_key_values_data": (keys, values),
    }

    if "position_ids" in batch_inputs:
        result["position_ids"] = batch_inputs["position_ids"][:, -1:]

    return result


def pretokenize_prompts(prompts: List[str], tokenizer, device: str) -> Dict:
    """Pre-tokenize all prompts once (BPE encoding)."""
    tokenized = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        return_attention_mask=True
    )
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    sorted_order = sorted(range(len(prompts)), key=lambda i: lengths[i])

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "lengths": lengths,
        "sorted_order": sorted_order,
        "device": device,
        "tokenizer": tokenizer,
    }


def build_padded_gpu_batches(
    cached_inputs: Dict,
    tokenizer,
    device: str,
    batch_size: int,
) -> List[Tuple[List[int], Dict[str, torch.Tensor]]]:
    """Pad each length-sorted batch once and keep tensors on-device."""
    sorted_order = cached_inputs["sorted_order"]
    batches: List[Tuple[List[int], Dict[str, torch.Tensor]]] = []

    for batch_start in range(0, len(sorted_order), batch_size):
        batch_indices = sorted_order[batch_start:batch_start + batch_size]
        batch_input_ids = [cached_inputs["input_ids"][i] for i in batch_indices]
        batch_attention = [cached_inputs["attention_mask"][i] for i in batch_indices]

        batch_inputs = tokenizer.pad(
            {"input_ids": batch_input_ids, "attention_mask": batch_attention},
            return_tensors="pt",
            padding=True,
        )
        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}
        batches.append((batch_indices, batch_inputs))

    return batches


# ============================================================================
# BATCH STEERING/ABLATION HOOKS
# ============================================================================

class BatchSteeringHook:
    """Hook that adds a *per-example* steering delta to activations.

    Designed for "multiplier sweep in one pass" by expanding the batch:
    each prompt is duplicated for each multiplier, and this hook adds a different
    delta vector for each expanded example.
    """

    def __init__(self, delta_bh: Optional[torch.Tensor] = None):
        self.delta_bh = delta_bh  # (batch, hidden)
        self.handle = None

    def set_delta(self, delta_bh: torch.Tensor):
        self.delta_bh = delta_bh

    def __call__(self, module, input, output):
        if self.delta_bh is None:
            return output

        hs = output[0] if isinstance(output, tuple) else output
        delta = self.delta_bh.to(device=hs.device, dtype=hs.dtype)

        if INTERVENTION_POSITION == "last":
            hs = hs.clone()
            hs[:, -1, :] = hs[:, -1, :] + delta
        else:
            hs = hs + delta[:, None, :]

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class BatchAblationHook:
    """Hook that projects out a *per-example* direction from activations.

    For batched ablation: each prompt is duplicated for each direction,
    and this hook removes a different direction for each expanded example.
    """

    def __init__(self, directions_bh: Optional[torch.Tensor] = None):
        self.directions_bh = directions_bh
        self.handle = None

    def set_directions(self, directions_bh: torch.Tensor):
        self.directions_bh = directions_bh

    def __call__(self, module, input, output):
        if self.directions_bh is None:
            return output

        hs = output[0] if isinstance(output, tuple) else output
        dirs = self.directions_bh.to(device=hs.device, dtype=hs.dtype)

        if INTERVENTION_POSITION == "last":
            hs = hs.clone()
            last_token = hs[:, -1, :]
            dots = torch.einsum('bh,bh->b', last_token, dirs)
            proj = dots.unsqueeze(-1) * dirs
            hs[:, -1, :] = last_token - proj
        else:
            dots = torch.einsum('bsh,bh->bs', hs, dirs)
            proj = dots.unsqueeze(-1) * dirs.unsqueeze(1)
            hs = hs - proj

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def precompute_direction_tensors(
    directions: Dict[int, np.ndarray],
    layers: List[int],
    num_controls: int,
    device: str,
    dtype: torch.dtype
) -> Dict:
    """Precompute normalized direction tensors on GPU for all layers and controls."""
    cached = {}
    for layer_idx in layers:
        direction = directions[layer_idx]
        direction = direction / np.linalg.norm(direction)
        direction_tensor = torch.tensor(direction, dtype=dtype, device=device)

        control_dirs = generate_orthogonal_directions(direction, num_controls)
        control_tensors = [
            torch.tensor(cd, dtype=dtype, device=device)
            for cd in control_dirs
        ]

        cached[layer_idx] = {
            "direction": direction_tensor,
            "controls": control_tensors,
        }

    return cached


def get_introspection_prefix() -> str:
    """Get prefix for introspection data files (from run_introspection_experiment.py)."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


def load_introspection_data(run_name: str = None, metric: str = "entropy") -> dict:
    """
    Load previously collected introspection data.

    Looks for:
    - {run_name}_paired_data.json (or computed from config)
    - {run_name}_meta_activations.npz

    Args:
        run_name: Optional run name prefix for data files
        metric: Which metric to load ("entropy", "top_prob", "margin", "logit_gap", "top_logit")
    """
    # Try to find data files
    if run_name:
        paired_path = Path(f"{run_name}_paired_data.json")
        acts_path = Path(f"{run_name}_meta_activations.npz")
    else:
        prefix = get_introspection_prefix()
        paired_path = Path(f"{prefix}_paired_data.json")
        acts_path = Path(f"{prefix}_meta_activations.npz")

    if not paired_path.exists():
        raise FileNotFoundError(
            f"Could not find {paired_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    if not acts_path.exists():
        raise FileNotFoundError(
            f"Could not find {acts_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    # Load paired data
    print(f"Loading paired data from {paired_path}...")
    with open(paired_path) as f:
        paired_data = json.load(f)

    # Extract arrays - handle both old format (list of dicts) and new format (dict with arrays)
    if isinstance(paired_data, list):
        # Old format: list of {"direct_entropy": ..., "stated_confidence": ...}
        metric_values = np.array([d["direct_entropy"] for d in paired_data])
        stated_confidences = np.array([d["stated_confidence"] for d in paired_data])
    else:
        # New format from run_introspection_experiment.py
        # Check for nested direct_metrics format first
        if "direct_metrics" in paired_data and isinstance(paired_data["direct_metrics"], dict):
            # Nested format: {"direct_metrics": {"entropy": [...], "top_prob": [...], ...}}
            direct_metrics = paired_data["direct_metrics"]
            if metric not in direct_metrics:
                available = list(direct_metrics.keys())
                raise ValueError(f"Metric '{metric}' not found in direct_metrics. Available: {available}")
            metric_values = np.array(direct_metrics[metric])
        else:
            # Flat format: {"direct_entropies": [...], "direct_top_probs": [...], ...}
            metric_key = METRIC_KEY_MAP.get(metric, "direct_entropies")
            if metric_key not in paired_data:
                available = [k for k in paired_data.keys() if k.startswith("direct_")]
                raise ValueError(f"Metric '{metric}' (key: {metric_key}) not found in data. Available: {available}")
            metric_values = np.array(paired_data[metric_key])

        meta_responses = paired_data["meta_responses"]
        meta_probs = paired_data.get("meta_probs")
        meta_mappings = paired_data.get("meta_mappings")
        meta_task = paired_data.get("config", {}).get("meta_task", "confidence")

        # Convert meta responses to confidence values
        if meta_task == "delegate":
            # For delegate task, confidence = P(Answer)
            stated_confidences = []
            for i, (response, probs, mapping) in enumerate(zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses),
                meta_mappings or [None] * len(meta_responses)
            )):
                if probs is not None and len(probs) >= 2 and mapping is not None:
                    if mapping.get("1") == "Answer":
                        stated_confidences.append(float(probs[0]))
                    else:
                        stated_confidences.append(float(probs[1]))
                elif probs is not None and len(probs) >= 1:
                    stated_confidences.append(float(probs[0]))
                else:
                    stated_confidences.append(1.0 if response == "1" else 0.0)
            stated_confidences = np.array(stated_confidences)
        else:
            # For confidence task, use logprob-weighted expected value
            # This computes E[confidence] = sum(P(option) * midpoint)
            META_RANGE_MIDPOINTS = {
                "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
                "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
            }
            options = list(META_RANGE_MIDPOINTS.keys())
            midpoints = np.array([META_RANGE_MIDPOINTS[opt] for opt in options])

            stated_confidences = []
            for i, (response, probs) in enumerate(zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses)
            )):
                if probs is not None and len(probs) == len(midpoints):
                    # Logprob-weighted expected value: sum(P(option) * midpoint)
                    probs_arr = np.array(probs)
                    expected_conf = float(np.dot(probs_arr, midpoints))
                    stated_confidences.append(expected_conf)
                else:
                    # Fallback to argmax response midpoint if probs unavailable
                    stated_confidences.append(META_RANGE_MIDPOINTS.get(response, 0.5))
            stated_confidences = np.array(stated_confidences)

    # Load meta activations
    print(f"Loading meta activations from {acts_path}...")
    acts_data = np.load(acts_path)

    # activations are stored as layer_0, layer_1, etc.
    layer_keys = sorted([k for k in acts_data.keys() if k.startswith("layer_")])
    num_layers = len(layer_keys)

    meta_activations = {
        int(k.split("_")[1]): acts_data[k]
        for k in layer_keys
    }

    n_examples = len(metric_values) if isinstance(paired_data, dict) else len(paired_data)
    print(f"Loaded {n_examples} examples with {num_layers} layers")
    print(f"{metric.capitalize()} range: [{metric_values.min():.3f}, {metric_values.max():.3f}]")
    print(f"Confidence range: [{stated_confidences.min():.3f}, {stated_confidences.max():.3f}]")

    return {
        "paired_data": paired_data,
        "metric_values": metric_values,
        "metric_name": metric,
        "stated_confidences": stated_confidences,
        "meta_activations": meta_activations,
        "num_layers": num_layers,
    }


# ============================================================================
# STEERING HELPERS (wrappers around tasks.py functions)
# ============================================================================

# Import utility functions from core
from core.model_utils import is_base_model, has_chat_template


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> str:
    """Format a meta/confidence question using centralized tasks.py logic."""
    full_prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
    return full_prompt


def format_delegate_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    trial_index: int = 0
) -> Tuple[str, List[str], Dict[str, str]]:
    """Format a delegate question using centralized tasks.py logic."""
    return format_answer_or_delegate_prompt(
        question, tokenizer, trial_index=trial_index,
        alternate_mapping=True, use_chat_template=use_chat_template
    )


def response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None
) -> float:
    """
    Convert a meta response to a confidence value.

    Wrapper around tasks.response_to_confidence that passes the correct task_type.
    """
    task_type = "delegate" if META_TASK == "delegate" else "confidence"
    return tasks_response_to_confidence(response, probs, mapping, task_type)


def get_confidence_response(
    model,
    tokenizer,
    question: Dict,
    layer_idx: Optional[int],
    steering_direction: Optional[np.ndarray],
    multiplier: float,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response with optional steering."""
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    if layer_idx is not None and steering_direction is not None and multiplier != 0.0:
        steering_tensor = torch.tensor(
            steering_direction,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        hook = SteeringHook(steering_tensor, multiplier)
        handle = layer_module.register_forward_hook(hook)

        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
        finally:
            handle.remove()
    else:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

    final_logits = outputs.logits[0, -1, :]
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    option_logits = final_logits[option_token_ids]
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


def get_batch_confidence_responses(
    model,
    tokenizer,
    questions: List[Dict],
    layer_idx: Optional[int],
    steering_direction: Optional[np.ndarray],
    multiplier: float,
    use_chat_template: bool,
    batch_size: int = 8
) -> List[Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]]:
    """
    Get confidence responses for a batch of questions with optional steering.

    Much more efficient than calling get_confidence_response one at a time.
    """
    results = []

    # Pre-compute option token IDs
    if META_TASK == "delegate":
        options = DELEGATE_OPTIONS
    else:
        options = META_OPTIONS
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]

    # Set up steering hook once if needed
    handle = None
    if layer_idx is not None and steering_direction is not None and multiplier != 0.0:
        steering_tensor = torch.tensor(
            steering_direction,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        hook = SteeringHook(steering_tensor, multiplier)
        handle = layer_module.register_forward_hook(hook)

    try:
        # Process in batches
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

            # Format prompts and collect mappings
            prompts = []
            mappings = []
            for i, q in enumerate(batch_questions):
                trial_idx = batch_start + i
                if META_TASK == "delegate":
                    prompt, _, mapping = format_delegate_prompt(q, tokenizer, use_chat_template, trial_idx)
                else:
                    prompt = format_meta_prompt(q, tokenizer, use_chat_template)
                    mapping = None
                prompts.append(prompt)
                mappings.append(mapping)

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)

            # Extract confidence for each item in batch
            # With left-padding, the last position (-1) is always the final real token
            for i in range(len(batch_questions)):
                final_logits = outputs.logits[i, -1, :]
                option_logits = final_logits[option_token_ids]
                option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

                response = options[np.argmax(option_probs)]
                confidence = response_to_confidence(response, option_probs, mappings[i])

                results.append((response, confidence, option_probs, mappings[i]))

            del inputs, outputs

    finally:
        if handle is not None:
            handle.remove()

    return results


def get_batch_confidence_with_ablation(
    model,
    tokenizer,
    questions: List[Dict],
    layer_idx: int,
    ablation_direction: np.ndarray,
    use_chat_template: bool,
    batch_size: int = 8
) -> List[Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]]:
    """
    Get confidence responses for a batch of questions with ablation.
    """
    results = []

    # Pre-compute option token IDs
    if META_TASK == "delegate":
        options = DELEGATE_OPTIONS
    else:
        options = META_OPTIONS
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]

    # Set up ablation hook
    ablation_tensor = torch.tensor(
        ablation_direction,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    if hasattr(model, 'get_base_model'):
        layer_module = model.get_base_model().model.layers[layer_idx]
    else:
        layer_module = model.model.layers[layer_idx]

    hook = AblationHook(ablation_tensor)
    handle = layer_module.register_forward_hook(hook)

    try:
        # Process in batches
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

            # Format prompts and collect mappings
            prompts = []
            mappings = []
            for i, q in enumerate(batch_questions):
                trial_idx = batch_start + i
                if META_TASK == "delegate":
                    prompt, _, mapping = format_delegate_prompt(q, tokenizer, use_chat_template, trial_idx)
                else:
                    prompt = format_meta_prompt(q, tokenizer, use_chat_template)
                    mapping = None
                prompts.append(prompt)
                mappings.append(mapping)

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)

            # Extract confidence for each item in batch
            # With left-padding, the last position (-1) is always the final real token
            for i in range(len(batch_questions)):
                final_logits = outputs.logits[i, -1, :]
                option_logits = final_logits[option_token_ids]
                option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

                response = options[np.argmax(option_probs)]
                confidence = response_to_confidence(response, option_probs, mappings[i])

                results.append((response, confidence, option_probs, mappings[i]))

            del inputs, outputs

    finally:
        handle.remove()

    return results


def get_confidence_with_ablation(
    model,
    tokenizer,
    question: Dict,
    layer_idx: int,
    ablation_direction: np.ndarray,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response with ablation (direction zeroed out)."""
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    ablation_tensor = torch.tensor(
        ablation_direction,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    if hasattr(model, 'get_base_model'):
        layer_module = model.get_base_model().model.layers[layer_idx]
    else:
        layer_module = model.model.layers[layer_idx]

    hook = AblationHook(ablation_tensor)
    handle = layer_module.register_forward_hook(hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    final_logits = outputs.logits[0, -1, :]
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    option_logits = final_logits[option_token_ids]
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


# ============================================================================
# STEERING AND ABLATION EXPERIMENTS
# ============================================================================

def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    directions: Dict[int, np.ndarray],
    multipliers: List[float],
    num_controls: int,
    use_chat_template: bool,
    batch_size: int = 8
) -> Dict:
    """Run steering experiment with KV cache optimization + batched multipliers.

    Key optimizations:
    1. KV cache: Compute prefix KV cache once per batch, reuse for each condition
    2. Batched multipliers: Process all multipliers in one forward pass by expanding batch
    3. Pre-tokenization: Tokenize all prompts once upfront
    """
    print(f"\nRunning steering experiment (KV Cache + batched multipliers)...")
    print(f"  Layers: {layers}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Control directions: {num_controls}")
    print(f"  Batch size: {batch_size}")

    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    # Pre-format and tokenize all prompts
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "delegate":
            prompt, _, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index=q_idx)
        else:
            prompt = format_meta_prompt(question, tokenizer, use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)

    # Get option token IDs
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = DELEGATE_OPTIONS
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = META_OPTIONS

    # Precompute direction tensors
    cached_directions = precompute_direction_tensors(
        directions, layers, num_controls, DEVICE,
        torch.float16 if DEVICE == "cuda" else torch.float32
    )

    nonzero_multipliers = [m for m in multipliers if m != 0.0]
    k_mult = len(nonzero_multipliers)

    # Initialize results
    shared_baseline = [None] * len(questions)
    final_layer_results = {}
    for l in layers:
        final_layer_results[l] = {
            "baseline": shared_baseline,
            "contrastive": {m: [None] * len(questions) for m in multipliers},
            "controls": {f"control_{i}": {m: [None] * len(questions) for m in multipliers} for i in range(num_controls)},
        }
        if 0.0 in multipliers:
            final_layer_results[l]["contrastive"][0.0] = shared_baseline
            for k in final_layer_results[l]["controls"]:
                final_layer_results[l]["controls"][k][0.0] = shared_baseline

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, batch_size)
    n_dirs = 1 + num_controls
    print(f"Processing {len(gpu_batches)} batches × {len(layers)} layers × {n_dirs} directions...")
    print(f"  (batch size {batch_size}, {k_mult} multipliers per pass)")

    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Batches"):
        B = len(batch_indices)

        # 1. Compute KV Cache (Snapshot)
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        # 2. Compute Baseline (No steering)
        fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)

        baseline_inputs = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "past_key_values": fresh_cache,
            "use_cache": True
        }
        if "position_ids" in base_step_data:
            baseline_inputs["position_ids"] = base_step_data["position_ids"]

        with torch.inference_mode():
            out = model(**baseline_inputs)
            logits = out.logits[:, -1, :][:, option_token_ids]
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

        for i, q_idx in enumerate(batch_indices):
            p = probs[i]
            resp = options[np.argmax(p)]
            conf = response_to_confidence(resp, p, mappings[q_idx])
            m_val = direct_metric_values[q_idx]
            align = compute_alignment((m_val - metric_mean)/metric_std, (conf - 0.5)/0.25, METRIC)
            shared_baseline[q_idx] = {
                "question_idx": q_idx, "response": resp, "confidence": conf,
                "metric": float(m_val), "alignment": float(align)
            }

        # 3. Iterate Layers with batched direction processing
        # Expand inputs once for multiplier batching (same memory as run_introspection_steering)
        expanded_input_ids = base_step_data["input_ids"].repeat_interleave(k_mult, dim=0)
        expanded_attention_mask = base_step_data["attention_mask"].repeat_interleave(k_mult, dim=0)
        expanded_position_ids = None
        if "position_ids" in base_step_data:
            expanded_position_ids = base_step_data["position_ids"].repeat_interleave(k_mult, dim=0)

        for layer_idx in layers:
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer_idx]
            else:
                layer_module = model.model.layers[layer_idx]

            direction_tensor = cached_directions[layer_idx]["direction"]
            control_tensors = cached_directions[layer_idx]["controls"]
            all_directions = [direction_tensor] + control_tensors  # primary + controls

            hook = BatchSteeringHook()
            hook.register(layer_module)

            def run_batched_mult_sweep(direction_vector, result_dict):
                # 1. Create FRESH cache expanded for this pass
                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_mult)

                # 2. Attach cache to inputs
                current_inputs = {
                    "input_ids": expanded_input_ids,
                    "attention_mask": expanded_attention_mask,
                    "past_key_values": pass_cache,
                    "use_cache": True
                }
                if expanded_position_ids is not None:
                    current_inputs["position_ids"] = expanded_position_ids

                # 3. Set Deltas
                deltas = []
                for _ in range(B):
                    for mult in nonzero_multipliers:
                        deltas.append(direction_vector * mult)
                delta_bh = torch.stack(deltas, dim=0)
                hook.set_delta(delta_bh)

                # 4. Run Model
                with torch.inference_mode():
                    out = model(**current_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                # 5. Store Results
                for i, q_idx in enumerate(batch_indices):
                    for j, mult in enumerate(nonzero_multipliers):
                        idx = i * k_mult + j
                        p = probs[idx]
                        resp = options[np.argmax(p)]
                        conf = response_to_confidence(resp, p, mappings[q_idx])
                        m_val = direct_metric_values[q_idx]
                        align = compute_alignment((m_val - metric_mean)/metric_std, (conf - 0.5)/0.25, METRIC)

                        result_dict[mult][q_idx] = {
                            "question_idx": q_idx, "response": resp, "confidence": conf,
                            "metric": float(m_val), "alignment": float(align)
                        }

            try:
                # Contrastive direction
                run_batched_mult_sweep(direction_tensor, final_layer_results[layer_idx]["contrastive"])
                # Control directions
                for i_c, ctrl_dir in enumerate(control_tensors):
                    run_batched_mult_sweep(ctrl_dir, final_layer_results[layer_idx]["controls"][f"control_{i_c}"])
            finally:
                hook.remove()

            # Clear cache after each layer to prevent memory accumulation on large models
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results,
        "direction_key": "contrastive",  # For shared analysis functions
    }


def run_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    directions: Dict[int, np.ndarray],
    num_controls: int,
    use_chat_template: bool,
    baseline_results: Optional[List[Dict]] = None,
    batch_size: int = 8
) -> Dict:
    """Run ablation experiment with KV cache optimization.

    Key optimizations:
    1. KV cache: Compute prefix KV cache once per batch, reuse for each condition
    2. Pre-tokenization: Tokenize all prompts once upfront
    3. BatchAblationHook: Apply per-example ablation in batched forward passes
    """
    print(f"\nRunning ablation experiment (KV Cache)...")
    print(f"  Layers: {layers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Control directions: {num_controls}")
    print(f"  Batch size: {batch_size}")
    if baseline_results is not None:
        print(f"  Reusing baseline from steering experiment")

    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    # Pre-format and tokenize all prompts
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "delegate":
            prompt, _, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index=q_idx)
        else:
            prompt = format_meta_prompt(question, tokenizer, use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)

    # Get option token IDs
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = DELEGATE_OPTIONS
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = META_OPTIONS

    # Precompute direction tensors
    cached_directions = precompute_direction_tensors(
        directions, layers, num_controls, DEVICE,
        torch.float16 if DEVICE == "cuda" else torch.float32
    )

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, batch_size)

    # Initialize results
    if baseline_results is None:
        baseline_results = [None] * len(questions)
        need_baseline = True
    else:
        need_baseline = False

    final_layer_results = {}
    for l in layers:
        final_layer_results[l] = {
            "baseline": baseline_results,
            "contrastive_ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    n_dirs = 1 + num_controls
    print(f"Processing {len(gpu_batches)} batches × {len(layers)} layers × {n_dirs} directions...")
    print(f"  (batch size {batch_size})")

    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Batches"):
        B = len(batch_indices)

        # 1. Compute KV Cache (Snapshot)
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        inputs_template = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "use_cache": True
        }
        if "position_ids" in base_step_data:
            inputs_template["position_ids"] = base_step_data["position_ids"]

        # 2. Compute Baseline if needed
        if need_baseline:
            fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
            baseline_inputs = inputs_template.copy()
            baseline_inputs["past_key_values"] = fresh_cache

            with torch.inference_mode():
                out = model(**baseline_inputs)
                logits = out.logits[:, -1, :][:, option_token_ids]
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

            for i, q_idx in enumerate(batch_indices):
                p = probs[i]
                resp = options[np.argmax(p)]
                conf = response_to_confidence(resp, p, mappings[q_idx])
                m_val = direct_metric_values[q_idx]
                align = compute_alignment((m_val - metric_mean)/metric_std, (conf - 0.5)/0.25, METRIC)
                baseline_results[q_idx] = {
                    "question_idx": q_idx, "response": resp, "confidence": conf,
                    "metric": float(m_val), "alignment": float(align)
                }

        # 3. Iterate Layers - process one direction at a time (same memory as introspection script)
        for layer_idx in layers:
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer_idx]
            else:
                layer_module = model.model.layers[layer_idx]

            direction_tensor = cached_directions[layer_idx]["direction"]
            control_tensors = cached_directions[layer_idx]["controls"]

            hook = BatchAblationHook()
            hook.register(layer_module)

            def run_ablation_for_direction(dir_tensor, result_list):
                # Create fresh cache for this direction
                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)

                current_inputs = {
                    "input_ids": base_step_data["input_ids"],
                    "attention_mask": base_step_data["attention_mask"],
                    "past_key_values": pass_cache,
                    "use_cache": True
                }
                if "position_ids" in base_step_data:
                    current_inputs["position_ids"] = base_step_data["position_ids"]

                # Build direction tensor for batch items
                dirs_batch = dir_tensor.unsqueeze(0).expand(B, -1)
                hook.set_directions(dirs_batch)

                # Forward pass for this direction
                with torch.inference_mode():
                    out = model(**current_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                # Unpack results
                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf = response_to_confidence(resp, p, mappings[q_idx])
                    m_val = direct_metric_values[q_idx]
                    align = compute_alignment((m_val - metric_mean)/metric_std, (conf - 0.5)/0.25, METRIC)

                    result_list[q_idx] = {
                        "question_idx": q_idx, "response": resp, "confidence": conf,
                        "metric": float(m_val), "alignment": float(align)
                    }

            try:
                # Contrastive direction ablation
                run_ablation_for_direction(direction_tensor, final_layer_results[layer_idx]["contrastive_ablated"])
                # Control direction ablations
                for i_c, ctrl_dir in enumerate(control_tensors):
                    run_ablation_for_direction(ctrl_dir, final_layer_results[layer_idx]["controls_ablated"][f"control_{i_c}"])
            finally:
                hook.remove()

            # Clear cache after each layer to prevent memory accumulation on large models
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results,
        "direction_key": "contrastive",  # For shared analysis functions
    }


# ============================================================================
# CONTRASTIVE DIRECTION ANALYSIS
# ============================================================================

def compute_calibration_direction_with_details(
    meta_activations: np.ndarray,
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    metric_higher_is_confident: bool = False,
) -> dict:
    """
    Compute calibration direction: calibrated vs uncalibrated examples.

    This is a "pure calibration" direction that captures the difference between
    examples where the model's stated confidence matches its actual uncertainty
    vs examples where they disagree.

    - Calibrated: introspection_score > 0 (stated confidence tracks model confidence)
    - Uncalibrated: introspection_score < 0 (stated confidence inversely tracks model confidence)

    Unlike the contrastive direction (which varies confidence within calibrated samples),
    this direction captures the calibration axis itself.

    Args:
        meta_activations: Activation vectors for each example
        metric_values: The uncertainty metric values (e.g., entropy, top_prob)
        stated_confidences: Model's stated confidence values
        metric_higher_is_confident: If True, higher metric = more confident (e.g., top_prob).
                                   If False, lower metric = more confident (e.g., entropy).

    Returns direction and detailed info about selected examples.
    """
    # Z-score normalize
    metric_z = stats.zscore(metric_values)
    confidence_z = stats.zscore(stated_confidences)

    # Normalize metric direction so positive = more confident
    if not metric_higher_is_confident:
        metric_z_conf = -metric_z  # Invert so positive = more confident
    else:
        metric_z_conf = metric_z

    # Introspection score: positive when calibrated (stated confidence tracks model confidence)
    introspection_scores = metric_z_conf * confidence_z

    # Split by calibration status
    calibrated_mask = introspection_scores > 0
    uncalibrated_mask = introspection_scores < 0

    calibrated_acts = meta_activations[calibrated_mask]
    uncalibrated_acts = meta_activations[uncalibrated_mask]

    if len(calibrated_acts) == 0 or len(uncalibrated_acts) == 0:
        raise ValueError(f"Not enough examples: calibrated={len(calibrated_acts)}, uncalibrated={len(uncalibrated_acts)}")

    # Compute direction: calibrated - uncalibrated
    calibrated_mean = calibrated_acts.mean(axis=0)
    uncalibrated_mean = uncalibrated_acts.mean(axis=0)

    direction = calibrated_mean - uncalibrated_mean
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm

    return {
        "direction": direction_normalized,
        "direction_magnitude": direction_norm,
        "n_calibrated": int(calibrated_mask.sum()),
        "n_uncalibrated": int(uncalibrated_mask.sum()),
        "calibrated_metric_mean": float(metric_values[calibrated_mask].mean()),
        "calibrated_confidence_mean": float(stated_confidences[calibrated_mask].mean()),
        "uncalibrated_metric_mean": float(metric_values[uncalibrated_mask].mean()),
        "uncalibrated_confidence_mean": float(stated_confidences[uncalibrated_mask].mean()),
        "calibrated_introspection_mean": float(introspection_scores[calibrated_mask].mean()),
        "uncalibrated_introspection_mean": float(introspection_scores[uncalibrated_mask].mean()),
    }


def compute_contrastive_direction_with_details(
    meta_activations: np.ndarray,
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    metric_higher_is_confident: bool = False,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25
) -> dict:
    """
    Compute contrastive direction based on confidence dimension.

    Contrasts correctly high-confidence examples vs correctly low-confidence examples:
    - High confidence group: high stated confidence AND high model confidence (correctly confident)
    - Low confidence group: low stated confidence AND low model confidence (correctly uncertain)

    This captures the confidence axis within calibrated examples only.

    Args:
        meta_activations: Activation vectors for each example
        metric_values: The uncertainty metric values (e.g., entropy, top_prob)
        stated_confidences: Model's stated confidence values
        metric_higher_is_confident: If True, higher metric = more confident (e.g., top_prob).
                                   If False, lower metric = more confident (e.g., entropy).
        top_quantile: Not used directly (for API compatibility)
        bottom_quantile: Not used directly (for API compatibility)

    Returns direction and detailed info about selected examples.
    """
    # Z-score normalize
    metric_z = stats.zscore(metric_values)
    confidence_z = stats.zscore(stated_confidences)

    # Normalize metric direction so positive = more confident
    # For entropy: low entropy = confident, so we negate
    # For top_prob: high top_prob = confident, so we keep as is
    if not metric_higher_is_confident:
        metric_z_conf = -metric_z  # Invert so positive = more confident
    else:
        metric_z_conf = metric_z

    # Introspection score: positive when calibrated (stated confidence tracks model confidence)
    introspection_scores = metric_z_conf * confidence_z

    # Only consider well-calibrated examples (positive introspection score)
    calibrated_mask = introspection_scores > 0

    # Within calibrated examples, split by confidence
    # High confidence: both stated and model confidence high
    high_conf_mask = calibrated_mask & (confidence_z > 0) & (metric_z_conf > 0)
    # Low confidence: both stated and model confidence low
    low_conf_mask = calibrated_mask & (confidence_z < 0) & (metric_z_conf < 0)

    high_conf_acts = meta_activations[high_conf_mask]
    low_conf_acts = meta_activations[low_conf_mask]

    if len(high_conf_acts) == 0 or len(low_conf_acts) == 0:
        raise ValueError(f"Not enough examples: high_conf={len(high_conf_acts)}, low_conf={len(low_conf_acts)}")

    # Compute direction: high confidence - low confidence
    high_conf_mean = high_conf_acts.mean(axis=0)
    low_conf_mean = low_conf_acts.mean(axis=0)

    direction = high_conf_mean - low_conf_mean
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm

    return {
        "direction": direction_normalized,
        "direction_magnitude": direction_norm,
        "n_high_conf": int(high_conf_mask.sum()),
        "n_low_conf": int(low_conf_mask.sum()),
        "n_calibrated": int(calibrated_mask.sum()),
        "high_conf_metric_mean": float(metric_values[high_conf_mask].mean()),
        "high_conf_confidence_mean": float(stated_confidences[high_conf_mask].mean()),
        "low_conf_metric_mean": float(metric_values[low_conf_mask].mean()),
        "low_conf_confidence_mean": float(stated_confidences[low_conf_mask].mean()),
    }


def analyze_selected_examples(
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    metric_name: str = "entropy",
    metric_higher_is_confident: bool = False
) -> dict:
    """
    Analyze the characteristics of selected high-confidence vs low-confidence examples.

    Both groups are calibrated (correctly high or correctly low confidence).
    """
    # Z-scores for interpretation
    metric_z = stats.zscore(metric_values)
    conf_z = stats.zscore(stated_confidences)

    # Normalize so positive = more confident
    if not metric_higher_is_confident:
        metric_z_conf = -metric_z
    else:
        metric_z_conf = metric_z

    # Introspection score: positive when calibrated
    introspection_scores = metric_z_conf * conf_z
    calibrated_mask = introspection_scores > 0

    # High confidence (correctly confident)
    high_conf_mask = calibrated_mask & (conf_z > 0) & (metric_z_conf > 0)
    # Low confidence (correctly uncertain)
    low_conf_mask = calibrated_mask & (conf_z < 0) & (metric_z_conf < 0)

    print("\n" + "="*60)
    print("SELECTED EXAMPLES ANALYSIS")
    print("="*60)

    print(f"\nCalibrated examples (n={calibrated_mask.sum()}):")

    print(f"\nHigh confidence (correctly confident, n={high_conf_mask.sum()}):")
    if high_conf_mask.sum() > 0:
        print(f"  Mean {metric_name}: {metric_values[high_conf_mask].mean():.3f}")
        print(f"  Mean confidence: {stated_confidences[high_conf_mask].mean():.3f}")
        print(f"  Mean {metric_name} z-score: {metric_z[high_conf_mask].mean():.2f}")
        print(f"  Mean confidence z-score: {conf_z[high_conf_mask].mean():.2f}")

    print(f"\nLow confidence (correctly uncertain, n={low_conf_mask.sum()}):")
    if low_conf_mask.sum() > 0:
        print(f"  Mean {metric_name}: {metric_values[low_conf_mask].mean():.3f}")
        print(f"  Mean confidence: {stated_confidences[low_conf_mask].mean():.3f}")
        print(f"  Mean {metric_name} z-score: {metric_z[low_conf_mask].mean():.2f}")
        print(f"  Mean confidence z-score: {conf_z[low_conf_mask].mean():.2f}")

    return {
        "n_calibrated": int(calibrated_mask.sum()),
        "high_conf": {
            "n": int(high_conf_mask.sum()),
            "metric_mean": float(metric_values[high_conf_mask].mean()) if high_conf_mask.sum() > 0 else None,
            "confidence_mean": float(stated_confidences[high_conf_mask].mean()) if high_conf_mask.sum() > 0 else None,
        },
        "low_conf": {
            "n": int(low_conf_mask.sum()),
            "metric_mean": float(metric_values[low_conf_mask].mean()) if low_conf_mask.sum() > 0 else None,
            "confidence_mean": float(stated_confidences[low_conf_mask].mean()) if low_conf_mask.sum() > 0 else None,
        },
    }


def analyze_calibration_selection(
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    metric_name: str = "entropy",
    metric_higher_is_confident: bool = False
) -> dict:
    """
    Analyze the characteristics of calibrated vs uncalibrated examples.

    - Calibrated: stated confidence tracks model confidence (introspection_score > 0)
    - Uncalibrated: stated confidence inversely tracks model confidence (introspection_score < 0)
    """
    # Z-scores for interpretation
    metric_z = stats.zscore(metric_values)
    conf_z = stats.zscore(stated_confidences)

    # Normalize so positive = more confident
    if not metric_higher_is_confident:
        metric_z_conf = -metric_z
    else:
        metric_z_conf = metric_z

    # Introspection score: positive when calibrated
    introspection_scores = metric_z_conf * conf_z
    calibrated_mask = introspection_scores > 0
    uncalibrated_mask = introspection_scores < 0

    print("\n" + "="*60)
    print("CALIBRATION DIRECTION SELECTION ANALYSIS")
    print("="*60)

    print(f"\nTotal examples: {len(metric_values)}")
    print(f"Calibrated (introspection > 0): n={calibrated_mask.sum()} ({100*calibrated_mask.mean():.1f}%)")
    print(f"Uncalibrated (introspection < 0): n={uncalibrated_mask.sum()} ({100*uncalibrated_mask.mean():.1f}%)")

    print(f"\nCalibrated examples (n={calibrated_mask.sum()}):")
    if calibrated_mask.sum() > 0:
        print(f"  Mean {metric_name}: {metric_values[calibrated_mask].mean():.3f}")
        print(f"  Mean confidence: {stated_confidences[calibrated_mask].mean():.3f}")
        print(f"  Mean introspection score: {introspection_scores[calibrated_mask].mean():.3f}")

    print(f"\nUncalibrated examples (n={uncalibrated_mask.sum()}):")
    if uncalibrated_mask.sum() > 0:
        print(f"  Mean {metric_name}: {metric_values[uncalibrated_mask].mean():.3f}")
        print(f"  Mean confidence: {stated_confidences[uncalibrated_mask].mean():.3f}")
        print(f"  Mean introspection score: {introspection_scores[uncalibrated_mask].mean():.3f}")

    return {
        "n_total": len(metric_values),
        "calibrated": {
            "n": int(calibrated_mask.sum()),
            "pct": float(100 * calibrated_mask.mean()),
            "metric_mean": float(metric_values[calibrated_mask].mean()) if calibrated_mask.sum() > 0 else None,
            "confidence_mean": float(stated_confidences[calibrated_mask].mean()) if calibrated_mask.sum() > 0 else None,
            "introspection_mean": float(introspection_scores[calibrated_mask].mean()) if calibrated_mask.sum() > 0 else None,
        },
        "uncalibrated": {
            "n": int(uncalibrated_mask.sum()),
            "pct": float(100 * uncalibrated_mask.mean()),
            "metric_mean": float(metric_values[uncalibrated_mask].mean()) if uncalibrated_mask.sum() > 0 else None,
            "confidence_mean": float(stated_confidences[uncalibrated_mask].mean()) if uncalibrated_mask.sum() > 0 else None,
            "introspection_mean": float(introspection_scores[uncalibrated_mask].mean()) if uncalibrated_mask.sum() > 0 else None,
        },
    }


def run_layer_analysis(
    meta_activations: dict,
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    metric_higher_is_confident: bool = False,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25,
    direction_type: str = "contrastive"
) -> dict:
    """
    Compute direction for each layer and analyze.

    Args:
        direction_type: "contrastive" (high_conf vs low_conf within calibrated)
                       or "calibration" (calibrated vs uncalibrated)
    """
    print("\n" + "="*60)
    print(f"LAYER-BY-LAYER ANALYSIS ({direction_type.upper()} direction)")
    print("="*60)

    # Compute z-scores for correlation analysis
    confidence_z = stats.zscore(stated_confidences)

    results = {}

    for layer_idx in tqdm(sorted(meta_activations.keys())):
        acts = meta_activations[layer_idx]

        # Compute mean activation norm for normalization
        mean_activation_norm = float(np.linalg.norm(acts, axis=1).mean())

        # Compute direction based on type
        if direction_type == "calibration":
            dir_info = compute_calibration_direction_with_details(
                acts, metric_values, stated_confidences,
                metric_higher_is_confident=metric_higher_is_confident
            )
        else:  # contrastive (default)
            dir_info = compute_contrastive_direction_with_details(
                acts, metric_values, stated_confidences,
                metric_higher_is_confident=metric_higher_is_confident,
                top_quantile=top_quantile, bottom_quantile=bottom_quantile
            )

        # Test how well projection correlates with stated confidence
        proj = acts @ dir_info["direction"]
        corr, pval = stats.pearsonr(proj, confidence_z)

        results[layer_idx] = {
            **dir_info,
            "projection_correlation": float(corr),
            "projection_pvalue": float(pval),
            "mean_activation_norm": mean_activation_norm,
        }

    # Print summary with appropriate headers
    if direction_type == "calibration":
        print(f"\n{'Layer':<8} {'Dir Mag':<12} {'Proj Corr':<12} {'R²':<12} {'p-value':<12} {'N calib':<8} {'N uncalib':<8}")
        print("-" * 76)
        for layer_idx in sorted(results.keys()):
            r = results[layer_idx]
            corr = r['projection_correlation']
            r2 = corr ** 2
            print(f"{layer_idx:<8} {r['direction_magnitude']:<12.4f} "
                  f"{corr:<12.4f} {r2:<12.4f} {r['projection_pvalue']:<12.2e} "
                  f"{r['n_calibrated']:<8} {r['n_uncalibrated']:<8}")
    else:  # contrastive
        print(f"\n{'Layer':<8} {'Dir Mag':<12} {'Proj Corr':<12} {'R²':<12} {'p-value':<12} {'N high':<8} {'N low':<8}")
        print("-" * 72)
        for layer_idx in sorted(results.keys()):
            r = results[layer_idx]
            corr = r['projection_correlation']
            r2 = corr ** 2
            print(f"{layer_idx:<8} {r['direction_magnitude']:<12.4f} "
                  f"{corr:<12.4f} {r2:<12.4f} {r['projection_pvalue']:<12.2e} "
                  f"{r['n_high_conf']:<8} {r['n_low_conf']:<8}")

    # Find best layer
    best_layer = max(results.keys(), key=lambda l: abs(results[l]["projection_correlation"]))
    print(f"\nBest layer: {best_layer} (correlation = {results[best_layer]['projection_correlation']:.4f})")

    return results


def plot_results(
    metric_values: np.ndarray,
    stated_confidences: np.ndarray,
    layer_results: dict,
    metric_name: str = "entropy",
    metric_higher_is_confident: bool = False,
    output_path: str = "contrastive_direction_results.png"
):
    """Plot analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Compute z-scores and masks for coloring
    metric_z = stats.zscore(metric_values)
    conf_z = stats.zscore(stated_confidences)

    # Normalize so positive = more confident
    if not metric_higher_is_confident:
        metric_z_conf = -metric_z
    else:
        metric_z_conf = metric_z

    introspection_scores = metric_z_conf * conf_z
    calibrated = introspection_scores > 0
    high_conf = calibrated & (conf_z > 0) & (metric_z_conf > 0)
    low_conf = calibrated & (conf_z < 0) & (metric_z_conf < 0)

    # 1. Introspection score distribution
    ax = axes[0, 0]
    ax.hist(introspection_scores, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='gray', linestyle='--', label='Calibration boundary')
    ax.set_xlabel('Introspection Score')
    ax.set_ylabel('Count')
    ax.set_title('Introspection Score Distribution (>0 = calibrated)')
    ax.legend()

    # 2. Metric vs Confidence with contrast group coloring
    ax = axes[0, 1]
    colors = ['green' if high_conf[i] else 'blue' if low_conf[i] else 'gray'
              for i in range(len(introspection_scores))]
    ax.scatter(metric_values, stated_confidences, c=colors, alpha=0.5, s=20)
    ax.set_xlabel(f'Direct {metric_name.capitalize()}')
    ax.set_ylabel('Stated Confidence')
    ax.set_title(f'{metric_name.capitalize()} vs Confidence\n(green=high conf, blue=low conf)')

    # Add trend line
    z = np.polyfit(metric_values, stated_confidences, 1)
    p = np.poly1d(z)
    x_line = np.linspace(metric_values.min(), metric_values.max(), 100)
    ax.plot(x_line, p(x_line), 'b--', alpha=0.5, label='Overall trend')
    ax.legend()

    # 3. Normalized direction magnitude by layer
    ax = axes[1, 0]
    layers = sorted(layer_results.keys())
    magnitudes = [layer_results[l]["direction_magnitude"] for l in layers]
    activation_norms = [layer_results[l]["mean_activation_norm"] for l in layers]
    normalized_magnitudes = [m / n for m, n in zip(magnitudes, activation_norms)]
    ax.plot(layers, normalized_magnitudes, 'o-')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Direction Magnitude / Activation Norm')
    ax.set_title('Normalized Contrastive Direction Magnitude by Layer')
    ax.grid(True, alpha=0.3)

    # 4. Projection correlation by layer
    ax = axes[1, 1]
    correlations = [layer_results[l]["projection_correlation"] for l in layers]
    ax.plot(layers, correlations, 'o-')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with Stated Confidence')
    ax.set_title('Direction Projection Correlation by Layer')
    ax.grid(True, alpha=0.3)

    # Highlight best layer
    best_layer = max(layers, key=lambda l: abs(layer_results[l]["projection_correlation"]))
    ax.scatter([best_layer], [layer_results[best_layer]["projection_correlation"]],
               color='red', s=100, zorder=5, label=f'Best: layer {best_layer}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


# ============================================================================
# DIRECTION COMPARISON
# ============================================================================

def compute_all_direction_types(
    meta_activations: Dict[int, np.ndarray],
    metric_values: np.ndarray,
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    metric_higher_is_confident: bool = False,
    n_clusters: int = 3,
    cluster_method: str = "quantile",
    direction_types: List[str] = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute multiple direction types for each layer.

    Direction types:
    - contrastive: High conf/low metric - Low conf/high metric (calibrated examples only)
    - calibration: Calibrated - Uncalibrated (pure calibration axis)
    - caa: Simple mean(high_metric) - mean(low_metric)
    - cluster: Cluster-based directions (low_to_high, low_to_mid, mid_to_high)

    Args:
        meta_activations: Dict mapping layer_idx to activations (n_samples, hidden_dim)
        metric_values: The metric to use for grouping (e.g., stated_confidences)
        direct_entropies: Metric values for contrastive direction (e.g., entropy, top_prob)
        stated_confidences: Confidence values (used for contrastive direction)
        metric_higher_is_confident: Whether higher metric value = more confident
        n_clusters: Number of clusters for cluster-based directions
        cluster_method: "quantile" or "kmeans"
        direction_types: List of direction types to compute. If None, compute all.
                        Options: "contrastive", "calibration", "caa", "cluster"

    Returns:
        Dict mapping layer_idx to Dict of direction_name -> direction_vector
    """
    # Default to all direction types
    if direction_types is None or len(direction_types) == 0:
        direction_types = ["contrastive", "calibration", "caa", "cluster"]

    print("\n" + "=" * 60)
    print("COMPUTING DIRECTION TYPES")
    print("=" * 60)
    print(f"Selected types: {direction_types}")

    all_directions = {}

    for layer_idx in tqdm(sorted(meta_activations.keys()), desc="Computing directions"):
        acts = meta_activations[layer_idx]
        layer_directions = {}

        # 1. Contrastive direction (calibrated high conf vs low conf)
        if "contrastive" in direction_types:
            try:
                contrastive_info = compute_contrastive_direction_with_details(
                    acts, direct_entropies, stated_confidences,
                    metric_higher_is_confident=metric_higher_is_confident
                )
                layer_directions["contrastive"] = contrastive_info["direction"]
            except ValueError as e:
                print(f"  Layer {layer_idx}: Could not compute contrastive direction: {e}")

        # 2. Calibration direction (calibrated vs uncalibrated)
        if "calibration" in direction_types:
            try:
                calibration_info = compute_calibration_direction_with_details(
                    acts, direct_entropies, stated_confidences,
                    metric_higher_is_confident=metric_higher_is_confident
                )
                layer_directions["calibration"] = calibration_info["direction"]
            except ValueError as e:
                print(f"  Layer {layer_idx}: Could not compute calibration direction: {e}")

        # 3. CAA direction (simple mean difference)
        if "caa" in direction_types:
            try:
                caa_direction, caa_info = compute_caa_direction(
                    acts, metric_values, high_quantile=0.25, low_quantile=0.25
                )
                layer_directions["caa"] = caa_direction
            except ValueError as e:
                print(f"  Layer {layer_idx}: Could not compute CAA direction: {e}")

        # 4. Cluster-based directions
        if "cluster" in direction_types:
            try:
                cluster_info = compute_cluster_centroids(
                    acts, metric_values, n_clusters=n_clusters, method=cluster_method
                )
                cluster_dirs = compute_cluster_directions(
                    cluster_info["centroids"], normalize=True
                )
                # Add relevant cluster directions
                for dir_name, direction in cluster_dirs.items():
                    layer_directions[f"cluster_{dir_name}"] = direction
            except Exception as e:
                print(f"  Layer {layer_idx}: Could not compute cluster directions: {e}")

        all_directions[layer_idx] = layer_directions

    # Print summary
    if all_directions:
        first_layer = list(all_directions.keys())[0]
        dir_types = list(all_directions[first_layer].keys())
        print(f"\nComputed direction types: {dir_types}")
        print(f"Number of layers: {len(all_directions)}")

    return all_directions


def evaluate_direction_quality(
    meta_activations: Dict[int, np.ndarray],
    all_directions: Dict[int, Dict[str, np.ndarray]],
    metric_values: np.ndarray,
    metric_name: str = "confidence"
) -> Dict[int, Dict[str, Dict]]:
    """
    Evaluate quality of each direction type by measuring correlation with metric.

    For each layer and direction type:
    - Project activations onto direction
    - Compute correlation of projection with metric values
    - Compute R² (variance explained)

    Returns:
        Dict mapping layer_idx to Dict of direction_name -> quality_metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING DIRECTION QUALITY")
    print("=" * 60)

    # Z-score normalize metric
    metric_z = stats.zscore(metric_values)

    quality_results = {}

    for layer_idx in sorted(all_directions.keys()):
        acts = meta_activations[layer_idx]
        layer_results = {}

        for dir_name, direction in all_directions[layer_idx].items():
            # Project onto direction
            proj = acts @ direction

            # Compute correlation with metric
            corr, pval = stats.pearsonr(proj, metric_z)
            r_squared = corr ** 2

            layer_results[dir_name] = {
                "correlation": float(corr),
                "r_squared": float(r_squared),
                "p_value": float(pval),
            }

        quality_results[layer_idx] = layer_results

    # Print summary table
    if quality_results:
        first_layer = list(quality_results.keys())[0]
        dir_types = list(quality_results[first_layer].keys())

        print(f"\n{'Layer':<8}", end="")
        for dt in dir_types:
            short_name = dt[:12]
            print(f"{short_name:<15}", end="")
        print()
        print("-" * (8 + 15 * len(dir_types)))

        for layer_idx in sorted(quality_results.keys()):
            print(f"{layer_idx:<8}", end="")
            for dt in dir_types:
                if dt in quality_results[layer_idx]:
                    corr = quality_results[layer_idx][dt]["correlation"]
                    print(f"{corr:+.4f}       ", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()

    return quality_results


def compare_direction_similarities(
    all_directions: Dict[int, Dict[str, np.ndarray]],
    layers: Optional[List[int]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compare similarity between different direction types at each layer.

    Returns:
        Dict mapping layer_idx to Dict of "dirA_vs_dirB" -> cosine_similarity
    """
    print("\n" + "=" * 60)
    print("DIRECTION SIMILARITY ANALYSIS")
    print("=" * 60)

    if layers is None:
        layers = sorted(all_directions.keys())

    similarity_results = {}

    for layer_idx in layers:
        if layer_idx not in all_directions:
            continue

        layer_dirs = all_directions[layer_idx]
        if len(layer_dirs) < 2:
            continue

        # Use the compare_directions utility from core.probes
        similarities = compare_directions(layer_dirs)
        similarity_results[layer_idx] = similarities

    # Print summary for key comparisons
    if similarity_results:
        first_layer = list(similarity_results.keys())[0]
        comparison_keys = list(similarity_results[first_layer].keys())

        # Focus on key comparisons
        key_comparisons = [k for k in comparison_keys if "contrastive" in k or "caa" in k]
        if key_comparisons:
            print(f"\nKey direction comparisons (cosine similarity):")
            print(f"{'Layer':<8}", end="")
            for comp in key_comparisons[:5]:  # Limit to 5 for readability
                print(f"{comp[:18]:<20}", end="")
            print()
            print("-" * (8 + 20 * min(5, len(key_comparisons))))

            for layer_idx in sorted(similarity_results.keys()):
                print(f"{layer_idx:<8}", end="")
                for comp in key_comparisons[:5]:
                    if comp in similarity_results[layer_idx]:
                        sim = similarity_results[layer_idx][comp]
                        print(f"{sim:.4f}              ", end="")
                    else:
                        print(f"{'N/A':<20}", end="")
                print()

    return similarity_results


def run_direction_comparison_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    layers: List[int],
    all_directions: Dict[int, Dict[str, np.ndarray]],
    multipliers: List[float],
    use_chat_template: bool,
    batch_size: int = 8
) -> Dict:
    """
    Run steering experiment comparing different direction types.

    For each layer and direction type, run steering and measure effect.
    This allows comparing the causal efficacy of different direction computation methods.

    Returns:
        Dict with per-layer, per-direction-type steering results
    """
    print("\n" + "=" * 70)
    print("DIRECTION COMPARISON STEERING EXPERIMENT")
    print("=" * 70)
    print(f"  Layers: {layers}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Batch size: {batch_size}")

    # Get direction types from first layer
    if not all_directions:
        print("No directions to compare!")
        return {}

    first_layer = list(all_directions.keys())[0]
    direction_types = list(all_directions[first_layer].keys())
    print(f"  Direction types: {direction_types}")

    entropy_mean = direct_entropies.mean()
    entropy_std = direct_entropies.std()

    def compute_results_from_batch(batch_results):
        """Convert batch results to per-question result dicts."""
        out = []
        for q_idx, (response, confidence, probs, mapping) in enumerate(batch_results):
            metric_val = direct_entropies[q_idx]
            metric_z = (metric_val - entropy_mean) / entropy_std
            confidence_z = (confidence - 0.5) / 0.25
            alignment = compute_alignment(metric_z, confidence_z, METRIC)

            out.append({
                "question_idx": q_idx,
                "response": response,
                "confidence": confidence,
                "metric": float(metric_val),
                "alignment": float(alignment),
            })
        return out

    results = {
        "layers": layers,
        "multipliers": multipliers,
        "direction_types": direction_types,
        "num_questions": len(questions),
        "layer_results": {},
    }

    # Compute baseline once
    print("\nComputing baseline (no steering)...")
    baseline_batch = get_batch_confidence_responses(
        model, tokenizer, questions, None, None, 0.0, use_chat_template, batch_size
    )
    shared_baseline = compute_results_from_batch(baseline_batch)

    for layer_idx in tqdm(layers, desc="Layers"):
        if layer_idx not in all_directions:
            continue

        layer_dirs = all_directions[layer_idx]
        layer_results = {
            "baseline": shared_baseline,
            "direction_results": {},
        }

        for dir_type, direction in layer_dirs.items():
            dir_results = {m: [] for m in multipliers}

            for mult in tqdm(multipliers, desc=f"L{layer_idx}/{dir_type[:10]}", leave=False):
                if mult == 0.0:
                    dir_results[mult] = layer_results["baseline"]
                    continue

                batch_results = get_batch_confidence_responses(
                    model, tokenizer, questions, layer_idx, direction, mult,
                    use_chat_template, batch_size
                )
                dir_results[mult] = compute_results_from_batch(batch_results)

            layer_results["direction_results"][dir_type] = dir_results

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


def analyze_direction_comparison_results(results: Dict) -> Dict:
    """
    Analyze direction comparison experiment results.

    For each layer and direction type, compute:
    - Mean confidence change at each multiplier
    - Effect size (confidence change per unit multiplier)
    - Comparison to other direction types
    """
    analysis = {
        "layer_analysis": {},
        "direction_type_summary": {},
    }

    all_direction_types = results.get("direction_types", [])
    multipliers = results.get("multipliers", [])

    # Initialize summary accumulators
    for dt in all_direction_types:
        analysis["direction_type_summary"][dt] = {
            "mean_effect": [],
            "layers": [],
        }

    for layer_idx, layer_results in results.get("layer_results", {}).items():
        baseline_confidences = [r["confidence"] for r in layer_results["baseline"]]
        baseline_mean = np.mean(baseline_confidences)

        layer_analysis = {
            "baseline_mean_confidence": float(baseline_mean),
            "direction_effects": {},
        }

        for dir_type, dir_results in layer_results.get("direction_results", {}).items():
            effects = {}
            effect_per_mult = []

            for mult in multipliers:
                if mult == 0.0:
                    effects[mult] = {
                        "mean_confidence": float(baseline_mean),
                        "confidence_change": 0.0,
                    }
                    continue

                mult_confidences = [r["confidence"] for r in dir_results[mult]]
                mult_mean = np.mean(mult_confidences)
                change = mult_mean - baseline_mean

                effects[mult] = {
                    "mean_confidence": float(mult_mean),
                    "confidence_change": float(change),
                }

                # Track effect per unit multiplier (for effect size)
                effect_per_mult.append(change / abs(mult))

            # Compute overall effect size (mean absolute effect per unit multiplier)
            if effect_per_mult:
                mean_effect = np.mean(np.abs(effect_per_mult))
            else:
                mean_effect = 0.0

            layer_analysis["direction_effects"][dir_type] = {
                "multiplier_effects": effects,
                "mean_effect_per_unit": float(mean_effect),
            }

            # Accumulate for summary
            if dir_type in analysis["direction_type_summary"]:
                analysis["direction_type_summary"][dir_type]["mean_effect"].append(mean_effect)
                analysis["direction_type_summary"][dir_type]["layers"].append(layer_idx)

        analysis["layer_analysis"][layer_idx] = layer_analysis

    # Compute overall summary for each direction type
    for dt in all_direction_types:
        if dt in analysis["direction_type_summary"]:
            effects = analysis["direction_type_summary"][dt]["mean_effect"]
            if effects:
                analysis["direction_type_summary"][dt]["overall_mean_effect"] = float(np.mean(effects))
                analysis["direction_type_summary"][dt]["overall_std_effect"] = float(np.std(effects))
            else:
                analysis["direction_type_summary"][dt]["overall_mean_effect"] = 0.0
                analysis["direction_type_summary"][dt]["overall_std_effect"] = 0.0

    return analysis


def print_direction_comparison_summary(analysis: Dict):
    """Print summary of direction comparison analysis."""
    print("\n" + "=" * 70)
    print("DIRECTION COMPARISON SUMMARY")
    print("=" * 70)

    # Overall direction type ranking
    print("\nOverall Direction Type Effectiveness (mean effect per unit multiplier):")
    print(f"{'Direction Type':<25} {'Mean Effect':<15} {'Std':<15}")
    print("-" * 55)

    summary = analysis.get("direction_type_summary", {})
    sorted_types = sorted(
        summary.keys(),
        key=lambda dt: summary[dt].get("overall_mean_effect", 0),
        reverse=True
    )

    for dt in sorted_types:
        mean_eff = summary[dt].get("overall_mean_effect", 0)
        std_eff = summary[dt].get("overall_std_effect", 0)
        print(f"{dt:<25} {mean_eff:<15.4f} {std_eff:<15.4f}")

    # Per-layer breakdown
    print("\nPer-Layer Effect Comparison:")
    layer_analysis = analysis.get("layer_analysis", {})

    for layer_idx in sorted(layer_analysis.keys()):
        layer = layer_analysis[layer_idx]
        print(f"\n  Layer {layer_idx}:")
        print(f"    {'Direction':<20} {'Effect/mult':<15}")
        print(f"    {'-'*35}")

        effects = layer.get("direction_effects", {})
        sorted_dirs = sorted(
            effects.keys(),
            key=lambda d: effects[d].get("mean_effect_per_unit", 0),
            reverse=True
        )

        for dt in sorted_dirs:
            eff = effects[dt].get("mean_effect_per_unit", 0)
            print(f"    {dt:<20} {eff:.4f}")


def plot_direction_comparison(
    quality_results: Dict[int, Dict[str, Dict]],
    similarity_results: Dict[int, Dict[str, float]],
    steering_analysis: Optional[Dict] = None,
    output_path: str = "direction_comparison.png"
):
    """Plot direction comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get data
    layers = sorted(quality_results.keys())
    if not layers:
        print("No quality results to plot")
        return

    first_layer = layers[0]
    direction_types = list(quality_results[first_layer].keys())

    # 1. Correlation by layer for each direction type
    ax = axes[0, 0]
    for dt in direction_types:
        correlations = []
        valid_layers = []
        for layer_idx in layers:
            if dt in quality_results[layer_idx]:
                correlations.append(quality_results[layer_idx][dt]["correlation"])
                valid_layers.append(layer_idx)
        if correlations:
            ax.plot(valid_layers, correlations, 'o-', label=dt[:15], alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with Metric')
    ax.set_title('Direction Quality by Layer\n(Correlation of Projection with Metric)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. R² by layer
    ax = axes[0, 1]
    for dt in direction_types:
        r2_values = []
        valid_layers = []
        for layer_idx in layers:
            if dt in quality_results[layer_idx]:
                r2_values.append(quality_results[layer_idx][dt]["r_squared"])
                valid_layers.append(layer_idx)
        if r2_values:
            ax.plot(valid_layers, r2_values, 'o-', label=dt[:15], alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('R² (Variance Explained)')
    ax.set_title('Direction Quality by Layer\n(R² of Projection)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Direction similarity (contrastive vs others)
    ax = axes[1, 0]
    if similarity_results:
        # Find comparisons involving contrastive
        sample_layer = list(similarity_results.keys())[0]
        contrastive_comps = [k for k in similarity_results[sample_layer].keys()
                           if "contrastive" in k and k != "contrastive_vs_contrastive"]

        for comp in contrastive_comps[:5]:  # Limit for readability
            similarities = []
            valid_layers = []
            for layer_idx in layers:
                if layer_idx in similarity_results and comp in similarity_results[layer_idx]:
                    similarities.append(similarity_results[layer_idx][comp])
                    valid_layers.append(layer_idx)
            if similarities:
                ax.plot(valid_layers, similarities, 'o-', label=comp[:20], alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Direction Similarity\n(Contrastive vs Other Methods)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center')
        ax.set_title('Direction Similarity')

    # 4. Steering effect comparison (if available)
    ax = axes[1, 1]
    if steering_analysis and "layer_analysis" in steering_analysis:
        layer_analysis = steering_analysis["layer_analysis"]
        for dt in direction_types:
            effects = []
            valid_layers = []
            for layer_idx in layers:
                if (layer_idx in layer_analysis and
                    dt in layer_analysis[layer_idx].get("direction_effects", {})):
                    eff = layer_analysis[layer_idx]["direction_effects"][dt].get("mean_effect_per_unit", 0)
                    effects.append(eff)
                    valid_layers.append(layer_idx)
            if effects:
                ax.plot(valid_layers, effects, 'o-', label=dt[:15], alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Effect per Unit Multiplier')
        ax.set_title('Steering Effect by Direction Type')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No steering data\n(Run with RUN_STEERING=True)', ha='center', va='center')
        ax.set_title('Steering Effect Comparison')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDirection comparison plot saved to {output_path}")


def main():
    global METRIC  # Allow CLI to set the metric

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Compute contrastive introspection directions")
    parser.add_argument("--metric", choices=VALID_METRICS, default=METRIC,
                        help=f"Metric to use for contrastive direction (default: {METRIC})")
    args = parser.parse_args()
    METRIC = args.metric

    # Get metric properties
    metric_higher_is_confident = METRIC_HIGHER_IS_CONFIDENT[METRIC]

    print(f"Device: {DEVICE}")
    print(f"Metric: {METRIC} (higher = {'more' if metric_higher_is_confident else 'less'} confident)")

    # Print direction type info
    if DIRECTION_TYPES and len(DIRECTION_TYPES) == 1:
        dir_type = DIRECTION_TYPES[0]
        if dir_type == "calibration":
            print(f"Direction type: CALIBRATION (calibrated vs uncalibrated)")
        else:
            print(f"Direction type: CONTRASTIVE (top {TOP_QUANTILE*100:.0f}% vs bottom {BOTTOM_QUANTILE*100:.0f}%)")
    else:
        print(f"Direction types: {DIRECTION_TYPES or 'all'}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Load data
    data = load_introspection_data(metric=METRIC)

    metric_values = data["metric_values"]
    stated_confidences = data["stated_confidences"]
    meta_activations = data["meta_activations"]
    paired_data = data["paired_data"]
    num_layers = data["num_layers"]

    # Compute introspection scores
    # For entropy: low entropy = confident, so we negate before computing scores
    # For other metrics: high value = confident, so we negate the metric to match compute_introspection_scores
    print("\nComputing introspection scores...")
    if metric_higher_is_confident:
        # Higher metric = more confident, so we negate to make it like entropy
        introspection_scores = compute_introspection_scores(-metric_values, stated_confidences)
    else:
        introspection_scores = compute_introspection_scores(metric_values, stated_confidences)

    print(f"Introspection score range: [{introspection_scores.min():.3f}, {introspection_scores.max():.3f}]")
    print(f"Mean: {introspection_scores.mean():.3f}, Std: {introspection_scores.std():.3f}")

    # Determine which direction type to use for layer analysis
    if DIRECTION_TYPES and len(DIRECTION_TYPES) == 1:
        analysis_direction_type = DIRECTION_TYPES[0]
    else:
        analysis_direction_type = "contrastive"  # default

    # Run layer-by-layer analysis
    layer_results = run_layer_analysis(
        meta_activations, metric_values, stated_confidences,
        metric_higher_is_confident=metric_higher_is_confident,
        top_quantile=TOP_QUANTILE, bottom_quantile=BOTTOM_QUANTILE,
        direction_type=analysis_direction_type
    )

    # Get best layer for detailed analysis
    best_layer = max(layer_results.keys(), key=lambda l: abs(layer_results[l]["projection_correlation"]))

    # Detailed analysis on best layer
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS (Layer {best_layer})")
    print(f"{'='*60}")

    best_acts = meta_activations[best_layer]
    best_result = layer_results[best_layer]

    # Show layer-specific statistics
    print(f"\nDirection properties:")
    print(f"  Magnitude: {best_result['direction_magnitude']:.4f}")
    print(f"  Projection correlation: {best_result['projection_correlation']:.4f}")
    print(f"  Projection p-value: {best_result['projection_pvalue']:.2e}")
    print(f"  R²: {best_result['projection_correlation']**2:.4f}")

    # Projection distribution
    proj = best_acts @ best_result["direction"]
    print(f"\nProjection distribution:")
    print(f"  Mean: {proj.mean():.4f}")
    print(f"  Std: {proj.std():.4f}")
    print(f"  Range: [{proj.min():.4f}, {proj.max():.4f}]")

    # Analyze selected examples based on direction type
    example_analysis = None
    if analysis_direction_type == "calibration":
        # Show calibration selection diagnostics
        analyze_calibration_selection(
            metric_values,
            stated_confidences,
            metric_name=METRIC,
            metric_higher_is_confident=metric_higher_is_confident
        )
    elif DIRECTION_TYPES is None or "contrastive" in DIRECTION_TYPES:
        example_analysis = analyze_selected_examples(
            metric_values,
            stated_confidences,
            metric_name=METRIC,
            metric_higher_is_confident=metric_higher_is_confident
        )

    # Plot results
    plot_results(
        metric_values,
        stated_confidences,
        layer_results,
        metric_name=METRIC,
        metric_higher_is_confident=metric_higher_is_confident,
        output_path=f"{output_prefix}_results.png"
    )

    # Save results
    results = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "dataset": DATASET_NAME,
            "metric": METRIC,
            "metric_higher_is_confident": metric_higher_is_confident,
            "top_quantile": TOP_QUANTILE,
            "bottom_quantile": BOTTOM_QUANTILE,
            "seed": SEED,
            "meta_task": META_TASK,
        },
        "best_layer": best_layer,
        "layer_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "direction"}
            for k, v in layer_results.items()
        },
        "example_analysis": example_analysis,
    }

    results_path = f"{output_prefix}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
    print(f"\nResults saved to {results_path}")

    # Save directions
    directions_dict = {
        layer_idx: layer_results[layer_idx]["direction"]
        for layer_idx in layer_results.keys()
    }
    directions_for_save = {
        f"layer_{k}": v
        for k, v in directions_dict.items()
    }
    np.savez_compressed(f"{output_prefix}_directions.npz", **directions_for_save)
    print(f"Directions saved to {output_prefix}_directions.npz")

    # ========================================================================
    # DIRECTION COMPARISON (if enabled)
    # ========================================================================
    all_directions = None
    quality_results = None
    similarity_results = None
    direction_comparison_analysis = None

    if COMPARE_DIRECTIONS:
        print("\n" + "=" * 70)
        print("DIRECTION COMPARISON MODE")
        print("=" * 70)

        # Print diagnostics for calibration direction if requested
        if DIRECTION_TYPES is None or "calibration" in DIRECTION_TYPES:
            analyze_calibration_selection(
                metric_values,
                stated_confidences,
                metric_name=METRIC,
                metric_higher_is_confident=metric_higher_is_confident
            )

        # Print diagnostics for contrastive direction if requested
        if DIRECTION_TYPES is None or "contrastive" in DIRECTION_TYPES:
            analyze_selected_examples(
                metric_values,
                stated_confidences,
                metric_name=METRIC,
                metric_higher_is_confident=metric_higher_is_confident
            )

        # Compute all direction types for each layer
        all_directions = compute_all_direction_types(
            meta_activations,
            metric_values=stated_confidences,  # Used for CAA and cluster directions
            direct_entropies=metric_values,     # Used for contrastive direction
            stated_confidences=stated_confidences,
            metric_higher_is_confident=metric_higher_is_confident,
            n_clusters=N_CLUSTERS,
            cluster_method=CLUSTER_METHOD,
            direction_types=DIRECTION_TYPES
        )

        # Evaluate quality of each direction type
        quality_results = evaluate_direction_quality(
            meta_activations,
            all_directions,
            metric_values=stated_confidences,
            metric_name="confidence"
        )

        # Compare similarity between direction types
        similarity_results = compare_direction_similarities(all_directions)

        # Save all directions
        all_directions_for_save = {}
        for layer_idx, layer_dirs in all_directions.items():
            for dir_name, direction in layer_dirs.items():
                all_directions_for_save[f"layer_{layer_idx}_{dir_name}"] = direction
        np.savez_compressed(f"{output_prefix}_all_directions.npz", **all_directions_for_save)
        print(f"\nAll directions saved to {output_prefix}_all_directions.npz")

        # Save quality results
        quality_results_serializable = {
            str(k): v for k, v in quality_results.items()
        }
        quality_path = f"{output_prefix}_direction_quality.json"
        with open(quality_path, "w") as f:
            json.dump(quality_results_serializable, f, indent=2)
        print(f"Direction quality saved to {quality_path}")

        # Save similarity results
        similarity_results_serializable = {
            str(k): v for k, v in similarity_results.items()
        }
        similarity_path = f"{output_prefix}_direction_similarity.json"
        with open(similarity_path, "w") as f:
            json.dump(similarity_results_serializable, f, indent=2)
        print(f"Direction similarity saved to {similarity_path}")

    # ========================================================================
    # STEERING/ABLATION EXPERIMENTS
    # ========================================================================
    if RUN_STEERING:
        print("\n" + "=" * 70)
        print("STEERING/ABLATION EXPERIMENTS")
        print("=" * 70)

        # Free up memory before loading model for steering
        # meta_activations can be large (500 samples × 32 layers × 4096 dim = ~250MB+)
        del meta_activations
        import gc
        gc.collect()

        # Load model for steering
        print(f"\nLoading model: {BASE_MODEL_NAME}")
        adapter_path = MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
        model, tokenizer, _ = load_model_and_tokenizer(
            BASE_MODEL_NAME,
            adapter_path,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT
        )
        use_chat_template = not is_base_model(BASE_MODEL_NAME)

        # Initialize token cache for efficient inference
        print("Initializing token cache...")
        initialize_token_cache(tokenizer)

        # Determine which layers to steer
        if STEERING_LAYERS is not None:
            steering_layers = STEERING_LAYERS
        else:
            # Select layers using principled criteria:
            # 1. Statistical significance: p < MIN_PROJECTION_PVAL
            # 2. Effect size: R² > MIN_PROJECTION_R2 (i.e., |r| > sqrt(MIN_PROJECTION_R2))
            min_corr = MIN_PROJECTION_R2 ** 0.5  # Convert R² threshold to |r| threshold
            candidate_layers = [
                (layer_idx, abs(r["projection_correlation"]))
                for layer_idx, r in layer_results.items()
                if (r["projection_pvalue"] < MIN_PROJECTION_PVAL and
                    r["projection_correlation"] ** 2 > MIN_PROJECTION_R2)
            ]
            # Sort by correlation (descending) and take top N
            candidate_layers.sort(key=lambda x: x[1], reverse=True)
            steering_layers = [layer_idx for layer_idx, _ in candidate_layers[:MAX_STEERING_LAYERS]]
            # Sort numerically for consistent presentation
            steering_layers.sort()

        if not steering_layers:
            min_corr = MIN_PROJECTION_R2 ** 0.5
            print(f"No layers meet criteria (p < {MIN_PROJECTION_PVAL}, R² > {MIN_PROJECTION_R2}, i.e., |r| > {min_corr:.3f}). Skipping steering.")
        else:
            print(f"Selected layers for steering: {steering_layers}")

            # Compute number of control directions (dynamic or fixed)
            if NUM_CONTROL_DIRECTIONS is None:
                # Dynamic: ensure enough power for FDR correction
                # For FDR at α with N layers, need pooled_samples > N/α = 20*N (for α=0.05)
                # We use FDR_SAFETY_FACTOR * N for margin, so controls_per_layer = FDR_SAFETY_FACTOR
                num_controls = max(MIN_CONTROLS_PER_LAYER, FDR_SAFETY_FACTOR)
                target_pooled = num_controls * len(steering_layers)
                min_p_achievable = 1.0 / (target_pooled + 1)
                fdr_threshold = FDR_ALPHA / len(steering_layers)  # Approx threshold for best layer
                print(f"Dynamic control directions: {num_controls} per layer "
                      f"({target_pooled} pooled samples, min_p={min_p_achievable:.4f}, FDR_thresh≈{fdr_threshold:.4f})")
            else:
                num_controls = NUM_CONTROL_DIRECTIONS
                print(f"Fixed control directions: {num_controls} per layer")

            # Get questions for steering
            questions = paired_data.get("questions", [])
            if len(questions) > NUM_STEERING_QUESTIONS:
                questions = questions[:NUM_STEERING_QUESTIONS]
                steering_metric_values = metric_values[:NUM_STEERING_QUESTIONS]
            else:
                steering_metric_values = metric_values

            print(f"Using {len(questions)} questions for steering")

            # Run steering experiment
            steering_results = run_steering_experiment(
                model, tokenizer, questions, steering_metric_values,
                steering_layers, directions_dict, STEERING_MULTIPLIERS,
                num_controls, use_chat_template,
                batch_size=STEERING_BATCH_SIZE
            )

            # Analyze steering results (using shared analysis with full statistics)
            steering_analysis = shared_analyze_steering_results(steering_results)
            print_steering_summary(steering_analysis)

            # Save steering results
            steering_results_path = f"{output_prefix}_steering_results.json"
            with open(steering_results_path, "w") as f:
                json.dump(steering_results, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"\nSteering results saved to {steering_results_path}")

            steering_analysis_path = f"{output_prefix}_steering_analysis.json"
            with open(steering_analysis_path, "w") as f:
                json.dump(steering_analysis, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"Steering analysis saved to {steering_analysis_path}")

            # Run ablation experiment
            print("\n" + "=" * 70)
            print("RUNNING ABLATION EXPERIMENT")
            print("=" * 70)

            # Reuse baseline from steering
            first_layer = steering_layers[0]
            baseline_from_steering = steering_results["layer_results"][first_layer]["baseline"]

            ablation_results = run_ablation_experiment(
                model, tokenizer, questions, steering_metric_values,
                steering_layers, directions_dict, num_controls,
                use_chat_template, baseline_results=baseline_from_steering,
                batch_size=STEERING_BATCH_SIZE
            )

            # Analyze ablation results (using shared analysis with full statistics)
            ablation_analysis = shared_analyze_ablation_results(ablation_results)
            print_ablation_summary(ablation_analysis)

            # Save ablation results
            ablation_results_path = f"{output_prefix}_ablation_results.json"
            with open(ablation_results_path, "w") as f:
                json.dump(ablation_results, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"\nAblation results saved to {ablation_results_path}")

            ablation_analysis_path = f"{output_prefix}_ablation_analysis.json"
            with open(ablation_analysis_path, "w") as f:
                json.dump(ablation_analysis, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"Ablation analysis saved to {ablation_analysis_path}")

            # ================================================================
            # DIRECTION COMPARISON STEERING (if enabled)
            # ================================================================
            if COMPARE_DIRECTIONS and all_directions is not None:
                print("\n" + "=" * 70)
                print("DIRECTION COMPARISON STEERING EXPERIMENT")
                print("=" * 70)

                # Run steering experiment comparing all direction types
                direction_comparison_results = run_direction_comparison_experiment(
                    model, tokenizer, questions, steering_metric_values,
                    steering_layers, all_directions, STEERING_MULTIPLIERS,
                    use_chat_template, batch_size=STEERING_BATCH_SIZE
                )

                # Analyze results
                direction_comparison_analysis = analyze_direction_comparison_results(
                    direction_comparison_results
                )
                print_direction_comparison_summary(direction_comparison_analysis)

                # Save direction comparison results
                direction_comp_path = f"{output_prefix}_direction_comparison_results.json"
                with open(direction_comp_path, "w") as f:
                    json.dump(direction_comparison_results, f, indent=2,
                              default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
                print(f"\nDirection comparison results saved to {direction_comp_path}")

                direction_comp_analysis_path = f"{output_prefix}_direction_comparison_analysis.json"
                with open(direction_comp_analysis_path, "w") as f:
                    json.dump(direction_comparison_analysis, f, indent=2,
                              default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
                print(f"Direction comparison analysis saved to {direction_comp_analysis_path}")

    # ========================================================================
    # FINAL PLOTS (if direction comparison was enabled)
    # ========================================================================
    if COMPARE_DIRECTIONS and quality_results is not None:
        plot_direction_comparison(
            quality_results,
            similarity_results or {},
            steering_analysis=direction_comparison_analysis,
            output_path=f"{output_prefix}_direction_comparison.png"
        )

    print("\n" + "=" * 70)
    print("CONTRASTIVE DIRECTION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
