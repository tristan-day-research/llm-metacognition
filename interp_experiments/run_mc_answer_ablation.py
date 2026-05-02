"""
MC Answer Probe Causality Experiment.

Tests whether the MC answer probe direction is causally involved in introspection by:
1. Ablating the MC answer direction during meta task execution
2. Measuring impact on D2M transfer R² (entropy probe transfer)
3. Measuring impact on behavioral correlation (stated confidence vs entropy)
4. Computing direction similarity between MC answer and entropy probes

Optimized: Uses parallel ablation to run baseline, MC, and all controls simultaneously.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import random
from scipy import stats
import argparse
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from prompts import (
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    get_delegate_mapping,
    response_to_confidence,
)

# Import DynamicCache safely for KV cache optimization
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

# =============================================================================
# CONFIGURATION — edit values in experiment_config.MCAnswerAblationConfig
# =============================================================================
from experiment_config import MCAnswerAblationConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
MODEL_NAME = _C.MODEL_NAME
DATASET_NAME = _C.DATASET_NAME
AVAILABLE_METRICS = list(_C.AVAILABLE_METRICS)
METRIC = _C.METRIC
REVERSE_MODE = _C.REVERSE_MODE
META_TASK = _C.META_TASK
D2M_R2_THRESHOLD = _C.D2M_R2_THRESHOLD
D2D_R2_THRESHOLD = _C.D2D_R2_THRESHOLD
OUTPUTS_DIR = _C.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(exist_ok=True)
ABLATION_LAYERS = _C.ABLATION_LAYERS
NUM_QUESTIONS = _C.NUM_QUESTIONS
NUM_CONTROL_DIRECTIONS = _C.NUM_CONTROL_DIRECTIONS
FDR_ALPHA = _C.FDR_ALPHA
FDR_SAFETY_FACTOR = _C.FDR_SAFETY_FACTOR
MIN_CONTROLS_PER_LAYER = _C.MIN_CONTROLS_PER_LAYER
BATCH_SIZE = _C.BATCH_SIZE
VARIANT_BATCH_SIZE = _C.VARIANT_BATCH_SIZE
INTERVENTION_POSITION = _C.INTERVENTION_POSITION
LOAD_IN_4BIT = _C.LOAD_IN_4BIT
LOAD_IN_8BIT = _C.LOAD_IN_8BIT
SEED = _C.SEED

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Cached token IDs
_CACHED_TOKEN_IDS = {
    "meta_options": None,
    "delegate_options": None,
}


def get_output_prefix() -> str:
    """Generate output filename prefix."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


def get_directions_prefix() -> str:
    """Generate output filename prefix for direction files (task-independent)."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection")


def initialize_token_cache(tokenizer):
    """Precompute option token IDs once."""
    _CACHED_TOKEN_IDS["meta_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0]
        for opt in STATED_CONFIDENCE_OPTIONS.keys()
    ]
    _CACHED_TOKEN_IDS["delegate_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0]
        for opt in ANSWER_OR_DELEGATE_OPTIONS
    ]


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_mc_answer_directions(prefix: str) -> Dict[int, np.ndarray]:
    """Load MC answer probe directions."""
    path = Path(f"{prefix}_mc_answer_directions.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"MC answer directions not found: {path}\n"
            "Run run_introspection_experiment.py first to generate them."
        )
    print(f"Loading MC answer directions from {path}...")
    data = np.load(path)
    directions = {
        int(k.split("_")[1]): data[k]
        for k in data.files if k.startswith("layer_")
    }
    print(f"  Loaded {len(directions)} layer directions")
    return directions


def load_entropy_directions(prefix: str, metric: str) -> Dict[int, np.ndarray]:
    """Load entropy/metric probe directions."""
    path = Path(f"{prefix}_{metric}_directions.npz")
    if not path.exists():
        raise FileNotFoundError(f"Entropy directions not found: {path}")
    print(f"Loading {metric} directions from {path}...")
    data = np.load(path)
    directions = {
        int(k.split("_")[1]): data[k]
        for k in data.files if k.startswith("layer_")
    }
    print(f"  Loaded {len(directions)} layer directions")
    return directions


def load_probe_results(prefix: str, metric: str) -> Dict:
    """Load probe results for layer selection."""
    path = Path(f"{prefix}_{metric}_results.json")
    if not path.exists():
        raise FileNotFoundError(f"Probe results not found: {path}")
    print(f"Loading probe results from {path}...")
    with open(path) as f:
        return json.load(f)


def load_paired_data(prefix: str) -> Dict:
    """Load paired data including questions and direct metrics."""
    # Try base prefix first (without metric suffix)
    base_prefix = prefix.rsplit("_", 1)[0] if "_" in prefix else prefix
    path = Path(f"{base_prefix}_paired_data.json")

    if not path.exists():
        # Try with the full prefix
        path = Path(f"{prefix}_paired_data.json")

    if not path.exists():
        raise FileNotFoundError(
            f"Paired data not found at {path}\n"
            "Run run_introspection_experiment.py first."
        )

    print(f"Loading paired data from {path}...")
    with open(path) as f:
        data = json.load(f)

    # Convert direct_metrics back to numpy arrays
    direct_metrics = {k: np.array(v) for k, v in data["direct_metrics"].items()}

    return {
        "questions": data["questions"],
        "direct_metrics": direct_metrics,
        "direct_probs": data["direct_probs"],
        "meta_probs": data["meta_probs"],
        "meta_responses": data["meta_responses"],
        "meta_mappings": data.get("meta_mappings"),
        "config": data.get("config", {}),
    }


def load_activations(prefix: str, layers: List[int]) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load saved direct activations for specified layers.

    Returns:
        direct_activations: {layer_idx: activations array}
        entropy_values: Ground truth entropy from direct task
    """
    direct_path = Path(f"{prefix}_direct_activations.npz")

    if not direct_path.exists():
        raise FileNotFoundError(
            f"Activation file not found: {direct_path}\n"
            "Run run_introspection_experiment.py first to generate it."
        )

    print(f"Loading activations from {direct_path}...")
    direct_data = np.load(direct_path)

    direct_activations = {}

    for layer_idx in layers:
        key = f"layer_{layer_idx}"
        if key in direct_data.files:
            direct_activations[layer_idx] = direct_data[key]

    # Get entropy values (saved in direct_activations file)
    if "entropy" in direct_data.files:
        entropy_values = direct_data["entropy"]
    else:
        raise KeyError("Entropy values not found in direct_activations.npz")

    print(f"  Loaded {len(direct_activations)} layers, {len(direct_activations[layers[0]])} samples")

    return direct_activations, entropy_values


def load_all_activations(prefix: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[int]]:
    """
    Load ALL saved direct activations (all layers).

    Returns:
        direct_activations: {layer_idx: activations array}
        entropy_values: Ground truth entropy from direct task
        available_layers: List of all layers that have saved activations
    """
    direct_path = Path(f"{prefix}_direct_activations.npz")

    if not direct_path.exists():
        raise FileNotFoundError(
            f"Activation file not found: {direct_path}\n"
            "Run run_introspection_experiment.py first to generate it."
        )

    print(f"Loading ALL activations from {direct_path}...")
    direct_data = np.load(direct_path)

    direct_activations = {}
    available_layers = []

    for key in direct_data.files:
        if key.startswith("layer_"):
            layer_idx = int(key.split("_")[1])
            direct_activations[layer_idx] = direct_data[key]
            available_layers.append(layer_idx)

    available_layers = sorted(available_layers)

    # Get entropy values (saved in direct_activations file)
    if "entropy" in direct_data.files:
        entropy_values = direct_data["entropy"]
    else:
        raise KeyError("Entropy values not found in direct_activations.npz")

    n_samples = len(entropy_values)
    print(f"  Loaded {len(available_layers)} layers, {n_samples} samples")
    print(f"  Available layers: {available_layers[:5]}...{available_layers[-5:]}")

    return direct_activations, entropy_values, available_layers


def train_entropy_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 128
) -> Tuple[StandardScaler, PCA, Ridge]:
    """
    Train entropy probe on direct activations.

    Returns:
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        probe: Fitted Ridge regressor
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
    X_pca = pca.fit_transform(X_scaled)

    # Ridge regression
    probe = Ridge(alpha=1.0)
    probe.fit(X_pca, y_train)

    return scaler, pca, probe


def apply_probe_strict(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    probe: Ridge
) -> np.ndarray:
    """
    Apply a pre-trained probe using the ORIGINAL scaler (no domain adaptation).
    This is the strictest/fairest comparison for ablation experiments.
    """
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return probe.predict(X_pca)


def apply_probe_d2m_fixed(
    X: np.ndarray,
    pca: PCA,
    probe: Ridge
) -> np.ndarray:
    """
    Apply probe to meta activations using SEPARATE scaling (matches direct_to_meta_fixed).

    This fits a fresh StandardScaler on X before applying PCA and probe.
    This is what run_introspection_experiment.py does for its D2M transfer metric.
    """
    meta_scaler = StandardScaler()
    X_scaled = meta_scaler.fit_transform(X)
    X_pca = pca.transform(X_scaled)
    return probe.predict(X_pca)


def apply_probe_d2m_centered(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    probe: Ridge
) -> np.ndarray:
    """
    Apply probe to meta activations using CENTERED scaling (matches direct_to_meta_centered).

    1. Center X using its own mean
    2. Scale using the DIRECT scaler's variance (scaler.scale_)

    This is more rigorous than d2m_fixed because it doesn't refit the variance.
    """
    meta_mean = np.mean(X, axis=0)
    X_centered = X - meta_mean
    X_scaled = X_centered / scaler.scale_
    X_pca = pca.transform(X_scaled)
    return probe.predict(X_pca)


# =============================================================================
# MC ANSWER PROBE FUNCTIONS (for reverse analysis)
# =============================================================================

def train_mc_answer_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 256
) -> Tuple[StandardScaler, PCA, LogisticRegression]:
    """
    Train MC answer probe (4-class classification) on direct activations.

    Returns:
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        clf: Fitted LogisticRegression classifier
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
    X_pca = pca.fit_transform(X_scaled)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_pca, y_train)

    return scaler, pca, clf


def apply_mc_probe_d2m_fixed(
    X: np.ndarray,
    pca: PCA,
    clf: LogisticRegression
) -> np.ndarray:
    """
    Apply MC answer probe to meta activations using SEPARATE scaling.
    Returns predicted class labels (0-3 for A/B/C/D).
    """
    meta_scaler = StandardScaler()
    X_scaled = meta_scaler.fit_transform(X)
    X_pca = pca.transform(X_scaled)
    return clf.predict(X_pca)


def apply_mc_probe_d2m_centered(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    clf: LogisticRegression
) -> np.ndarray:
    """
    Apply MC answer probe using CENTERED scaling (meta mean, direct variance).
    Returns predicted class labels (0-3 for A/B/C/D).
    """
    meta_mean = np.mean(X, axis=0)
    X_centered = X - meta_mean
    X_scaled = X_centered / scaler.scale_
    X_pca = pca.transform(X_scaled)
    return clf.predict(X_pca)


# =============================================================================
# OPTIMIZED ABLATION INFRASTRUCTURE
# =============================================================================

class ParallelAblationHook:
    """
    Applies N different ablation vectors to a batch expanded by factor N.
    Can optionally capture the post-ablation activations.
    """
    def __init__(self, vectors: torch.Tensor, capture: bool = False):
        self.vectors = vectors # (N_vectors, Hidden)
        self.capture = capture
        self.captured_activations = None # [N_vec, B, Hidden]
        self.handle = None

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        
        total_batch = hs.shape[0]
        n_vectors = self.vectors.shape[0]
        
        if total_batch % n_vectors != 0:
            return output
            
        real_batch = total_batch // n_vectors
        
        # [N_vec, B, Seq, H]
        hs_view = hs.view(n_vectors, real_batch, -1, hs.shape[-1])
        vecs_view = self.vectors.view(n_vectors, 1, 1, -1).to(hs.dtype)
        
        if INTERVENTION_POSITION == "last":
            last_token = hs_view[:, :, -1:, :]
            dots = torch.sum(last_token * vecs_view, dim=-1, keepdim=True)
            proj = dots * vecs_view
            hs_view[:, :, -1:, :] = last_token - proj
            
            if self.capture:
                # Capture the modified last token: [N_vec, B, H]
                self.captured_activations = hs_view[:, :, -1, :].detach().cpu().numpy()
        else:
            dots = torch.sum(hs_view * vecs_view, dim=-1, keepdim=True)
            proj = dots * vecs_view
            hs_view = hs_view - proj
            
            if self.capture:
                 self.captured_activations = hs_view[:, :, -1, :].detach().cpu().numpy()

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ActivationCaptureHook:
    """Captures post-ablation activations at a downstream layer (no ablation, just capture)."""
    def __init__(self, n_variants: int):
        self.n_variants = n_variants
        self.captured = None  # Will be [N_variants, B, Hidden]
        self.handle = None

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        total_batch = hs.shape[0]
        if total_batch % self.n_variants != 0:
            return output
        real_batch = total_batch // self.n_variants
        # hs: [N_var * B, Seq, H] -> [N_var, B, H] (last token)
        hs_view = hs.view(self.n_variants, real_batch, -1, hs.shape[-1])
        self.captured = hs_view[:, :, -1, :].detach().cpu().numpy()
        return output

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class SimpleAblationHook:
    """
    Ablates a single direction from activations. No batch expansion.
    Used for simultaneous all-layer ablation.
    """
    def __init__(self, direction: torch.Tensor):
        self.direction = direction  # (Hidden,) normalized
        self.handle = None

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output

        vec = self.direction.view(1, 1, -1).to(hs.dtype)

        if INTERVENTION_POSITION == "last":
            last_token = hs[:, -1:, :]
            dot = torch.sum(last_token * vec, dim=-1, keepdim=True)
            proj = dot * vec
            hs[:, -1:, :] = last_token - proj
        else:
            dot = torch.sum(hs * vec, dim=-1, keepdim=True)
            proj = dot * vec
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


# =============================================================================
# KV CACHE HELPERS
# =============================================================================

def extract_cache_tensors(past_key_values):
    """
    Extract raw tensors from past_key_values (tuple or DynamicCache).
    Returns (key_tensors, value_tensors) where each is a list of tensors.
    """
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


def get_kv_cache(model, batch_inputs):
    """
    Run the prefix to generate KV cache tensors.
    Returns dictionary with next-step inputs and 'past_key_values_data' (snapshot).

    OPTIMIZATION: Uses base transformer instead of full CausalLM to avoid
    computing full-vocab logits for the prefix. We only need the KV cache.
    """
    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]

    # Get the base transformer (handles PEFT/LoRA models)
    if hasattr(model, 'get_base_model'):
        base_model = model.get_base_model().model
    else:
        base_model = model.model

    # Run Prefix (Tokens 0 to T-1)
    if input_ids.shape[1] > 1:
        prefix_ids = input_ids[:, :-1]
        prefix_mask = attention_mask[:, :-1]

        with torch.inference_mode():
            # Use base transformer to skip lm_head projection
            # This avoids computing full-vocab logits just to get KV cache
            outputs = base_model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        last_ids = input_ids[:, -1:]
    else:
        past_key_values = None
        last_ids = input_ids

    result = {
        "input_ids": last_ids,
        "attention_mask": attention_mask,  # Full mask
        "past_key_values": past_key_values,
    }

    if "position_ids" in batch_inputs:
        result["position_ids"] = batch_inputs["position_ids"][:, -1:]

    return result


def get_option_logits(model, inputs, option_token_ids):
    """
    Run forward pass and compute logits ONLY for the specified option tokens.

    OPTIMIZATION: Uses base transformer to get hidden states, then projects only
    the option tokens through lm_head. This avoids computing the full vocab
    projection (128K+ tokens) when we only need 4-5 option tokens.

    Args:
        model: The CausalLM model (e.g., LlamaForCausalLM)
        inputs: Dict with input_ids, attention_mask, past_key_values, etc.
        option_token_ids: List/tensor of token IDs to compute logits for

    Returns:
        logits: Tensor of shape [batch_size, len(option_token_ids)]
    """
    with torch.inference_mode():
        # Get the base transformer (handles PEFT/LoRA models)
        if hasattr(model, 'get_base_model'):
            base_model = model.get_base_model().model
            lm_head = model.get_base_model().lm_head
        else:
            base_model = model.model
            lm_head = model.lm_head

        # Run base transformer to get hidden states (no lm_head)
        outputs = base_model(**inputs)
        hidden_states = outputs.last_hidden_state[:, -1, :]  # [B, hidden_dim]

        # Project only option tokens through lm_head
        # lm_head.weight has shape [vocab_size, hidden_dim]
        option_weights = lm_head.weight[option_token_ids]  # [n_options, hidden_dim]
        logits = torch.matmul(hidden_states, option_weights.T)  # [B, n_options]

    return logits


def expand_inputs_for_parallel_ablation(step_data, n_expansions):
    """
    Expands input_ids and KV cache for parallel processing.
    Robustly handles DynamicCache vs Legacy Cache.
    """
    input_ids = step_data["input_ids"]
    pkv = step_data["past_key_values"]
    
    input_ids_exp = input_ids.repeat(n_expansions, 1)
    expanded_pkv = None
    
    if pkv is not None:
        if DynamicCache is not None and isinstance(pkv, DynamicCache):
            expanded_pkv = DynamicCache()
            
            # --- Robust Layer Extraction ---
            layers_data = []
            if hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0:
                # Standard DynamicCache with exposed lists
                layers_data = zip(pkv.key_cache, pkv.value_cache)
            elif hasattr(pkv, "to_legacy_cache"):
                # Safe fallback to tuple of tuples
                layers_data = pkv.to_legacy_cache()
            else:
                # Fallback: try iterating assuming it behaves like a list/tuple
                try:
                    layers_data = list(pkv)
                except Exception:
                    # Last resort: try generic indexing if len is available
                    try:
                        layers_data = [pkv[i] for i in range(len(pkv))]
                    except Exception as e:
                        raise ValueError(f"Could not extract layers from DynamicCache: {e}")

            # Expand and Update
            for layer_idx, (k, v) in enumerate(layers_data):
                # k, v are tensors [B, H, S, D]
                k_exp = k.repeat(n_expansions, 1, 1, 1)
                v_exp = v.repeat(n_expansions, 1, 1, 1)
                expanded_pkv.update(k_exp, v_exp, layer_idx)
        else:
            # Legacy Tuple of Tuples
            layers = []
            for k, v in pkv:
                k_exp = k.repeat(n_expansions, 1, 1, 1)
                v_exp = v.repeat(n_expansions, 1, 1, 1)
                layers.append((k_exp, v_exp))
            expanded_pkv = tuple(layers)
            
    mask = step_data["attention_mask"]
    mask_exp = mask.repeat(n_expansions, 1)
    
    return {
        "input_ids": input_ids_exp,
        "attention_mask": mask_exp,
        "past_key_values": expanded_pkv,
        "use_cache": True
    }


def generate_orthogonal_directions(direction: np.ndarray, num_directions: int) -> List[np.ndarray]:
    """Generate random directions orthogonal to the given direction."""
    hidden_dim = len(direction)
    mat = np.random.randn(num_directions, hidden_dim)
    dir_norm = direction / np.linalg.norm(direction)
    # Remove projection onto main direction
    projs = np.outer(mat @ dir_norm, dir_norm)
    mat = mat - projs
    # Normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / norms
    return list(mat)


def pretokenize_prompts(prompts: List[str], tokenizer, device: str) -> List[Dict]:
    """Pre-tokenize all prompts."""
    cached = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", padding=False, truncation=True)
        cached.append({
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        })
    return cached


def build_padded_gpu_batches(
    cached_inputs: List[Dict],
    tokenizer,
    device: str,
    batch_size: int
) -> List[Tuple[List[int], Dict]]:
    """Build padded batches for GPU processing."""
    batches = []
    n = len(cached_inputs)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = list(range(start, end))

        # Find max length in batch
        max_len = max(cached_inputs[i]["input_ids"].shape[1] for i in batch_indices)

        # Pad and stack
        input_ids_list = []
        attention_mask_list = []

        for i in batch_indices:
            ids = cached_inputs[i]["input_ids"]
            mask = cached_inputs[i]["attention_mask"]
            pad_len = max_len - ids.shape[1]

            if pad_len > 0:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                ids = torch.cat([
                    torch.full((1, pad_len), pad_id, dtype=ids.dtype, device=device),
                    ids
                ], dim=1)
                mask = torch.cat([
                    torch.zeros((1, pad_len), dtype=mask.dtype, device=device),
                    mask
                ], dim=1)

            input_ids_list.append(ids)
            attention_mask_list.append(mask)

        batch_inputs = {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
        }

        batches.append((batch_indices, batch_inputs))

    return batches


# =============================================================================
# CORE EXPERIMENT FUNCTIONS
# =============================================================================

def compute_direction_similarity(
    mc_directions: Dict[int, np.ndarray],
    entropy_directions: Dict[int, np.ndarray]
) -> Dict[int, float]:
    """Compute cosine similarity between MC answer and entropy directions."""
    similarities = {}
    common_layers = set(mc_directions.keys()) & set(entropy_directions.keys())

    for layer_idx in sorted(common_layers):
        mc_dir = mc_directions[layer_idx]
        ent_dir = entropy_directions[layer_idx]

        # Normalize (should already be normalized, but ensure)
        mc_dir = mc_dir / np.linalg.norm(mc_dir)
        ent_dir = ent_dir / np.linalg.norm(ent_dir)

        similarities[layer_idx] = float(np.dot(mc_dir, ent_dir))

    return similarities


def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation."""
    if len(confidences) < 2 or np.std(confidences) == 0 or np.std(metric_values) == 0:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def bootstrap_metric_change_ci(
    y_true: np.ndarray,
    baseline_preds: np.ndarray,
    ablated_preds: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for (ablated_metric - baseline_metric).

    Args:
        y_true: Ground truth values
        baseline_preds: Baseline predictions (no ablation)
        ablated_preds: Predictions after ablation
        metric_fn: Function(y_true, y_pred) -> float (e.g., accuracy_score, r2_score)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed

    Returns:
        (ci_low, ci_high, point_estimate): CI bounds and point estimate for the change
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Point estimate
    base_metric = metric_fn(y_true, baseline_preds)
    ablated_metric = metric_fn(y_true, ablated_preds)
    point_estimate = ablated_metric - base_metric

    # Bootstrap
    boot_changes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_base = metric_fn(y_true[idx], baseline_preds[idx])
        boot_abl = metric_fn(y_true[idx], ablated_preds[idx])
        boot_changes.append(boot_abl - boot_base)

    boot_changes = np.array(boot_changes)
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_changes, 100 * alpha / 2)
    ci_high = np.percentile(boot_changes, 100 * (1 - alpha / 2))

    return float(ci_low), float(ci_high), float(point_estimate)


def bootstrap_accuracy_change_ci(
    baseline_correct: np.ndarray,
    ablated_correct: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for accuracy change using per-question correctness.

    This is more efficient than bootstrap_metric_change_ci for accuracy since
    we already have binary correctness vectors.

    Args:
        baseline_correct: Binary array (1=correct, 0=incorrect) for baseline
        ablated_correct: Binary array for ablated condition
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        (ci_low, ci_high, point_estimate): CI bounds and point estimate
    """
    rng = np.random.default_rng(seed)
    n = len(baseline_correct)

    # Point estimate
    base_acc = np.mean(baseline_correct)
    ablated_acc = np.mean(ablated_correct)
    point_estimate = ablated_acc - base_acc

    # Bootstrap
    boot_changes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_base = np.mean(baseline_correct[idx])
        boot_abl = np.mean(ablated_correct[idx])
        boot_changes.append(boot_abl - boot_base)

    boot_changes = np.array(boot_changes)
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_changes, 100 * alpha / 2)
    ci_high = np.percentile(boot_changes, 100 * (1 - alpha / 2))

    return float(ci_low), float(ci_high), float(point_estimate)


def bootstrap_correlation_change_ci(
    baseline_conf: np.ndarray,
    baseline_metric: np.ndarray,
    ablated_conf: np.ndarray,
    ablated_metric: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for correlation change.

    Args:
        baseline_conf: Confidence values for baseline condition
        baseline_metric: Metric values for baseline condition
        ablated_conf: Confidence values for ablated condition
        ablated_metric: Metric values for ablated condition
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        (ci_low, ci_high, point_estimate): CI bounds and point estimate
    """
    rng = np.random.default_rng(seed)
    n = len(baseline_conf)

    baseline_corr = compute_correlation(baseline_conf, baseline_metric)
    ablated_corr = compute_correlation(ablated_conf, ablated_metric)
    point_estimate = ablated_corr - baseline_corr

    boot_changes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_base_corr = compute_correlation(baseline_conf[idx], baseline_metric[idx])
        boot_abl_corr = compute_correlation(ablated_conf[idx], ablated_metric[idx])
        boot_changes.append(boot_abl_corr - boot_base_corr)

    boot_changes = np.array(boot_changes)
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_changes, 100 * alpha / 2)
    ci_high = np.percentile(boot_changes, 100 * (1 - alpha / 2))

    return float(ci_low), float(ci_high), float(point_estimate)


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool) -> str:
    """Format a meta/confidence question."""
    prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
    return prompt


def local_response_to_confidence(response: str, probs: np.ndarray, mapping: Optional[Dict]) -> float:
    """Convert response to confidence value."""
    if META_TASK == "delegate":
        return response_to_confidence(response, probs, mapping)
    else:
        return get_stated_confidence_signal(probs)


def run_experiment_parallel(
    model,
    tokenizer,
    questions: List[Dict],
    direct_activations: Optional[Dict[int, np.ndarray]],
    d2m_target_values: Optional[np.ndarray],
    direct_metric_values: np.ndarray,
    layers: List[int],
    ablation_directions: Dict[int, np.ndarray],
    num_controls: int,
    use_chat_template: bool,
    test_indices: Optional[np.ndarray],
    run_d2m: bool,
    num_layers: int,
    all_measurement_layers: Optional[List[int]] = None,
    probe_type: str = "entropy_regression",  # or "mc_classification"
) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
    """
    Runs both Behavioral and D2M experiments in a parallelized manner.
    Includes memory optimization by batching the variants (chunking).

    Baseline D2M uses zero-vector "ablation" from captured activations (not pre-saved)
    to ensure consistency with control comparisons.

    Args:
        d2m_target_values: Target values for D2M probes. For entropy_regression mode,
                           this is entropy values. For mc_classification mode, this is
                           model_predicted_answer (0-3).
        ablation_directions: Directions to ablate (mc_directions for forward, entropy_directions for reverse)
        all_measurement_layers: If provided, measure D2M at ALL these layers for each ablation.
                               Creates a full ablation×measurement matrix.
        probe_type: "entropy_regression" (R² metric) or "mc_classification" (accuracy metric)
    """

    # --- SETUP D2M SPLITS IF NEEDED ---
    d2m_extra = None
    all_probes = {}  # {layer: (scaler, pca, probe)} - probes for ALL measurement layers
    final_layer = num_layers - 1
    measurement_layers_map = {}  # {ablation_layer: [(name, measure_layer), ...]}

    if run_d2m:
        n_samples = len(questions)
        all_indices = np.arange(n_samples)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]

        y_train = d2m_target_values[train_indices]
        y_test = d2m_target_values[test_indices]

        # Determine measurement layers
        if all_measurement_layers is not None:
            # Full matrix mode: measure at ALL downstream layers
            unique_measurement_layers = set(all_measurement_layers)
            for ablation_layer in layers:
                # Measure at all layers >= ablation_layer (downstream + same)
                measure_at = []
                for m_layer in sorted(all_measurement_layers):
                    if m_layer >= ablation_layer:
                        measure_at.append((f"layer_{m_layer}", m_layer))
                measurement_layers_map[ablation_layer] = measure_at
        else:
            # Legacy mode: just N+1 and final
            unique_measurement_layers = set()
            for ablation_layer in layers:
                measure_at = []
                # N+1 layer (if exists)
                if ablation_layer + 1 < num_layers:
                    measure_at.append(("n_plus_1", ablation_layer + 1))
                    unique_measurement_layers.add(ablation_layer + 1)
                # Final layer (if different from ablation layer)
                if final_layer != ablation_layer:
                    measure_at.append(("final", final_layer))
                    unique_measurement_layers.add(final_layer)
                # Also include same-layer for existing analysis
                unique_measurement_layers.add(ablation_layer)
                measurement_layers_map[ablation_layer] = measure_at

        # Pre-train probes for ALL measurement layers
        probe_desc = "MC answer (classification)" if probe_type == "mc_classification" else "entropy (regression)"
        print(f"  Pre-training {probe_desc} probes for {len(unique_measurement_layers)} measurement layers...")
        for measure_layer in sorted(unique_measurement_layers):
            if measure_layer in direct_activations:
                X_train = direct_activations[measure_layer][train_indices]
                if probe_type == "mc_classification":
                    scaler, pca, probe = train_mc_answer_probe(X_train, y_train)
                else:
                    scaler, pca, probe = train_entropy_probe(X_train, y_train)
                all_probes[measure_layer] = (scaler, pca, probe)

        # Storage for bootstrap CIs computation
        d2m_extra = {
            "y_test": y_test,
            "captured_activations": {},  # {layer: [baseline_acts, mc_acts]} for test set
            "probes": all_probes,  # All trained probes
            "measurement_layers_map": measurement_layers_map,  # For analysis
        }
    
    # --- PREPARE BATCHES (Shared) ---
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "delegate":
            prompt, _, mapping = format_answer_or_delegate_prompt(
                question, tokenizer, use_chat_template, trial_index=q_idx
            )
        else:
            prompt = format_meta_prompt(question, tokenizer, use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)
        
    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)
    
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = list(ANSWER_OR_DELEGATE_OPTIONS)
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = list(STATED_CONFIDENCE_OPTIONS.keys())
        
    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    # --- RESULTS STORAGE ---
    beh_results = {}
    d2m_results = {}

    num_variants = 2 + num_controls # Baseline (0), MC (1), Controls (2..101)
    n_test = len(test_indices) if test_indices is not None else 0

    # --- PRE-GENERATE ALL CONTROL DIRECTIONS (for deterministic RNG) ---
    # This allows us to invert the loops while maintaining reproducibility
    all_vectors = {}  # {layer_idx: torch.Tensor of shape [num_variants, hidden_dim]}
    all_layer_modules = {}  # {layer_idx: layer_module}

    print(f"  Pre-generating control directions for {len(layers)} layers...")
    for layer_idx in layers:
        ablation_dir = ablation_directions[layer_idx]
        ablation_dir = ablation_dir / np.linalg.norm(ablation_dir)
        control_dirs = generate_orthogonal_directions(ablation_dir, num_controls)

        # Stack for parallel: [Baseline(Zero), Primary ablation, Controls...]
        zeros = np.zeros_like(ablation_dir)
        all_vecs = np.vstack([zeros, ablation_dir] + control_dirs)
        all_vectors[layer_idx] = torch.tensor(all_vecs, dtype=torch.float16, device=DEVICE)

        # Cache layer modules
        if hasattr(model, 'get_base_model'):
            all_layer_modules[layer_idx] = model.get_base_model().model.layers[layer_idx]
        else:
            all_layer_modules[layer_idx] = model.model.layers[layer_idx]

    # --- INITIALIZE STORAGE FOR ALL LAYERS ---
    hidden_dim = direct_activations[layers[0]].shape[1] if direct_activations else 0

    for layer_idx in layers:
        beh_results[layer_idx] = {
            "baseline": [None] * len(questions),
            "primary_ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    # D2M storage: only allocate for test_indices (MEMORY OPTIMIZATION)
    d2m_captured_all = {}  # {layer_idx: np.ndarray of shape [num_variants, n_test, hidden_dim]}
    downstream_captured_all = {}  # {layer_idx: {name: np.ndarray}}

    if run_d2m:
        for layer_idx in layers:
            # Use float16 for memory efficiency (MEMORY OPTIMIZATION)
            d2m_captured_all[layer_idx] = np.zeros((num_variants, n_test, hidden_dim), dtype=np.float16)

            measure_at = measurement_layers_map.get(layer_idx, [])
            downstream_captured_all[layer_idx] = {}
            for name, measure_layer in measure_at:
                if measure_layer in all_probes:
                    downstream_captured_all[layer_idx][name] = np.zeros((num_variants, n_test, hidden_dim), dtype=np.float16)

    # Build test_indices set for fast lookup
    test_indices_set = set(test_indices) if test_indices is not None else set()
    # Map from question index to test index
    q_idx_to_test_idx = {q_idx: t_idx for t_idx, q_idx in enumerate(test_indices)} if test_indices is not None else {}

    # --- MAIN LOOP: BATCHES FIRST, THEN LAYERS (KV CACHE OPTIMIZATION) ---
    # This computes KV cache once per batch, reused across all layers
    print(f"  Running {len(gpu_batches)} batches × {len(layers)} layers...")

    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Batches"):
        B_local = len(batch_indices)

        # Compute KV cache ONCE per batch (MAJOR SPEEDUP)
        step_data = get_kv_cache(model, batch_inputs)

        # Now iterate over layers, reusing the same KV cache
        for layer_idx in layers:
            vectors = all_vectors[layer_idx]
            layer_module = all_layer_modules[layer_idx]
            measure_at = measurement_layers_map.get(layer_idx, [])

            # Get downstream capture storage for this layer
            downstream_captured = downstream_captured_all.get(layer_idx, {})

            # Split the variants into chunks to save memory
            for v_start in range(0, num_variants, VARIANT_BATCH_SIZE):
                v_end = min(v_start + VARIANT_BATCH_SIZE, num_variants)
                chunk_size = v_end - v_start

                # 1. Expand inputs for this chunk
                expanded_inputs = expand_inputs_for_parallel_ablation(step_data, chunk_size)

                # 2. Slice vectors
                chunk_vectors = vectors[v_start:v_end]

                # 3. Hook for ablation at this layer
                hook = ParallelAblationHook(chunk_vectors, capture=run_d2m)
                hook.register(layer_module)

                # 4. Hooks for capturing at downstream layers
                capture_hooks = {}  # {name: ActivationCaptureHook}
                if run_d2m:
                    for name, measure_layer in measure_at:
                        if name in downstream_captured:
                            if hasattr(model, 'get_base_model'):
                                measure_module = model.get_base_model().model.layers[measure_layer]
                            else:
                                measure_module = model.model.layers[measure_layer]
                            cap_hook = ActivationCaptureHook(chunk_size)
                            cap_hook.register(measure_module)
                            capture_hooks[name] = cap_hook

                try:
                    # OPTIMIZATION: Use get_option_logits to avoid full vocab projection
                    logits = get_option_logits(model, expanded_inputs, option_token_ids)

                    # -- BEHAVIORAL PROCESSING --
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                    # Reshape: [ChunkSize * B, Options] -> [ChunkSize, B, Options]
                    probs_view = probs.reshape(chunk_size, B_local, -1)

                    for local_v_idx in range(chunk_size):
                        global_v_idx = v_start + local_v_idx

                        for b_i, q_idx in enumerate(batch_indices):
                            p = probs_view[local_v_idx, b_i]
                            resp = options[np.argmax(p)]
                            conf = local_response_to_confidence(resp, p, mappings[q_idx])
                            m_val = direct_metric_values[q_idx]
                            align = -((m_val - metric_mean) / metric_std) * ((conf - 0.5) / 0.25)

                            data = {
                                "question_idx": q_idx,
                                "response": resp,
                                "confidence": conf,
                                "metric": float(m_val),
                                "alignment": float(align)
                            }

                            if global_v_idx == 0:
                                beh_results[layer_idx]["baseline"][q_idx] = data
                            elif global_v_idx == 1:
                                beh_results[layer_idx]["primary_ablated"][q_idx] = data
                            else:
                                c_idx = global_v_idx - 2
                                beh_results[layer_idx]["controls_ablated"][f"control_{c_idx}"][q_idx] = data

                    # -- D2M CAPTURE (same-layer) - only for test indices --
                    if run_d2m:
                        acts = hook.captured_activations
                        # acts shape: [ChunkSize, B, Hidden]
                        for local_v_idx in range(chunk_size):
                            global_v_idx = v_start + local_v_idx
                            for i, q_idx in enumerate(batch_indices):
                                if q_idx in test_indices_set:
                                    t_idx = q_idx_to_test_idx[q_idx]
                                    # acts is already numpy (converted in hook)
                                    d2m_captured_all[layer_idx][global_v_idx, t_idx] = acts[local_v_idx, i].astype(np.float16)

                        # -- DOWNSTREAM CAPTURE --
                        for name, cap_hook in capture_hooks.items():
                            if cap_hook.captured is not None:
                                for local_v_idx in range(chunk_size):
                                    global_v_idx = v_start + local_v_idx
                                    for i, q_idx in enumerate(batch_indices):
                                        if q_idx in test_indices_set:
                                            t_idx = q_idx_to_test_idx[q_idx]
                                            # captured is already numpy (converted in hook)
                                            downstream_captured[name][global_v_idx, t_idx] = cap_hook.captured[local_v_idx, i].astype(np.float16)

                finally:
                    hook.remove()
                    for cap_hook in capture_hooks.values():
                        cap_hook.remove()

    # Clear GPU cache once at the end (not per-layer)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --- COMPUTE D2M SCORES (after all batches/layers are done) ---
    if run_d2m:
        print(f"  Computing D2M scores for {len(layers)} layers...")
        for layer_idx in layers:
            # Get captured activations for this layer (already sliced to test indices)
            captured_test = d2m_captured_all[layer_idx]  # Shape: [num_variants, n_test, hidden_dim]
            downstream_captured = downstream_captured_all.get(layer_idx, {})
            measure_at = measurement_layers_map.get(layer_idx, [])

            # Get probe for this layer
            scaler, pca, probe = all_probes.get(layer_idx, (None, None, None))

            # CRITICAL: Use captured_test[0] (zero-vector ablation) as baseline
            # NOT pre-saved meta activations, because control scores are computed from
            # captured activations. Using pre-saved would introduce systematic bias.
            baseline_captured = captured_test[0]  # Zero-vector "ablation" = no change

            if probe_type == "mc_classification":
                # Classification mode: use accuracy
                # Store per-question predictions for bootstrap CIs
                base_preds = apply_mc_probe_d2m_fixed(baseline_captured, pca, probe)
                primary_preds = apply_mc_probe_d2m_fixed(captured_test[1], pca, probe)
                base_acc = accuracy_score(y_test, base_preds)
                primary_acc = accuracy_score(y_test, primary_preds)
                ctrl_accs = []
                for c_idx in range(num_controls):
                    c_acc = accuracy_score(y_test, apply_mc_probe_d2m_fixed(captured_test[c_idx + 2], pca, probe))
                    ctrl_accs.append(c_acc)

                # Per-question correctness for bootstrap
                base_correct = (base_preds == y_test).astype(int)
                primary_correct = (primary_preds == y_test).astype(int)

                d2m_results[layer_idx] = {
                    "same_layer": {
                        "baseline_acc": float(base_acc),
                        "primary_ablated_acc": float(primary_acc),
                        "primary_acc_change": float(primary_acc - base_acc),
                        "control_accs": ctrl_accs,
                        "control_acc_changes": [a - base_acc for a in ctrl_accs],
                        # Per-question data for bootstrap
                        "baseline_correct": base_correct.tolist(),
                        "primary_correct": primary_correct.tolist(),
                    },
                    "downstream": {},
                    "probe_type": "mc_classification",
                }
            else:
                # Regression mode: use R²
                # Store per-question predictions for bootstrap CIs
                base_preds_fixed = apply_probe_d2m_fixed(baseline_captured, pca, probe)
                primary_preds_fixed = apply_probe_d2m_fixed(captured_test[1], pca, probe)
                base_r2_fixed = r2_score(y_test, base_preds_fixed)
                primary_r2_fixed = r2_score(y_test, primary_preds_fixed)
                ctrl_r2s_fixed = []
                for c_idx in range(num_controls):
                    c_r2 = r2_score(y_test, apply_probe_d2m_fixed(captured_test[c_idx + 2], pca, probe))
                    ctrl_r2s_fixed.append(c_r2)

                # 2. Centered: meta mean, direct variance
                base_preds_centered = apply_probe_d2m_centered(baseline_captured, scaler, pca, probe)
                primary_preds_centered = apply_probe_d2m_centered(captured_test[1], scaler, pca, probe)
                base_r2_centered = r2_score(y_test, base_preds_centered)
                primary_r2_centered = r2_score(y_test, primary_preds_centered)
                ctrl_r2s_centered = []
                for c_idx in range(num_controls):
                    c_r2 = r2_score(y_test, apply_probe_d2m_centered(captured_test[c_idx + 2], scaler, pca, probe))
                    ctrl_r2s_centered.append(c_r2)

                d2m_results[layer_idx] = {
                    "same_layer": {
                        "baseline_r2": float(base_r2_fixed),
                        "primary_ablated_r2": float(primary_r2_fixed),
                        "primary_r2_change": float(primary_r2_fixed - base_r2_fixed),
                        "control_r2s": ctrl_r2s_fixed,
                        "control_r2_changes": [r - base_r2_fixed for r in ctrl_r2s_fixed],
                        "baseline_r2_centered": float(base_r2_centered),
                        "primary_ablated_r2_centered": float(primary_r2_centered),
                        "primary_r2_change_centered": float(primary_r2_centered - base_r2_centered),
                        "control_r2s_centered": ctrl_r2s_centered,
                        "control_r2_changes_centered": [r - base_r2_centered for r in ctrl_r2s_centered],
                        # Per-question data for bootstrap (y_test is shared, store predictions)
                        "y_test": y_test.tolist(),
                        "baseline_preds_fixed": base_preds_fixed.tolist(),
                        "primary_preds_fixed": primary_preds_fixed.tolist(),
                        "baseline_preds_centered": base_preds_centered.tolist(),
                        "primary_preds_centered": primary_preds_centered.tolist(),
                    },
                    "downstream": {},
                    "probe_type": "entropy_regression",
                }

            # Compute downstream D2M scores (already sliced to test indices)
            for name, measure_layer in measure_at:
                if name in downstream_captured and measure_layer in all_probes:
                    ds_captured_test = downstream_captured[name]  # Already [num_variants, n_test, hidden]
                    ds_scaler, ds_pca, ds_probe = all_probes[measure_layer]

                    if probe_type == "mc_classification":
                        # Get per-question predictions for bootstrap CIs
                        ds_base_preds = apply_mc_probe_d2m_fixed(ds_captured_test[0], ds_pca, ds_probe)
                        ds_primary_preds = apply_mc_probe_d2m_fixed(ds_captured_test[1], ds_pca, ds_probe)
                        ds_base_correct = (ds_base_preds == y_test).astype(int)
                        ds_primary_correct = (ds_primary_preds == y_test).astype(int)
                        ds_base = float(np.mean(ds_base_correct))
                        ds_primary = float(np.mean(ds_primary_correct))
                        ds_ctrls = []
                        for c_idx in range(num_controls):
                            c_score = accuracy_score(y_test, apply_mc_probe_d2m_fixed(ds_captured_test[c_idx + 2], ds_pca, ds_probe))
                            ds_ctrls.append(c_score)

                        d2m_results[layer_idx]["downstream"][name] = {
                            "measure_layer": measure_layer,
                            "baseline_acc": float(ds_base),
                            "primary_ablated_acc": float(ds_primary),
                            "primary_acc_change": float(ds_primary - ds_base),
                            "control_accs": ds_ctrls,
                            "control_acc_changes": [a - ds_base for a in ds_ctrls],
                            # Per-question data for bootstrap CIs
                            "baseline_correct": ds_base_correct.tolist(),
                            "primary_correct": ds_primary_correct.tolist(),
                        }
                    else:
                        # Get per-question predictions for bootstrap CIs
                        ds_base_preds = apply_probe_d2m_fixed(ds_captured_test[0], ds_pca, ds_probe)
                        ds_primary_preds = apply_probe_d2m_fixed(ds_captured_test[1], ds_pca, ds_probe)
                        ds_base_r2 = r2_score(y_test, ds_base_preds)
                        ds_primary_r2 = r2_score(y_test, ds_primary_preds)
                        ds_ctrl_r2s = []
                        for c_idx in range(num_controls):
                            c_r2 = r2_score(y_test, apply_probe_d2m_fixed(ds_captured_test[c_idx + 2], ds_pca, ds_probe))
                            ds_ctrl_r2s.append(c_r2)

                        d2m_results[layer_idx]["downstream"][name] = {
                            "measure_layer": measure_layer,
                            "baseline_r2": float(ds_base_r2),
                            "primary_ablated_r2": float(ds_primary_r2),
                            "primary_r2_change": float(ds_primary_r2 - ds_base_r2),
                            "control_r2s": ds_ctrl_r2s,
                            "control_r2_changes": [r - ds_base_r2 for r in ds_ctrl_r2s],
                            # Per-question data for bootstrap CIs
                            "y_test": y_test.tolist(),
                            "baseline_preds": ds_base_preds.tolist(),
                            "primary_preds": ds_primary_preds.tolist(),
                        }

            # Store captured baseline and primary for diagnostics
            d2m_extra["captured_activations"][layer_idx] = np.stack([baseline_captured, captured_test[1]])  # [2, n_test, hidden]

    return beh_results, (d2m_results if run_d2m else None), d2m_extra


def run_simultaneous_all_layer_ablation(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    ablation_directions: Dict[int, np.ndarray],
    use_chat_template: bool,
    num_controls: int = 25,
    ablation_name: str = "MC answer",
) -> Dict:
    """
    Ablate a direction at ALL layers simultaneously.
    Returns a SINGLE behavioral correlation (stated confidence vs entropy).

    Also runs control ablations (random orthogonal directions at each layer)
    to compute p-value and CI.

    Args:
        ablation_directions: Directions to ablate (mc_directions or entropy_directions)
        ablation_name: Name for logging ("MC answer" or "entropy")

    This answers: "What happens to introspection if we remove the [direction]
    representation from the ENTIRE model?"
    """
    print("\n" + "=" * 70)
    print("SIMULTANEOUS ALL-LAYER ABLATION")
    print(f"Ablating {ablation_name} direction at {len(ablation_directions)} layers simultaneously")
    print(f"Running {num_controls} control ablations for CI")
    print("=" * 70)

    # Prepare prompts
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "delegate":
            prompt, _, mapping = format_answer_or_delegate_prompt(
                question, tokenizer, use_chat_template, trial_index=q_idx
            )
        else:
            prompt = format_meta_prompt(question, tokenizer, use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = list(ANSWER_OR_DELEGATE_OPTIONS)
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = list(STATED_CONFIDENCE_OPTIONS.keys())

    # Get layer modules
    layer_modules = {}
    sorted_layers = sorted(ablation_directions.keys())
    for layer_idx in sorted_layers:
        if hasattr(model, 'get_base_model'):
            layer_modules[layer_idx] = model.get_base_model().model.layers[layer_idx]
        else:
            layer_modules[layer_idx] = model.model.layers[layer_idx]

    # Prepare ablation directions as tensors
    primary_direction_tensors = {}
    for layer_idx, direction in ablation_directions.items():
        dir_norm = direction / np.linalg.norm(direction)
        primary_direction_tensors[layer_idx] = torch.tensor(dir_norm, dtype=torch.float16, device=DEVICE)

    # Generate control directions for each layer
    # Each control set is {layer: random_orthogonal_direction}
    print(f"  Generating {num_controls} control direction sets...")
    control_direction_sets = []
    for ctrl_idx in range(num_controls):
        ctrl_set = {}
        for layer_idx, abl_dir in ablation_directions.items():
            # Generate one random orthogonal direction per layer
            ctrl_dirs = generate_orthogonal_directions(abl_dir / np.linalg.norm(abl_dir), 1)
            ctrl_set[layer_idx] = torch.tensor(ctrl_dirs[0], dtype=torch.float16, device=DEVICE)
        control_direction_sets.append(ctrl_set)

    def run_with_directions(direction_tensors: Dict[int, torch.Tensor], desc: str) -> List[Dict]:
        """Run inference with given directions ablated at all layers."""
        results = []
        for batch_indices, batch_inputs in tqdm(gpu_batches, desc=desc, leave=False):
            hooks = []
            for layer_idx in sorted_layers:
                hook = SimpleAblationHook(direction_tensors[layer_idx])
                hook.register(layer_modules[layer_idx])
                hooks.append(hook)

            try:
                # OPTIMIZATION: Use get_option_logits to avoid full vocab projection
                logits = get_option_logits(model, batch_inputs, option_token_ids)
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf = local_response_to_confidence(resp, p, mappings[q_idx])
                    results.append({
                        "question_idx": q_idx,
                        "confidence": conf,
                        "metric": float(direct_metric_values[q_idx]),
                    })
            finally:
                for hook in hooks:
                    hook.remove()
        return results

    # Run baseline (no ablation)
    print("  Running baseline (no ablation)...")
    baseline_results = []
    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Baseline", leave=False):
        # OPTIMIZATION: Use get_option_logits to avoid full vocab projection
        logits = get_option_logits(model, batch_inputs, option_token_ids)
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

        for i, q_idx in enumerate(batch_indices):
            p = probs[i]
            resp = options[np.argmax(p)]
            conf = local_response_to_confidence(resp, p, mappings[q_idx])
            baseline_results.append({
                "question_idx": q_idx,
                "confidence": conf,
                "metric": float(direct_metric_values[q_idx]),
            })

    # Run with primary direction ablation
    print(f"  Running with {ablation_name} direction ablated at ALL layers...")
    primary_ablated_results = run_with_directions(primary_direction_tensors, f"{ablation_name} ablation")

    # Run control ablations
    print(f"  Running {num_controls} control ablations...")
    control_results = []
    for ctrl_idx, ctrl_set in enumerate(control_direction_sets):
        ctrl_res = run_with_directions(ctrl_set, f"Control {ctrl_idx+1}/{num_controls}")
        control_results.append(ctrl_res)

    # Compute correlations
    baseline_conf = np.array([r["confidence"] for r in baseline_results])
    baseline_metric = np.array([r["metric"] for r in baseline_results])
    primary_conf = np.array([r["confidence"] for r in primary_ablated_results])

    baseline_corr = compute_correlation(baseline_conf, baseline_metric)
    primary_corr = compute_correlation(primary_conf, baseline_metric)
    primary_change = primary_corr - baseline_corr

    # Control correlations
    control_corrs = []
    control_changes = []
    for ctrl_res in control_results:
        ctrl_conf = np.array([r["confidence"] for r in ctrl_res])
        ctrl_corr = compute_correlation(ctrl_conf, baseline_metric)
        control_corrs.append(ctrl_corr)
        control_changes.append(ctrl_corr - baseline_corr)

    control_changes = np.array(control_changes)
    ctrl_mean = np.mean(control_changes)
    ctrl_std = np.std(control_changes)

    # Z-score
    z_score = (primary_change - ctrl_mean) / ctrl_std if ctrl_std > 0 else 0.0

    # Two-tailed p-value: use parametric (from z-score) since empirical is floored at 1/(n+1)
    # With only 25 controls, empirical p can't go below ~0.038, but z=-4 should be p<0.0001
    p_value = 2 * stats.norm.sf(np.abs(z_score))  # two-tailed from z

    # 95% CI from control distribution (null distribution of correlation changes)
    ci_low, ci_high = np.percentile(control_changes, [2.5, 97.5])

    result = {
        "ablation_name": ablation_name,
        "num_layers_ablated": len(ablation_directions),
        "layers_ablated": sorted_layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "baseline_correlation": float(baseline_corr),
        "primary_ablated_correlation": float(primary_corr),
        "primary_correlation_change": float(primary_change),
        "control_change_mean": float(ctrl_mean),
        "control_change_std": float(ctrl_std),
        "control_changes": [float(c) for c in control_changes],
        "p_value": float(p_value),
        "z_score": float(z_score),
        "null_ci_95": (float(ci_low), float(ci_high)),
    }

    print(f"\n  Baseline correlation (conf vs {METRIC}): {baseline_corr:.4f}")
    print(f"  {ablation_name}-ablated correlation:      {primary_corr:.4f}")
    print(f"  Change: {primary_change:+.4f}")
    print(f"  Null distribution: mean={ctrl_mean:+.4f}, std={ctrl_std:.4f}")
    print(f"  Null 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  z-score: {z_score:+.2f}, p-value: {p_value:.2e}")

    sig = "SIGNIFICANT" if p_value < 0.05 else "not significant"
    print(f"\n  Result: {sig} (p={p_value:.2e})")

    return result


def analyze_d2m_transfer_ablation_results(
    results_map: Dict,
    layers: List[int],
    num_test: int,
    num_controls: int,
) -> Dict:
    """Analyze D2M transfer ablation results with statistical testing.

    Uses control-based CIs (null distribution range) instead of bootstrap CIs.

    Handles both probe types:
    - entropy_regression: Uses R² metric with fixed and centered variants
    - mc_classification: Uses accuracy metric (no centered variant)

    Handles new structure with same_layer and downstream results.
    """
    # Detect probe type from first layer's results
    first_layer = layers[0]
    probe_type = results_map[first_layer].get("probe_type", "entropy_regression")

    analysis = {
        "layers": layers,
        "num_test": num_test,
        "num_controls": num_controls,
        "probe_type": probe_type,
        "effects": {},  # Same-layer (fixed for regression, only option for classification)
        "effects_centered": {},  # Same-layer centered (regression only)
        "effects_downstream_n_plus_1": {},  # Downstream N+1
        "effects_downstream_final": {},  # Downstream final layer
    }

    if probe_type == "mc_classification":
        # Classification mode: use accuracy
        # Per-layer analysis (not pooled across layers)
        raw_p_values = []
        for layer_idx in layers:
            lr = results_map[layer_idx]["same_layer"]
            primary_change = lr["primary_acc_change"]
            control_changes = np.array(lr["control_acc_changes"])

            # Per-layer null distribution
            layer_null_mean = np.mean(control_changes)
            layer_null_std = np.std(control_changes)
            layer_null_ci = np.percentile(control_changes, [2.5, 97.5])

            # Rank-based p-value: where does primary rank among controls?
            # This is the "direction specificity" test
            primary_abs = np.abs(primary_change)
            n_extreme = np.sum(np.abs(control_changes) >= primary_abs)
            p_value_rank = (n_extreme + 1) / (len(control_changes) + 1)

            # Z-score using per-layer null
            z_score = (primary_change - layer_null_mean) / layer_null_std if layer_null_std > 0 else 0.0

            # Bootstrap CI over questions (sampling uncertainty)
            boot_ci_low, boot_ci_high = None, None
            if "baseline_correct" in lr and "primary_correct" in lr:
                baseline_correct = np.array(lr["baseline_correct"])
                primary_correct = np.array(lr["primary_correct"])
                boot_ci_low, boot_ci_high, _ = bootstrap_accuracy_change_ci(
                    baseline_correct, primary_correct, n_bootstrap=1000, seed=42 + layer_idx
                )

            raw_p_values.append((layer_idx, p_value_rank))
            analysis["effects"][layer_idx] = {
                "baseline_acc": lr["baseline_acc"],
                "ablated_acc": lr["primary_ablated_acc"],
                "acc_change": primary_change,
                # Control distribution statistics
                "control_acc_mean": float(np.mean(lr["control_accs"])),
                "control_change_mean": float(layer_null_mean),
                "control_change_std": float(layer_null_std),
                "control_change_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                # Uniqueness: how much bigger than random?
                "uniqueness": float(primary_change - layer_null_mean),
                # Statistical tests
                "p_value_rank": p_value_rank,  # Direction specificity (rank among controls)
                "z_score": z_score,
                "null_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                # Bootstrap CI (sampling uncertainty over questions)
                "bootstrap_ci_95": (boot_ci_low, boot_ci_high) if boot_ci_low is not None else None,
            }

        # FDR correction across layers
        sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
        n_tests = len(sorted_pvals)
        fdr_adjusted = {}
        for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
            fdr_adjusted[layer_idx] = min(1.0, p_val * n_tests / rank)
        sorted_keys = [k for k, _ in sorted_pvals]
        min_q = 1.0
        for key in reversed(sorted_keys):
            fdr_adjusted[key] = min(min_q, fdr_adjusted[key])
            min_q = fdr_adjusted[key]
        for layer_idx in layers:
            analysis["effects"][layer_idx]["p_value_fdr"] = fdr_adjusted[layer_idx]

    else:
        # Regression mode: use R² with fixed and centered variants
        # Per-layer analysis (not pooled across layers)
        raw_p_values_fixed = []
        raw_p_values_centered = []

        for layer_idx in layers:
            lr = results_map[layer_idx]["same_layer"]

            # === FIXED (separate scaler) ===
            primary_r2_change = lr["primary_r2_change"]
            control_r2_changes = np.array(lr["control_r2_changes"])

            # Per-layer null distribution
            layer_null_mean = np.mean(control_r2_changes)
            layer_null_std = np.std(control_r2_changes)
            layer_null_ci = np.percentile(control_r2_changes, [2.5, 97.5])

            # Rank-based p-value
            primary_abs = np.abs(primary_r2_change)
            n_extreme = np.sum(np.abs(control_r2_changes) >= primary_abs)
            p_value_rank = (n_extreme + 1) / (len(control_r2_changes) + 1)
            z_score = (primary_r2_change - layer_null_mean) / layer_null_std if layer_null_std > 0 else 0.0

            # Bootstrap CI over questions (sampling uncertainty) - fixed
            boot_ci_low_fixed, boot_ci_high_fixed = None, None
            if "y_test" in lr and "baseline_preds_fixed" in lr and "primary_preds_fixed" in lr:
                y_test = np.array(lr["y_test"])
                base_preds = np.array(lr["baseline_preds_fixed"])
                primary_preds = np.array(lr["primary_preds_fixed"])
                boot_ci_low_fixed, boot_ci_high_fixed, _ = bootstrap_metric_change_ci(
                    y_test, base_preds, primary_preds, r2_score, n_bootstrap=1000, seed=42 + layer_idx
                )

            raw_p_values_fixed.append((layer_idx, p_value_rank))
            analysis["effects"][layer_idx] = {
                "baseline_r2": lr["baseline_r2"],
                "ablated_r2": lr["primary_ablated_r2"],
                "r2_change": primary_r2_change,
                # Control distribution statistics
                "control_r2_mean": float(np.mean(lr["control_r2s"])),
                "control_change_mean": float(layer_null_mean),
                "control_change_std": float(layer_null_std),
                "control_change_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                # Uniqueness
                "uniqueness": float(primary_r2_change - layer_null_mean),
                # Statistical tests
                "p_value_rank": p_value_rank,
                "z_score": z_score,
                "null_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                # Bootstrap CI (sampling uncertainty over questions)
                "bootstrap_ci_95": (boot_ci_low_fixed, boot_ci_high_fixed) if boot_ci_low_fixed is not None else None,
            }

            # === CENTERED (meta mean, direct variance) ===
            primary_r2_change_cen = lr["primary_r2_change_centered"]
            control_r2_changes_cen = np.array(lr["control_r2_changes_centered"])

            # Per-layer null distribution (centered)
            layer_null_mean_cen = np.mean(control_r2_changes_cen)
            layer_null_std_cen = np.std(control_r2_changes_cen)
            layer_null_ci_cen = np.percentile(control_r2_changes_cen, [2.5, 97.5])

            # Rank-based p-value (centered)
            primary_abs_cen = np.abs(primary_r2_change_cen)
            n_extreme_cen = np.sum(np.abs(control_r2_changes_cen) >= primary_abs_cen)
            p_value_rank_cen = (n_extreme_cen + 1) / (len(control_r2_changes_cen) + 1)
            z_score_cen = (primary_r2_change_cen - layer_null_mean_cen) / layer_null_std_cen if layer_null_std_cen > 0 else 0.0

            # Bootstrap CI over questions (sampling uncertainty) - centered
            boot_ci_low_cen, boot_ci_high_cen = None, None
            if "y_test" in lr and "baseline_preds_centered" in lr and "primary_preds_centered" in lr:
                y_test = np.array(lr["y_test"])
                base_preds_cen = np.array(lr["baseline_preds_centered"])
                primary_preds_cen = np.array(lr["primary_preds_centered"])
                boot_ci_low_cen, boot_ci_high_cen, _ = bootstrap_metric_change_ci(
                    y_test, base_preds_cen, primary_preds_cen, r2_score, n_bootstrap=1000, seed=42 + layer_idx
                )

            raw_p_values_centered.append((layer_idx, p_value_rank_cen))
            analysis["effects_centered"][layer_idx] = {
                "baseline_r2": lr["baseline_r2_centered"],
                "ablated_r2": lr["primary_ablated_r2_centered"],
                "r2_change": primary_r2_change_cen,
                # Control distribution statistics
                "control_r2_mean": float(np.mean(lr["control_r2s_centered"])),
                "control_change_mean": float(layer_null_mean_cen),
                "control_change_std": float(layer_null_std_cen),
                "control_change_ci_95": (float(layer_null_ci_cen[0]), float(layer_null_ci_cen[1])),
                # Uniqueness
                "uniqueness": float(primary_r2_change_cen - layer_null_mean_cen),
                # Statistical tests
                "p_value_rank": p_value_rank_cen,
                "z_score": z_score_cen,
                "null_ci_95": (float(layer_null_ci_cen[0]), float(layer_null_ci_cen[1])),
                # Bootstrap CI (sampling uncertainty over questions)
                "bootstrap_ci_95": (boot_ci_low_cen, boot_ci_high_cen) if boot_ci_low_cen is not None else None,
            }

        # FDR correction for same-layer FIXED
        sorted_pvals_fixed = sorted(raw_p_values_fixed, key=lambda x: x[1])
        n_tests = len(sorted_pvals_fixed)
        fdr_adjusted_fixed = {}
        for rank, (layer_idx, p_val) in enumerate(sorted_pvals_fixed, 1):
            fdr_adjusted_fixed[layer_idx] = min(1.0, p_val * n_tests / rank)
        min_q = 1.0
        for key in reversed([k for k, _ in sorted_pvals_fixed]):
            fdr_adjusted_fixed[key] = min(min_q, fdr_adjusted_fixed[key])
            min_q = fdr_adjusted_fixed[key]
        for layer_idx in layers:
            analysis["effects"][layer_idx]["p_value_fdr"] = fdr_adjusted_fixed[layer_idx]

        # FDR correction for same-layer CENTERED
        sorted_pvals_centered = sorted(raw_p_values_centered, key=lambda x: x[1])
        fdr_adjusted_centered = {}
        for rank, (layer_idx, p_val) in enumerate(sorted_pvals_centered, 1):
            fdr_adjusted_centered[layer_idx] = min(1.0, p_val * n_tests / rank)
        min_q = 1.0
        for key in reversed([k for k, _ in sorted_pvals_centered]):
            fdr_adjusted_centered[key] = min(min_q, fdr_adjusted_centered[key])
            min_q = fdr_adjusted_centered[key]
        for layer_idx in layers:
            analysis["effects_centered"][layer_idx]["p_value_fdr"] = fdr_adjusted_centered[layer_idx]

    # === DOWNSTREAM ANALYSIS ===
    # Per-layer analysis (not pooled across layers)
    for downstream_type in ["n_plus_1", "final"]:
        effects_key = f"effects_downstream_{downstream_type}"

        # Find layers with this downstream type
        layers_with_downstream = []
        for layer_idx in layers:
            downstream = results_map[layer_idx].get("downstream", {})
            if downstream_type in downstream:
                layers_with_downstream.append(layer_idx)

        if not layers_with_downstream:
            continue

        raw_p_values_ds = []
        for layer_idx in layers_with_downstream:
            ds = results_map[layer_idx]["downstream"][downstream_type]
            ctrl_key = "control_acc_changes" if probe_type == "mc_classification" else "control_r2_changes"
            control_changes = np.array(ds[ctrl_key])

            # Per-layer null distribution
            layer_null_mean = np.mean(control_changes)
            layer_null_std = np.std(control_changes)
            layer_null_ci = np.percentile(control_changes, [2.5, 97.5])

            if probe_type == "mc_classification":
                primary_change = ds["primary_acc_change"]
                primary_abs = np.abs(primary_change)
                n_extreme = np.sum(np.abs(control_changes) >= primary_abs)
                p_value_rank = (n_extreme + 1) / (len(control_changes) + 1)
                z_score = (primary_change - layer_null_mean) / layer_null_std if layer_null_std > 0 else 0.0

                # Bootstrap CI over questions (sampling uncertainty)
                boot_ci_low, boot_ci_high = None, None
                if "baseline_correct" in ds and "primary_correct" in ds:
                    baseline_correct = np.array(ds["baseline_correct"])
                    primary_correct = np.array(ds["primary_correct"])
                    boot_ci_low, boot_ci_high, _ = bootstrap_accuracy_change_ci(
                        baseline_correct, primary_correct, n_bootstrap=1000, seed=42 + layer_idx
                    )

                raw_p_values_ds.append((layer_idx, p_value_rank))
                analysis[effects_key][layer_idx] = {
                    "measure_layer": ds["measure_layer"],
                    "baseline_acc": ds["baseline_acc"],
                    "ablated_acc": ds["primary_ablated_acc"],
                    "acc_change": primary_change,
                    # Control distribution statistics
                    "control_acc_mean": float(np.mean(ds["control_accs"])),
                    "control_change_mean": float(layer_null_mean),
                    "control_change_std": float(layer_null_std),
                    "control_change_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                    # Uniqueness
                    "uniqueness": float(primary_change - layer_null_mean),
                    # Statistical tests
                    "p_value_rank": p_value_rank,
                    "z_score": z_score,
                    "null_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                    # Bootstrap CI (sampling uncertainty over questions)
                    "bootstrap_ci_95": (boot_ci_low, boot_ci_high) if boot_ci_low is not None else None,
                }
            else:
                primary_change = ds["primary_r2_change"]
                primary_abs = np.abs(primary_change)
                n_extreme = np.sum(np.abs(control_changes) >= primary_abs)
                p_value_rank = (n_extreme + 1) / (len(control_changes) + 1)
                z_score = (primary_change - layer_null_mean) / layer_null_std if layer_null_std > 0 else 0.0

                # Bootstrap CI over questions (sampling uncertainty)
                boot_ci_low, boot_ci_high = None, None
                if "y_test" in ds and "baseline_preds" in ds and "primary_preds" in ds:
                    y_test_ds = np.array(ds["y_test"])
                    base_preds = np.array(ds["baseline_preds"])
                    primary_preds = np.array(ds["primary_preds"])
                    boot_ci_low, boot_ci_high, _ = bootstrap_metric_change_ci(
                        y_test_ds, base_preds, primary_preds, r2_score, n_bootstrap=1000, seed=42 + layer_idx
                    )

                raw_p_values_ds.append((layer_idx, p_value_rank))
                analysis[effects_key][layer_idx] = {
                    "measure_layer": ds["measure_layer"],
                    "baseline_r2": ds["baseline_r2"],
                    "ablated_r2": ds["primary_ablated_r2"],
                    "r2_change": primary_change,
                    # Control distribution statistics
                    "control_r2_mean": float(np.mean(ds["control_r2s"])),
                    "control_change_mean": float(layer_null_mean),
                    "control_change_std": float(layer_null_std),
                    "control_change_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                    # Uniqueness
                    "uniqueness": float(primary_change - layer_null_mean),
                    # Statistical tests
                    "p_value_rank": p_value_rank,
                    "z_score": z_score,
                    "null_ci_95": (float(layer_null_ci[0]), float(layer_null_ci[1])),
                    # Bootstrap CI (sampling uncertainty over questions)
                    "bootstrap_ci_95": (boot_ci_low, boot_ci_high) if boot_ci_low is not None else None,
                }

        # FDR correction for this downstream type
        if raw_p_values_ds:
            sorted_pvals_ds = sorted(raw_p_values_ds, key=lambda x: x[1])
            n_tests_ds = len(sorted_pvals_ds)
            fdr_adjusted_ds = {}
            for rank, (layer_idx, p_val) in enumerate(sorted_pvals_ds, 1):
                fdr_adjusted_ds[layer_idx] = min(1.0, p_val * n_tests_ds / rank)
            min_q = 1.0
            for key in reversed([k for k, _ in sorted_pvals_ds]):
                fdr_adjusted_ds[key] = min(min_q, fdr_adjusted_ds[key])
                min_q = fdr_adjusted_ds[key]
            for layer_idx in layers_with_downstream:
                analysis[effects_key][layer_idx]["p_value_fdr"] = fdr_adjusted_ds[layer_idx]

    # Summary
    significant_layers_fixed = [l for l in layers if analysis["effects"][l]["p_value_fdr"] < 0.05]
    significant_layers_n_plus_1 = [l for l in analysis["effects_downstream_n_plus_1"]
                                   if analysis["effects_downstream_n_plus_1"][l].get("p_value_fdr", 1.0) < 0.05]
    significant_layers_final = [l for l in analysis["effects_downstream_final"]
                                if analysis["effects_downstream_final"][l].get("p_value_fdr", 1.0) < 0.05]

    analysis["summary"] = {
        "significant_layers_fdr": significant_layers_fixed,
        "significant_layers_fdr_n_plus_1": significant_layers_n_plus_1,
        "significant_layers_fdr_final": significant_layers_final,
        "n_significant": len(significant_layers_fixed),
        "n_significant_n_plus_1": len(significant_layers_n_plus_1),
        "n_significant_final": len(significant_layers_final),
    }

    # Add centered summary only for regression mode
    if probe_type != "mc_classification":
        significant_layers_centered = [l for l in layers if analysis["effects_centered"][l]["p_value_fdr"] < 0.05]
        analysis["summary"]["significant_layers_fdr_centered"] = significant_layers_centered
        analysis["summary"]["n_significant_centered"] = len(significant_layers_centered)

    return analysis


def analyze_behavioral_ablation_results(layer_results: Dict, layers: List[int], num_questions: int, num_controls: int, metric: str) -> Dict:
    """Analyze behavioral ablation results with statistical testing.

    Args:
        layer_results: Raw results from ablation experiment
        layers: List of layer indices
        num_questions: Number of questions used
        num_controls: Number of control directions per layer
        metric: The metric being used (affects p-value direction)
    """
    analysis = {
        "layers": layers,
        "num_questions": num_questions,
        "num_controls": num_controls,
        "metric": metric,
        "effects": {},
    }

    all_control_corr_changes = []
    layer_data = {}

    for layer_idx in layers:
        lr = layer_results[layer_idx]
        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_metric = np.array([r["metric"] for r in lr["baseline"]])
        ablated_conf = np.array([r["confidence"] for r in lr["primary_ablated"]])
        ablated_metric = np.array([r["metric"] for r in lr["primary_ablated"]])

        baseline_corr = compute_correlation(baseline_conf, baseline_metric)
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)

        control_corrs = []
        for ctrl_key in lr["controls_ablated"]:
            ctrl_conf = np.array([r["confidence"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_metric = np.array([r["metric"] for r in lr["controls_ablated"][ctrl_key]])
            control_corrs.append(compute_correlation(ctrl_conf, ctrl_metric))

        primary_corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]
        all_control_corr_changes.extend(control_corr_changes)

        # Bootstrap CI over questions (sampling uncertainty)
        boot_ci_low, boot_ci_high, _ = bootstrap_correlation_change_ci(
            baseline_conf, baseline_metric, ablated_conf, ablated_metric,
            n_bootstrap=1000, seed=42 + layer_idx
        )

        layer_data[layer_idx] = {
            "baseline_corr": baseline_corr,
            "ablated_corr": ablated_corr,
            "primary_corr_change": primary_corr_change,
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
            "bootstrap_ci_95": (boot_ci_low, boot_ci_high),
        }

    pooled_null = np.array(all_control_corr_changes)
    raw_p_values = []

    # Two-tailed test: We're testing whether MC direction ablation has ANY significant effect
    # compared to random control directions, regardless of direction.
    # This is appropriate because we don't know a priori whether ablating the MC answer
    # representation will make introspection better or worse.

    # Compute control-based CI from pooled null distribution
    # This answers: "What range of effects do random directions produce?"
    null_ci_low, null_ci_high = np.percentile(pooled_null, [2.5, 97.5])

    # Compute pooled stats for z-scores (to match p-value computation)
    pooled_mean = np.mean(pooled_null)
    pooled_std = np.std(pooled_null)

    for layer_idx in layers:
        ld = layer_data[layer_idx]
        # Two-tailed: count controls with |change| >= |primary change|
        primary_abs = np.abs(ld["primary_corr_change"])
        n_extreme = np.sum(np.abs(pooled_null) >= primary_abs)
        p_value_pooled = (n_extreme + 1) / (len(pooled_null) + 1)

        # Z-score using POOLED null (consistent with p-value)
        z_score = (ld["primary_corr_change"] - pooled_mean) / pooled_std if pooled_std > 0 else 0.0

        raw_p_values.append((layer_idx, p_value_pooled))
        analysis["effects"][layer_idx] = {
            "baseline_correlation": ld["baseline_corr"],
            "ablated_correlation": ld["ablated_corr"],
            "correlation_change": ld["primary_corr_change"],
            "control_correlation_mean": float(np.mean(ld["control_corrs"])),
            "control_change_mean": float(np.mean(ld["control_corr_changes"])),
            "control_change_std": float(np.std(ld["control_corr_changes"])),
            "p_value_pooled": p_value_pooled,
            "z_score": z_score,
            # CI from control distribution - "what range do random directions produce?"
            "null_ci_95": (float(null_ci_low), float(null_ci_high)),
            # Bootstrap CI - sampling uncertainty for the primary effect
            "bootstrap_ci_95": ld["bootstrap_ci_95"],
        }

    # FDR correction
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}
    for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer_idx] = adjusted

    sorted_keys = [k for k, _ in sorted_pvals]
    min_q = 1.0
    for key in reversed(sorted_keys):
        fdr_adjusted[key] = min(min_q, fdr_adjusted[key])
        min_q = fdr_adjusted[key]

    for layer_idx in layers:
        analysis["effects"][layer_idx]["p_value_fdr"] = fdr_adjusted[layer_idx]

    significant_layers = [l for l in layers if analysis["effects"][l]["p_value_fdr"] < 0.05]
    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_fdr": significant_layers,
        "n_significant": len(significant_layers),
    }

    return analysis


def analyze_d2m_full_matrix(
    results_map: Dict,
    ablation_layers: List[int],
    measurement_layers: List[int],
    num_test: int,
    num_controls: int,
) -> Dict:
    """
    Analyze D2M ablation results in full matrix mode.

    Creates an ablation×measurement matrix showing how ablation at each layer
    affects D2M at all downstream layers.

    Handles both probe types:
    - entropy_regression: Uses R² metric
    - mc_classification: Uses accuracy metric

    Args:
        results_map: {ablation_layer: {"downstream": {f"layer_{m}": {...}, ...}}}
        ablation_layers: List of layers where ablation was performed
        measurement_layers: List of all possible measurement layers
        num_test: Number of test samples
        num_controls: Number of control directions
    """
    # Detect probe type from first result
    probe_type = results_map.get(ablation_layers[0], {}).get("probe_type", "entropy_regression")

    # Set key names based on probe type
    if probe_type == "mc_classification":
        change_key = "primary_acc_change"
        baseline_key = "baseline_acc"
        ablated_key = "primary_ablated_acc"
        control_key = "control_acc_changes"
        metric_name = "acc"
    else:
        change_key = "primary_r2_change"
        baseline_key = "baseline_r2"
        ablated_key = "primary_ablated_r2"
        control_key = "control_r2_changes"
        metric_name = "r2"

    analysis = {
        "ablation_layers": ablation_layers,
        "measurement_layers": measurement_layers,
        "num_test": num_test,
        "num_controls": num_controls,
        "probe_type": probe_type,
        "matrix": {},  # {ablation_layer: {measure_layer: {...}}}
    }

    # Pool ALL control changes across the entire matrix for a single pooled null
    all_control_changes = []
    for abl_layer in ablation_layers:
        downstream = results_map.get(abl_layer, {}).get("downstream", {})
        for key, data in downstream.items():
            if key.startswith("layer_"):
                all_control_changes.extend(data[control_key])

    if not all_control_changes:
        print("WARNING: No downstream data found for full matrix analysis")
        return analysis

    pooled_null = np.array(all_control_changes)
    null_ci = np.percentile(pooled_null, [2.5, 97.5])
    pooled_mean = np.mean(pooled_null)
    pooled_std = np.std(pooled_null)

    analysis["pooled_null_stats"] = {
        "size": len(pooled_null),
        "mean": float(pooled_mean),
        "std": float(pooled_std),
        "ci_95": (float(null_ci[0]), float(null_ci[1])),
    }

    # Build the matrix with statistics
    raw_p_values = []  # For FDR correction

    for abl_layer in ablation_layers:
        analysis["matrix"][abl_layer] = {}
        downstream = results_map.get(abl_layer, {}).get("downstream", {})

        for m_layer in measurement_layers:
            if m_layer < abl_layer:
                continue  # Can't measure upstream of ablation

            key = f"layer_{m_layer}"
            if key in downstream:
                data = downstream[key]
                primary_change = data[change_key]
                primary_abs = np.abs(primary_change)
                n_extreme = np.sum(np.abs(pooled_null) >= primary_abs)
                p_value = (n_extreme + 1) / (len(pooled_null) + 1)
                z_score = (primary_change - pooled_mean) / pooled_std if pooled_std > 0 else 0.0

                raw_p_values.append(((abl_layer, m_layer), p_value))
                analysis["matrix"][abl_layer][m_layer] = {
                    f"baseline_{metric_name}": data[baseline_key],
                    f"ablated_{metric_name}": data[ablated_key],
                    f"{metric_name}_change": primary_change,
                    "control_change_mean": float(np.mean(data[control_key])),
                    "control_change_std": float(np.std(data[control_key])),
                    "p_value_pooled": p_value,
                    "z_score": z_score,
                }

    # FDR correction across entire matrix
    if raw_p_values:
        sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
        n_tests = len(sorted_pvals)
        fdr_adjusted = {}
        for rank, (key, p_val) in enumerate(sorted_pvals, 1):
            fdr_adjusted[key] = min(1.0, p_val * n_tests / rank)
        # Ensure monotonicity
        min_q = 1.0
        for key, _ in reversed(sorted_pvals):
            fdr_adjusted[key] = min(min_q, fdr_adjusted[key])
            min_q = fdr_adjusted[key]
        # Assign back
        for (abl_layer, m_layer), _ in raw_p_values:
            analysis["matrix"][abl_layer][m_layer]["p_value_fdr"] = fdr_adjusted[(abl_layer, m_layer)]

    # Summary statistics
    significant_cells = []
    for abl_layer in ablation_layers:
        for m_layer in analysis["matrix"].get(abl_layer, {}):
            cell = analysis["matrix"][abl_layer][m_layer]
            if cell.get("p_value_fdr", 1.0) < 0.05:
                significant_cells.append((abl_layer, m_layer, cell[f"{metric_name}_change"], cell["z_score"]))

    analysis["summary"] = {
        "total_cells": len(raw_p_values),
        "significant_cells_fdr": significant_cells,
        "n_significant": len(significant_cells),
    }

    return analysis


def plot_d2m_heatmap(
    matrix_analysis: Dict,
    behavioral_analysis: Optional[Dict],
    output_prefix: str,
    simultaneous_result: Optional[Dict] = None,
):
    """
    Plot full ablation×measurement matrix as a heatmap.

    Shows:
    1. R² change heatmap (ablation layer × measurement layer)
    2. Z-score heatmap
    3. Behavioral correlation change (simultaneous all-layer if available, else per-layer)
    """
    # Sort layers by index for interpretable axes
    ablation_layers = sorted(matrix_analysis["ablation_layers"])
    measurement_layers = sorted(matrix_analysis["measurement_layers"])

    # Detect probe type and set key name
    probe_type = matrix_analysis.get("probe_type", "entropy_regression")
    change_key = "acc_change" if probe_type == "mc_classification" else "r2_change"
    metric_label = "Acc" if probe_type == "mc_classification" else "R²"

    # Build matrices
    n_abl = len(ablation_layers)
    n_meas = len(measurement_layers)

    r2_change_matrix = np.full((n_abl, n_meas), np.nan)
    z_score_matrix = np.full((n_abl, n_meas), np.nan)

    for i, abl_layer in enumerate(ablation_layers):
        for j, m_layer in enumerate(measurement_layers):
            if m_layer >= abl_layer and abl_layer in matrix_analysis["matrix"]:
                cell = matrix_analysis["matrix"][abl_layer].get(m_layer, {})
                if cell:
                    r2_change_matrix[i, j] = cell[change_key]
                    z_score_matrix[i, j] = cell["z_score"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: Metric change heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(r2_change_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-0.1, vmax=0.1)
    ax1.set_xlabel('Measurement Layer')
    ax1.set_ylabel('Ablation Layer')
    ax1.set_title(f'D2M {metric_label} Change (Ablation - Baseline)')
    ax1.set_xticks(np.arange(n_meas))
    ax1.set_yticks(np.arange(n_abl))
    ax1.set_xticklabels(measurement_layers, fontsize=8, rotation=45)
    ax1.set_yticklabels(ablation_layers, fontsize=8)
    plt.colorbar(im1, ax=ax1, label=f'{metric_label} Change')

    # Panel 2: Z-score heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(z_score_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-3, vmax=3)
    ax2.set_xlabel('Measurement Layer')
    ax2.set_ylabel('Ablation Layer')
    ax2.set_title('D2M Z-Score (vs Per-Cell Null)')
    ax2.set_xticks(np.arange(n_meas))
    ax2.set_yticks(np.arange(n_abl))
    ax2.set_xticklabels(measurement_layers, fontsize=8, rotation=45)
    ax2.set_yticklabels(ablation_layers, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='Z-Score')

    # Panel 3: Behavioral correlation change
    ax3 = axes[1, 0]
    if simultaneous_result is not None:
        # Full-matrix mode: show single all-layer ablation result
        beh_change = simultaneous_result["primary_correlation_change"]
        null_ci = simultaneous_result["null_ci_95"]
        ax3.axhspan(null_ci[0], null_ci[1], alpha=0.2, color='gray', label='Null 95% CI')
        ax3.bar([0], [beh_change], color='tab:red', alpha=0.7, width=0.5)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('')
        ax3.set_ylabel('Correlation Change')
        ax3.set_title(f'Behavioral: All-Layer Ablation\nΔr = {beh_change:+.4f}, p = {simultaneous_result["p_value"]:.4f}')
        ax3.set_xticks([0])
        ax3.set_xticklabels(['All Layers'])
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
    elif behavioral_analysis is not None:
        # Per-layer mode (legacy)
        beh_changes = [behavioral_analysis["effects"][l]["correlation_change"] for l in ablation_layers]
        null_ci = behavioral_analysis["effects"][ablation_layers[0]]["null_ci_95"]
        x = np.arange(len(ablation_layers))
        ax3.axhspan(null_ci[0], null_ci[1], alpha=0.2, color='gray', label='Null 95% CI')
        ax3.bar(x, beh_changes, color='tab:red', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Ablation Layer')
        ax3.set_ylabel('Correlation Change')
        ax3.set_title('Behavioral: Conf-Entropy Corr Change per Ablation Layer')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ablation_layers, fontsize=8, rotation=45)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No behavioral data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Behavioral: N/A')

    # Panel 4: Aggregate effect by measurement layer (mean across ablation layers)
    ax4 = axes[1, 1]
    mean_effect_by_meas = []
    stderr_effect_by_meas = []
    for j, m_layer in enumerate(measurement_layers):
        effects = []
        for i, abl_layer in enumerate(ablation_layers):
            if not np.isnan(r2_change_matrix[i, j]):
                effects.append(r2_change_matrix[i, j])
        if effects:
            mean_effect_by_meas.append(np.mean(effects))
            stderr_effect_by_meas.append(np.std(effects) / np.sqrt(len(effects)) if len(effects) > 1 else 0)
        else:
            mean_effect_by_meas.append(np.nan)
            stderr_effect_by_meas.append(0)

    x_meas = np.arange(len(measurement_layers))
    ax4.bar(x_meas, mean_effect_by_meas, yerr=stderr_effect_by_meas,
            color='tab:blue', alpha=0.7, capsize=2)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add null CI from pooled stats
    if "pooled_null_stats" in matrix_analysis:
        ci = matrix_analysis["pooled_null_stats"]["ci_95"]
        ax4.axhspan(ci[0], ci[1], alpha=0.2, color='gray', label='Null 95% CI')

    ax4.set_xlabel('Measurement Layer')
    ax4.set_ylabel('Mean R² Change')
    ax4.set_title('Mean D2M R² Change by Measurement Layer\n(averaged across ablation layers)')
    ax4.set_xticks(x_meas)
    ax4.set_xticklabels(measurement_layers, fontsize=8, rotation=45)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_mc_answer_ablation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_prefix}_mc_answer_ablation_matrix.png")


def plot_results(
    behavioral_analysis: Dict,
    d2m_analysis: Optional[Dict],
    direction_similarity: Dict[int, float],
    output_prefix: str
):
    """Generate plots for the experiment results."""

    # Sort layers by index for interpretable x-axis
    layers = sorted(behavioral_analysis["layers"])

    # Detect probe type from d2m_analysis if available
    if d2m_analysis is not None:
        probe_type = d2m_analysis.get("probe_type", "entropy_regression")
        change_key = "acc_change" if probe_type == "mc_classification" else "r2_change"
        baseline_key = "baseline_acc" if probe_type == "mc_classification" else "baseline_r2"
        metric_label = "Acc" if probe_type == "mc_classification" else "R²"
    else:
        change_key = "r2_change"
        baseline_key = "baseline_r2"
        metric_label = "R²"

    # Determine layout: 6 panels with D2M (3x2), 2 panels without
    if d2m_analysis is not None:
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(layers))
    width = 0.35

    # Panel 1: Behavioral correlation change
    ax1 = axes[0]
    mc_changes = [behavioral_analysis["effects"][l]["correlation_change"] for l in layers]

    # Get null CI (same for all layers since it's pooled)
    null_ci = behavioral_analysis["effects"][layers[0]]["null_ci_95"]

    ctrl_means = [behavioral_analysis["effects"][l]["control_change_mean"] for l in layers]
    ctrl_stds = [behavioral_analysis["effects"][l]["control_change_std"] for l in layers]

    # Bootstrap CIs (sampling uncertainty) for behavioral
    boot_ci_lows = [behavioral_analysis["effects"][l]["bootstrap_ci_95"][0] for l in layers]
    boot_ci_highs = [behavioral_analysis["effects"][l]["bootstrap_ci_95"][1] for l in layers]

    # Draw shaded region for null distribution 95% CI
    ax1.axhspan(null_ci[0], null_ci[1], alpha=0.2, color='gray', label='Null 95% CI (pooled)')

    # MC changes as points with bootstrap CI error bars
    mc_changes_arr = np.array(mc_changes)
    boot_err_low = mc_changes_arr - np.array(boot_ci_lows)
    boot_err_high = np.array(boot_ci_highs) - mc_changes_arr
    ax1.errorbar(x, mc_changes, yerr=[boot_err_low, boot_err_high],
                 fmt='o', markersize=8, color='tab:red', capsize=4, capthick=1.5,
                 elinewidth=1.5, zorder=5, label='MC Ablation ± 95% CI')
    ax1.bar(x, ctrl_means, width, yerr=ctrl_stds, label='Control (mean±std)',
            color='tab:gray', alpha=0.4, capsize=3)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Correlation Change')
    ax1.set_title('Behavioral: Conf-Entropy Corr Change')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2 onward: D2M results
    if d2m_analysis is not None:
        # Panel 2: D2M Same-layer Fixed
        ax2 = axes[1]
        d2m_changes = [d2m_analysis["effects"][l][change_key] for l in layers]
        d2m_ctrl_means = [d2m_analysis["effects"][l]["control_change_mean"] for l in layers]

        # Per-layer control CIs (direction specificity)
        ctrl_ci_lows = [d2m_analysis["effects"][l]["control_change_ci_95"][0] for l in layers]
        ctrl_ci_highs = [d2m_analysis["effects"][l]["control_change_ci_95"][1] for l in layers]

        # Bootstrap CIs (sampling uncertainty) - if available
        boot_ci_lows = []
        boot_ci_highs = []
        has_boot_ci = False
        for l in layers:
            boot_ci = d2m_analysis["effects"][l].get("bootstrap_ci_95")
            if boot_ci and boot_ci[0] is not None:
                boot_ci_lows.append(boot_ci[0])
                boot_ci_highs.append(boot_ci[1])
                has_boot_ci = True
            else:
                boot_ci_lows.append(np.nan)
                boot_ci_highs.append(np.nan)

        # Draw per-layer control CI as error bars on control mean
        ax2.bar(x, d2m_ctrl_means, width, color='tab:gray', alpha=0.4, label='Control mean')
        ax2.errorbar(x, d2m_ctrl_means, yerr=[np.array(d2m_ctrl_means) - np.array(ctrl_ci_lows),
                                               np.array(ctrl_ci_highs) - np.array(d2m_ctrl_means)],
                     fmt='none', color='gray', capsize=3, label='Ctrl 95% CI')

        # Main effect as points with bootstrap CI error bars
        if has_boot_ci:
            ax2.errorbar(x, d2m_changes, yerr=[np.array(d2m_changes) - np.array(boot_ci_lows),
                                                np.array(boot_ci_highs) - np.array(d2m_changes)],
                         fmt='o', color='tab:orange', markersize=8, capsize=4, capthick=2,
                         label='Main effect ± Boot CI', zorder=5)
        else:
            ax2.scatter(x, d2m_changes, s=80, color='tab:orange', zorder=5, label='Main effect')

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel(f'{metric_label} Change')
        ax2.set_title('D2M Same-Layer (ablate & measure at N)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: D2M Downstream N+1
        ax3 = axes[2]
        if d2m_analysis["effects_downstream_n_plus_1"]:
            ds_n1 = d2m_analysis["effects_downstream_n_plus_1"]

            layers_n1 = [l for l in layers if l in ds_n1]
            x_n1 = np.arange(len(layers_n1))
            mc_changes_n1 = [ds_n1[l][change_key] for l in layers_n1]
            ctrl_ci_lows_n1 = [ds_n1[l]["control_change_ci_95"][0] for l in layers_n1]
            ctrl_ci_highs_n1 = [ds_n1[l]["control_change_ci_95"][1] for l in layers_n1]

            # Bootstrap CIs (sampling uncertainty) - if available
            boot_ci_lows_n1 = []
            boot_ci_highs_n1 = []
            has_boot_ci_n1 = False
            for l in layers_n1:
                boot_ci = ds_n1[l].get("bootstrap_ci_95")
                if boot_ci and boot_ci[0] is not None:
                    boot_ci_lows_n1.append(boot_ci[0])
                    boot_ci_highs_n1.append(boot_ci[1])
                    has_boot_ci_n1 = True
                else:
                    boot_ci_lows_n1.append(np.nan)
                    boot_ci_highs_n1.append(np.nan)

            # Draw control CI as shaded region
            ax3.fill_between(x_n1, ctrl_ci_lows_n1, ctrl_ci_highs_n1, alpha=0.3, color='gray', label='Ctrl 95% CI')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Main effect with bootstrap CI error bars
            if has_boot_ci_n1:
                ax3.errorbar(x_n1, mc_changes_n1, yerr=[np.array(mc_changes_n1) - np.array(boot_ci_lows_n1),
                                                        np.array(boot_ci_highs_n1) - np.array(mc_changes_n1)],
                             fmt='o', color='tab:purple', markersize=8, capsize=4, capthick=2,
                             label='Main effect ± Boot CI', zorder=5)
            else:
                ax3.scatter(x_n1, mc_changes_n1, s=80, color='tab:purple', zorder=5, label='Main effect')

            ax3.set_xlabel('Ablation Layer')
            ax3.set_ylabel(f'{metric_label} Change')
            ax3.set_title('D2M Downstream N+1 (ablate at N, measure at N+1)')
            ax3.set_xticks(x_n1)
            ax3.set_xticklabels(layers_n1)
            ax3.legend(loc='best', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No N+1 data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('D2M Downstream N+1')
        ax3.grid(True, alpha=0.3, axis='y')

        # Panel 4: D2M Downstream Final
        ax4 = axes[3]
        if d2m_analysis["effects_downstream_final"]:
            ds_final = d2m_analysis["effects_downstream_final"]
            first_layer = list(ds_final.keys())[0]
            final_layer = ds_final[first_layer]["measure_layer"]

            layers_final = [l for l in layers if l in ds_final]
            x_final = np.arange(len(layers_final))
            mc_changes_final = [ds_final[l][change_key] for l in layers_final]
            ctrl_ci_lows_final = [ds_final[l]["control_change_ci_95"][0] for l in layers_final]
            ctrl_ci_highs_final = [ds_final[l]["control_change_ci_95"][1] for l in layers_final]

            # Bootstrap CIs (sampling uncertainty) - if available
            boot_ci_lows_final = []
            boot_ci_highs_final = []
            has_boot_ci_final = False
            for l in layers_final:
                boot_ci = ds_final[l].get("bootstrap_ci_95")
                if boot_ci and boot_ci[0] is not None:
                    boot_ci_lows_final.append(boot_ci[0])
                    boot_ci_highs_final.append(boot_ci[1])
                    has_boot_ci_final = True
                else:
                    boot_ci_lows_final.append(np.nan)
                    boot_ci_highs_final.append(np.nan)

            # Draw control CI as shaded region
            ax4.fill_between(x_final, ctrl_ci_lows_final, ctrl_ci_highs_final, alpha=0.3, color='gray', label='Ctrl 95% CI')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Main effect with bootstrap CI error bars
            if has_boot_ci_final:
                ax4.errorbar(x_final, mc_changes_final, yerr=[np.array(mc_changes_final) - np.array(boot_ci_lows_final),
                                                              np.array(boot_ci_highs_final) - np.array(mc_changes_final)],
                             fmt='o', color='tab:cyan', markersize=8, capsize=4, capthick=2,
                             label='Main effect ± Boot CI', zorder=5)
            else:
                ax4.scatter(x_final, mc_changes_final, s=80, color='tab:cyan', zorder=5, label='Main effect')

            ax4.set_xlabel('Ablation Layer')
            ax4.set_ylabel(f'{metric_label} Change')
            ax4.set_title(f'D2M Downstream Final (ablate at N, measure at {final_layer})')
            ax4.set_xticks(x_final)
            ax4.set_xticklabels(layers_final)
            ax4.legend(loc='best', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No final layer data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('D2M Downstream Final')
        ax4.grid(True, alpha=0.3, axis='y')

        ax_sim = axes[4]

        # Panel 6: Effect size comparison (same-layer vs downstream)
        ax6 = axes[5]
        if d2m_analysis["effects_downstream_final"]:
            ds_final = d2m_analysis["effects_downstream_final"]
            layers_final = [l for l in layers if l in ds_final]
            x_compare = np.arange(len(layers_final))

            # Show CHANGES (effect sizes), not baseline values
            same_layer_changes = [d2m_analysis["effects"][l][change_key] for l in layers_final]
            final_changes = [ds_final[l][change_key] for l in layers_final]

            ax6.bar(x_compare - 0.2, same_layer_changes, 0.4, label=f'Same-layer Δ{metric_label}', color='tab:orange', alpha=0.7)
            ax6.bar(x_compare + 0.2, final_changes, 0.4, label=f'Final layer Δ{metric_label}', color='tab:cyan', alpha=0.7)
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax6.set_xlabel('Ablation Layer')
            ax6.set_ylabel(f'{metric_label} Change')
            ax6.set_title(f'D2M Effect: Same-Layer vs Final-Layer Measurement')
            ax6.set_xticks(x_compare)
            ax6.set_xticklabels(layers_final)
            ax6.legend(loc='best', fontsize=8)
        else:
            ax6.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('D2M Effect Comparison')
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax_sim = axes[1]

    # Panel 5 (or 2): Direction similarity
    sim_layers = sorted(direction_similarity.keys())
    similarities = [direction_similarity[l] for l in sim_layers]

    ax_sim.plot(sim_layers, similarities, 'o-', color='tab:blue', markersize=6, linewidth=2)
    ax_sim.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax_sim.set_xlabel('Layer')
    ax_sim.set_ylabel('Cosine Similarity')
    ax_sim.set_title('MC Answer vs Entropy Direction')
    ax_sim.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_mc_answer_ablation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_prefix}_mc_answer_ablation.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global METRIC

    parser = argparse.ArgumentParser(description="MC Answer Probe Causality Experiment")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Metric for entropy probe (default: {METRIC})")
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS,
                        help=f"Number of questions (default: {NUM_QUESTIONS})")
    parser.add_argument("--num-controls", type=int, default=None,
                        help="Number of control directions per layer (default: 25, use 100+ for tighter p-values)")
    parser.add_argument("--similarity-only", action="store_true",
                        help="Only compute direction similarity (skip ablation)")
    parser.add_argument("--full-matrix", action="store_true",
                        help="Full matrix mode: ablate at EVERY layer, measure D2M at ALL downstream layers")
    args = parser.parse_args()

    METRIC = args.metric
    num_questions = args.num_questions
    full_matrix_mode = args.full_matrix

    global NUM_CONTROL_DIRECTIONS
    if args.num_controls is not None:
        NUM_CONTROL_DIRECTIONS = args.num_controls

    print("=" * 70)
    print("MC ANSWER PROBE CAUSALITY EXPERIMENT")
    if full_matrix_mode:
        print("  *** FULL MATRIX MODE: Ablate at all layers, measure at all downstream layers ***")
    if REVERSE_MODE:
        print("  *** REVERSE MODE: Ablate ENTROPY direction, measure MC ANSWER D2M transfer ***")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Metric: {METRIC}")
    print(f"Device: {DEVICE}")

    output_prefix = get_output_prefix()
    directions_prefix = get_directions_prefix()

    mc_directions = load_mc_answer_directions(directions_prefix)
    entropy_directions = load_entropy_directions(directions_prefix, METRIC)
    probe_results = load_probe_results(output_prefix, METRIC)
    paired_data = load_paired_data(output_prefix)
    
    questions = paired_data["questions"]
    direct_metrics = paired_data["direct_metrics"]
    if METRIC not in direct_metrics:
        raise ValueError(f"Metric '{METRIC}' not found in paired data.")
    direct_metric_values = direct_metrics[METRIC]
    
    # Subsample - track indices for D2M alignment
    subsample_indices = None
    if len(questions) > num_questions:
        np.random.seed(SEED)
        subsample_indices = np.random.choice(len(questions), num_questions, replace=False)
        subsample_indices = np.sort(subsample_indices)  # Keep sorted for easier debugging
        questions = [questions[i] for i in subsample_indices]
        direct_metric_values = direct_metric_values[subsample_indices]

    # Select Layers
    if full_matrix_mode:
        # In full matrix mode, ablate at ALL layers with MC directions
        layers = sorted([l for l in mc_directions.keys() if l in entropy_directions])
        print(f"\n[Full Matrix] Using ALL {len(layers)} layers with MC directions: {layers[:5]}...{layers[-3:]}")
    else:
        # Default: select layers based on D2M R² threshold
        layer_candidates = []
        if "probe_results" in probe_results:
            for layer_str, lr in probe_results["probe_results"].items():
                d2m_r2 = lr.get("direct_to_meta_fixed", {}).get("r2", 0)
                d2d_r2 = lr.get("direct_to_direct", {}).get("test_r2", 0)
                if d2m_r2 >= D2M_R2_THRESHOLD and d2d_r2 >= D2D_R2_THRESHOLD:
                    layer_candidates.append((int(layer_str), d2m_r2))
        if layer_candidates:
            layer_candidates.sort(key=lambda x: -x[1])
            layers = [l[0] for l in layer_candidates]
        else:
            layers = sorted(mc_directions.keys())
        layers = [l for l in layers if l in mc_directions and l in entropy_directions]
        print(f"\nSelected {len(layers)} layers: {layers}")

    # Similarity
    direction_similarity = compute_direction_similarity(mc_directions, entropy_directions)
    if args.similarity_only:
        results = {
            "config": {"base_model": BASE_MODEL_NAME, "dataset": DATASET_NAME, "metric": METRIC},
            "direction_similarity": {str(k): v for k, v in direction_similarity.items()},
        }
        with open(f"{output_prefix}_{METRIC}_mc_answer_similarity.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # Load Activations & Test Indices
    all_measurement_layers = None  # For full matrix mode
    try:
        if full_matrix_mode:
            # In full matrix mode, load ALL activations
            direct_activations, entropy_values, available_layers = load_all_activations(directions_prefix)
            # Use all available layers as measurement layers
            all_measurement_layers = available_layers
            print(f"  [Full Matrix] Will measure D2M at {len(all_measurement_layers)} layers")
        else:
            direct_activations, entropy_values = load_activations(directions_prefix, layers)

        # If we subsampled questions, we must also subsample activations to match
        if subsample_indices is not None:
            print(f"  Subsampling activations to match {len(subsample_indices)} questions...")
            entropy_values = entropy_values[subsample_indices]
            direct_activations = {
                layer: acts[subsample_indices] for layer, acts in direct_activations.items()
            }

        # Recompute test indices for the (possibly subsampled) data
        # We create a fresh 80/20 split because original test_indices don't align after subsampling
        n_samples = len(entropy_values)
        np.random.seed(SEED + 1)  # Different seed to avoid correlation with subsample selection
        test_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
        print(f"  Using {len(test_indices)} test samples for D2M evaluation")

        run_d2m_experiment = True
    except FileNotFoundError:
        print("  Skipping D2M transfer experiment (activations not found)")
        run_d2m_experiment = False
        direct_activations, entropy_values, test_indices = None, None, None

    # Load Model
    adapter_path = MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME, adapter_path=adapter_path, load_in_4bit=LOAD_IN_4BIT
    )
    model.eval()
    initialize_token_cache(tokenizer)
    use_chat_template = should_use_chat_template(BASE_MODEL_NAME, tokenizer)

    # Determine which directions to ablate and which D2M target to use based on mode
    if REVERSE_MODE:
        ablation_directions = entropy_directions
        ablation_name = "entropy"
        probe_type = "mc_classification"
        # For reverse mode, D2M target is model's predicted answer
        direct_probs = paired_data.get("direct_probs", [])
        if subsample_indices is not None:
            direct_probs = [direct_probs[i] for i in subsample_indices]
        direct_probs_array = np.array([p if p else [0.25, 0.25, 0.25, 0.25] for p in direct_probs])
        d2m_target_values = np.argmax(direct_probs_array, axis=1)
        print("\n[REVERSE MODE] Will ablate ENTROPY directions, measure MC ANSWER D2M transfer")
        print(f"  Model predicted answers: {len(d2m_target_values)} questions")
        print(f"  Answer distribution: A={np.sum(d2m_target_values==0)}, B={np.sum(d2m_target_values==1)}, "
              f"C={np.sum(d2m_target_values==2)}, D={np.sum(d2m_target_values==3)}")
    else:
        ablation_directions = mc_directions
        ablation_name = "MC answer"
        probe_type = "entropy_regression"
        d2m_target_values = entropy_values  # For forward mode, D2M target is entropy

    # ===== SIMULTANEOUS ALL-LAYER ABLATION =====
    # Ablate direction at ALL layers at once, measure single behavioral correlation change
    simultaneous_result = None
    if full_matrix_mode:
        simultaneous_result = run_simultaneous_all_layer_ablation(
            model, tokenizer, questions, direct_metric_values, ablation_directions, use_chat_template,
            ablation_name=ablation_name
        )

    # Run Parallel Experiment
    # In full_matrix_mode: only needed for D2M matrix (skip per-layer behavioral analysis)
    # In normal mode: run both behavioral and D2M per-layer analysis
    beh_results, d2m_results, d2m_extra = None, None, None
    behavioral_analysis = None
    d2m_analysis = None

    if full_matrix_mode:
        # Full matrix mode: ONLY run D2M matrix experiment (skip behavioral per-layer)
        if run_d2m_experiment:
            print(f"\n[Full Matrix] Running D2M matrix experiment (per-layer behavioral skipped)...")
            beh_results, d2m_results, d2m_extra = run_experiment_parallel(
                model, tokenizer, questions, direct_activations, d2m_target_values, direct_metric_values,
                layers, ablation_directions, NUM_CONTROL_DIRECTIONS, use_chat_template, test_indices, run_d2m_experiment,
                num_layers, all_measurement_layers=all_measurement_layers, probe_type=probe_type
            )
    else:
        # Normal mode: run both behavioral and D2M per-layer experiments
        beh_results, d2m_results, d2m_extra = run_experiment_parallel(
            model, tokenizer, questions, direct_activations, d2m_target_values, direct_metric_values,
            layers, ablation_directions, NUM_CONTROL_DIRECTIONS, use_chat_template, test_indices, run_d2m_experiment,
            num_layers, all_measurement_layers=all_measurement_layers, probe_type=probe_type
        )

        # Analyze and print per-layer behavioral results (only in normal mode)
        behavioral_analysis = analyze_behavioral_ablation_results(beh_results, layers, len(questions), NUM_CONTROL_DIRECTIONS, METRIC)
        null_ci = behavioral_analysis["effects"][layers[0]]["null_ci_95"]

        print("\n" + "=" * 110)
        print("BEHAVIORAL ABLATION RESULTS (per-layer)")
        print(f"Null 95% CI (pooled): [{null_ci[0]:.4f}, {null_ci[1]:.4f}]")
        print("=" * 110)
        print(f"{'Layer':<6} {'Base r':<10} {'Ablated r':<12} {'Delta':<12} {'p_raw':<8} {'p_FDR':<8} {'Z-score':<10}")
        print("-" * 110)
        for l in layers:
            eff = behavioral_analysis["effects"][l]
            sig = "*" if eff["p_value_fdr"] < 0.05 else ""
            print(f"{l:<6} {eff['baseline_correlation']:+.4f}     {eff['ablated_correlation']:+.4f}       "
                  f"{eff['correlation_change']:+.4f}       {eff['p_value_pooled']:.4f}  {eff['p_value_fdr']:.4f}  {eff['z_score']:+.2f} {sig}")

        # Analyze and print per-layer D2M results (only in normal mode)
        if run_d2m_experiment:
            d2m_analysis = analyze_d2m_transfer_ablation_results(
                d2m_results, layers, len(test_indices), NUM_CONTROL_DIRECTIONS,
            )

            if probe_type == "mc_classification":
                # Classification mode: print accuracy results
                print("\n" + "=" * 180)
                print(f"D2M TRANSFER ABLATION RESULTS - MC ANSWER ACCURACY (ablating {ablation_name})")
                print(f"Per-layer null distribution from {NUM_CONTROL_DIRECTIONS} control directions")
                print("Ctrl CI = control distribution 95% CI (direction specificity), Boot CI = bootstrap 95% CI (sampling uncertainty)")
                print("=" * 180)
                print(f"{'Layer':<6} {'Base':<8} {'Ablated':<8} {'Delta':<8} {'Ctrl Mean':<10} {'Ctrl CI':<16} {'Boot CI':<16} {'Unique':<8} {'p_rank':<8} {'p_FDR':<8} {'z':<6}")
                print("-" * 180)
                for l in layers:
                    eff = d2m_analysis["effects"][l]
                    sig = "*" if eff["p_value_fdr"] < 0.05 else ""
                    ctrl_ci = eff["control_change_ci_95"]
                    boot_ci = eff.get("bootstrap_ci_95")
                    boot_ci_str = f"[{boot_ci[0]:+.3f},{boot_ci[1]:+.3f}]" if boot_ci else "N/A"
                    print(f"{l:<6} {eff['baseline_acc']:.4f}   {eff['ablated_acc']:.4f}   "
                          f"{eff['acc_change']:+.4f}   {eff['control_change_mean']:+.4f}    "
                          f"[{ctrl_ci[0]:+.3f},{ctrl_ci[1]:+.3f}]  {boot_ci_str:<16} {eff['uniqueness']:+.4f}   "
                          f"{eff['p_value_rank']:.4f}   {eff['p_value_fdr']:.4f}   {eff['z_score']:+.1f} {sig}")
            else:
                # Regression mode: print R² results (both fixed and centered)
                print("\n" + "=" * 180)
                print(f"D2M TRANSFER ABLATION RESULTS - FIXED (ablating {ablation_name})")
                print(f"Per-layer null distribution from {NUM_CONTROL_DIRECTIONS} control directions")
                print("Ctrl CI = control distribution 95% CI (direction specificity), Boot CI = bootstrap 95% CI (sampling uncertainty)")
                print("=" * 180)
                print(f"{'Layer':<6} {'Base':<8} {'Ablated':<8} {'Delta':<8} {'Ctrl Mean':<10} {'Ctrl CI':<16} {'Boot CI':<16} {'Unique':<8} {'p_rank':<8} {'p_FDR':<8} {'z':<6}")
                print("-" * 180)
                for l in layers:
                    eff = d2m_analysis["effects"][l]
                    sig = "*" if eff["p_value_fdr"] < 0.05 else ""
                    ctrl_ci = eff["control_change_ci_95"]
                    boot_ci = eff.get("bootstrap_ci_95")
                    boot_ci_str = f"[{boot_ci[0]:+.3f},{boot_ci[1]:+.3f}]" if boot_ci else "N/A"
                    print(f"{l:<6} {eff['baseline_r2']:.4f}   {eff['ablated_r2']:.4f}   "
                          f"{eff['r2_change']:+.4f}   {eff['control_change_mean']:+.4f}    "
                          f"[{ctrl_ci[0]:+.3f},{ctrl_ci[1]:+.3f}]  {boot_ci_str:<16} {eff['uniqueness']:+.4f}   "
                          f"{eff['p_value_rank']:.4f}   {eff['p_value_fdr']:.4f}   {eff['z_score']:+.1f} {sig}")

                print("\n" + "=" * 180)
                print(f"D2M TRANSFER ABLATION RESULTS - CENTERED (ablating {ablation_name})")
                print(f"Per-layer null distribution from {NUM_CONTROL_DIRECTIONS} control directions")
                print("Ctrl CI = control distribution 95% CI (direction specificity), Boot CI = bootstrap 95% CI (sampling uncertainty)")
                print("=" * 180)
                print(f"{'Layer':<6} {'Base':<8} {'Ablated':<8} {'Delta':<8} {'Ctrl Mean':<10} {'Ctrl CI':<16} {'Boot CI':<16} {'Unique':<8} {'p_rank':<8} {'p_FDR':<8} {'z':<6}")
                print("-" * 180)
                for l in layers:
                    eff = d2m_analysis["effects_centered"][l]
                    sig = "*" if eff["p_value_fdr"] < 0.05 else ""
                    ctrl_ci = eff["control_change_ci_95"]
                    boot_ci = eff.get("bootstrap_ci_95")
                    boot_ci_str = f"[{boot_ci[0]:+.3f},{boot_ci[1]:+.3f}]" if boot_ci else "N/A"
                    print(f"{l:<6} {eff['baseline_r2']:.4f}   {eff['ablated_r2']:.4f}   "
                          f"{eff['r2_change']:+.4f}   {eff['control_change_mean']:+.4f}    "
                          f"[{ctrl_ci[0]:+.3f},{ctrl_ci[1]:+.3f}]  {boot_ci_str:<16} {eff['uniqueness']:+.4f}   "
                          f"{eff['p_value_rank']:.4f}   {eff['p_value_fdr']:.4f}   {eff['z_score']:+.1f} {sig}")

            # Print downstream results (N+1) - handle both probe types
            if d2m_analysis["effects_downstream_n_plus_1"]:
                print("\n" + "=" * 170)
                print(f"D2M DOWNSTREAM ABLATION RESULTS - N+1 (ablate {ablation_name} at N, measure at N+1)")
                print(f"Per-layer null distribution from {NUM_CONTROL_DIRECTIONS} control directions")
                print("=" * 170)
                if probe_type == "mc_classification":
                    print(f"{'Ablate':<7} {'Meas':<6} {'Base':<8} {'Ablated':<8} {'Delta':<8} {'Ctrl Mean':<10} {'Ctrl CI':<16} {'Unique':<8} {'p_rank':<8} {'p_FDR':<8} {'z':<6}")
                    print("-" * 170)
                    for l in layers:
                        if l in d2m_analysis["effects_downstream_n_plus_1"]:
                            eff = d2m_analysis["effects_downstream_n_plus_1"][l]
                            sig = "*" if eff.get("p_value_fdr", 1.0) < 0.05 else ""
                            ctrl_ci = eff["control_change_ci_95"]
                            print(f"{l:<7} {eff['measure_layer']:<6} {eff['baseline_acc']:.4f}   {eff['ablated_acc']:.4f}   "
                                  f"{eff['acc_change']:+.4f}   {eff['control_change_mean']:+.4f}    "
                                  f"[{ctrl_ci[0]:+.3f},{ctrl_ci[1]:+.3f}]  {eff['uniqueness']:+.4f}   "
                                  f"{eff['p_value_rank']:.4f}   {eff.get('p_value_fdr', 1.0):.4f}   {eff['z_score']:+.1f} {sig}")
                else:
                    print(f"{'Ablate':<7} {'Meas':<6} {'Base':<8} {'Ablated':<8} {'Delta':<8} {'Ctrl Mean':<10} {'Ctrl CI':<16} {'Unique':<8} {'p_rank':<8} {'p_FDR':<8} {'z':<6}")
                    print("-" * 170)
                    for l in layers:
                        if l in d2m_analysis["effects_downstream_n_plus_1"]:
                            eff = d2m_analysis["effects_downstream_n_plus_1"][l]
                            sig = "*" if eff.get("p_value_fdr", 1.0) < 0.05 else ""
                            ctrl_ci = eff["control_change_ci_95"]
                            print(f"{l:<7} {eff['measure_layer']:<6} {eff['baseline_r2']:.4f}   {eff['ablated_r2']:.4f}   "
                                  f"{eff['r2_change']:+.4f}   {eff['control_change_mean']:+.4f}    "
                                  f"[{ctrl_ci[0]:+.3f},{ctrl_ci[1]:+.3f}]  {eff['uniqueness']:+.4f}   "
                                  f"{eff['p_value_rank']:.4f}   {eff.get('p_value_fdr', 1.0):.4f}   {eff['z_score']:+.1f} {sig}")

            # Print downstream results (Final layer) - handle both probe types
            if d2m_analysis["effects_downstream_final"]:
                print("\n" + "=" * 170)
                print(f"D2M DOWNSTREAM ABLATION RESULTS - FINAL LAYER (ablate {ablation_name} at N, measure at final)")
                print(f"Per-layer null distribution from {NUM_CONTROL_DIRECTIONS} control directions")
                print("=" * 170)
                if probe_type == "mc_classification":
                    print(f"{'Ablate':<7} {'Meas':<6} {'Base':<8} {'Ablated':<8} {'Delta':<8} {'Ctrl Mean':<10} {'Ctrl CI':<16} {'Unique':<8} {'p_rank':<8} {'p_FDR':<8} {'z':<6}")
                    print("-" * 170)
                    for l in layers:
                        if l in d2m_analysis["effects_downstream_final"]:
                            eff = d2m_analysis["effects_downstream_final"][l]
                            sig = "*" if eff.get("p_value_fdr", 1.0) < 0.05 else ""
                            ctrl_ci = eff["control_change_ci_95"]
                            print(f"{l:<7} {eff['measure_layer']:<6} {eff['baseline_acc']:.4f}   {eff['ablated_acc']:.4f}   "
                                  f"{eff['acc_change']:+.4f}   {eff['control_change_mean']:+.4f}    "
                                  f"[{ctrl_ci[0]:+.3f},{ctrl_ci[1]:+.3f}]  {eff['uniqueness']:+.4f}   "
                                  f"{eff['p_value_rank']:.4f}   {eff.get('p_value_fdr', 1.0):.4f}   {eff['z_score']:+.1f} {sig}")
                else:
                    print(f"{'Ablate':<7} {'Meas':<6} {'Base':<8} {'Ablated':<8} {'Delta':<8} {'Ctrl Mean':<10} {'Ctrl CI':<16} {'Unique':<8} {'p_rank':<8} {'p_FDR':<8} {'z':<6}")
                    print("-" * 170)
                    for l in layers:
                        if l in d2m_analysis["effects_downstream_final"]:
                            eff = d2m_analysis["effects_downstream_final"][l]
                            sig = "*" if eff.get("p_value_fdr", 1.0) < 0.05 else ""
                            ctrl_ci = eff["control_change_ci_95"]
                            print(f"{l:<7} {eff['measure_layer']:<6} {eff['baseline_r2']:.4f}   {eff['ablated_r2']:.4f}   "
                                  f"{eff['r2_change']:+.4f}   {eff['control_change_mean']:+.4f}    "
                                  f"[{ctrl_ci[0]:+.3f},{ctrl_ci[1]:+.3f}]  {eff['uniqueness']:+.4f}   "
                                  f"{eff['p_value_rank']:.4f}   {eff.get('p_value_fdr', 1.0):.4f}   {eff['z_score']:+.1f} {sig}")

    # Full Matrix Analysis (if enabled)
    matrix_analysis = None
    if full_matrix_mode and run_d2m_experiment and d2m_results:
        print("\n" + "=" * 100)
        print("FULL MATRIX D2M ANALYSIS (Ablation Layer × Measurement Layer)")
        print("=" * 100)

        matrix_analysis = analyze_d2m_full_matrix(
            d2m_results, layers, all_measurement_layers, len(test_indices), NUM_CONTROL_DIRECTIONS
        )

        if "pooled_null_stats" in matrix_analysis:
            ci = matrix_analysis["pooled_null_stats"]["ci_95"]
            print(f"Pooled null: mean={matrix_analysis['pooled_null_stats']['mean']:.4f}, "
                  f"std={matrix_analysis['pooled_null_stats']['std']:.4f}, "
                  f"95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

        print(f"\nMatrix dimensions: {len(layers)} ablation layers × {len(all_measurement_layers)} measurement layers")
        print(f"Total cells analyzed: {matrix_analysis['summary']['total_cells']}")
        print(f"Significant cells (FDR < 0.05): {matrix_analysis['summary']['n_significant']}")

        # Determine metric name based on probe type
        matrix_probe_type = matrix_analysis.get("probe_type", "entropy_regression")
        metric_label = "Acc" if matrix_probe_type == "mc_classification" else "R²"
        change_key = "acc_change" if matrix_probe_type == "mc_classification" else "r2_change"

        if matrix_analysis["summary"]["significant_cells_fdr"]:
            print("\nSignificant cells:")
            for abl_l, meas_l, metric_change, z_score in matrix_analysis["summary"]["significant_cells_fdr"]:
                print(f"  Ablate {abl_l} → Measure {meas_l}: Δ{metric_label}={metric_change:+.4f}, z={z_score:+.2f}")

        # Print a summary heatmap-style table for key measurement layers
        print(f"\n--- D2M {metric_label} Change Matrix (rows=ablation, cols=measurement) ---")
        # Show subset of measurement layers for readability
        sample_meas = all_measurement_layers[::max(1, len(all_measurement_layers)//10)]  # ~10 columns
        header = "Ablate\\Meas " + " ".join([f"{m:>6}" for m in sample_meas])
        print(header)
        for abl_layer in layers:
            row_data = []
            for m_layer in sample_meas:
                if m_layer >= abl_layer and abl_layer in matrix_analysis["matrix"]:
                    cell = matrix_analysis["matrix"][abl_layer].get(m_layer, {})
                    if cell:
                        row_data.append(f"{cell[change_key]:+.3f}")
                    else:
                        row_data.append("   -  ")
                else:
                    row_data.append("   -  ")
            print(f"{abl_layer:>6}       " + " ".join([f"{v:>6}" for v in row_data]))

    # Save
    results = {
        "config": {
            "base_model": BASE_MODEL_NAME, "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "dataset": DATASET_NAME, "metric": METRIC, "num_questions": len(questions),
            "num_controls": NUM_CONTROL_DIRECTIONS, "layers": layers,
            "full_matrix_mode": full_matrix_mode,
            "reverse_mode": REVERSE_MODE,
        },
        "direction_similarity": {str(k): v for k, v in direction_similarity.items()},
        "behavioral_ablation": behavioral_analysis,
    }
    if d2m_analysis:
        results["d2m_transfer_ablation"] = d2m_analysis
    if simultaneous_result:
        results["simultaneous_all_layer_ablation"] = simultaneous_result
    if matrix_analysis:
        # Convert matrix keys to strings for JSON serialization
        matrix_json = {}
        for abl_layer in matrix_analysis.get("matrix", {}):
            matrix_json[str(abl_layer)] = {
                str(m): v for m, v in matrix_analysis["matrix"][abl_layer].items()
            }
        results["d2m_full_matrix"] = {
            "ablation_layers": matrix_analysis["ablation_layers"],
            "measurement_layers": matrix_analysis["measurement_layers"],
            "pooled_null_stats": matrix_analysis.get("pooled_null_stats"),
            "matrix": matrix_json,
            "summary": matrix_analysis["summary"],
        }
    results_path = f"{output_prefix}_{METRIC}_mc_answer_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plotting
    if full_matrix_mode and matrix_analysis:
        plot_d2m_heatmap(matrix_analysis, behavioral_analysis, output_prefix + f"_{METRIC}",
                         simultaneous_result=simultaneous_result)
    elif behavioral_analysis:
        plot_results(behavioral_analysis, d2m_analysis, direction_similarity, output_prefix + f"_{METRIC}")

if __name__ == "__main__":
    main()