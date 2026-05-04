"""
Steering and ablation experiments using probe directions.

This script supports three types of directions:
1. "introspection" - From run_introspection_probe.py (meta activations → introspection_score)
                     NOTE: Only ablation is run for this direction type (steering skipped).
                     The introspection direction captures calibration quality, not a direct
                     uncertainty signal, so steering doesn't make conceptual sense.
2. "entropy" - From run_introspection_experiment.py (direct activations → entropy)
3. "shared" - From analyze_shared_unique.py (shared MC entropy direction across datasets)

Set DIRECTION_TYPE at the top to choose which direction to use.

For "shared" direction type:
- Loads shared component from *_shared_unique_directions.npz
- Uses META_R2_THRESHOLD to filter layers (only tests layers where direct→meta R² >= threshold)
- Tests whether the shared uncertainty signal (common across datasets) is causal for
  the model's confidence judgments

The script:
1. Loads probe results and directions from probe training
2. Automatically selects layers based on probe performance or transfer R²
3. Runs steering experiments with the probe direction and control directions
4. Runs ablation experiments to test causality (zeroing out the direction)
5. Measures effect on alignment between stated confidence and actual entropy
6. Computes p-values vs random control directions for statistical significance

Ablation tests the hypothesis: if the direction is causal for the model's
confidence judgments, removing it should degrade the correlation between
stated confidence and actual entropy.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import random
from scipy import stats

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from prompts import (
    # Confidence task
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    # Other-confidence task (control)
    OTHER_CONFIDENCE_OPTIONS,
    format_other_confidence_prompt,
    get_other_confidence_signal,
    # Delegate task
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    get_delegate_mapping,
    # Unified conversion
    response_to_confidence,
)

# =============================================================================
# CONFIGURATION — edit values in experiment_config.IntrospectionSteeringConfig
# =============================================================================
from experiment_config import IntrospectionSteeringConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
MODEL_NAME = _C.MODEL_NAME
DATASET_NAME = _C.DATASET_NAME
DIRECTION_TYPE = _C.DIRECTION_TYPE
AVAILABLE_METRICS = list(_C.AVAILABLE_METRICS)
METRIC = _C.METRIC
D2M_R2_THRESHOLD = _C.D2M_R2_THRESHOLD
D2D_R2_THRESHOLD = _C.D2D_R2_THRESHOLD
META_R2_THRESHOLD = _C.META_R2_THRESHOLD
META_TASK = _C.META_TASK
OUTPUTS_DIR = _C.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(exist_ok=True)


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    # Add delegate suffix if using delegate task (matches run_introspection_experiment.py)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


def get_directions_prefix() -> str:
    """Generate output filename prefix for direction files (task-independent).

    Direction files are task-independent because they predict metrics from
    direct task activations - the meta task doesn't affect direction computation.
    """
    model_short = get_model_short_name(BASE_MODEL_NAME)
    # NO task suffix - directions are task-independent
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection")


# Steering config
STEERING_LAYERS = None  # None = auto-select from probe results
STEERING_MULTIPLIERS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
NUM_STEERING_QUESTIONS = 100
# Number of control directions per layer. Set to None for dynamic scaling.
# Dynamic scaling ensures enough power for FDR correction:
#   - For FDR at α=0.05 with N layers, need min p-value < 0.05/N
#   - min p-value ≈ 1/(pooled_samples), so need pooled_samples > 20*N
#   - We use 25*N for safety margin, giving controls_per_layer = 25
NUM_CONTROL_DIRECTIONS = None  # None = dynamic based on num_layers, or set explicit value
FDR_ALPHA = 0.05              # Target FDR significance level
FDR_SAFETY_FACTOR = 25        # Multiplier: pooled_samples = FDR_SAFETY_FACTOR * num_layers
MIN_CONTROLS_PER_LAYER = 10   # Minimum controls even for few layers

BATCH_SIZE = 8  # Batch size for baseline/single-direction forward passes

# Intervention position: "last" or "all"
# - "last": Only modify the final token position (more precise, comparable to patching)
# - "all": Modify all token positions (standard steering approach)
INTERVENTION_POSITION = "last"

# Expanded batch target for multi-multiplier sweeps.
# When sweeping k multipliers simultaneously, we expand each base batch by k.
# This sets the TARGET total expanded batch size (base_batch * k_mult).
# Higher values = better GPU utilization but more memory.
# With k_mult=6 and EXPANDED_BATCH_TARGET=48, base batch = 8, expanded = 48.
EXPANDED_BATCH_TARGET = 48

# Quantization (for large models like 70B)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Aliases for backward compatibility with local code
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS

# Cached token IDs - populated once at startup to avoid repeated tokenization
_CACHED_TOKEN_IDS = {
    "meta_options": None,      # List of token IDs for S, T, U, V, W, X, Y, Z
    "delegate_options": None,  # List of token IDs for "1", "2"
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
# PROMPT FORMATTING WRAPPERS
# ============================================================================

def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> str:
    """Format a meta/confidence question using centralized tasks.py logic."""
    full_prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
    return full_prompt


def format_other_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> str:
    """Format an other-confidence (human difficulty estimation) question."""
    full_prompt, _ = format_other_confidence_prompt(question, tokenizer, use_chat_template)
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


def get_meta_options() -> List[str]:
    """Return the meta options based on META_TASK setting."""
    if META_TASK == "delegate":
        return DELEGATE_OPTIONS
    else:
        return META_OPTIONS


def local_response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None
) -> float:
    """
    Convert a meta response to a confidence value.

    Wrapper around tasks.response_to_confidence that passes the correct task_type.
    """
    task_type = "delegate" if META_TASK == "delegate" else "confidence"
    return response_to_confidence(response, probs, mapping, task_type)


# ============================================================================
# STEERING AND ABLATION
# ============================================================================

# Import DynamicCache safely
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

def extract_cache_tensors(past_key_values):
    """
    Extract raw tensors from past_key_values (tuple or DynamicCache).
    Uses robust indexing (cache[i]) instead of attribute access.
    Returns (key_tensors, value_tensors) where each is a list of tensors.
    """
    keys = []
    values = []
    
    # Robustly determine number of layers
    try:
        num_layers = len(past_key_values)
    except TypeError:
        # Fallback for weird objects (e.g. some PEFT proxies)
        if hasattr(past_key_values, "to_legacy_cache"):
             return extract_cache_tensors(past_key_values.to_legacy_cache())
        raise ValueError(f"Cannot determine length of cache: {type(past_key_values)}")
        
    for i in range(num_layers):
        # Indexing returns (key_state, value_state) for layer i
        # This is supported by both legacy Tuples and DynamicCache.__getitem__
        k, v = past_key_values[i]
        keys.append(k)
        values.append(v)
        
    return keys, values


def create_fresh_cache(key_tensors, value_tensors, expand_size=1):
    """
    Create a fresh DynamicCache (or tuple) from tensors.
    Uses the public .update() API to populate, avoiding attribute errors.
    """
    if DynamicCache is not None:
        cache = DynamicCache()
        for i, (k, v) in enumerate(zip(key_tensors, value_tensors)):
            if expand_size > 1:
                k = k.repeat_interleave(expand_size, dim=0)
                v = v.repeat_interleave(expand_size, dim=0)
            
            # Use public update API to populate the fresh cache.
            # When the cache is empty, update() appends these tensors as the layer state.
            cache.update(k, v, i)
            
        return cache
    else:
        # Fallback to tuple for older transformers
        layers = []
        for k, v in zip(key_tensors, value_tensors):
            if expand_size > 1:
                k = k.repeat_interleave(expand_size, dim=0)
                v = v.repeat_interleave(expand_size, dim=0)
            layers.append((k, v))
        return tuple(layers)
        
def get_kv_cache(model, batch_inputs):
    """
    Run the prefix to generate KV cache tensors.
    Returns dictionary with next-step inputs and 'past_key_values_data' (snapshot).
    """
    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]
    
    # 1. Run Prefix (Tokens 0 to T-1)
    prefix_ids = input_ids[:, :-1]
    prefix_mask = attention_mask[:, :-1]
    
    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
        )
    
    # 2. Extract Immutable Snapshot (List of Tensors)
    # This prevents the "tuple has no attribute get_seq_length" AND mutation bugs
    keys, values = extract_cache_tensors(outputs.past_key_values)
    
    # 3. Prepare next step inputs
    last_ids = input_ids[:, -1:]
    
    result = {
        "input_ids": last_ids,
        "attention_mask": attention_mask, # Full mask (History + Current)
        "past_key_values_data": (keys, values), # Store tensors, not the object
    }
    
    # Preserve position_ids if available (Critical for RoPE)
    if "position_ids" in batch_inputs:
        result["position_ids"] = batch_inputs["position_ids"][:, -1:]
        
    return result
    
class SteeringHook:
    """Hook that adds a steering vector to activations.

    Respects INTERVENTION_POSITION setting:
    - "last": Only modify the final token position
    - "all": Modify all token positions
    """

    def __init__(self, steering_vector: torch.Tensor, multiplier: float, pre_normalized: bool = False):
        # Ensure normalized so multiplier has consistent meaning across directions
        if pre_normalized:
            self.steering_vector = steering_vector
        else:
            self.steering_vector = steering_vector / steering_vector.norm()
        self.multiplier = multiplier
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        delta = self.multiplier * self.steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)

        if INTERVENTION_POSITION == "last":
            # Only modify the last token position
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + delta
        else:
            # Modify all positions (original behavior)
            hidden_states = hidden_states + delta.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def set_multiplier(self, multiplier: float):
        """Update multiplier without recreating the hook."""
        self.multiplier = multiplier

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()



class BatchSteeringHook:
    """Hook that adds a *per-example* steering delta to activations.

    This is designed for "multiplier sweep in one pass" by expanding the batch:
    each prompt is duplicated for each multiplier, and this hook adds a different
    delta vector for each expanded example.

    Respects INTERVENTION_POSITION setting:
    - "last": Only modify the final token position
    - "all": Modify all token positions
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

        # hs: (batch, seq, hidden); delta_bh: (batch, hidden)
        # Must cast both device and dtype for compatibility with device_map="auto"
        delta = self.delta_bh.to(device=hs.device, dtype=hs.dtype)

        if INTERVENTION_POSITION == "last":
            # Only modify the last token position
            hs = hs.clone()
            hs[:, -1, :] = hs[:, -1, :] + delta
        else:
            # Broadcast delta across all sequence positions (original behavior)
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



class AblationHook:
    """
    Hook that removes the component of activations along a direction.

    Projects out the direction: x' = x - (x · d) * d
    This tests whether the direction is causally involved in the behavior.

    Respects INTERVENTION_POSITION setting:
    - "last": Only modify the final token position
    - "all": Modify all token positions
    """

    def __init__(self, direction: torch.Tensor, pre_normalized: bool = False):
        # Ensure normalized
        if pre_normalized:
            self.direction = direction
        else:
            self.direction = direction / direction.norm()
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Ensure direction is on correct device/dtype
        direction = self.direction.to(device=hidden_states.device, dtype=hidden_states.dtype)

        if INTERVENTION_POSITION == "last":
            # Only modify the last token position
            hidden_states = hidden_states.clone()
            last_token = hidden_states[:, -1, :]  # (batch, hidden)
            proj = (last_token @ direction).unsqueeze(-1) * direction  # (batch, hidden)
            hidden_states[:, -1, :] = last_token - proj
        else:
            # Project out the direction from all tokens (original behavior)
            # hidden_states: (batch, seq_len, hidden_dim)
            # direction: (hidden_dim,)
            proj = (hidden_states @ direction).unsqueeze(-1) * direction
            hidden_states = hidden_states - proj

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


class BatchAblationHook:
    """Hook that projects out a *per-example* direction from activations.

    For batched ablation: each prompt is duplicated for each direction,
    and this hook removes a different direction for each expanded example.
    This allows processing multiple ablation conditions in a single forward pass.

    Respects INTERVENTION_POSITION setting:
    - "last": Only modify the final token position
    - "all": Modify all token positions
    """

    def __init__(self, directions_bh: Optional[torch.Tensor] = None):
        """
        Args:
            directions_bh: (batch, hidden_dim) tensor of normalized directions
                          Each row is a direction to project out for that example
        """
        self.directions_bh = directions_bh
        self.handle = None
        self._diag_printed = False

    def set_directions(self, directions_bh: torch.Tensor):
        self.directions_bh = directions_bh
        self._diag_printed = False  # Reset diagnostic flag for new directions

    def __call__(self, module, input, output):
        if self.directions_bh is None:
            return output

        hs = output[0] if isinstance(output, tuple) else output
        # hs: (batch, seq, hidden); directions_bh: (batch, hidden)

        # Cast directions to match device and dtype
        dirs = self.directions_bh.to(device=hs.device, dtype=hs.dtype)

        if INTERVENTION_POSITION == "last":
            # Only modify the last token position
            hs = hs.clone()
            last_token = hs[:, -1, :]  # (batch, hidden)
            # Dot product for last token only: (batch, hidden) * (batch, hidden) -> (batch,)
            dots = torch.einsum('bh,bh->b', last_token, dirs)
            # Projection: dots[:, None] * dirs -> (batch, hidden)
            proj = dots.unsqueeze(-1) * dirs

            # Diagnostic: track projection magnitude (first call only per direction set)
            if not self._diag_printed:
                proj_norm = proj.norm(dim=-1).mean().item()
                hs_norm = last_token.norm(dim=-1).mean().item()
                dir_norm = dirs.norm(dim=-1).mean().item()
                dot_mean = dots.abs().mean().item()
                # Always print first diagnostic per layer to confirm hook is running
                if not hasattr(self, '_first_diag_done'):
                    print(f"  [BatchAblationHook] dir_norm={dir_norm:.4f}, hs_norm={hs_norm:.2f}, dot={dot_mean:.4f}, proj_norm={proj_norm:.4f}")
                    self._first_diag_done = True
                elif proj_norm < 1e-6 and hs_norm > 1e-3:
                    print(f"  [BatchAblationHook] WARNING: Near-zero projection ({proj_norm:.2e}) vs hs_norm ({hs_norm:.2e})")
                self._diag_printed = True

            hs[:, -1, :] = last_token - proj
        else:
            # Project out direction from all tokens (original behavior):
            # For each example i: proj_i = (hs_i @ d_i) * d_i
            # Dot product: (batch, seq, hidden) einsum with (batch, hidden) -> (batch, seq)
            dots = torch.einsum('bsh,bh->bs', hs, dirs)
            # Projection: dots[:, :, None] * dirs[:, None, :] -> (batch, seq, hidden)
            proj = dots.unsqueeze(-1) * dirs.unsqueeze(1)
            # Remove projection
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


def generate_orthogonal_directions(direction: np.ndarray, num_directions: int) -> List[np.ndarray]:
    """Generate random directions orthogonal to the given direction."""
    hidden_dim = len(direction)
    orthogonal = []

    for _ in range(num_directions):
        random_vec = np.random.randn(hidden_dim)
        random_vec = random_vec - np.dot(random_vec, direction) * direction
        for prev in orthogonal:
            random_vec = random_vec - np.dot(random_vec, prev) * prev
        random_vec = random_vec / np.linalg.norm(random_vec)
        orthogonal.append(random_vec)

    return orthogonal


def pretokenize_prompts(
    prompts: List[str],
    tokenizer,
    device: str
) -> Dict:
    """
    Pre-tokenize all prompts once (BPE encoding).

    Returns dict with:
        - input_ids: List of token ID lists (variable length, no padding yet)
        - attention_mask: List of attention mask lists
        - lengths: List of sequence lengths
        - sorted_order: Indices sorted by length (for efficient batching)

    Padding is deferred to batch time to avoid padding short prompts to global max.
    """
    # Tokenize without padding - just BPE encode once
    tokenized = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        return_attention_mask=True
    )

    lengths = [len(ids) for ids in tokenized["input_ids"]]
    # Sort indices by length for efficient batching (similar lengths together)
    sorted_order = sorted(range(len(prompts)), key=lambda i: lengths[i])

    return {
        "input_ids": tokenized["input_ids"],  # List of lists
        "attention_mask": tokenized["attention_mask"],  # List of lists
        "lengths": lengths,
        "sorted_order": sorted_order,
        "device": device,
        "tokenizer": tokenizer,  # Keep reference for padding
    }


def build_padded_gpu_batches(
    cached_inputs: Dict,
    tokenizer,
    device: str,
    batch_size: int,
) -> List[Tuple[List[int], Dict[str, torch.Tensor]]]:
    """Pad each length-sorted batch once and keep tensors on-device.

    This eliminates repeated tokenizer.pad() and CPU→GPU copies for every
    (layer × multiplier × control) forward pass.
    """
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
        # Keep on-device for reuse across many sweeps.
        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}
        batches.append((batch_indices, batch_inputs))

    return batches


def _get_transformer_and_lm_head(model):
    """Best-effort access to (transformer, lm_head) for fast option-only logits."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    transformer = getattr(base, "model", None)
    lm_head = getattr(base, "lm_head", None)
    if transformer is None or lm_head is None or not hasattr(lm_head, "weight"):
        return None, None
    return transformer, lm_head


def _prepare_option_weight(lm_head, model, option_token_ids: List[int]) -> Optional[torch.Tensor]:
    """Extract lm_head rows for the option token IDs: (n_opt, hidden_dim).

    For models with device_map="auto", the lm_head weight may be on meta device.
    In that case, we do a dummy forward pass to materialize the weight.
    """
    if lm_head is None or not hasattr(lm_head, "weight"):
        return None
    W = lm_head.weight
    if W is None or W.ndim != 2:
        return None

    # Check if weight is on meta device (happens with device_map="auto" for large models)
    if W.device.type == "meta":
        # The lm_head hasn't been materialized yet. We need to run a forward pass
        # to trigger the weight loading. This is a one-time cost.
        print("  Note: lm_head on meta device, triggering materialization...")
        try:
            # Create minimal dummy input to trigger weight loading
            dummy_input = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            with torch.no_grad():
                _ = model(dummy_input, use_cache=False)
            # Now check again
            W = lm_head.weight
            if W.device.type == "meta":
                print("  Warning: lm_head still on meta after forward pass, using slow path")
                return None
        except Exception as e:
            print(f"  Warning: Failed to materialize lm_head: {e}, using slow path")
            return None

    option_ids = torch.tensor(option_token_ids, dtype=torch.long, device=W.device)
    return W.index_select(0, option_ids)


def _compute_batch_option_logits(
    model,
    transformer,
    W_opt: Optional[torch.Tensor],
    option_token_ids: List[int],
    batch_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return (batch, n_opt) logits for the next token.

    Fast path: transformer forward → last_hidden_state[:, -1] → matmul with W_opt.
    Fallback: model forward → full logits → index option_token_ids.
    """
    if transformer is None or W_opt is None:
        outputs = model(**batch_inputs, use_cache=False)
        batch_logits = outputs.logits[:, -1, :]
        # Ensure we're on a real device (not meta) before indexing
        if batch_logits.device.type == "meta":
            raise RuntimeError("Model logits are on meta device - model may not be fully loaded")
        return batch_logits[:, option_token_ids]

    out = transformer(**batch_inputs, use_cache=False, return_dict=True)
    last_h = out.last_hidden_state[:, -1, :]
    # With device_map="auto", lm_head may live on a different device.
    if last_h.device != W_opt.device:
        last_h = last_h.to(W_opt.device)
    return last_h @ W_opt.T


def precompute_direction_tensors(
    directions: Dict,
    layers: List[int],
    num_controls: int,
    device: str,
    dtype: torch.dtype
) -> Dict:
    """
    Precompute normalized direction tensors on GPU for all layers and controls.

    Returns dict with structure:
    {
        layer_idx: {
            "introspection": tensor,  # normalized, on GPU
            "controls": [tensor, ...]  # normalized, on GPU
        }
    }
    """
    cached = {}
    for layer_idx in layers:
        introspection_dir = np.array(directions[f"layer_{layer_idx}_introspection"])
        # Normalize in numpy, then convert to tensor
        introspection_dir = introspection_dir / np.linalg.norm(introspection_dir)
        introspection_tensor = torch.tensor(introspection_dir, dtype=dtype, device=device)

        # Generate and cache control directions
        control_dirs = generate_orthogonal_directions(introspection_dir, num_controls)
        control_tensors = [
            torch.tensor(cd, dtype=dtype, device=device)
            for cd in control_dirs
        ]

        cached[layer_idx] = {
            "introspection": introspection_tensor,
            "controls": control_tensors,
        }

    return cached


def get_confidence_response(
    model,
    tokenizer,
    question: Dict,
    layer_idx: Optional[int],
    steering_vector: Optional[np.ndarray],
    multiplier: float,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response, optionally with steering.

    Returns (response, confidence, option_probs, mapping) where mapping is only
    set for delegate task.
    """
    # Format prompt based on task type
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    if layer_idx is not None and steering_vector is not None and multiplier != 0.0:
        # Steering
        steering_tensor = torch.tensor(
            steering_vector,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        hook = SteeringHook(steering_tensor, multiplier)
        hook.register(layer_module)

        # Prepare fast option-only projection
        if META_TASK == "delegate":
            option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        else:
            option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        transformer, lm_head = _get_transformer_and_lm_head(model)
        W_opt = _prepare_option_weight(lm_head, model, option_token_ids)

        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
            with torch.inference_mode():
                option_logits = _compute_batch_option_logits(
                    model, transformer, W_opt, option_token_ids, inputs
                )[0]
        finally:
            hook.remove()
    else:
        # No steering
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        # Prepare fast option-only projection
        if META_TASK == "delegate":
            option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        else:
            option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        transformer, lm_head = _get_transformer_and_lm_head(model)
        W_opt = _prepare_option_weight(lm_head, model, option_token_ids)
        with torch.inference_mode():
            option_logits = _compute_batch_option_logits(
                model, transformer, W_opt, option_token_ids, inputs
            )[0]

    option_probs = torch.softmax(option_logits, dim=-1).float().cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = local_response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


def get_confidence_with_ablation(
    model,
    tokenizer,
    question: Dict,
    layer_idx: int,
    ablation_direction: np.ndarray,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response with ablation (direction zeroed out).

    Returns (response, confidence, option_probs, mapping) where mapping is only
    set for delegate task.
    """
    # Format prompt based on task type
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    # Create ablation tensor
    ablation_tensor = torch.tensor(
        ablation_direction,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    if hasattr(model, 'get_base_model'):
        layer_module = model.get_base_model().model.layers[layer_idx]
    else:
        layer_module = model.model.layers[layer_idx]

    hook = AblationHook(ablation_tensor)
    hook.register(layer_module)

    # Prepare fast option-only projection
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
    transformer, lm_head = _get_transformer_and_lm_head(model)
    W_opt = _prepare_option_weight(lm_head, model, option_token_ids)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.inference_mode():
            option_logits = _compute_batch_option_logits(
                model, transformer, W_opt, option_token_ids, inputs
            )[0]
    finally:
        hook.remove()

    option_probs = torch.softmax(option_logits, dim=-1).float().cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = local_response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


# ============================================================================
# OTHER-CONFIDENCE CONTROL
# ============================================================================

def run_other_confidence_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    layers: List[int],
    directions: Dict,
    use_chat_template: bool,
    steering_multiplier: float = 0.0,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """
    Run other-confidence (human difficulty estimation) prompts with optional steering.

    This is a control task to test whether the introspection direction specifically
    affects self-confidence or also affects general confidence-like judgments.

    Args:
        steering_multiplier: If 0.0, runs baseline. Otherwise applies steering at this multiplier.

    Returns:
        Dict with per-question results: response, signal (expected %), etc.
    """
    # Only run for confidence task (not delegate)
    if META_TASK != "confidence":
        return None

    print(f"\nRunning other-confidence control (multiplier={steering_multiplier})...")

    # Pre-format other-confidence prompts
    other_prompts = []
    for question in questions:
        prompt = format_other_meta_prompt(question, tokenizer, use_chat_template)
        other_prompts.append(prompt)

    # Pre-tokenize
    cached_other_inputs = pretokenize_prompts(other_prompts, tokenizer, DEVICE)

    # Get option token IDs (same S-Z scale as self-confidence)
    other_option_token_ids = _CACHED_TOKEN_IDS["meta_options"]  # S, T, U, V, W, X, Y, Z
    other_options = META_OPTIONS

    # Build batches
    gpu_batches = build_padded_gpu_batches(cached_other_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Get model components for efficient forward
    transformer, lm_head = _get_transformer_and_lm_head(model)
    W_opt = _prepare_option_weight(lm_head, model, other_option_token_ids)

    results = {
        "layers": layers,
        "steering_multiplier": steering_multiplier,
        "num_questions": len(questions),
        "layer_results": {},
    }

    for layer_idx in tqdm(layers, desc="Other-confidence layers"):
        # Get layer module (handle adapter models)
        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        # Get direction tensor
        if cached_directions is not None and layer_idx in cached_directions:
            introspection_tensor = cached_directions[layer_idx]["introspection"]
        else:
            direction_key = f"layer_{layer_idx}_introspection"
            introspection_dir = directions.get(direction_key)
            if introspection_dir is None:
                continue
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            introspection_dir = introspection_dir / np.linalg.norm(introspection_dir)
            introspection_tensor = torch.tensor(introspection_dir, dtype=dtype, device=DEVICE)

        # Run with or without steering
        if steering_multiplier != 0.0:
            hook = SteeringHook(introspection_tensor, multiplier=steering_multiplier, pre_normalized=True)
            hook.register(layer_module)

        question_results = [None] * len(other_prompts)

        try:
            for batch_indices, batch_inputs in gpu_batches:
                with torch.inference_mode():
                    batch_option_logits = _compute_batch_option_logits(
                        model, transformer, W_opt, other_option_token_ids, batch_inputs
                    )
                    batch_option_probs = torch.softmax(batch_option_logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    option_probs = batch_option_probs[i]
                    response = other_options[np.argmax(option_probs)]
                    signal = get_other_confidence_signal(option_probs)

                    question_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": response,
                        "signal": signal,  # Expected % of humans who would know
                        "probs": option_probs.tolist(),
                    }
        finally:
            if steering_multiplier != 0.0:
                hook.remove()

        results["layer_results"][layer_idx] = question_results

    return results


def analyze_other_confidence_effect(
    baseline_other: Dict,
    steered_other: Dict,
    baseline_self: List[Dict],
    steered_self: List[Dict],
    layer_idx: int
) -> Dict:
    """
    Compare steering effect on self-confidence vs other-confidence.

    Returns dict with:
    - self_effect: mean change in self-confidence signal
    - other_effect: mean change in other-confidence signal
    - self_vs_other_ratio: how much more self is affected than other
    - p_value_paired: paired t-test p-value for |self_delta| > |other_delta|
    - p_value_permutation: permutation test p-value
    """
    if baseline_other is None or steered_other is None:
        return None

    # Get per-question changes
    self_baseline_signals = np.array([r["confidence"] for r in baseline_self])
    self_steered_signals = np.array([r["confidence"] for r in steered_self])
    self_delta = self_steered_signals - self_baseline_signals

    other_baseline_signals = np.array([r["signal"] for r in baseline_other["layer_results"].get(layer_idx, [])])
    other_steered_signals = np.array([r["signal"] for r in steered_other["layer_results"].get(layer_idx, [])])

    if len(other_baseline_signals) == 0 or len(other_steered_signals) == 0:
        return None

    other_delta = other_steered_signals - other_baseline_signals

    self_effect = float(np.mean(self_delta))
    other_effect = float(np.mean(other_delta))

    # Compute ratio (avoid division by zero)
    if abs(other_effect) > 1e-6:
        ratio = abs(self_effect) / abs(other_effect)
    elif abs(self_effect) > 1e-6:
        ratio = float('inf')
    else:
        # Both effects are ~0 - no meaningful comparison
        ratio = float('nan')

    # Compute correlation safely (avoid warning when std=0)
    if len(self_delta) > 1 and np.std(self_delta) > 1e-10 and np.std(other_delta) > 1e-10:
        self_other_corr = float(np.corrcoef(self_delta, other_delta)[0, 1])
    else:
        self_other_corr = float('nan')

    # Statistical test: is |self_delta| > |other_delta| per question?
    # Paired t-test on absolute effects
    abs_self = np.abs(self_delta)
    abs_other = np.abs(other_delta)
    diff = abs_self - abs_other  # positive if self effect larger

    # Paired t-test (one-sided: self > other)
    if len(diff) > 1 and np.std(diff) > 1e-10:
        t_stat, p_two_sided = stats.ttest_rel(abs_self, abs_other)
        # One-sided p-value: we want self > other
        p_paired = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    else:
        p_paired = float('nan')

    # Permutation test: shuffle self/other labels
    n_permutations = 1000
    observed_diff = np.mean(abs_self) - np.mean(abs_other)
    count_extreme = 0
    combined = np.stack([abs_self, abs_other], axis=1)  # (n_questions, 2)
    for _ in range(n_permutations):
        # Randomly swap self/other for each question
        swaps = np.random.randint(0, 2, size=len(combined))
        perm_self = np.array([combined[i, swaps[i]] for i in range(len(combined))])
        perm_other = np.array([combined[i, 1 - swaps[i]] for i in range(len(combined))])
        perm_diff = np.mean(perm_self) - np.mean(perm_other)
        if perm_diff >= observed_diff:
            count_extreme += 1
    p_permutation = (count_extreme + 1) / (n_permutations + 1)

    # Bootstrap 95% CI for the difference
    n_bootstrap = 1000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(diff), size=len(diff), replace=True)
        bootstrap_diffs.append(np.mean(diff[idx]))
    ci_low = float(np.percentile(bootstrap_diffs, 2.5))
    ci_high = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "self_effect_mean": self_effect,
        "self_effect_std": float(np.std(self_delta)),
        "other_effect_mean": other_effect,
        "other_effect_std": float(np.std(other_delta)),
        "self_vs_other_ratio": ratio,
        "self_other_correlation": self_other_corr,
        "diff_abs_effect": float(np.mean(diff)),  # mean(|self| - |other|)
        "diff_abs_effect_ci95": [ci_low, ci_high],
        "p_value_paired": float(p_paired),
        "p_value_permutation": float(p_permutation),
    }


def run_other_confidence_with_ablation(
    model,
    tokenizer,
    questions: List[Dict],
    layers: List[int],
    directions: Dict,
    use_chat_template: bool,
    ablate: bool = False,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """
    Run other-confidence (human difficulty estimation) prompts with optional ablation.

    Args:
        ablate: If True, ablates the introspection direction. If False, runs baseline.

    Returns:
        Dict with per-question results: response, signal (expected %), etc.
    """
    # Only run for confidence task (not delegate)
    if META_TASK != "confidence":
        return None

    condition = "with ablation" if ablate else "baseline"
    print(f"\nRunning other-confidence control ({condition})...")

    # Pre-format other-confidence prompts
    other_prompts = []
    for question in questions:
        prompt = format_other_meta_prompt(question, tokenizer, use_chat_template)
        other_prompts.append(prompt)

    # Pre-tokenize
    cached_other_inputs = pretokenize_prompts(other_prompts, tokenizer, DEVICE)

    # Get option token IDs (same S-Z scale as self-confidence)
    other_option_token_ids = _CACHED_TOKEN_IDS["meta_options"]  # S, T, U, V, W, X, Y, Z
    other_options = META_OPTIONS

    # Build batches
    gpu_batches = build_padded_gpu_batches(cached_other_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Get model components for efficient forward
    transformer, lm_head = _get_transformer_and_lm_head(model)
    W_opt = _prepare_option_weight(lm_head, model, other_option_token_ids)

    results = {
        "layers": layers,
        "ablated": ablate,
        "num_questions": len(questions),
        "layer_results": {},
    }

    for layer_idx in tqdm(layers, desc="Other-confidence layers"):
        # Get layer module (handle adapter models)
        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        # Get direction tensor if ablating
        if ablate:
            if cached_directions is not None and layer_idx in cached_directions:
                introspection_tensor = cached_directions[layer_idx]["introspection"]
            else:
                direction_key = f"layer_{layer_idx}_introspection"
                introspection_dir = directions.get(direction_key)
                if introspection_dir is None:
                    continue
                dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                introspection_dir = introspection_dir / np.linalg.norm(introspection_dir)
                introspection_tensor = torch.tensor(introspection_dir, dtype=dtype, device=DEVICE)

            hook = AblationHook(introspection_tensor, pre_normalized=True)
            hook.register(layer_module)

        question_results = [None] * len(other_prompts)

        try:
            for batch_indices, batch_inputs in gpu_batches:
                with torch.inference_mode():
                    batch_option_logits = _compute_batch_option_logits(
                        model, transformer, W_opt, other_option_token_ids, batch_inputs
                    )
                    batch_option_probs = torch.softmax(batch_option_logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    option_probs = batch_option_probs[i]
                    response = other_options[np.argmax(option_probs)]
                    signal = get_other_confidence_signal(option_probs)

                    question_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": response,
                        "signal": signal,  # Expected % of humans who would know
                        "probs": option_probs.tolist(),
                    }
        finally:
            if ablate:
                hook.remove()

        results["layer_results"][layer_idx] = question_results

    return results


def analyze_other_confidence_ablation_effect(
    baseline_other: Dict,
    ablated_other: Dict,
    baseline_self: List[Dict],
    ablated_self: List[Dict],
    layer_idx: int
) -> Dict:
    """
    Compare ablation effect on self-confidence vs other-confidence.

    Returns dict with:
    - self_effect: mean change in self-confidence correlation with metric
    - other_effect: mean change in other-confidence signal
    - self_vs_other_ratio: how much more self is affected than other
    """
    if baseline_other is None or ablated_other is None:
        return None

    # Get per-question changes in raw signals
    self_baseline_signals = np.array([r["confidence"] for r in baseline_self])
    self_ablated_signals = np.array([r["confidence"] for r in ablated_self])
    self_delta = self_ablated_signals - self_baseline_signals

    other_baseline_signals = np.array([r["signal"] for r in baseline_other["layer_results"].get(layer_idx, [])])
    other_ablated_signals = np.array([r["signal"] for r in ablated_other["layer_results"].get(layer_idx, [])])

    if len(other_baseline_signals) == 0 or len(other_ablated_signals) == 0:
        return None

    other_delta = other_ablated_signals - other_baseline_signals

    self_effect = float(np.mean(np.abs(self_delta)))  # Mean absolute change
    other_effect = float(np.mean(np.abs(other_delta)))

    # Compute ratio (avoid division by zero)
    if other_effect > 1e-6:
        ratio = self_effect / other_effect
    elif self_effect > 1e-6:
        ratio = float('inf')
    else:
        # Both effects are ~0 - no meaningful comparison
        ratio = float('nan')

    # Compute correlation safely (avoid warning when std=0)
    if len(self_delta) > 1 and np.std(self_delta) > 1e-10 and np.std(other_delta) > 1e-10:
        self_other_corr = float(np.corrcoef(self_delta, other_delta)[0, 1])
    else:
        self_other_corr = float('nan')

    # Statistical test: is |self_delta| > |other_delta| per question?
    abs_self = np.abs(self_delta)
    abs_other = np.abs(other_delta)
    diff = abs_self - abs_other  # positive if self effect larger

    # Paired t-test (one-sided: self > other)
    if len(diff) > 1 and np.std(diff) > 1e-10:
        t_stat, p_two_sided = stats.ttest_rel(abs_self, abs_other)
        # One-sided p-value: we want self > other
        p_paired = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    else:
        p_paired = float('nan')

    # Permutation test: shuffle self/other labels
    n_permutations = 1000
    observed_diff = np.mean(abs_self) - np.mean(abs_other)
    count_extreme = 0
    combined = np.stack([abs_self, abs_other], axis=1)  # (n_questions, 2)
    for _ in range(n_permutations):
        # Randomly swap self/other for each question
        swaps = np.random.randint(0, 2, size=len(combined))
        perm_self = np.array([combined[i, swaps[i]] for i in range(len(combined))])
        perm_other = np.array([combined[i, 1 - swaps[i]] for i in range(len(combined))])
        perm_diff = np.mean(perm_self) - np.mean(perm_other)
        if perm_diff >= observed_diff:
            count_extreme += 1
    p_permutation = (count_extreme + 1) / (n_permutations + 1)

    # Bootstrap 95% CI for the difference
    n_bootstrap = 1000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(diff), size=len(diff), replace=True)
        bootstrap_diffs.append(np.mean(diff[idx]))
    ci_low = float(np.percentile(bootstrap_diffs, 2.5))
    ci_high = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "self_effect_mean_abs": self_effect,
        "self_effect_std": float(np.std(self_delta)),
        "other_effect_mean_abs": other_effect,
        "other_effect_std": float(np.std(other_delta)),
        "self_vs_other_ratio": ratio,
        "self_other_correlation": self_other_corr,
        "diff_abs_effect": float(np.mean(diff)),  # mean(|self| - |other|)
        "diff_abs_effect_ci95": [ci_low, ci_high],
        "p_value_paired": float(p_paired),
        "p_value_permutation": float(p_permutation),
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def _run_steering_full_forward(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    multipliers: List[float],
    num_controls: int,
    cached_directions: Dict,
    option_token_ids: List[int],
    options: List[str],
    mappings: List,
    cached_inputs: List[Dict],
) -> Dict:
    """Fallback: Full forward pass implementation for INTERVENTION_POSITION='all'."""
    print("Using full forward passes (INTERVENTION_POSITION='all')...")

    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    # Initialize results
    shared_baseline = [None] * len(questions)
    final_layer_results = {}
    for l in layers:
        final_layer_results[l] = {
            "baseline": shared_baseline,
            "introspection": {m: [None] * len(questions) for m in multipliers},
            "controls": {f"control_{i}": {m: [None] * len(questions) for m in multipliers} for i in range(num_controls)},
        }
        if 0.0 in multipliers:
            final_layer_results[l]["introspection"][0.0] = shared_baseline
            for k in final_layer_results[l]["controls"]:
                final_layer_results[l]["controls"][k][0.0] = shared_baseline

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Compute baseline first
    print("Computing baseline...")
    for batch_indices, batch_inputs in gpu_batches:
        with torch.inference_mode():
            out = model(**batch_inputs)
            logits = out.logits[:, -1, :][:, option_token_ids]
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        for i, q_idx in enumerate(batch_indices):
            p = probs[i]
            resp = options[np.argmax(p)]
            conf = local_response_to_confidence(resp, p, mappings[q_idx])
            m_val = direct_metric_values[q_idx]
            align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)
            shared_baseline[q_idx] = {
                "question_idx": q_idx, "response": resp, "confidence": conf,
                "metric": float(m_val), "alignment": float(align)
            }

    nonzero_multipliers = [m for m in multipliers if m != 0.0]

    # Process each layer
    for layer_idx in tqdm(layers, desc="Layers"):
        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        intro_dir = cached_directions[layer_idx]["introspection"]
        control_dirs = cached_directions[layer_idx]["controls"]

        # Helper to run all questions for one direction+multiplier
        def run_condition(direction_vector, mult, result_storage):
            hook = SteeringHook(direction_vector, mult, pre_normalized=True)
            hook.register(layer_module)
            try:
                for batch_indices, batch_inputs in gpu_batches:
                    with torch.inference_mode():
                        out = model(**batch_inputs)
                        logits = out.logits[:, -1, :][:, option_token_ids]
                        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                    for i, q_idx in enumerate(batch_indices):
                        p = probs[i]
                        resp = options[np.argmax(p)]
                        conf = local_response_to_confidence(resp, p, mappings[q_idx])
                        m_val = direct_metric_values[q_idx]
                        align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)
                        result_storage[mult][q_idx] = {
                            "question_idx": q_idx, "response": resp, "confidence": conf,
                            "metric": float(m_val), "alignment": float(align)
                        }
            finally:
                hook.remove()

        # Introspection
        for mult in nonzero_multipliers:
            run_condition(intro_dir, mult, final_layer_results[layer_idx]["introspection"])

        # Controls
        for i_c, ctrl_dir in enumerate(control_dirs):
            for mult in nonzero_multipliers:
                run_condition(ctrl_dir, mult, final_layer_results[layer_idx]["controls"][f"control_{i_c}"])

    return {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results
    }


def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    directions: Dict,
    multipliers: List[float],
    num_controls: int,
    use_chat_template: bool,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """Run steering experiment with KV cache optimization (Robust Fix)."""

    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

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

    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = DELEGATE_OPTIONS
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = META_OPTIONS

    if cached_directions is None:
        cached_directions = precompute_direction_tensors(
            directions, layers, num_controls, DEVICE,
            torch.float16 if DEVICE == "cuda" else torch.float32
        )

    # Fallback if not intervening on last token
    if INTERVENTION_POSITION != "last":
        print(f"\nRunning steering experiment (full forward fallback)...")
        # Ensure _run_steering_full_forward is defined or just return/error
        return _run_steering_full_forward(
            model, tokenizer, questions, direct_metric_values,
            layers, multipliers, num_controls,
            cached_directions, option_token_ids, options, mappings, cached_inputs
        )

    print(f"\nRunning steering experiment (KV Cache + batched multipliers)...")

    nonzero_multipliers = [m for m in multipliers if m != 0.0]
    k_mult = len(nonzero_multipliers)

    # Initialize results
    shared_baseline = [None] * len(questions)
    final_layer_results = {}
    for l in layers:
        final_layer_results[l] = {
            "baseline": shared_baseline,
            "introspection": {m: [None] * len(questions) for m in multipliers},
            "controls": {f"control_{i}": {m: [None] * len(questions) for m in multipliers} for i in range(num_controls)},
        }
        if 0.0 in multipliers:
            final_layer_results[l]["introspection"][0.0] = shared_baseline
            for k in final_layer_results[l]["controls"]:
                final_layer_results[l]["controls"][k][0.0] = shared_baseline

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)
    print(f"Processing {len(gpu_batches)} batches, {k_mult} multipliers batched per forward pass...")

    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Batches"):
        B = len(batch_indices)

        # 1. Compute KV Cache (Snapshot)
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        # 2. Compute Baseline (No steering)
        # Reconstruct fresh cache (size 1)
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
            conf = local_response_to_confidence(resp, p, mappings[q_idx])
            m_val = direct_metric_values[q_idx]
            align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)
            shared_baseline[q_idx] = {
                "question_idx": q_idx, "response": resp, "confidence": conf,
                "metric": float(m_val), "alignment": float(align)
            }

        # 3. Prepare Inputs for Steering (Expansion)
        expanded_input_ids = base_step_data["input_ids"].repeat_interleave(k_mult, dim=0)
        expanded_attention_mask = base_step_data["attention_mask"].repeat_interleave(k_mult, dim=0)
        
        expanded_inputs_template = {
            "input_ids": expanded_input_ids,
            "attention_mask": expanded_attention_mask,
            "use_cache": True
        }
        if "position_ids" in base_step_data:
            expanded_inputs_template["position_ids"] = base_step_data["position_ids"].repeat_interleave(k_mult, dim=0)

        # 4. Iterate Layers
        for layer_idx in layers:
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer_idx]
            else:
                layer_module = model.model.layers[layer_idx]

            intro_dir = cached_directions[layer_idx]["introspection"]
            control_dirs = cached_directions[layer_idx]["controls"]

            hook = BatchSteeringHook()
            hook.register(layer_module)

            def run_batched_mult_sweep(direction_vector, result_dict):
                # 1. Create FRESH cache expanded for this pass
                # (Re-creating the cache object is fast compared to the model run)
                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_mult)
                
                # 2. Attach cache to inputs
                current_inputs = expanded_inputs_template.copy()
                current_inputs["past_key_values"] = pass_cache

                # 3. Set Deltas
                # delta_bh: [B * k_mult, Hidden]
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
                        conf = local_response_to_confidence(resp, p, mappings[q_idx])
                        m_val = direct_metric_values[q_idx]
                        align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)

                        result_dict[mult][q_idx] = {
                            "question_idx": q_idx, "response": resp, "confidence": conf,
                            "metric": float(m_val), "alignment": float(align)
                        }

            try:
                # Introspection
                run_batched_mult_sweep(intro_dir, final_layer_results[layer_idx]["introspection"])
                # Controls
                for i_c, ctrl_dir in enumerate(control_dirs):
                    run_batched_mult_sweep(ctrl_dir, final_layer_results[layer_idx]["controls"][f"control_{i_c}"])
            finally:
                hook.remove()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results
    }
    
# ============================================================================
# ABLATION EXPERIMENT
# ============================================================================

def _run_ablation_full_forward(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    num_controls: int,
    cached_directions: Dict,
    option_token_ids: List[int],
    options: List[str],
    mappings: List,
    cached_inputs: List[Dict],
    baseline_results: Optional[List[Dict]] = None,
) -> Dict:
    """Fallback: Full forward pass implementation for INTERVENTION_POSITION='all'."""
    print("Using full forward passes (INTERVENTION_POSITION='all')...")

    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Compute baseline if not provided
    if baseline_results is None:
        print("Computing baseline...")
        baseline_results = [None] * len(questions)
        for batch_indices, batch_inputs in gpu_batches:
            with torch.inference_mode():
                out = model(**batch_inputs)
                logits = out.logits[:, -1, :][:, option_token_ids]
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            for i, q_idx in enumerate(batch_indices):
                p = probs[i]
                resp = options[np.argmax(p)]
                conf = local_response_to_confidence(resp, p, mappings[q_idx])
                m_val = direct_metric_values[q_idx]
                align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)
                baseline_results[q_idx] = {
                    "question_idx": q_idx, "response": resp, "confidence": conf,
                    "metric": float(m_val), "alignment": float(align)
                }

    final_layer_results = {}
    for l in layers:
        final_layer_results[l] = {
            "baseline": baseline_results,
            "introspection_ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    # Process each layer
    for layer_idx in tqdm(layers, desc="Layers"):
        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        intro_dir = cached_directions[layer_idx]["introspection"]
        control_dirs = cached_directions[layer_idx]["controls"]

        # Helper to run all questions for one direction
        def run_ablation_condition(direction_vector, result_storage, key=None):
            hook = AblationHook(direction_vector, pre_normalized=True)
            hook.register(layer_module)
            try:
                for batch_indices, batch_inputs in gpu_batches:
                    with torch.inference_mode():
                        out = model(**batch_inputs)
                        logits = out.logits[:, -1, :][:, option_token_ids]
                        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                    for i, q_idx in enumerate(batch_indices):
                        p = probs[i]
                        resp = options[np.argmax(p)]
                        conf = local_response_to_confidence(resp, p, mappings[q_idx])
                        m_val = direct_metric_values[q_idx]
                        align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)
                        data = {
                            "question_idx": q_idx, "response": resp, "confidence": conf,
                            "metric": float(m_val), "alignment": float(align)
                        }
                        if key:
                            result_storage[key][q_idx] = data
                        else:
                            result_storage[q_idx] = data
            finally:
                hook.remove()

        # Introspection ablation
        run_ablation_condition(intro_dir, final_layer_results[layer_idx]["introspection_ablated"])

        # Control ablations
        for i_c, ctrl_dir in enumerate(control_dirs):
            run_ablation_condition(ctrl_dir, final_layer_results[layer_idx]["controls_ablated"], key=f"control_{i_c}")

    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results
    }


def run_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    directions: Dict,
    num_controls: int,
    use_chat_template: bool,
    baseline_results: Optional[List[Dict]] = None,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """Run ablation experiment with KV cache optimization (Robust Fix)."""

    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

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
    
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = DELEGATE_OPTIONS
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = META_OPTIONS

    if cached_directions is None:
        cached_directions = precompute_direction_tensors(
            directions, layers, num_controls, DEVICE,
            torch.float16 if DEVICE == "cuda" else torch.float32
        )

    if INTERVENTION_POSITION != "last":
        print(f"\nRunning ablation experiment (full forward fallback)...")
        return _run_ablation_full_forward(
            model, tokenizer, questions, direct_metric_values,
            layers, num_controls, cached_directions,
            option_token_ids, options, mappings, cached_inputs, baseline_results
        )

    print(f"\nRunning ablation experiment (KV Cache)...")

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)
    
    # Compute baseline if not provided
    if baseline_results is None:
        print("Computing baseline...")
        baseline_results = [None] * len(questions)
        for batch_indices, batch_inputs in gpu_batches:
            with torch.inference_mode():
                out = model(**batch_inputs)
                logits = out.logits[:, -1, :][:, option_token_ids]
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            for i, q_idx in enumerate(batch_indices):
                p = probs[i]
                resp = options[np.argmax(p)]
                conf = local_response_to_confidence(resp, p, mappings[q_idx])
                m_val = direct_metric_values[q_idx]
                align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)
                baseline_results[q_idx] = {
                    "question_idx": q_idx, "response": resp, "confidence": conf,
                    "metric": float(m_val), "alignment": float(align)
                }

    final_layer_results = {}
    for l in layers:
        final_layer_results[l] = {
            "baseline": baseline_results,
            "introspection_ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    print(f"Processing {len(gpu_batches)} batches (KV-Cached)...")

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

        # Iterate Layers
        for layer_idx in layers:
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer_idx]
            else:
                layer_module = model.model.layers[layer_idx]

            intro_dir = cached_directions[layer_idx]["introspection"]
            control_dirs = cached_directions[layer_idx]["controls"]

            hook = BatchAblationHook()
            hook.register(layer_module)

            def run_single_ablation(direction_vector, result_storage, key=None):
                # 1. Reconstruct Cache
                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                
                current_inputs = inputs_template.copy()
                current_inputs["past_key_values"] = pass_cache
                
                # 2. Set Direction
                dirs_batch = direction_vector.unsqueeze(0).expand(B, -1)
                hook.set_directions(dirs_batch)

                # 3. Run
                with torch.inference_mode():
                    out = model(**current_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf = local_response_to_confidence(resp, p, mappings[q_idx])
                    m_val = direct_metric_values[q_idx]
                    align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)

                    data = {
                        "question_idx": q_idx, "response": resp, "confidence": conf,
                        "metric": float(m_val), "alignment": float(align)
                    }
                    if key: result_storage[key][q_idx] = data
                    else: result_storage[q_idx] = data

            try:
                run_single_ablation(intro_dir, final_layer_results[layer_idx]["introspection_ablated"])
                for i_c, ctrl_dir in enumerate(control_dirs):
                    run_single_ablation(ctrl_dir, final_layer_results[layer_idx]["controls_ablated"], key=f"control_{i_c}")
            finally:
                hook.remove()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results
    }
    
def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and uncertainty metric."""
    # We expect negative correlation for entropy-like metrics: high metric = low confidence
    # For logit_gap etc., sign depends on metric definition
    if len(confidences) < 2 or np.std(confidences) == 0 or np.std(metric_values) == 0:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def analyze_ablation_results(results: Dict) -> Dict:
    """Compute ablation effect statistics with proper statistical testing.

    Statistical improvements:
    1. Pooled null distribution: Collect all control effects across all layers
       to build a larger null distribution for more robust p-values
    2. Per-layer permutation test: Compare introspection effect to layer-specific controls
    3. FDR correction: Benjamini-Hochberg correction for multiple layer testing
    4. Bootstrap CIs: 95% confidence intervals on control effects
    """
    analysis = {
        "layers": results["layers"],
        "num_questions": results["num_questions"],
        "num_controls": results["num_controls"],
        "effects": {},
    }

    # First pass: collect all effects for pooled null distribution
    all_control_corr_changes = []  # Pooled across all layers
    layer_data = {}  # Store extracted data for second pass

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        # Extract data - results now use "metric" key instead of "entropy"
        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_metric = np.array([r["metric"] for r in lr["baseline"]])
        baseline_align = np.array([r["alignment"] for r in lr["baseline"]])

        ablated_conf = np.array([r["confidence"] for r in lr["introspection_ablated"]])
        ablated_metric = np.array([r["metric"] for r in lr["introspection_ablated"]])
        ablated_align = np.array([r["alignment"] for r in lr["introspection_ablated"]])

        # Compute correlations (confidence vs selected metric)
        baseline_corr = compute_correlation(baseline_conf, baseline_metric)
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)

        # Control ablations
        control_corrs = []
        control_aligns = []
        control_confs = []
        for ctrl_key in lr["controls_ablated"]:
            ctrl_conf = np.array([r["confidence"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_metric = np.array([r["metric"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_align = np.array([r["alignment"] for r in lr["controls_ablated"][ctrl_key]])
            control_corrs.append(compute_correlation(ctrl_conf, ctrl_metric))
            control_aligns.append(ctrl_align.mean())
            control_confs.append(ctrl_conf.mean())

        intro_corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]

        # Add to pooled null distribution
        all_control_corr_changes.extend(control_corr_changes)

        # Store for second pass
        layer_data[layer_idx] = {
            "baseline_corr": baseline_corr,
            "baseline_conf": baseline_conf,
            "baseline_metric": baseline_metric,
            "baseline_align": baseline_align,
            "ablated_corr": ablated_corr,
            "ablated_conf": ablated_conf,
            "ablated_align": ablated_align,
            "intro_corr_change": intro_corr_change,
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
            "control_aligns": control_aligns,
            "control_confs": control_confs,
        }

    # Convert pooled null to array for efficient computation
    pooled_null = np.array(all_control_corr_changes)

    # Second pass: compute statistics with pooled null
    raw_p_values = []

    for layer_idx in results["layers"]:
        ld = layer_data[layer_idx]

        avg_control_corr = np.mean(ld["control_corrs"])
        avg_control_align = np.mean(ld["control_aligns"])
        std_control_corr = np.std(ld["control_corr_changes"])

        # Per-layer p-value (original method, kept for comparison)
        n_controls_worse_local = sum(1 for c in ld["control_corr_changes"] if c >= ld["intro_corr_change"])
        p_value_local = (n_controls_worse_local + 1) / (len(ld["control_corrs"]) + 1)

        # Pooled p-value: compare to all control effects across all layers
        # This gives much finer granularity (e.g., with 20 controls × 7 layers = 140 samples)
        n_pooled_worse = np.sum(pooled_null >= ld["intro_corr_change"])
        p_value_pooled = (n_pooled_worse + 1) / (len(pooled_null) + 1)

        # Bootstrap 95% CI for control effect
        n_bootstrap = 1000
        bootstrap_means = []
        ctrl_changes = np.array(ld["control_corr_changes"])
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(ctrl_changes, size=len(ctrl_changes), replace=True)
            bootstrap_means.append(np.mean(boot_sample))
        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)

        # Effect size: how many SDs away from control mean?
        if std_control_corr > 0:
            effect_size_z = (ld["intro_corr_change"] - np.mean(ld["control_corr_changes"])) / std_control_corr
        else:
            effect_size_z = 0.0

        raw_p_values.append((layer_idx, p_value_pooled))

        analysis["effects"][layer_idx] = {
            "baseline": {
                "correlation": ld["baseline_corr"],
                "mean_alignment": float(ld["baseline_align"].mean()),
                "mean_confidence": float(ld["baseline_conf"].mean()),
            },
            "introspection_ablated": {
                "correlation": ld["ablated_corr"],
                "correlation_change": ld["intro_corr_change"],
                "mean_alignment": float(ld["ablated_align"].mean()),
                "alignment_change": float(ld["ablated_align"].mean() - ld["baseline_align"].mean()),
                "mean_confidence": float(ld["ablated_conf"].mean()),
                "p_value_local": p_value_local,  # Per-layer (old method)
                "p_value_pooled": p_value_pooled,  # Pooled null (more powerful)
                "effect_size_z": effect_size_z,  # Z-score vs controls
            },
            "control_ablated": {
                "correlation_mean": avg_control_corr,
                "correlation_std": float(np.std(ld["control_corrs"])),
                "correlation_change_mean": float(np.mean(ld["control_corr_changes"])),
                "correlation_change_std": std_control_corr,
                "correlation_change_ci95": [float(ci_low), float(ci_high)],
                "mean_alignment": avg_control_align,
                "alignment_change": avg_control_align - float(ld["baseline_align"].mean()),
            },
            "individual_controls": {
                f"control_{i}": {
                    "correlation": ld["control_corrs"][i],
                    "correlation_change": ld["control_corr_changes"][i],
                }
                for i in range(len(ld["control_corrs"]))
            },
        }

    # FDR correction (Benjamini-Hochberg)
    # Sort p-values, compute adjusted p-values
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
        # BH adjusted p-value: p * n / rank, but capped at 1 and monotonic
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer_idx] = adjusted

    # Make monotonic (each p-value >= all smaller p-values)
    prev_adjusted = 0.0
    for layer_idx, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer_idx] = max(fdr_adjusted[layer_idx], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer_idx]

    # Add FDR-adjusted p-values to analysis
    for layer_idx in results["layers"]:
        analysis["effects"][layer_idx]["introspection_ablated"]["p_value_fdr"] = fdr_adjusted[layer_idx]

    # Summary statistics
    significant_layers_pooled = [l for l in results["layers"]
                                  if analysis["effects"][l]["introspection_ablated"]["p_value_pooled"] < 0.05]
    significant_layers_fdr = [l for l in results["layers"]
                              if analysis["effects"][l]["introspection_ablated"]["p_value_fdr"] < 0.05]

    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_pooled_p05": significant_layers_pooled,
        "significant_layers_fdr_p05": significant_layers_fdr,
        "n_significant_pooled": len(significant_layers_pooled),
        "n_significant_fdr": len(significant_layers_fdr),
    }

    return analysis


def get_expected_slope_sign(metric: str) -> int:
    """
    Get the expected sign of the confidence slope for a given metric.

    The probe direction points toward *increasing* the metric value.
    - entropy: HIGH = uncertain → steering +direction should DECREASE confidence → expected slope < 0
    - logit_gap, top_prob, margin, top_logit: HIGH = confident → expected slope > 0

    Returns:
        +1 if +direction should increase confidence
        -1 if +direction should decrease confidence
    """
    if metric == "entropy":
        return -1  # +direction = more uncertain = less confident
    else:
        return +1  # +direction = more confident


def analyze_results(results: Dict, metric: str = None) -> Dict:
    """Compute summary statistics with statistical significance testing.

    Args:
        results: Raw steering results
        metric: The uncertainty metric used (for sign interpretation). If None,
                sign interpretation is skipped.

    Statistical approach (mirrors ablation analysis):
    1. Compute confidence slope for introspection direction
    2. Compute confidence slope for each control direction (not just average)
    3. Pool all control slopes across layers for null distribution
    4. Per-layer p-value: fraction of control slopes with |slope| >= |introspection slope|
    5. FDR correction for multiple testing
    6. Bootstrap CIs for control slope distribution
    """
    analysis = {
        "layers": results["layers"],
        "multipliers": results["multipliers"],
        "metric": metric,
        "effects": {},
    }

    multipliers = results["multipliers"]

    # First pass: collect all slopes for pooled null distribution
    all_control_slopes = []  # Pooled across all layers
    layer_data = {}  # Store for second pass

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        baseline_align = np.mean([r["alignment"] for r in lr["baseline"]])
        baseline_conf = np.mean([r["confidence"] for r in lr["baseline"]])

        # Introspection: confidence change per multiplier
        intro_conf_changes = []
        intro_align_changes = []
        for mult in multipliers:
            intro_conf = np.mean([r["confidence"] for r in lr["introspection"][mult]])
            intro_align = np.mean([r["alignment"] for r in lr["introspection"][mult]])
            intro_conf_changes.append(intro_conf - baseline_conf)
            intro_align_changes.append(intro_align - baseline_align)

        intro_conf_slope = np.polyfit(multipliers, intro_conf_changes, 1)[0]
        intro_align_slope = np.polyfit(multipliers, intro_align_changes, 1)[0]

        # Per-control slopes (not just average)
        control_slopes = []
        control_align_slopes = []
        for ctrl_key in lr["controls"]:
            ctrl_conf_changes = []
            ctrl_align_changes = []
            for mult in multipliers:
                ctrl_conf = np.mean([r["confidence"] for r in lr["controls"][ctrl_key][mult]])
                ctrl_align = np.mean([r["alignment"] for r in lr["controls"][ctrl_key][mult]])
                ctrl_conf_changes.append(ctrl_conf - baseline_conf)
                ctrl_align_changes.append(ctrl_align - baseline_align)
            ctrl_slope = np.polyfit(multipliers, ctrl_conf_changes, 1)[0]
            ctrl_align_slope = np.polyfit(multipliers, ctrl_align_changes, 1)[0]
            control_slopes.append(ctrl_slope)
            control_align_slopes.append(ctrl_align_slope)

        # Add to pooled null
        all_control_slopes.extend(control_slopes)

        # Build by_multiplier dict for backward compatibility
        effects = {"introspection": {}, "control_avg": {}}
        for mult in multipliers:
            intro_conf = np.mean([r["confidence"] for r in lr["introspection"][mult]])
            intro_align = np.mean([r["alignment"] for r in lr["introspection"][mult]])
            effects["introspection"][mult] = {
                "alignment": float(intro_align),
                "alignment_change": float(intro_align - baseline_align),
                "confidence": float(intro_conf),
                "confidence_change": float(intro_conf - baseline_conf),
            }
            ctrl_confs = []
            ctrl_aligns = []
            for ctrl_key in lr["controls"]:
                ctrl_confs.extend([r["confidence"] for r in lr["controls"][ctrl_key][mult]])
                ctrl_aligns.extend([r["alignment"] for r in lr["controls"][ctrl_key][mult]])
            effects["control_avg"][mult] = {
                "alignment": float(np.mean(ctrl_aligns)),
                "alignment_change": float(np.mean(ctrl_aligns) - baseline_align),
                "confidence": float(np.mean(ctrl_confs)),
                "confidence_change": float(np.mean(ctrl_confs) - baseline_conf),
            }

        layer_data[layer_idx] = {
            "intro_conf_slope": intro_conf_slope,
            "intro_align_slope": intro_align_slope,
            "control_slopes": control_slopes,
            "control_align_slopes": control_align_slopes,
            "baseline_align": baseline_align,
            "baseline_conf": baseline_conf,
            "effects": effects,
        }

    # Convert pooled null to array
    pooled_null = np.array(all_control_slopes)
    pooled_null_abs = np.abs(pooled_null)

    # Second pass: compute p-values and statistics
    raw_p_values = []

    for layer_idx in results["layers"]:
        ld = layer_data[layer_idx]

        intro_slope = ld["intro_conf_slope"]
        intro_slope_abs = abs(intro_slope)
        control_slopes = np.array(ld["control_slopes"])

        # Average control slope (for backward compatibility)
        avg_ctrl_slope = float(np.mean(control_slopes))
        std_ctrl_slope = float(np.std(control_slopes))

        # Per-layer p-value: two-tailed test (|introspection| vs |controls|)
        n_controls_larger_local = np.sum(np.abs(control_slopes) >= intro_slope_abs)
        p_value_local = (n_controls_larger_local + 1) / (len(control_slopes) + 1)

        # Pooled p-value: compare to all control slopes across all layers
        n_pooled_larger = np.sum(pooled_null_abs >= intro_slope_abs)
        p_value_pooled = (n_pooled_larger + 1) / (len(pooled_null) + 1)

        # Bootstrap 95% CI for control slope magnitude
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(control_slopes, size=len(control_slopes), replace=True)
            bootstrap_means.append(np.mean(np.abs(boot_sample)))
        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)

        # Effect size: how many SDs away from control mean?
        if std_ctrl_slope > 0:
            effect_size_z = (intro_slope_abs - np.mean(np.abs(control_slopes))) / np.std(np.abs(control_slopes))
        else:
            effect_size_z = 0.0

        raw_p_values.append((layer_idx, p_value_pooled))

        analysis["effects"][layer_idx] = {
            "by_multiplier": ld["effects"],
            "slopes": {
                "introspection": float(intro_slope),
                "control_avg": avg_ctrl_slope,
                "control_std": std_ctrl_slope,
                "introspection_alignment": float(ld["intro_align_slope"]),
                "control_avg_alignment": float(np.mean(ld["control_align_slopes"])),
            },
            "statistics": {
                "p_value_local": float(p_value_local),
                "p_value_pooled": float(p_value_pooled),
                "effect_size_z": float(effect_size_z),
                "control_slope_ci95": [float(ci_low), float(ci_high)],
                "n_controls": len(control_slopes),
            },
            "baseline_alignment": float(ld["baseline_align"]),
            "baseline_confidence": float(ld["baseline_conf"]),
        }

    # FDR correction (Benjamini-Hochberg)
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer_idx] = adjusted

    # Make monotonic
    prev_adjusted = 0.0
    for layer_idx, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer_idx] = max(fdr_adjusted[layer_idx], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer_idx]

    # Add FDR-adjusted p-values
    for layer_idx in results["layers"]:
        analysis["effects"][layer_idx]["statistics"]["p_value_fdr"] = fdr_adjusted[layer_idx]

    # Summary statistics
    significant_layers_pooled = [l for l in results["layers"]
                                  if analysis["effects"][l]["statistics"]["p_value_pooled"] < 0.05]
    significant_layers_fdr = [l for l in results["layers"]
                              if analysis["effects"][l]["statistics"]["p_value_fdr"] < 0.05]

    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_pooled_p05": significant_layers_pooled,
        "significant_layers_fdr_p05": significant_layers_fdr,
        "n_significant_pooled": len(significant_layers_pooled),
        "n_significant_fdr": len(significant_layers_fdr),
    }

    # Add sign interpretation metadata
    if metric:
        expected_sign = get_expected_slope_sign(metric)
        analysis["sign_interpretation"] = {
            "metric": metric,
            "expected_slope_sign": expected_sign,
            "expected_slope_sign_str": "positive" if expected_sign > 0 else "negative",
            "explanation": (
                f"For {metric}, +direction should → "
                f"{'higher' if expected_sign > 0 else 'lower'} confidence"
            ),
        }

        # Check best layer's sign
        if analysis["effects"]:
            best_layer = max(
                analysis["layers"],
                key=lambda l: abs(analysis["effects"][l]["slopes"]["introspection"])
            )
            best_slope = analysis["effects"][best_layer]["slopes"]["introspection"]
            actual_sign = 1 if best_slope > 0 else -1
            analysis["sign_interpretation"]["best_layer"] = best_layer
            analysis["sign_interpretation"]["actual_slope_sign"] = actual_sign
            analysis["sign_interpretation"]["sign_matches_expected"] = (actual_sign == expected_sign)

    return analysis


def plot_results(analysis: Dict, output_prefix: str):
    """Create visualizations with statistical significance."""
    layers = analysis["layers"]
    multipliers = analysis["multipliers"]

    if not layers:
        print("  Skipping plot - no layers to visualize")
        return

    # Check if we have statistics (new format)
    has_stats = "statistics" in analysis["effects"].get(layers[0], {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Confidence slopes by layer with error bars and significance
    ax1 = axes[0]
    intro_slopes = [analysis["effects"][l]["slopes"]["introspection"] for l in layers]
    ctrl_slopes = [analysis["effects"][l]["slopes"]["control_avg"] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    if has_stats:
        ctrl_stds = [analysis["effects"][l]["slopes"]["control_std"] for l in layers]
        # Color bars by significance
        intro_colors = []
        for l in layers:
            p_fdr = analysis["effects"][l]["statistics"]["p_value_fdr"]
            if p_fdr < 0.01:
                intro_colors.append('darkgreen')
            elif p_fdr < 0.05:
                intro_colors.append('green')
            else:
                intro_colors.append('lightgreen')

        ax1.bar(x - width/2, intro_slopes, width, label='Introspection', color=intro_colors, alpha=0.8)
        ax1.bar(x + width/2, ctrl_slopes, width, yerr=ctrl_stds, label='Control (avg±SD)',
                color='gray', alpha=0.7, capsize=3)

        # Add significance markers
        for i, l in enumerate(layers):
            p_fdr = analysis["effects"][l]["statistics"]["p_value_fdr"]
            y_pos = max(abs(intro_slopes[i]), abs(ctrl_slopes[i]) + ctrl_stds[i]) * 1.1
            if intro_slopes[i] < 0:
                y_pos = -y_pos
            if p_fdr < 0.01:
                ax1.text(x[i] - width/2, y_pos, '**', ha='center', va='bottom' if y_pos > 0 else 'top', fontsize=12, fontweight='bold')
            elif p_fdr < 0.05:
                ax1.text(x[i] - width/2, y_pos, '*', ha='center', va='bottom' if y_pos > 0 else 'top', fontsize=12, fontweight='bold')
    else:
        ax1.bar(x - width/2, intro_slopes, width, label='Introspection', color='green', alpha=0.7)
        ax1.bar(x + width/2, ctrl_slopes, width, label='Control (avg)', color='gray', alpha=0.7)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Confidence Slope (Δconf / Δmult)")
    ax1.set_title("Steering Effect on Confidence" + (" (* p<.05, ** p<.01 FDR)" if has_stats else ""))
    ax1.legend()

    # Plot 2: Best layer detail - show confidence change
    best_layer = max(layers, key=lambda l: abs(analysis["effects"][l]["slopes"]["introspection"]))
    ax2 = axes[1]

    intro_conf = [analysis["effects"][best_layer]["by_multiplier"]["introspection"][m]["confidence_change"] for m in multipliers]
    ctrl_conf = [analysis["effects"][best_layer]["by_multiplier"]["control_avg"][m]["confidence_change"] for m in multipliers]

    ax2.plot(multipliers, intro_conf, 'o-', label='Introspection', linewidth=2, color='green')
    ax2.plot(multipliers, ctrl_conf, '^--', label='Control', linewidth=2, color='gray', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Steering Multiplier")
    ax2.set_ylabel("Δ Confidence")
    ax2.set_title(f"Confidence Change (Layer {best_layer})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary
    ax3 = axes[2]
    ax3.axis('off')

    intro_slope = analysis["effects"][best_layer]["slopes"]["introspection"]
    ctrl_slope = analysis["effects"][best_layer]["slopes"]["control_avg"]
    metric = analysis.get("metric", "unknown")

    summary = f"""
STEERING EXPERIMENT SUMMARY

Metric: {metric}
Best Layer: {best_layer}
  Confidence slope: {intro_slope:.4f}
  Control slope: {ctrl_slope:.4f}
  Difference: {abs(intro_slope) - abs(ctrl_slope):.4f}
"""

    if has_stats:
        best_stats = analysis["effects"][best_layer]["statistics"]
        summary += f"""
Statistics:
  p-value (pooled): {best_stats['p_value_pooled']:.4f}
  p-value (FDR): {best_stats['p_value_fdr']:.4f}
  Effect size (Z): {best_stats['effect_size_z']:.2f}
"""
        # Add overall summary
        summ = analysis.get("summary", {})
        if summ:
            n_sig = summ.get("n_significant_fdr", 0)
            summary += f"""
Significant layers: {n_sig}/{len(layers)} (FDR<0.05)
"""

    summary += """
Interpretation:
"""
    # Check if introspection direction causes systematic confidence shift
    if abs(intro_slope) > abs(ctrl_slope) + 0.01:
        direction_str = "lower" if intro_slope < 0 else "higher"
        summary += f"""  ✓ Steering shifts confidence
  +mult → {direction_str} confidence"""

        # Check sign against expectation
        if metric and metric != "unknown":
            expected_sign = get_expected_slope_sign(metric)
            actual_sign = 1 if intro_slope > 0 else -1
            if actual_sign == expected_sign:
                summary += f"""
  ✓ Sign correct for {metric}"""
            else:
                summary += f"""
  ⚠ Sign OPPOSITE!"""
    elif abs(intro_slope) > 0.01:
        summary += """  ⚠ Weak steering effect"""
    else:
        summary += """  ✗ No steering effect"""

    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_steering_results.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_steering_results.png")
    plt.close()


def print_summary(analysis: Dict):
    """Print summary of results with statistical significance."""
    print("\n" + "=" * 70)
    print("STEERING EXPERIMENT RESULTS")
    print("=" * 70)

    if not analysis["layers"]:
        print("\n⚠ No layers were tested - check layer selection criteria")
        return

    metric = analysis.get("metric")
    if metric:
        expected_sign = get_expected_slope_sign(metric)
        expected_direction = "negative" if expected_sign < 0 else "positive"
        print(f"\nMetric: {metric}")
        print(f"Expected slope sign: {expected_direction} (+direction → {'less' if expected_sign < 0 else 'more'} confident)")

    # Check if we have statistics (new format)
    has_stats = "statistics" in analysis["effects"].get(analysis["layers"][0], {})

    if has_stats:
        print("\n--- Confidence Slopes by Layer (with significance) ---")
        print(f"{'Layer':<8} {'Intro Slope':<12} {'Ctrl (±SD)':<16} {'p(pooled)':<10} {'p(FDR)':<10} {'Z-score':<8} {'Sig':<5}")
        print("-" * 75)

        for layer in analysis["layers"]:
            s = analysis["effects"][layer]["slopes"]
            stats = analysis["effects"][layer]["statistics"]
            p_pooled = stats["p_value_pooled"]
            p_fdr = stats["p_value_fdr"]
            z = stats["effect_size_z"]

            # Significance markers
            if p_fdr < 0.01:
                sig = "**"
            elif p_fdr < 0.05:
                sig = "*"
            elif p_pooled < 0.05:
                sig = "~"  # nominally significant but not FDR-corrected
            else:
                sig = ""

            ctrl_str = f"{s['control_avg']:.4f}±{s['control_std']:.4f}"
            print(f"{layer:<8} {s['introspection']:<12.4f} {ctrl_str:<16} {p_pooled:<10.4f} {p_fdr:<10.4f} {z:<8.2f} {sig:<5}")

        # Print summary
        summary = analysis.get("summary", {})
        if summary:
            print(f"\nPooled null size: {summary.get('pooled_null_size', 'N/A')} control slopes")
            n_sig_pooled = summary.get("n_significant_pooled", 0)
            n_sig_fdr = summary.get("n_significant_fdr", 0)
            print(f"Significant layers (p<0.05 pooled): {n_sig_pooled}/{len(analysis['layers'])}")
            print(f"Significant layers (FDR<0.05): {n_sig_fdr}/{len(analysis['layers'])}")
            if summary.get("significant_layers_fdr_p05"):
                print(f"  FDR-significant: {summary['significant_layers_fdr_p05']}")

        print("\nLegend: ** p_FDR<0.01, * p_FDR<0.05, ~ p_pooled<0.05 (not FDR-corrected)")
    else:
        # Old format without statistics
        print("\n--- Confidence Slopes by Layer ---")
        print(f"{'Layer':<8} {'Introspection':<15} {'Control':<15}")
        print("-" * 40)

        for layer in analysis["layers"]:
            s = analysis["effects"][layer]["slopes"]
            print(f"{layer:<8} {s['introspection']:<15.4f} {s['control_avg']:<15.4f}")

    # Best layer - pick largest magnitude slope
    best_layer = max(analysis["layers"], key=lambda l: abs(analysis["effects"][l]["slopes"]["introspection"]))
    best_intro = analysis["effects"][best_layer]["slopes"]["introspection"]
    best_ctrl = analysis["effects"][best_layer]["slopes"]["control_avg"]

    print(f"\nStrongest steering effect: Layer {best_layer}")
    print(f"  Confidence slope: {best_intro:.4f}")
    print(f"  Control slope: {best_ctrl:.4f}")

    if has_stats:
        best_stats = analysis["effects"][best_layer]["statistics"]
        print(f"  p-value (pooled): {best_stats['p_value_pooled']:.4f}")
        print(f"  p-value (FDR): {best_stats['p_value_fdr']:.4f}")
        print(f"  Effect size (Z): {best_stats['effect_size_z']:.2f}")

    if abs(best_intro) > abs(best_ctrl) + 0.01:
        direction = "lower" if best_intro < 0 else "higher"
        print(f"\n✓ Steering systematically shifts confidence {direction}!")
        print("  Effect stronger than random controls.")

        # Check if sign matches expectation
        if metric:
            expected_sign = get_expected_slope_sign(metric)
            actual_sign = 1 if best_intro > 0 else -1
            if actual_sign == expected_sign:
                print(f"  ✓ Sign matches expectation for {metric} (direction transfers correctly)")
            else:
                print(f"  ⚠ Sign is OPPOSITE to expectation for {metric}!")
                print(f"    This suggests the direction may not transfer from direct→meta context,")
                print(f"    or the representation differs between contexts.")
    elif abs(best_intro) > 0.01:
        print("\n⚠ Weak effect, not clearly separable from controls")
    else:
        print("\n✗ No steering effect found")


def plot_ablation_results(analysis: Dict, output_prefix: str):
    """Create improved ablation visualizations.

    Three panels:
    1. Absolute correlations (baseline, introspection-ablated, control-ablated with CI)
    2. Effect size with 95% CI - shows if introspection effect differs from controls
    3. Distribution plot - violin/box of control effects with introspection overlay
    """
    layers = analysis["layers"]

    if not layers:
        print("  Skipping ablation plot - no layers to visualize")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ==========================================================================
    # Plot 1: Absolute correlation values (not deltas)
    # Shows baseline, introspection-ablated, and control-ablated correlations
    # ==========================================================================
    ax1 = axes[0]

    baseline_corrs = [analysis["effects"][l]["baseline"]["correlation"] for l in layers]
    intro_corrs = [analysis["effects"][l]["introspection_ablated"]["correlation"] for l in layers]
    ctrl_corrs = [analysis["effects"][l]["control_ablated"]["correlation_mean"] for l in layers]
    ctrl_stds = [analysis["effects"][l]["control_ablated"]["correlation_std"] for l in layers]

    x = np.arange(len(layers))

    # Plot lines with markers
    ax1.plot(x, baseline_corrs, 'o-', label='Baseline (no ablation)', color='blue', linewidth=2, markersize=8)
    ax1.plot(x, intro_corrs, 's-', label='Introspection ablated', color='red', linewidth=2, markersize=8)
    ax1.errorbar(x, ctrl_corrs, yerr=ctrl_stds, fmt='^--', label='Control ablated (mean±SD)',
                 color='gray', linewidth=1.5, markersize=7, capsize=3, alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (confidence vs metric)")
    ax1.set_title("Correlation Values by Condition")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add annotation about expected direction
    ax1.text(0.02, 0.02, "Negative corr = well-calibrated\n(high metric → low confidence)",
             transform=ax1.transAxes, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ==========================================================================
    # Plot 2: Effect size with 95% CI
    # Shows (introspection_change - control_mean) with CI from bootstrap
    # ==========================================================================
    ax2 = axes[1]

    # Compute differential effect: introspection effect minus control effect
    diff_effects = []
    diff_ci_low = []
    diff_ci_high = []
    p_values_fdr = []

    for l in layers:
        intro_change = analysis["effects"][l]["introspection_ablated"]["correlation_change"]
        ctrl_mean = analysis["effects"][l]["control_ablated"]["correlation_change_mean"]
        ctrl_ci = analysis["effects"][l]["control_ablated"]["correlation_change_ci95"]

        # Differential effect
        diff = intro_change - ctrl_mean
        diff_effects.append(diff)

        # CI on the difference (approximate: introspection is fixed, so CI comes from control variance)
        diff_ci_low.append(intro_change - ctrl_ci[1])  # intro - upper_ctrl = lower bound of diff
        diff_ci_high.append(intro_change - ctrl_ci[0])  # intro - lower_ctrl = upper bound of diff

        p_values_fdr.append(analysis["effects"][l]["introspection_ablated"].get("p_value_fdr", 1.0))

    # Color bars by significance
    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_values_fdr]

    # Plot bars with error bars
    bars = ax2.bar(x, diff_effects, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.errorbar(x, diff_effects,
                 yerr=[np.array(diff_effects) - np.array(diff_ci_low),
                       np.array(diff_ci_high) - np.array(diff_effects)],
                 fmt='none', color='black', capsize=4, capthick=1.5)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Differential Effect\n(intro_Δcorr − control_Δcorr)")
    ax2.set_title("Introspection Effect vs Controls (with 95% CI)")
    ax2.grid(True, alpha=0.3, axis='y')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='p < 0.05 (FDR)'),
        Patch(facecolor='orange', alpha=0.7, edgecolor='black', label='p < 0.10 (FDR)'),
        Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='n.s.'),
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=9)

    # Annotation
    ax2.text(0.02, 0.98, "Positive = ablation hurts\nintrospection more than controls",
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ==========================================================================
    # Plot 3: Distribution plot - show where introspection falls in null distribution
    # ==========================================================================
    ax3 = axes[2]

    # Collect control correlation changes for each layer
    positions = []
    control_data = []
    intro_points = []

    for i, l in enumerate(layers):
        ctrl_changes = [
            analysis["effects"][l]["individual_controls"][f"control_{j}"]["correlation_change"]
            for j in range(len(analysis["effects"][l]["individual_controls"]))
        ]
        control_data.append(ctrl_changes)
        intro_points.append(analysis["effects"][l]["introspection_ablated"]["correlation_change"])
        positions.append(i)

    # Create violin plot for controls
    parts = ax3.violinplot(control_data, positions=positions, showmeans=True, showmedians=False)

    # Style the violins
    for pc in parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_edgecolor('gray')
        pc.set_alpha(0.7)
    # Style stat lines (keys may vary by matplotlib version)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('gray')
        parts['cmeans'].set_linewidth(2)
    for key in ['cbars', 'cmins', 'cmaxs']:
        if key in parts:
            parts[key].set_color('gray')

    # Overlay introspection points
    for i, (pos, intro_val, p_val) in enumerate(zip(positions, intro_points, p_values_fdr)):
        color = 'red' if p_val < 0.05 else 'orange' if p_val < 0.1 else 'darkred'
        marker = '*' if p_val < 0.05 else 'o'
        size = 200 if p_val < 0.05 else 100
        ax3.scatter(pos, intro_val, color=color, s=size, marker=marker, zorder=5,
                    edgecolor='black', linewidth=1)

    ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(layers)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Δ Correlation (ablated − baseline)")
    ax3.set_title("Introspection Effect in Null Distribution")
    ax3.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='gray', alpha=0.7, label='Control distribution'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15,
               markeredgecolor='black', label='Introspection (p<0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10,
               markeredgecolor='black', label='Introspection (n.s.)'),
    ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_ablation_results.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_ablation_results.png")
    plt.close()

    # ==========================================================================
    # Also create a summary figure with text
    # ==========================================================================
    fig2, ax_summary = plt.subplots(1, 1, figsize=(8, 6))
    ax_summary.axis('off')

    # Get summary statistics
    summary_stats = analysis.get("summary", {})
    sig_pooled = summary_stats.get("significant_layers_pooled_p05", [])
    sig_fdr = summary_stats.get("significant_layers_fdr_p05", [])
    pooled_n = summary_stats.get("pooled_null_size", 0)

    # Find best layer by FDR p-value
    best_layer = min(layers, key=lambda l: analysis["effects"][l]["introspection_ablated"].get("p_value_fdr", 1.0))
    best_stats = analysis["effects"][best_layer]
    best_p_fdr = best_stats["introspection_ablated"].get("p_value_fdr", 1.0)
    best_p_pooled = best_stats["introspection_ablated"].get("p_value_pooled", 1.0)
    best_effect_z = best_stats["introspection_ablated"].get("effect_size_z", 0.0)

    summary_text = f"""
ABLATION ANALYSIS SUMMARY
{'='*50}

Statistical Method:
  • Pooled null distribution: {pooled_n} control effects
    (all layers × all control directions)
  • Multiple comparisons: Benjamini-Hochberg FDR correction
  • Bootstrap 95% CIs on control effects (n=1000)

Results:
  • Layers tested: {len(layers)}
  • Significant (pooled p<0.05): {len(sig_pooled)} layers
    {sig_pooled if sig_pooled else 'None'}
  • Significant (FDR p<0.05): {len(sig_fdr)} layers
    {sig_fdr if sig_fdr else 'None'}

Best Layer: {best_layer}
  • Baseline correlation: {best_stats['baseline']['correlation']:.4f}
  • After introspection ablation: {best_stats['introspection_ablated']['correlation']:.4f}
  • Δcorr (introspection): {best_stats['introspection_ablated']['correlation_change']:.4f}
  • Δcorr (controls mean): {best_stats['control_ablated']['correlation_change_mean']:.4f}
  • Effect size (Z): {best_effect_z:.2f} SD
  • p-value (pooled): {best_p_pooled:.4f}
  • p-value (FDR-adjusted): {best_p_fdr:.4f}

Interpretation:
"""
    if len(sig_fdr) > 0:
        summary_text += f"""  ✓ SIGNIFICANT after FDR correction
  {len(sig_fdr)} layer(s) show introspection ablation
  degrades calibration more than random directions.
  This is evidence for a causal role of the direction."""
    elif len(sig_pooled) > 0:
        summary_text += f"""  ⚠ Significant before FDR correction only
  {len(sig_pooled)} layer(s) significant at pooled p<0.05
  but not after multiple comparison correction.
  Suggestive but not definitive evidence."""
    elif best_p_pooled < 0.1:
        summary_text += """  ⚠ Marginal effect (p < 0.10)
  Some trend toward introspection ablation
  having larger effect, but not significant.
  May need more statistical power."""
    else:
        summary_text += """  ✗ No significant effect detected
  Introspection ablation does not clearly
  differ from control ablations.
  Direction may not be causally involved."""

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.savefig(f"{output_prefix}_ablation_summary.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_ablation_summary.png")
    plt.close()


def plot_other_confidence_comparison(
    steering_analysis: Optional[Dict],
    ablation_analysis: Optional[Dict],
    output_prefix: str
):
    """
    Plot comparison of self-confidence vs other-confidence effects.

    Creates a figure showing how interventions affect self-reported confidence
    vs "other-confidence" (human difficulty estimation) - a control task.

    If the intervention is introspection-specific, it should affect self-confidence
    more than other-confidence.
    """
    has_steering = steering_analysis is not None and len(steering_analysis) > 0
    has_ablation = ablation_analysis is not None and len(ablation_analysis) > 0

    if not has_steering and not has_ablation:
        print("No other-confidence data to plot")
        return

    n_plots = (1 if has_steering else 0) + (1 if has_ablation else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Steering comparison
    if has_steering:
        ax = axes[plot_idx]
        layers = sorted([int(k) for k in steering_analysis.keys()])
        x = np.arange(len(layers))
        width = 0.35

        self_effects = [steering_analysis[str(l)]["self_effect_mean"] for l in layers]
        other_effects = [steering_analysis[str(l)]["other_effect_mean"] for l in layers]
        ratios = [steering_analysis[str(l)]["self_vs_other_ratio"] for l in layers]

        bars1 = ax.bar(x - width/2, self_effects, width, label='Self-confidence', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, other_effects, width, label='Other-confidence', color='coral', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Δ Confidence (steered - baseline)')
        ax.set_title('Steering: Self vs Other Confidence Effect')
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers])
        ax.legend()

        # Add ratio annotations
        for i, (l, ratio) in enumerate(zip(layers, ratios)):
            if not np.isinf(ratio):
                ax.annotate(f'{ratio:.1f}x', xy=(i, max(self_effects[i], other_effects[i])),
                           ha='center', va='bottom', fontsize=8, color='gray')

        # Add summary text
        mean_ratio = np.mean([r for r in ratios if not np.isinf(r)])
        if mean_ratio > 2.0:
            interpretation = "Introspection-specific"
        elif mean_ratio > 1.2:
            interpretation = "Self > Other"
        elif mean_ratio > 0.8:
            interpretation = "Similar effect"
        else:
            interpretation = "Other > Self (!)"
        ax.text(0.02, 0.98, f'Mean ratio: {mean_ratio:.2f}x\n{interpretation}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx += 1

    # Ablation comparison
    if has_ablation:
        ax = axes[plot_idx]
        layers = sorted([int(k) for k in ablation_analysis.keys()])
        x = np.arange(len(layers))
        width = 0.35

        self_effects = [ablation_analysis[str(l)]["self_effect_mean_abs"] for l in layers]
        other_effects = [ablation_analysis[str(l)]["other_effect_mean_abs"] for l in layers]
        ratios = [ablation_analysis[str(l)]["self_vs_other_ratio"] for l in layers]

        bars1 = ax.bar(x - width/2, self_effects, width, label='Self-confidence', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, other_effects, width, label='Other-confidence', color='coral', alpha=0.8)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean |Δ Confidence| (ablated vs baseline)')
        ax.set_title('Ablation: Self vs Other Confidence Effect')
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers])
        ax.legend()

        # Add ratio annotations
        for i, (l, ratio) in enumerate(zip(layers, ratios)):
            if not np.isinf(ratio):
                ax.annotate(f'{ratio:.1f}x', xy=(i, max(self_effects[i], other_effects[i])),
                           ha='center', va='bottom', fontsize=8, color='gray')

        # Add summary text
        mean_ratio = np.mean([r for r in ratios if not np.isinf(r)])
        if mean_ratio > 2.0:
            interpretation = "Introspection-specific"
        elif mean_ratio > 1.2:
            interpretation = "Self > Other"
        elif mean_ratio > 0.8:
            interpretation = "Similar effect"
        else:
            interpretation = "Other > Self (!)"
        ax.text(0.02, 0.98, f'Mean ratio: {mean_ratio:.2f}x\n{interpretation}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_other_confidence_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_other_confidence_comparison.png")
    plt.close()


def print_ablation_summary(analysis: Dict):
    """Print summary of ablation results with improved statistics."""
    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENT RESULTS")
    print("=" * 70)

    if not analysis["layers"]:
        print("\n⚠ No layers were tested - check layer selection criteria")
        return

    # Get summary stats
    summary = analysis.get("summary", {})
    pooled_n = summary.get("pooled_null_size", 0)
    sig_pooled = summary.get("significant_layers_pooled_p05", [])
    sig_fdr = summary.get("significant_layers_fdr_p05", [])

    print(f"\nStatistical method: Pooled null distribution ({pooled_n} control effects)")
    print("                    + Benjamini-Hochberg FDR correction")

    print("\n--- Correlation by Layer ---")
    print(f"{'Layer':<6} {'Baseline':<10} {'Ablated':<10} {'Ctrl±SD':<14} {'Δcorr':<8} {'EffectZ':<8} {'p(pool)':<8} {'p(FDR)':<8}")
    print("-" * 82)

    for layer in analysis["layers"]:
        e = analysis["effects"][layer]
        ctrl_mean = e['control_ablated']['correlation_mean']
        ctrl_std = e['control_ablated']['correlation_std']
        intro_change = e['introspection_ablated']['correlation_change']
        effect_z = e['introspection_ablated'].get('effect_size_z', 0.0)
        p_pooled = e['introspection_ablated'].get('p_value_pooled', float('nan'))
        p_fdr = e['introspection_ablated'].get('p_value_fdr', float('nan'))

        # Mark significant layers
        sig_marker = "**" if p_fdr < 0.05 else "*" if p_pooled < 0.05 else ""

        print(f"{layer:<6} {e['baseline']['correlation']:<10.4f} "
              f"{e['introspection_ablated']['correlation']:<10.4f} "
              f"{ctrl_mean:.3f}±{ctrl_std:.3f}  "
              f"{intro_change:<8.4f} {effect_z:<8.2f} "
              f"{p_pooled:<8.4f} {p_fdr:<8.4f} {sig_marker}")

    # Find best layer by FDR p-value
    best_layer = min(
        analysis["layers"],
        key=lambda l: analysis["effects"][l]["introspection_ablated"].get("p_value_fdr", 1.0)
    )
    best_stats = analysis["effects"][best_layer]
    best_p_fdr = best_stats["introspection_ablated"].get("p_value_fdr", 1.0)
    best_p_pooled = best_stats["introspection_ablated"].get("p_value_pooled", 1.0)
    best_effect_z = best_stats["introspection_ablated"].get("effect_size_z", 0.0)

    print(f"\n--- Summary ---")
    print(f"Significant layers (pooled p<0.05): {len(sig_pooled)} {sig_pooled if sig_pooled else ''}")
    print(f"Significant layers (FDR p<0.05):    {len(sig_fdr)} {sig_fdr if sig_fdr else ''}")

    print(f"\nBest layer by FDR p-value: Layer {best_layer}")
    print(f"  Baseline correlation:     {best_stats['baseline']['correlation']:.4f}")
    print(f"  After introspection abl:  {best_stats['introspection_ablated']['correlation']:.4f}")
    print(f"  Δcorr (introspection):    {best_stats['introspection_ablated']['correlation_change']:.4f}")
    print(f"  Δcorr (controls mean):    {best_stats['control_ablated']['correlation_change_mean']:.4f}")
    print(f"  Effect size (Z-score):    {best_effect_z:.2f} SD from control mean")
    print(f"  p-value (pooled):         {best_p_pooled:.4f}")
    print(f"  p-value (FDR-adjusted):   {best_p_fdr:.4f}")

    # Interpretation
    if len(sig_fdr) > 0:
        print(f"\n✓ SIGNIFICANT CAUSAL EFFECT (FDR-corrected p < 0.05)")
        print(f"  {len(sig_fdr)} layer(s) survive multiple comparison correction.")
        print("  Ablating the introspection direction degrades calibration")
        print("  significantly more than ablating random orthogonal directions.")
        print("  This is evidence the direction is causally involved in confidence.")
    elif len(sig_pooled) > 0:
        print(f"\n⚠ SUGGESTIVE but not FDR-significant")
        print(f"  {len(sig_pooled)} layer(s) significant at pooled p<0.05,")
        print("  but not after multiple comparison correction.")
        print("  May indicate a real but weak effect, or need more power.")
    elif best_p_pooled < 0.1:
        print("\n⚠ MARGINAL TREND (p < 0.10)")
        print("  Some suggestion of effect but not statistically reliable.")
        print("  Consider more control directions or more questions for power.")
    else:
        print("\n✗ NO SIGNIFICANT EFFECT DETECTED")
        print("  Cannot distinguish introspection ablation from random ablations.")
        print("  The direction may not be causally involved in confidence judgments.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global METRIC

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run steering and ablation experiments")
    parser.add_argument("--metric", type=str, choices=AVAILABLE_METRICS, default=METRIC,
                        help=f"Metric to use for directions when DIRECTION_TYPE='entropy' or 'introspection' (default: {METRIC})")
    parser.add_argument("--mode", type=str, choices=["both", "steering", "ablation"], default="both",
                        help="Which experiments to run: 'both', 'steering', or 'ablation' (default: both)")
    args = parser.parse_args()
    METRIC = args.metric
    RUN_MODE = args.mode

    print(f"Device: {DEVICE}")
    print(f"Direction type: {DIRECTION_TYPE}")
    if DIRECTION_TYPE in ("entropy", "introspection"):
        print(f"Metric: {METRIC}")
    print(f"Meta-judgment task: {META_TASK}")
    print(f"Intervention position: {INTERVENTION_POSITION}")
    print(f"Run mode: {RUN_MODE}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Compute input/output paths based on direction type
    paired_data_path = f"{output_prefix}_paired_data.json"

    if DIRECTION_TYPE == "shared":
        # Shared MC entropy direction from analyze_shared_unique.py
        # Use same prefix logic as analyze_shared_unique.py (includes adapter if set)
        model_short = get_model_short_name(BASE_MODEL_NAME)
        if MODEL_NAME != BASE_MODEL_NAME:
            adapter_short = get_model_short_name(MODEL_NAME)
            shared_prefix = OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}"
        else:
            shared_prefix = OUTPUTS_DIR / f"{model_short}"
        shared_directions_path = Path(f"{shared_prefix}_shared_unique_directions.npz")
        shared_transfer_path = Path(f"{shared_prefix}_{DATASET_NAME}_shared_unique_transfer.json")
        directions_path = str(shared_directions_path)
        transfer_results_path = str(shared_transfer_path)
        direction_key_template = "layer_{}_shared"
        probe_results_path = None  # Not used for shared directions
    elif DIRECTION_TYPE == "entropy":
        # Metric directions from run_introspection_experiment.py
        # Note: output file naming changed to include metric (e.g., *_logit_gap_results.json)
        # Probe results are task-specific, but directions are task-independent
        probe_results_path = f"{output_prefix}_{METRIC}_results.json"
        directions_prefix = get_directions_prefix()
        directions_path = f"{directions_prefix}_{METRIC}_directions.npz"
        direction_key_template = "layer_{}_{}"  # Will be formatted with metric name
        transfer_results_path = None
    else:
        # Introspection directions from run_introspection_probe.py
        # Note: output file naming changed to include metric (e.g., *_logit_gap_probe_results.json)
        # Probe results are task-specific, but directions are task-independent
        probe_results_path = f"{output_prefix}_{METRIC}_probe_results.json"
        directions_prefix = get_directions_prefix()
        directions_path = f"{directions_prefix}_{METRIC}_probe_directions.npz"
        direction_key_template = "layer_{}_introspection"
        transfer_results_path = None

    # Load probe results or transfer results depending on direction type
    if DIRECTION_TYPE == "shared":
        print(f"\nLoading transfer results from {transfer_results_path}...")
        with open(transfer_results_path, "r") as f:
            transfer_results = json.load(f)
        probe_results = None
    else:
        print(f"\nLoading probe results from {probe_results_path}...")
        with open(probe_results_path, "r") as f:
            probe_results = json.load(f)
        transfer_results = None

    # Load directions
    print(f"Loading directions from {directions_path}...")
    directions_data = np.load(directions_path)
    # Remap keys to consistent format for the rest of the script
    directions = {}
    for k in directions_data.files:
        # Extract layer number and remap to "layer_{idx}_introspection" format
        # (the steering functions expect this format)
        parts = k.split("_")
        layer_idx = parts[1]
        directions[f"layer_{layer_idx}_introspection"] = directions_data[k]

    # Determine layers to steer
    if STEERING_LAYERS is not None:
        layers = STEERING_LAYERS
    else:
        if DIRECTION_TYPE == "shared":
            # For shared directions, select layers where meta R² exceeds threshold
            layer_candidates = []
            layer_list = transfer_results["layers"]
            meta_r2_list = transfer_results["shared"]["meta_r2"]
            for layer_idx, meta_r2 in zip(layer_list, meta_r2_list):
                if meta_r2 >= META_R2_THRESHOLD:
                    layer_candidates.append((layer_idx, meta_r2))
            # Sort by meta R² descending
            layer_candidates.sort(key=lambda x: -x[1])
            layers = [l[0] for l in layer_candidates]
            if not layers:
                print(f"  Warning: No layers with meta R² >= {META_R2_THRESHOLD}")
                print(f"  Using top 5 layers by meta R² instead")
                all_layers = [(l, r) for l, r in zip(layer_list, meta_r2_list)]
                all_layers.sort(key=lambda x: -x[1])
                layers = [l[0] for l in all_layers[:5]]
            layers = sorted(layers)
            print(f"  Meta R² threshold: {META_R2_THRESHOLD}")
            print(f"  Layers above threshold: {len(layers)}")
        elif DIRECTION_TYPE == "entropy":
            # For entropy directions, select layers based on probe performance
            # The probe results file may come from run_introspection_experiment.py ("probe_results" key)
            # or from run_introspection_probe.py ("layer_results" key)
            layer_candidates = []

            if "probe_results" in probe_results:
                # Structure from run_introspection_experiment.py
                # Find layers with good direct→meta transfer AND good probe fit
                for layer_str, lr in probe_results["probe_results"].items():
                    d2m_r2 = lr.get("direct_to_meta_fixed", {}).get("r2", 0)
                    d2d_r2 = lr.get("direct_to_direct", {}).get("test_r2", 0)
                    # Include layer if it exceeds both thresholds
                    if d2m_r2 >= D2M_R2_THRESHOLD and d2d_r2 >= D2D_R2_THRESHOLD:
                        layer_candidates.append((int(layer_str), d2m_r2))
                # Sort by direct→meta R² descending to prioritize best transfer
                layer_candidates.sort(key=lambda x: -x[1])
                print(f"  D2M threshold: {D2M_R2_THRESHOLD}, D2D threshold: {D2D_R2_THRESHOLD}")
                print(f"  Layers passing both thresholds: {len(layer_candidates)}")
                layers = [l[0] for l in layer_candidates]
                # If no good layers found, use layers with best direct→direct
                if not layers:
                    print(f"  Warning: No layers passed thresholds, using top 5 by D2D R²")
                    all_layers = []
                    for layer_str, lr in probe_results["probe_results"].items():
                        d2d_r2 = lr.get("direct_to_direct", {}).get("test_r2", 0)
                        all_layers.append((int(layer_str), d2d_r2))
                    all_layers.sort(key=lambda x: -x[1])  # Sort by R² descending
                    layers = [l[0] for l in all_layers[:5]]  # Top 5
            elif "layer_results" in probe_results:
                # Structure from run_introspection_probe.py
                # Use significant layers or best R² layers
                for layer_str, lr in probe_results["layer_results"].items():
                    test_r2 = lr.get("test_r2", 0)
                    if lr.get("significant_p05", False) or test_r2 > 0.1:
                        layer_candidates.append((int(layer_str), test_r2))
                layer_candidates.sort(key=lambda x: -x[1])
                layers = [l[0] for l in layer_candidates]
                # If no good layers found, use top 5 by R²
                if not layers:
                    all_layers = []
                    for layer_str, lr in probe_results["layer_results"].items():
                        test_r2 = lr.get("test_r2", 0)
                        all_layers.append((int(layer_str), test_r2))
                    all_layers.sort(key=lambda x: -x[1])
                    layers = [l[0] for l in all_layers[:5]]
            else:
                # Fallback: use all layers from directions file
                print("  Warning: Unknown probe results structure, using all available layers")
                layers = [int(k.split("_")[1]) for k in directions.keys() if k.startswith("layer_")]
                layers = sorted(layers)[:10]  # Limit to first 10

            layers = sorted(layers)
        else:
            # Use significant layers from introspection probe
            layers = set()
            for layer_str, lr in probe_results.get("layer_results", {}).items():
                if lr.get("significant_p05", False):
                    layers.add(int(layer_str))
            if "best_layer" in probe_results:
                layers.add(probe_results["best_layer"]["layer"])
            if not layers:
                all_layers = [int(l) for l in probe_results.get("layer_results", {}).keys()]
                mid = len(all_layers) // 2
                layers = all_layers[max(0, mid-3):mid+4]
            layers = sorted(layers)

    print(f"Steering layers: {layers}")

    # Compute number of control directions (dynamic or fixed)
    if NUM_CONTROL_DIRECTIONS is None:
        # Dynamic: ensure enough power for FDR correction
        # For FDR at α with N layers, need pooled_samples > N/α = 20*N (for α=0.05)
        # We use FDR_SAFETY_FACTOR * N for margin, so controls_per_layer = FDR_SAFETY_FACTOR
        num_controls = max(MIN_CONTROLS_PER_LAYER, FDR_SAFETY_FACTOR)
        target_pooled = num_controls * len(layers)
        min_p_achievable = 1.0 / (target_pooled + 1)
        fdr_threshold = FDR_ALPHA / len(layers)  # Approx threshold for best layer
        print(f"Dynamic control directions: {num_controls} per layer "
              f"({target_pooled} pooled samples, min_p={min_p_achievable:.4f}, FDR_thresh≈{fdr_threshold:.4f})")
    else:
        num_controls = NUM_CONTROL_DIRECTIONS
        print(f"Fixed control directions: {num_controls} per layer")

    # Load paired data
    print(f"\nLoading paired data from {paired_data_path}...")
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)

    questions = paired_data["questions"][:NUM_STEERING_QUESTIONS]

    # Load the metric values for alignment calculation
    # New format has direct_metrics (dict of metric_name -> list of values)
    # Old format has direct_entropies (list of values)
    if "direct_metrics" in paired_data and METRIC in paired_data["direct_metrics"]:
        direct_metric_values = np.array(paired_data["direct_metrics"][METRIC])[:NUM_STEERING_QUESTIONS]
        print(f"Using metric '{METRIC}' from paired data")
    elif "direct_entropies" in paired_data:
        # Backward compatibility: fall back to direct_entropies
        direct_metric_values = np.array(paired_data["direct_entropies"])[:NUM_STEERING_QUESTIONS]
        print(f"Using 'entropy' (backward compatible fallback)")
    else:
        raise ValueError("Paired data missing both 'direct_metrics' and 'direct_entropies'")

    print(f"Using {len(questions)} questions")

    # Load model using centralized utility
    adapter_path = MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        adapter_path=adapter_path,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )

    # Initialize token ID cache once (avoids repeated tokenization)
    initialize_token_cache(tokenizer)

    # Ensure deterministic inference (no dropout) and a tiny speedup.
    model.eval()

    # Determine chat template usage (check once, not per prompt)
    use_chat_template = should_use_chat_template(BASE_MODEL_NAME, tokenizer)
    print(f"Using chat template: {use_chat_template}")

    # Precompute direction tensors on GPU
    print("Precomputing direction tensors...")
    direction_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    cached_directions = precompute_direction_tensors(
        directions, layers, num_controls, DEVICE, direction_dtype
    )
    print(f"  Cached {len(layers)} layers with {num_controls} controls each")

    # Add direction_type suffix to output files to distinguish them
    # Include metric name when using entropy or introspection directions
    if DIRECTION_TYPE == "entropy":
        direction_suffix = f"_{METRIC}"
    elif DIRECTION_TYPE == "introspection":
        direction_suffix = f"_{METRIC}_introspection"
    else:
        direction_suffix = f"_{DIRECTION_TYPE}"

    # Determine which experiments to run based on RUN_MODE and DIRECTION_TYPE
    run_steering = RUN_MODE in ("both", "steering")
    run_ablation = RUN_MODE in ("both", "ablation")

    # Skip steering for introspection directions - steering doesn't make sense conceptually.
    # The introspection direction captures calibration quality (metric-confidence alignment),
    # not a direct uncertainty signal. We can't causally steer toward "being well-calibrated"
    # without knowing a question's actual uncertainty. Ablation still makes sense: removing
    # "awareness of calibration" should degrade the metric-confidence correlation.
    baseline_from_steering = None
    other_confidence_steering_analysis = None  # Track for plotting
    other_confidence_ablation_analysis = None  # Track for plotting

    if run_steering:
        if DIRECTION_TYPE == "introspection":
            print("\n" + "=" * 70)
            print("SKIPPING STEERING EXPERIMENT (introspection directions)")
            print("=" * 70)
            print("Steering with introspection directions doesn't make conceptual sense.")
            print("The direction captures calibration quality, not a causal uncertainty signal.")
        else:
            # Run steering experiment
            results = run_steering_experiment(
                model, tokenizer, questions, direct_metric_values,
                layers, directions, STEERING_MULTIPLIERS, num_controls,
                use_chat_template, cached_directions
            )

            # Analyze
            analysis = analyze_results(results, metric=METRIC)

            # Save results
            output_results = f"{output_prefix}_steering{direction_suffix}_results.json"
            with open(output_results, "w") as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            print(f"\nSaved {output_results}")

            output_analysis = f"{output_prefix}_steering{direction_suffix}_analysis.json"
            with open(output_analysis, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"Saved {output_analysis}")

            # Print and plot steering results
            print_summary(analysis)
            plot_results(analysis, f"{output_prefix}{direction_suffix}")

            print("\n✓ Steering experiment complete!")

            # Extract baseline from steering results for ablation (first layer's baseline, they're all the same)
            first_layer = layers[0]
            baseline_from_steering = results["layer_results"][first_layer]["baseline"]

            # ==================================================================
            # OTHER-CONFIDENCE CONTROL (for confidence task only)
            # ==================================================================
            if META_TASK == "confidence":
                print("\n" + "-" * 50)
                print("OTHER-CONFIDENCE CONTROL EXPERIMENT")
                print("-" * 50)
                print("Testing whether steering affects self-confidence specifically,")
                print("or also affects general confidence-like judgments (human difficulty estimation).")

                # Run other-confidence at baseline (no steering)
                other_baseline = run_other_confidence_experiment(
                    model, tokenizer, questions, layers, directions, use_chat_template,
                    steering_multiplier=0.0, cached_directions=cached_directions
                )

                # Run other-confidence with max positive steering multiplier
                max_mult = max(STEERING_MULTIPLIERS)
                other_steered = run_other_confidence_experiment(
                    model, tokenizer, questions, layers, directions, use_chat_template,
                    steering_multiplier=max_mult, cached_directions=cached_directions
                )

                # Analyze: compare self vs other effects for each layer
                other_confidence_steering_analysis = {}
                for layer_idx in layers:
                    # Get self-confidence baseline and steered results for this layer
                    # Structure: layer_results[layer_idx]["baseline"] and layer_results[layer_idx]["introspection"][multiplier]
                    layer_results = results["layer_results"].get(layer_idx, {})
                    self_baseline = layer_results.get("baseline", [])
                    self_steered = layer_results.get("introspection", {}).get(max_mult, [])

                    if self_baseline and self_steered and None not in self_baseline and None not in self_steered:
                        effect = analyze_other_confidence_effect(
                            other_baseline, other_steered,
                            self_baseline, self_steered, layer_idx
                        )
                        if effect is not None:
                            other_confidence_steering_analysis[str(layer_idx)] = effect

                # Store in results
                results["other_confidence"] = {
                    "baseline": other_baseline,
                    "steered": other_steered,
                    "steering_multiplier": max_mult,
                    "analysis": other_confidence_steering_analysis,
                }

                # Print other-confidence summary with statistics
                print("\n--- Other-Confidence Control Results (Steering) ---")
                print(f"Comparing steering effect (multiplier={max_mult}) on self vs other confidence:")
                print(f"{'Layer':<8} {'Self':<10} {'Other':<10} {'Ratio':<8} {'Diff':<10} {'95% CI':<16} {'p(perm)':<10} {'Sig':<5}")
                print("-" * 85)
                for layer_str, effect in other_confidence_steering_analysis.items():
                    self_eff = effect["self_effect_mean"]
                    other_eff = effect["other_effect_mean"]
                    ratio = effect["self_vs_other_ratio"]
                    ratio_str = f"{ratio:.2f}x" if not np.isnan(ratio) else "N/A"
                    diff = effect.get("diff_abs_effect", 0)
                    ci = effect.get("diff_abs_effect_ci95", [0, 0])
                    p_perm = effect.get("p_value_permutation", float('nan'))
                    sig = "*" if p_perm < 0.05 else ""
                    ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
                    print(f"  {layer_str:<6} {self_eff:+.4f}   {other_eff:+.4f}   {ratio_str:<8} {diff:+.4f}   {ci_str:<16} {p_perm:.4f}    {sig}")

                # Overall assessment
                if other_confidence_steering_analysis:
                    # Filter out inf and nan ratios
                    valid_ratios = [e["self_vs_other_ratio"] for e in other_confidence_steering_analysis.values()
                                   if not np.isinf(e["self_vs_other_ratio"]) and not np.isnan(e["self_vs_other_ratio"])]
                    mean_ratio = np.mean(valid_ratios) if valid_ratios else float('nan')
                    mean_self = np.mean([e["self_effect_mean"] for e in other_confidence_steering_analysis.values()])
                    mean_other = np.mean([e["other_effect_mean"] for e in other_confidence_steering_analysis.values()])

                    if np.isnan(mean_ratio):
                        print(f"\n  Mean across layers: self={mean_self:+.3f}, other={mean_other:+.3f}, ratio=N/A")
                        print("  → No measurable effect on either self or other confidence")
                    else:
                        print(f"\n  Mean across layers: self={mean_self:+.3f}, other={mean_other:+.3f}, ratio={mean_ratio:.2f}x")
                        if mean_ratio > 2.0:
                            print("  → Steering primarily affects SELF-confidence (introspection-specific)")
                        elif mean_ratio > 1.2:
                            print("  → Steering affects self-confidence more than other-confidence")
                        elif mean_ratio > 0.8:
                            print("  → Steering affects self and other confidence similarly (general effect)")
                        else:
                            print("  → Steering affects other-confidence more than self (unexpected)")

    # ==========================================================================
    # ABLATION EXPERIMENT
    # ==========================================================================
    if run_ablation:
        print("\n" + "=" * 70)
        print("RUNNING ABLATION EXPERIMENT")
        print("=" * 70)

        # baseline_from_steering is set above (None if we skipped steering, otherwise from steering results)
        # run_ablation_experiment will compute its own baseline if baseline_results=None
        ablation_results = run_ablation_experiment(
            model, tokenizer, questions, direct_metric_values,
            layers, directions, num_controls,
            use_chat_template,
            baseline_results=baseline_from_steering,
            cached_directions=cached_directions
        )

        # Analyze ablation results
        ablation_analysis = analyze_ablation_results(ablation_results)

        # Save ablation results
        ablation_results_path = f"{output_prefix}_ablation{direction_suffix}_results.json"
        with open(ablation_results_path, "w") as f:
            json.dump(ablation_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\nSaved {ablation_results_path}")

        ablation_analysis_path = f"{output_prefix}_ablation{direction_suffix}_analysis.json"
        with open(ablation_analysis_path, "w") as f:
            json.dump(ablation_analysis, f, indent=2)
        print(f"Saved {ablation_analysis_path}")

        # Print and plot ablation results
        print_ablation_summary(ablation_analysis)
        plot_ablation_results(ablation_analysis, f"{output_prefix}{direction_suffix}")

        print("\n✓ Ablation experiment complete!")

        # ==================================================================
        # OTHER-CONFIDENCE CONTROL FOR ABLATION (for confidence task only)
        # ==================================================================
        if META_TASK == "confidence":
            print("\n" + "-" * 50)
            print("OTHER-CONFIDENCE CONTROL (ABLATION)")
            print("-" * 50)
            print("Testing whether ablation affects self-confidence specifically,")
            print("or also affects general confidence-like judgments.")

            # Run other-confidence at baseline (no ablation)
            other_baseline_abl = run_other_confidence_with_ablation(
                model, tokenizer, questions, layers, directions, use_chat_template,
                ablate=False, cached_directions=cached_directions
            )

            # Run other-confidence with ablation
            other_ablated = run_other_confidence_with_ablation(
                model, tokenizer, questions, layers, directions, use_chat_template,
                ablate=True, cached_directions=cached_directions
            )

            # Analyze: compare self vs other ablation effects for each layer
            other_confidence_ablation_analysis = {}
            for layer_idx in layers:
                # Get self-confidence baseline and ablated results for this layer
                # Structure: layer_results[layer_idx]["baseline"] and layer_results[layer_idx]["introspection_ablated"]
                layer_results = ablation_results["layer_results"].get(layer_idx, {})
                self_baseline = layer_results.get("baseline", [])
                self_ablated = layer_results.get("introspection_ablated", [])

                if self_baseline and self_ablated and None not in self_baseline and None not in self_ablated:
                    effect = analyze_other_confidence_ablation_effect(
                        other_baseline_abl, other_ablated,
                        self_baseline, self_ablated, layer_idx
                    )
                    if effect is not None:
                        other_confidence_ablation_analysis[str(layer_idx)] = effect

            # Store in ablation results
            ablation_results["other_confidence"] = {
                "baseline": other_baseline_abl,
                "ablated": other_ablated,
                "analysis": other_confidence_ablation_analysis,
            }

            # Re-save ablation results with other-confidence data
            with open(ablation_results_path, "w") as f:
                json.dump(ablation_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            print(f"\nRe-saved {ablation_results_path} with other-confidence data")

            # Print other-confidence ablation summary with statistics
            print("\n--- Other-Confidence Control Results (Ablation) ---")
            print("Comparing ablation effect on self vs other confidence:")
            print(f"{'Layer':<8} {'|Δself|':<10} {'|Δother|':<10} {'Ratio':<8} {'Diff':<10} {'95% CI':<16} {'p(perm)':<10} {'Sig':<5}")
            print("-" * 85)
            for layer_str, effect in other_confidence_ablation_analysis.items():
                self_eff = effect["self_effect_mean_abs"]
                other_eff = effect["other_effect_mean_abs"]
                ratio = effect["self_vs_other_ratio"]
                ratio_str = f"{ratio:.2f}x" if not np.isnan(ratio) else "N/A"
                diff = effect.get("diff_abs_effect", 0)
                ci = effect.get("diff_abs_effect_ci95", [0, 0])
                p_perm = effect.get("p_value_permutation", float('nan'))
                sig = "*" if p_perm < 0.05 else ""
                ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
                print(f"  {layer_str:<6} {self_eff:.4f}    {other_eff:.4f}    {ratio_str:<8} {diff:+.4f}   {ci_str:<16} {p_perm:.4f}    {sig}")

            # Overall assessment
            if other_confidence_ablation_analysis:
                # Filter out inf and nan ratios
                valid_ratios = [e["self_vs_other_ratio"] for e in other_confidence_ablation_analysis.values()
                               if not np.isinf(e["self_vs_other_ratio"]) and not np.isnan(e["self_vs_other_ratio"])]
                mean_ratio = np.mean(valid_ratios) if valid_ratios else float('nan')
                mean_self = np.mean([e["self_effect_mean_abs"] for e in other_confidence_ablation_analysis.values()])
                mean_other = np.mean([e["other_effect_mean_abs"] for e in other_confidence_ablation_analysis.values()])

                if np.isnan(mean_ratio):
                    print(f"\n  Mean across layers: |Δself|={mean_self:.3f}, |Δother|={mean_other:.3f}, ratio=N/A")
                    print("  → No measurable effect on either self or other confidence")
                else:
                    print(f"\n  Mean across layers: |Δself|={mean_self:.3f}, |Δother|={mean_other:.3f}, ratio={mean_ratio:.2f}x")
                    if mean_ratio > 2.0:
                        print("  → Ablation primarily affects SELF-confidence (introspection-specific)")
                    elif mean_ratio > 1.2:
                        print("  → Ablation affects self-confidence more than other-confidence")
                    elif mean_ratio > 0.8:
                        print("  → Ablation affects self and other confidence similarly (general effect)")
                    else:
                        print("  → Ablation affects other-confidence more than self (unexpected)")

    # ==================================================================
    # PLOT OTHER-CONFIDENCE COMPARISON (if any data available)
    # ==================================================================
    if other_confidence_steering_analysis or other_confidence_ablation_analysis:
        print("\n" + "-" * 50)
        print("PLOTTING OTHER-CONFIDENCE COMPARISON")
        print("-" * 50)
        plot_other_confidence_comparison(
            other_confidence_steering_analysis,
            other_confidence_ablation_analysis,
            f"{output_prefix}{direction_suffix}"
        )

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
