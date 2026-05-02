"""
Steering and ablation experiment runners with statistical analysis.

This module provides shared infrastructure for running steering/ablation experiments
across different scripts (run_introspection_steering.py, run_contrastive_direction.py).

Provides:
- KV cache utilities for efficient batch processing
- Batch steering/ablation hooks
- Experiment runners (steering sweep, ablation)
- Statistical analysis (slopes, pooled null, FDR correction)
- Visualization functions
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import DynamicCache safely
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SteeringExperimentConfig:
    """Configuration for steering/ablation experiments."""
    # Direction naming - key used in results dict
    direction_key: str = "direction"  # e.g., "introspection" or "contrastive"

    # Batch processing
    batch_size: int = 8
    intervention_position: str = "last"  # "last" or "all"

    # Device (auto-detected if None)
    device: str = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# =============================================================================
# KV CACHE UTILITIES
# =============================================================================

def extract_cache_tensors(past_key_values) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extract raw tensors from past_key_values (tuple or DynamicCache).
    Returns (key_tensors, value_tensors) where each is a list of tensors.

    Hugging Face cache layouts differ by version: some DynamicCache objects
    support len() but not integer __getitem__; others expose .key_cache /
    .value_cache or .layers instead of legacy (k, v) tuples per layer.
    """
    if past_key_values is None:
        raise ValueError("past_key_values is None")

    # 1. DynamicCache with .key_cache / .value_cache (common)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(past_key_values.key_cache), list(past_key_values.value_cache)

    # 2. DynamicCache with .layers (newer transformers)
    if hasattr(past_key_values, "layers"):
        keys, values = [], []
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                keys.append(layer.keys)
                values.append(layer.values)
            elif hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
                keys.append(layer.key_cache)
                values.append(layer.value_cache)
            elif isinstance(layer, tuple) and len(layer) == 2:
                keys.append(layer[0])
                values.append(layer[1])
            else:
                tensor_attrs = [
                    a
                    for a in dir(layer)
                    if not a.startswith("_")
                    and isinstance(getattr(layer, a, None), torch.Tensor)
                ]
                if len(tensor_attrs) >= 2:
                    keys.append(getattr(layer, tensor_attrs[0]))
                    values.append(getattr(layer, tensor_attrs[1]))
                else:
                    raise ValueError(
                        f"Cannot extract k/v from {type(layer).__name__}. "
                        f"Tensor attrs: {tensor_attrs}"
                    )
        return keys, values

    # 3. Iterable cache yielding (key, value) per layer
    if hasattr(past_key_values, "__iter__") and hasattr(past_key_values, "__len__"):
        keys, values = [], []
        for item in past_key_values:
            if isinstance(item, tuple) and len(item) == 2:
                keys.append(item[0])
                values.append(item[1])
            else:
                raise ValueError(f"Unexpected cache item type: {type(item)}")
        return keys, values

    # 4. Legacy: to_legacy_cache() then tuple indexing
    if hasattr(past_key_values, "to_legacy_cache"):
        return extract_cache_tensors(past_key_values.to_legacy_cache())

    keys, values = [], []
    try:
        n = len(past_key_values)
    except TypeError as e:
        raise ValueError(f"Cannot determine length of cache: {type(past_key_values)}") from e
    for i in range(n):
        k, v = past_key_values[i]
        keys.append(k)
        values.append(v)
    return keys, values


def create_fresh_cache(key_tensors: List[torch.Tensor], value_tensors: List[torch.Tensor], expand_size: int = 1):
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


def get_kv_cache(model, batch_inputs: Dict[str, torch.Tensor]) -> Dict:
    """
    Run the prefix to generate KV cache tensors.
    Returns dictionary with next-step inputs and 'past_key_values_data' (snapshot).
    """
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

    # Extract Immutable Snapshot (List of Tensors)
    keys, values = extract_cache_tensors(outputs.past_key_values)

    # Prepare next step inputs
    last_ids = input_ids[:, -1:]

    result = {
        "input_ids": last_ids,
        "attention_mask": attention_mask,  # Full mask (History + Current)
        "past_key_values_data": (keys, values),
    }

    # Preserve position_ids if available
    if "position_ids" in batch_inputs:
        result["position_ids"] = batch_inputs["position_ids"][:, -1:]

    return result


# =============================================================================
# BATCH HOOKS
# =============================================================================

class BatchSteeringHook:
    """Hook that adds a *per-example* steering delta to activations.

    This is designed for "multiplier sweep in one pass" by expanding the batch:
    each prompt is duplicated for each multiplier, and this hook adds a different
    delta vector for each expanded example.

    intervention_position can be:
        - "last": Only intervene at the final token (compatible with KV cache)
        - "all": Intervene at all token positions
        - "indexed": Use position_indices tensor for per-example positions
    """

    def __init__(self, delta_bh: Optional[torch.Tensor] = None, intervention_position: str = "last"):
        self.delta_bh = delta_bh  # (batch, hidden)
        self.handle = None
        self.intervention_position = intervention_position
        self.position_indices = None  # Optional: per-example position indices

    def set_delta(self, delta_bh: torch.Tensor):
        self.delta_bh = delta_bh

    def set_position_indices(self, indices: torch.Tensor):
        """Set per-example position indices for intervention."""
        self.position_indices = indices

    def __call__(self, module, input, output):
        if self.delta_bh is None:
            return output

        hs = output[0] if isinstance(output, tuple) else output
        delta = self.delta_bh.to(device=hs.device, dtype=hs.dtype)

        if self.intervention_position == "last":
            hs = hs.clone()
            hs[:, -1, :] = hs[:, -1, :] + delta
        elif self.intervention_position == "indexed" and self.position_indices is not None:
            # Per-example position indices
            hs = hs.clone()
            batch_indices = torch.arange(hs.shape[0], device=hs.device)
            pos_indices = self.position_indices.to(device=hs.device)
            hs[batch_indices, pos_indices, :] = hs[batch_indices, pos_indices, :] + delta
        else:
            # "all" positions
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

    intervention_position can be:
        - "last": Only intervene at the final token (compatible with KV cache)
        - "all": Intervene at all token positions
        - "indexed": Use position_indices tensor for per-example positions
    """

    def __init__(self, directions_bh: Optional[torch.Tensor] = None, intervention_position: str = "last"):
        self.directions_bh = directions_bh
        self.handle = None
        self.intervention_position = intervention_position
        self.position_indices = None  # Optional: per-example position indices
        self._diag_printed = False

    def set_directions(self, directions_bh: torch.Tensor):
        self.directions_bh = directions_bh
        self._diag_printed = False

    def set_position_indices(self, indices: torch.Tensor):
        """Set per-example position indices for intervention."""
        self.position_indices = indices

    def __call__(self, module, input, output):
        if self.directions_bh is None:
            return output

        hs = output[0] if isinstance(output, tuple) else output
        dirs = self.directions_bh.to(device=hs.device, dtype=hs.dtype)

        if self.intervention_position == "last":
            hs = hs.clone()
            last_token = hs[:, -1, :]
            dots = torch.einsum('bh,bh->b', last_token, dirs)
            proj = dots.unsqueeze(-1) * dirs
            hs[:, -1, :] = last_token - proj
        elif self.intervention_position == "indexed" and self.position_indices is not None:
            # Per-example position indices
            hs = hs.clone()
            batch_indices = torch.arange(hs.shape[0], device=hs.device)
            pos_indices = self.position_indices.to(device=hs.device)
            target_tokens = hs[batch_indices, pos_indices, :]  # (batch, hidden)
            dots = torch.einsum('bh,bh->b', target_tokens, dirs)
            proj = dots.unsqueeze(-1) * dirs
            hs[batch_indices, pos_indices, :] = target_tokens - proj
        else:
            # "all" positions
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


# =============================================================================
# TOKENIZATION UTILITIES
# =============================================================================

def pretokenize_prompts(prompts: List[str], tokenizer, device: str) -> Dict:
    """
    Pre-tokenize all prompts once (BPE encoding).

    Returns dict with:
        - input_ids: List of token ID lists (variable length, no padding yet)
        - attention_mask: List of attention mask lists
        - lengths: List of sequence lengths
        - sorted_order: Indices sorted by length (for efficient batching)
    """
    tokenized = tokenizer(
        prompts,
        padding=False,
        truncation=False,  # Don't truncate - breaks position finding
        add_special_tokens=False,  # Prompts already have special tokens from chat template
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


# =============================================================================
# DIRECTION PREPARATION
# =============================================================================

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


def precompute_direction_tensors(
    directions: Dict,
    layers: List[int],
    num_controls: int,
    device: str,
    dtype: torch.dtype,
    direction_key: str = "direction"
) -> Dict:
    """
    Precompute normalized direction tensors on GPU for all layers and controls.

    Args:
        directions: Dict with direction arrays. Keys can be:
            - f"layer_{layer_idx}_{direction_key}" (introspection format)
            - layer_idx directly (contrastive format where directions[layer_idx] is the array)
        layers: List of layer indices
        num_controls: Number of control directions to generate
        device: Target device
        dtype: Target dtype
        direction_key: Key name in output dict (e.g., "introspection" or "contrastive")

    Returns:
        {layer_idx: {direction_key: tensor, "controls": [tensor, ...]}}
    """
    cached = {}
    for layer_idx in layers:
        # Try both key formats
        key = f"layer_{layer_idx}_{direction_key}"
        if key in directions:
            dir_array = np.array(directions[key])
        elif layer_idx in directions:
            dir_array = np.array(directions[layer_idx])
        else:
            raise KeyError(f"Direction not found for layer {layer_idx}. Tried keys: {key}, {layer_idx}")

        # Normalize
        dir_array = dir_array / np.linalg.norm(dir_array)
        dir_tensor = torch.tensor(dir_array, dtype=dtype, device=device)

        # Generate control directions
        control_dirs = generate_orthogonal_directions(dir_array, num_controls)
        control_tensors = [
            torch.tensor(cd, dtype=dtype, device=device)
            for cd in control_dirs
        ]

        cached[layer_idx] = {
            direction_key: dir_tensor,
            "controls": control_tensors,
        }

    return cached


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    layers: List[int],
    directions: Dict,
    multipliers: List[float],
    num_controls: int,
    use_chat_template: bool,
    format_prompt_fn: Callable,
    response_to_confidence_fn: Callable,
    option_token_ids: List[int],
    options: List[str],
    config: SteeringExperimentConfig,
    mappings_fn: Optional[Callable] = None,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """
    Run steering experiment with KV cache optimization + batched multipliers.

    Args:
        model: The model
        tokenizer: Tokenizer
        questions: List of question dicts
        metric_values: Array of metric values (e.g., entropy) for each question
        layers: Layer indices to steer
        directions: Direction dict (see precompute_direction_tensors for format)
        multipliers: Steering multipliers to test
        num_controls: Number of control directions
        use_chat_template: Whether to use chat template
        format_prompt_fn: Function(question, tokenizer, use_chat_template, trial_index) -> (prompt, options) or (prompt, options, mapping)
        response_to_confidence_fn: Function(response, probs, mapping) -> confidence
        option_token_ids: Token IDs for the options
        options: Option strings
        config: SteeringExperimentConfig
        mappings_fn: Optional function to get mapping for delegate task
        cached_directions: Pre-computed direction tensors (optional)

    Returns:
        Results dict with layer_results[layer_idx][direction_key][mult] structure
    """
    device = config.device
    direction_key = config.direction_key
    batch_size = config.batch_size
    intervention_position = config.intervention_position

    metric_mean = metric_values.mean()
    metric_std = metric_values.std()

    # Format and tokenize prompts
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        result = format_prompt_fn(question, tokenizer, use_chat_template, q_idx)
        if len(result) == 3:
            prompt, _, mapping = result
        else:
            prompt, _ = result
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, device)

    # Precompute directions if not provided
    if cached_directions is None:
        cached_directions = precompute_direction_tensors(
            directions, layers, num_controls, device,
            torch.float16 if device == "cuda" else torch.float32,
            direction_key
        )

    # Handle KV cache vs full forward
    if intervention_position != "last":
        print(f"\nRunning steering experiment (full forward fallback)...")
        return _run_steering_full_forward(
            model, tokenizer, questions, metric_values, layers, multipliers,
            num_controls, cached_directions, option_token_ids, options,
            mappings, cached_inputs, config, response_to_confidence_fn
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
            direction_key: {m: [None] * len(questions) for m in multipliers},
            "controls": {f"control_{i}": {m: [None] * len(questions) for m in multipliers} for i in range(num_controls)},
        }
        if 0.0 in multipliers:
            final_layer_results[l][direction_key][0.0] = shared_baseline
            for k in final_layer_results[l]["controls"]:
                final_layer_results[l]["controls"][k][0.0] = shared_baseline

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, device, batch_size)
    print(f"Processing {len(gpu_batches)} batches, {k_mult} multipliers batched per forward pass...")

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
            conf = response_to_confidence_fn(resp, p, mappings[q_idx])
            m_val = metric_values[q_idx]
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

            dir_tensor = cached_directions[layer_idx][direction_key]
            control_dirs = cached_directions[layer_idx]["controls"]

            hook = BatchSteeringHook(intervention_position=intervention_position)
            hook.register(layer_module)

            def run_batched_mult_sweep(direction_vector, result_dict):
                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_mult)

                current_inputs = expanded_inputs_template.copy()
                current_inputs["past_key_values"] = pass_cache

                deltas = []
                for _ in range(B):
                    for mult in nonzero_multipliers:
                        deltas.append(direction_vector * mult)
                delta_bh = torch.stack(deltas, dim=0)
                hook.set_delta(delta_bh)

                with torch.inference_mode():
                    out = model(**current_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    for j, mult in enumerate(nonzero_multipliers):
                        idx = i * k_mult + j
                        p = probs[idx]
                        resp = options[np.argmax(p)]
                        conf = response_to_confidence_fn(resp, p, mappings[q_idx])
                        m_val = metric_values[q_idx]
                        align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)

                        result_dict[mult][q_idx] = {
                            "question_idx": q_idx, "response": resp, "confidence": conf,
                            "metric": float(m_val), "alignment": float(align)
                        }

            try:
                run_batched_mult_sweep(dir_tensor, final_layer_results[layer_idx][direction_key])
                for i_c, ctrl_dir in enumerate(control_dirs):
                    run_batched_mult_sweep(ctrl_dir, final_layer_results[layer_idx]["controls"][f"control_{i_c}"])
            finally:
                hook.remove()

        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results,
        "direction_key": direction_key,
    }


def _run_steering_full_forward(
    model, tokenizer, questions, metric_values, layers, multipliers,
    num_controls, cached_directions, option_token_ids, options,
    mappings, cached_inputs, config, response_to_confidence_fn
) -> Dict:
    """Fallback: Full forward pass implementation for INTERVENTION_POSITION='all'."""
    # Implementation placeholder - copies pattern from run_introspection_steering.py
    raise NotImplementedError("Full forward fallback not yet implemented in shared module")


def run_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    layers: List[int],
    directions: Dict,
    num_controls: int,
    use_chat_template: bool,
    format_prompt_fn: Callable,
    response_to_confidence_fn: Callable,
    option_token_ids: List[int],
    options: List[str],
    config: SteeringExperimentConfig,
    baseline_results: Optional[List[Dict]] = None,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """
    Run ablation experiment with KV cache optimization.

    Similar args to run_steering_experiment but without multipliers.
    """
    device = config.device
    direction_key = config.direction_key
    batch_size = config.batch_size
    intervention_position = config.intervention_position

    metric_mean = metric_values.mean()
    metric_std = metric_values.std()

    # Format and tokenize prompts
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        result = format_prompt_fn(question, tokenizer, use_chat_template, q_idx)
        if len(result) == 3:
            prompt, _, mapping = result
        else:
            prompt, _ = result
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, device)

    if cached_directions is None:
        cached_directions = precompute_direction_tensors(
            directions, layers, num_controls, device,
            torch.float16 if device == "cuda" else torch.float32,
            direction_key
        )

    if intervention_position != "last":
        print(f"\nRunning ablation experiment (full forward fallback)...")
        raise NotImplementedError("Full forward fallback not yet implemented")

    print(f"\nRunning ablation experiment (KV Cache)...")

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, device, batch_size)

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
                conf = response_to_confidence_fn(resp, p, mappings[q_idx])
                m_val = metric_values[q_idx]
                align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)
                baseline_results[q_idx] = {
                    "question_idx": q_idx, "response": resp, "confidence": conf,
                    "metric": float(m_val), "alignment": float(align)
                }

    ablated_key = f"{direction_key}_ablated"
    final_layer_results = {}
    for l in layers:
        final_layer_results[l] = {
            "baseline": baseline_results,
            ablated_key: [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    print(f"Processing {len(gpu_batches)} batches (KV-Cached)...")

    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Batches"):
        B = len(batch_indices)

        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        inputs_template = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "use_cache": True
        }
        if "position_ids" in base_step_data:
            inputs_template["position_ids"] = base_step_data["position_ids"]

        for layer_idx in layers:
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer_idx]
            else:
                layer_module = model.model.layers[layer_idx]

            dir_tensor = cached_directions[layer_idx][direction_key]
            control_dirs = cached_directions[layer_idx]["controls"]

            hook = BatchAblationHook(intervention_position=intervention_position)
            hook.register(layer_module)

            def run_single_ablation(direction_vector, result_storage, key=None):
                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)

                current_inputs = inputs_template.copy()
                current_inputs["past_key_values"] = pass_cache

                dirs_batch = direction_vector.unsqueeze(0).expand(B, -1)
                hook.set_directions(dirs_batch)

                with torch.inference_mode():
                    out = model(**current_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf = response_to_confidence_fn(resp, p, mappings[q_idx])
                    m_val = metric_values[q_idx]
                    align = -((m_val - metric_mean)/metric_std) * ((conf - 0.5)/0.25)

                    data = {
                        "question_idx": q_idx, "response": resp, "confidence": conf,
                        "metric": float(m_val), "alignment": float(align)
                    }
                    if key:
                        result_storage[key][q_idx] = data
                    else:
                        result_storage[q_idx] = data

            try:
                run_single_ablation(dir_tensor, final_layer_results[layer_idx][ablated_key])
                for i_c, ctrl_dir in enumerate(control_dirs):
                    run_single_ablation(ctrl_dir, final_layer_results[layer_idx]["controls_ablated"], key=f"control_{i_c}")
            finally:
                hook.remove()

        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": final_layer_results,
        "direction_key": direction_key,
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and metric."""
    if len(confidences) < 2 or np.std(confidences) == 0 or np.std(metric_values) == 0:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def get_expected_slope_sign(metric: str) -> int:
    """
    Get the expected sign of the confidence slope for a given metric.

    Returns:
        +1 if +direction should increase confidence
        -1 if +direction should decrease confidence
    """
    if metric == "entropy":
        return -1  # +direction = more uncertain = less confident
    else:
        return +1  # +direction = more confident


def analyze_steering_results(results: Dict, metric: str = None) -> Dict:
    """
    Analyze steering results with full statistical significance testing.

    Statistical approach:
    1. Compute confidence slope for primary direction
    2. Compute confidence slope for each control direction
    3. Pool all control slopes across layers for null distribution
    4. Per-layer p-value: fraction of control slopes with |slope| >= |primary slope|
    5. FDR correction for multiple testing
    6. Bootstrap CIs for control slope distribution
    """
    direction_key = results.get("direction_key", "introspection")

    analysis = {
        "layers": results["layers"],
        "multipliers": results["multipliers"],
        "metric": metric,
        "direction_key": direction_key,
        "effects": {},
    }

    multipliers = results["multipliers"]

    # First pass: collect all slopes for pooled null distribution
    all_control_slopes = []
    layer_data = {}

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        baseline_align = np.mean([r["alignment"] for r in lr["baseline"]])
        baseline_conf = np.mean([r["confidence"] for r in lr["baseline"]])

        # Primary direction: confidence change per multiplier
        primary_conf_changes = []
        primary_align_changes = []
        for mult in multipliers:
            primary_conf = np.mean([r["confidence"] for r in lr[direction_key][mult]])
            primary_align = np.mean([r["alignment"] for r in lr[direction_key][mult]])
            primary_conf_changes.append(primary_conf - baseline_conf)
            primary_align_changes.append(primary_align - baseline_align)

        primary_conf_slope = np.polyfit(multipliers, primary_conf_changes, 1)[0]
        primary_align_slope = np.polyfit(multipliers, primary_align_changes, 1)[0]

        # Per-control slopes
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

        all_control_slopes.extend(control_slopes)

        # Build by_multiplier dict
        effects = {direction_key: {}, "control_avg": {}}
        for mult in multipliers:
            primary_conf = np.mean([r["confidence"] for r in lr[direction_key][mult]])
            primary_align = np.mean([r["alignment"] for r in lr[direction_key][mult]])
            effects[direction_key][mult] = {
                "alignment": float(primary_align),
                "alignment_change": float(primary_align - baseline_align),
                "confidence": float(primary_conf),
                "confidence_change": float(primary_conf - baseline_conf),
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
            "primary_conf_slope": primary_conf_slope,
            "primary_align_slope": primary_align_slope,
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

        primary_slope = ld["primary_conf_slope"]
        primary_slope_abs = abs(primary_slope)
        control_slopes = np.array(ld["control_slopes"])

        avg_ctrl_slope = float(np.mean(control_slopes))
        std_ctrl_slope = float(np.std(control_slopes))

        # Per-layer p-value: two-tailed test
        n_controls_larger_local = np.sum(np.abs(control_slopes) >= primary_slope_abs)
        p_value_local = (n_controls_larger_local + 1) / (len(control_slopes) + 1)

        # Pooled p-value
        n_pooled_larger = np.sum(pooled_null_abs >= primary_slope_abs)
        p_value_pooled = (n_pooled_larger + 1) / (len(pooled_null) + 1)

        # Bootstrap 95% CI for control slope magnitude
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(control_slopes, size=len(control_slopes), replace=True)
            bootstrap_means.append(np.mean(np.abs(boot_sample)))
        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)

        # Effect size
        if std_ctrl_slope > 0:
            effect_size_z = (primary_slope_abs - np.mean(np.abs(control_slopes))) / np.std(np.abs(control_slopes))
        else:
            effect_size_z = 0.0

        raw_p_values.append((layer_idx, p_value_pooled))

        analysis["effects"][layer_idx] = {
            "by_multiplier": ld["effects"],
            "slopes": {
                direction_key: float(primary_slope),
                "control_avg": avg_ctrl_slope,
                "control_std": std_ctrl_slope,
                f"{direction_key}_alignment": float(ld["primary_align_slope"]),
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

    # Sign interpretation
    if metric:
        expected_sign = get_expected_slope_sign(metric)
        analysis["sign_interpretation"] = {
            "metric": metric,
            "expected_slope_sign": expected_sign,
            "expected_slope_sign_str": "positive" if expected_sign > 0 else "negative",
        }

        if analysis["effects"]:
            best_layer = max(
                analysis["layers"],
                key=lambda l: abs(analysis["effects"][l]["slopes"][direction_key])
            )
            best_slope = analysis["effects"][best_layer]["slopes"][direction_key]
            actual_sign = 1 if best_slope > 0 else -1
            analysis["sign_interpretation"]["best_layer"] = best_layer
            analysis["sign_interpretation"]["actual_slope_sign"] = actual_sign
            analysis["sign_interpretation"]["sign_matches_expected"] = (actual_sign == expected_sign)

    return analysis


def analyze_ablation_results(results: Dict) -> Dict:
    """
    Analyze ablation results with full statistical significance testing.

    Statistical approach:
    1. Pooled null distribution from all control effects across all layers
    2. Per-layer permutation test
    3. FDR correction for multiple layer testing
    4. Bootstrap CIs on control effects
    """
    direction_key = results.get("direction_key", "introspection")
    ablated_key = f"{direction_key}_ablated"

    analysis = {
        "layers": results["layers"],
        "num_questions": results["num_questions"],
        "num_controls": results["num_controls"],
        "direction_key": direction_key,
        "effects": {},
    }

    # First pass: collect all effects for pooled null
    all_control_corr_changes = []
    layer_data = {}

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_metric = np.array([r["metric"] for r in lr["baseline"]])
        baseline_align = np.array([r["alignment"] for r in lr["baseline"]])

        ablated_conf = np.array([r["confidence"] for r in lr[ablated_key]])
        ablated_metric = np.array([r["metric"] for r in lr[ablated_key]])
        ablated_align = np.array([r["alignment"] for r in lr[ablated_key]])

        baseline_corr = compute_correlation(baseline_conf, baseline_metric)
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)

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

        primary_corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]

        all_control_corr_changes.extend(control_corr_changes)

        layer_data[layer_idx] = {
            "baseline_corr": baseline_corr,
            "baseline_conf": baseline_conf,
            "baseline_metric": baseline_metric,
            "baseline_align": baseline_align,
            "ablated_corr": ablated_corr,
            "ablated_conf": ablated_conf,
            "ablated_align": ablated_align,
            "primary_corr_change": primary_corr_change,
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
            "control_aligns": control_aligns,
            "control_confs": control_confs,
        }

    pooled_null = np.array(all_control_corr_changes)

    # Second pass: compute statistics
    raw_p_values = []

    for layer_idx in results["layers"]:
        ld = layer_data[layer_idx]

        avg_control_corr = np.mean(ld["control_corrs"])
        avg_control_align = np.mean(ld["control_aligns"])
        std_control_corr = np.std(ld["control_corr_changes"])

        # Per-layer p-value
        n_controls_worse_local = sum(1 for c in ld["control_corr_changes"] if c >= ld["primary_corr_change"])
        p_value_local = (n_controls_worse_local + 1) / (len(ld["control_corrs"]) + 1)

        # Pooled p-value
        n_pooled_worse = np.sum(pooled_null >= ld["primary_corr_change"])
        p_value_pooled = (n_pooled_worse + 1) / (len(pooled_null) + 1)

        # Bootstrap 95% CI
        n_bootstrap = 1000
        bootstrap_means = []
        ctrl_changes = np.array(ld["control_corr_changes"])
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(ctrl_changes, size=len(ctrl_changes), replace=True)
            bootstrap_means.append(np.mean(boot_sample))
        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)

        # Effect size
        if std_control_corr > 0:
            effect_size_z = (ld["primary_corr_change"] - np.mean(ld["control_corr_changes"])) / std_control_corr
        else:
            effect_size_z = 0.0

        raw_p_values.append((layer_idx, p_value_pooled))

        analysis["effects"][layer_idx] = {
            "baseline": {
                "correlation": ld["baseline_corr"],
                "mean_alignment": float(ld["baseline_align"].mean()),
                "mean_confidence": float(ld["baseline_conf"].mean()),
            },
            f"{direction_key}_ablated": {
                "correlation": ld["ablated_corr"],
                "correlation_change": ld["primary_corr_change"],
                "mean_alignment": float(ld["ablated_align"].mean()),
                "alignment_change": float(ld["ablated_align"].mean() - ld["baseline_align"].mean()),
                "mean_confidence": float(ld["ablated_conf"].mean()),
                "p_value_local": p_value_local,
                "p_value_pooled": p_value_pooled,
                "effect_size_z": effect_size_z,
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

    # FDR correction
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer_idx] = adjusted

    prev_adjusted = 0.0
    for layer_idx, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer_idx] = max(fdr_adjusted[layer_idx], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer_idx]

    for layer_idx in results["layers"]:
        analysis["effects"][layer_idx][f"{direction_key}_ablated"]["p_value_fdr"] = fdr_adjusted[layer_idx]

    # Summary
    significant_layers_pooled = [l for l in results["layers"]
                                  if analysis["effects"][l][f"{direction_key}_ablated"]["p_value_pooled"] < 0.05]
    significant_layers_fdr = [l for l in results["layers"]
                              if analysis["effects"][l][f"{direction_key}_ablated"]["p_value_fdr"] < 0.05]

    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_pooled_p05": significant_layers_pooled,
        "significant_layers_fdr_p05": significant_layers_fdr,
        "n_significant_pooled": len(significant_layers_pooled),
        "n_significant_fdr": len(significant_layers_fdr),
    }

    return analysis


# =============================================================================
# PRINTING FUNCTIONS
# =============================================================================

def print_steering_summary(analysis: Dict):
    """Print summary of steering results with statistical significance."""
    direction_key = analysis.get("direction_key", "introspection")

    print("\n" + "=" * 70)
    print("STEERING EXPERIMENT RESULTS")
    print("=" * 70)

    if not analysis["layers"]:
        print("\n⚠ No layers were tested")
        return

    metric = analysis.get("metric")
    if metric:
        expected_sign = get_expected_slope_sign(metric)
        expected_direction = "negative" if expected_sign < 0 else "positive"
        print(f"\nMetric: {metric}")
        print(f"Expected slope sign: {expected_direction}")

    has_stats = "statistics" in analysis["effects"].get(analysis["layers"][0], {})

    if has_stats:
        print(f"\n--- Confidence Slopes by Layer (with significance) ---")
        print(f"{'Layer':<8} {f'{direction_key[:12]} Slope':<15} {'Ctrl (±SD)':<16} {'p(pooled)':<10} {'p(FDR)':<10} {'Z':<8} {'Sig':<5}")
        print("-" * 80)

        for layer in analysis["layers"]:
            s = analysis["effects"][layer]["slopes"]
            stats = analysis["effects"][layer]["statistics"]
            p_pooled = stats["p_value_pooled"]
            p_fdr = stats["p_value_fdr"]
            z = stats["effect_size_z"]

            if p_fdr < 0.01:
                sig = "**"
            elif p_fdr < 0.05:
                sig = "*"
            elif p_pooled < 0.05:
                sig = "~"
            else:
                sig = ""

            ctrl_str = f"{s['control_avg']:.4f}±{s['control_std']:.4f}"
            print(f"{layer:<8} {s[direction_key]:<15.4f} {ctrl_str:<16} {p_pooled:<10.4f} {p_fdr:<10.4f} {z:<8.2f} {sig:<5}")

        summary = analysis.get("summary", {})
        if summary:
            print(f"\nPooled null size: {summary.get('pooled_null_size', 'N/A')} control slopes")
            n_sig_fdr = summary.get("n_significant_fdr", 0)
            print(f"Significant layers (FDR<0.05): {n_sig_fdr}/{len(analysis['layers'])}")
            if summary.get("significant_layers_fdr_p05"):
                print(f"  FDR-significant: {summary['significant_layers_fdr_p05']}")

        print("\nLegend: ** p_FDR<0.01, * p_FDR<0.05, ~ p_pooled<0.05")

    # Best layer
    best_layer = max(analysis["layers"], key=lambda l: abs(analysis["effects"][l]["slopes"][direction_key]))
    best_slope = analysis["effects"][best_layer]["slopes"][direction_key]
    best_ctrl = analysis["effects"][best_layer]["slopes"]["control_avg"]

    print(f"\nStrongest steering effect: Layer {best_layer}")
    print(f"  Confidence slope: {best_slope:.4f}")
    print(f"  Control slope: {best_ctrl:.4f}")

    if has_stats:
        best_stats = analysis["effects"][best_layer]["statistics"]
        print(f"  p-value (FDR): {best_stats['p_value_fdr']:.4f}")
        print(f"  Effect size (Z): {best_stats['effect_size_z']:.2f}")


def print_ablation_summary(analysis: Dict):
    """Print summary of ablation results with statistical significance."""
    direction_key = analysis.get("direction_key", "introspection")
    ablated_key = f"{direction_key}_ablated"

    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENT RESULTS")
    print("=" * 70)

    if not analysis["layers"]:
        print("\n⚠ No layers were tested")
        return

    print(f"\n--- Correlation Change by Layer ---")
    print(f"{'Layer':<8} {'Baseline':<10} {f'{direction_key[:8]} Δ':<12} {'Ctrl Δ (±SD)':<16} {'p(FDR)':<10} {'Z':<8} {'Sig':<5}")
    print("-" * 80)

    for layer in analysis["layers"]:
        e = analysis["effects"][layer]
        baseline = e["baseline"]["correlation"]
        primary = e[ablated_key]
        ctrl = e["control_ablated"]

        p_fdr = primary.get("p_value_fdr", 1.0)
        z = primary.get("effect_size_z", 0.0)

        if p_fdr < 0.01:
            sig = "**"
        elif p_fdr < 0.05:
            sig = "*"
        else:
            sig = ""

        ctrl_str = f"{ctrl['correlation_change_mean']:.4f}±{ctrl['correlation_change_std']:.4f}"
        print(f"{layer:<8} {baseline:<10.4f} {primary['correlation_change']:<12.4f} {ctrl_str:<16} {p_fdr:<10.4f} {z:<8.2f} {sig:<5}")

    summary = analysis.get("summary", {})
    if summary:
        print(f"\nPooled null size: {summary.get('pooled_null_size', 'N/A')} control effects")
        n_sig_fdr = summary.get("n_significant_fdr", 0)
        print(f"Significant layers (FDR<0.05): {n_sig_fdr}/{len(analysis['layers'])}")
        if summary.get("significant_layers_fdr_p05"):
            print(f"  FDR-significant: {summary['significant_layers_fdr_p05']}")

    print("\nLegend: ** p_FDR<0.01, * p_FDR<0.05")
