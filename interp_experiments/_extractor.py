"""
Batched activation/logit extraction with optional KV-cache prefix sharing.

This module isolates the GPU-touching machinery used by the introspection
runners:
  - `extract_cache_tensors` / `create_fresh_cache`: shape-shifting helpers for
    transformers' evolving cache types (DynamicCache attr layouts, legacy
    tuple-of-tuples, iterables) so callers don't need to special-case each.
  - `BatchedExtractor`: registers forward hooks that grab last-token residuals
    at every layer in a single forward pass, and computes option-level
    probabilities/metrics from the final logits with one CPU transfer.

Pure of project-level config (no `MODEL_NAME`, `OUTPUTS_DIR`, etc.). Imports
only `_metrics` for the option-level metric computation.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from _metrics import compute_uncertainty_metrics

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


def extract_cache_tensors(past_key_values):
    """Extract immutable tensors from cache object (compatible with all transformers versions)."""
    # 1. DynamicCache with .key_cache / .value_cache attrs
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(past_key_values.key_cache), list(past_key_values.value_cache)
    # 2. DynamicCache with .layers attr (newer transformers)
    if hasattr(past_key_values, "layers"):
        keys, values = [], []
        for layer in past_key_values.layers:
            # DynamicLayer may store as .key_cache/.value_cache or as a tuple
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
                # Inspect what attributes the layer actually has
                tensor_attrs = [a for a in dir(layer) if not a.startswith("_") and isinstance(getattr(layer, a, None), torch.Tensor)]
                if len(tensor_attrs) >= 2:
                    keys.append(getattr(layer, tensor_attrs[0]))
                    values.append(getattr(layer, tensor_attrs[1]))
                else:
                    raise ValueError(
                        f"Cannot extract k/v from {type(layer).__name__}. "
                        f"Attrs: {[a for a in dir(layer) if not a.startswith('_')]}"
                    )
        return keys, values
    # 3. Iterable cache (yields (key, value) tuples)
    if hasattr(past_key_values, "__iter__") and hasattr(past_key_values, "__len__"):
        keys, values = [], []
        for item in past_key_values:
            if isinstance(item, tuple) and len(item) == 2:
                keys.append(item[0])
                values.append(item[1])
            else:
                raise ValueError(f"Unexpected cache item type: {type(item)}")
        return keys, values
    # 4. Legacy tuple-of-tuples
    if hasattr(past_key_values, "to_legacy_cache"):
        return extract_cache_tensors(past_key_values.to_legacy_cache())
    keys, values = [], []
    for i in range(len(past_key_values)):
        k, v = past_key_values[i]
        keys.append(k)
        values.append(v)
    return keys, values


def create_fresh_cache(key_tensors, value_tensors, expand_size=1):
    """Reconstruct a fresh cache object from tensors."""
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


class BatchedExtractor:
    """Extract activations and logits in a single batched forward pass.

    Optimized to:
    1. Store only last-token activations in hooks (reduces memory by seq_len×)
    2. Do single CPU transfer per batch (reduces GPU syncs from layers×batch to 1)
    """

    def __init__(self, model, num_layers: int, tokenizer=None):
        self.model = model
        self.num_layers = num_layers
        self.tokenizer = tokenizer
        self.activations = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # With left-padding, last position (-1) is always the final real token
            # Store only last-token activations: (batch_size, hidden_dim)
            self.activations[layer_idx] = hidden_states[:, -1, :].detach()
        return hook

    def _compute_extended_metrics(self, final_logits: torch.Tensor, option_token_ids: List[List[int]], metrics: dict):
        """Add full-vocab metrics to an existing metrics dict. Operates on GPU tensors.

        `option_token_ids` is a list of token-id GROUPS, one per option (e.g. [' A', 'A']
        for option "A" if both encode as single tokens). Sub-token aggregation matches
        finetune.run_evaluations → run_mcq_forward_pass: logsumexp BEFORE log_softmax,
        so option_logprobs sum the probability mass across all surface variants of each
        option.
        """
        full_log_probs = torch.log_softmax(final_logits, dim=-1)
        metrics["option_logprobs"] = [
            torch.logsumexp(full_log_probs[ids], dim=-1).item()
            for ids in option_token_ids
        ]

        full_probs = torch.softmax(final_logits, dim=-1)
        metrics["entropy_full_vocab"] = -(full_probs * full_probs.log().nan_to_num()).sum().item()

        sorted_olp = sorted(metrics["option_logprobs"], reverse=True)
        metrics["top2_margin_logprob"] = (sorted_olp[0] - sorted_olp[1]) if len(sorted_olp) > 1 else sorted_olp[0]

        top20_vals, top20_ids = torch.topk(final_logits, 20)
        top20_ids_list = top20_ids.cpu().tolist()
        top20_vals_list = top20_vals.cpu().tolist()
        if self.tokenizer:
            top20_strs = [self.tokenizer.decode([tid]) for tid in top20_ids_list]
        else:
            top20_strs = [""] * 20
        metrics["top20_logits"] = list(zip(top20_ids_list, top20_strs, top20_vals_list))

    def register_hooks(self):
        if hasattr(self.model, 'get_base_model'):
            base = self.model.get_base_model()
            layers = base.model.layers
        else:
            layers = self.model.model.layers

        for i, layer in enumerate(layers):
            hook = self._make_hook(i)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_token_ids: List[List[int]]
    ) -> Tuple[List[Dict[int, np.ndarray]], List[np.ndarray], List[np.ndarray], List[Dict[str, float]]]:
        """
        Extract activations AND compute option probabilities/metrics in one forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            option_token_ids: list of token-id GROUPS, one group per option. Each
                group is a list of vocabulary IDs whose probability should be
                summed for that option (e.g. [" A", "A"] both → option "A").
                Aggregation is done with logsumexp BEFORE softmax to match
                finetune/run_evaluations → run_mcq_forward_pass.

        Returns:
            layer_activations: List of {layer_idx: activation} dicts, one per batch item
            option_probs: List of probability arrays, one per batch item
            option_logits: List of (aggregated) logit arrays, one per batch item
            all_metrics: List of metric dicts, one per batch item
        """
        self.activations = {}
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Single CPU transfer: stack all layers and transfer at once
        # self.activations[layer_idx] is already (batch_size, hidden_dim) from optimized hook
        stacked = torch.stack([self.activations[i] for i in range(self.num_layers)], dim=0)
        # stacked shape: (num_layers, batch_size, hidden_dim)
        stacked_cpu = stacked.cpu().numpy()

        # Distribute to per-batch-item dicts
        all_layer_activations = []
        for batch_idx in range(batch_size):
            item_activations = {
                layer_idx: stacked_cpu[layer_idx, batch_idx]
                for layer_idx in range(self.num_layers)
            }
            all_layer_activations.append(item_activations)

        # Extract logits and compute probabilities/metrics for each batch item.
        # Per-option logit = logsumexp over the option's sub-token IDs.
        all_probs = []
        all_logits = []
        all_metrics = []
        for batch_idx in range(batch_size):
            final_logits = outputs.logits[batch_idx, -1, :]
            option_logits = torch.stack([
                torch.logsumexp(final_logits[ids], dim=-1) for ids in option_token_ids
            ], dim=0)
            option_logits_np = option_logits.cpu().numpy()
            probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
            metrics = compute_uncertainty_metrics(probs, option_logits_np)
            self._compute_extended_metrics(final_logits, option_token_ids, metrics)
            all_probs.append(probs)
            all_logits.append(option_logits_np)
            all_metrics.append(metrics)

        return all_layer_activations, all_probs, all_logits, all_metrics

    def compute_prefix_cache(self, input_ids: torch.Tensor):
        """Run shared prefix once to get cache snapshot (no hooks)."""
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        return extract_cache_tensors(outputs.past_key_values)

    def extract_batch_with_cache(
        self,
        suffix_ids: torch.Tensor,
        prefix_cache_data: Tuple,
        option_token_ids: List[List[int]],
        pad_token_id: int = 0
    ) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Extract activations AND compute option probabilities/metrics using KV cache prefix.

        Args:
            suffix_ids: (batch_size, suffix_len) - left-padded suffix token IDs
            prefix_cache_data: (keys, values) tuple from compute_prefix_cache
            option_token_ids: list of token-id GROUPS, one group per option.
                Each group's probabilities are summed via logsumexp before
                softmax (matches run_evaluations.run_mcq_forward_pass).
            pad_token_id: Token ID used for padding (default 0)

        Returns:
            Same as extract_batch
        """
        self.activations = {}
        batch_size = suffix_ids.shape[0]

        keys, values = prefix_cache_data
        prefix_len = keys[0].shape[2]
        suffix_len = suffix_ids.shape[1]

        # Build attention mask for prefix + suffix
        mask = torch.ones((batch_size, prefix_len + suffix_len), dtype=torch.long, device=suffix_ids.device)
        # Handle left-padding in suffix
        mask[:, prefix_len:] = (suffix_ids != pad_token_id).long()

        with torch.no_grad():
            outputs = self.model(
                input_ids=suffix_ids,
                attention_mask=mask,
                past_key_values=create_fresh_cache(keys, values, expand_size=batch_size),
                use_cache=False
            )

        # Single CPU transfer: stack all layers and transfer at once
        stacked = torch.stack([self.activations[i] for i in range(self.num_layers)], dim=0)
        stacked_cpu = stacked.cpu().numpy()

        # Distribute to per-batch-item dicts
        all_layer_activations = []
        for batch_idx in range(batch_size):
            item_activations = {
                layer_idx: stacked_cpu[layer_idx, batch_idx]
                for layer_idx in range(self.num_layers)
            }
            all_layer_activations.append(item_activations)

        # Extract logits and compute probabilities/metrics for each batch item.
        # Per-option logit = logsumexp over the option's sub-token IDs.
        all_probs = []
        all_logits = []
        all_metrics = []
        for batch_idx in range(batch_size):
            final_logits = outputs.logits[batch_idx, -1, :]
            option_logits = torch.stack([
                torch.logsumexp(final_logits[ids], dim=-1) for ids in option_token_ids
            ], dim=0)
            option_logits_np = option_logits.cpu().numpy()
            probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
            metrics = compute_uncertainty_metrics(probs, option_logits_np)
            self._compute_extended_metrics(final_logits, option_token_ids, metrics)
            all_probs.append(probs)
            all_logits.append(option_logits_np)
            all_metrics.append(metrics)

        return all_layer_activations, all_probs, all_logits, all_metrics
