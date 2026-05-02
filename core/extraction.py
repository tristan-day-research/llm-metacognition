"""
Activation and logit extraction utilities.

Provides BatchedExtractor for efficient combined extraction of:
- Layer activations (at last token position)
- Logits over specified option tokens
- Entropy computation
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute entropy from a probability distribution."""
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy)


class BatchedExtractor:
    """
    Extract activations and logits in a single batched forward pass.

    This class registers forward hooks on all model layers to capture
    hidden states, then extracts both activations and option probabilities
    in one forward pass (instead of separate passes).
    """

    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.activations = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures hidden states for a layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach()
        return hook

    def register_hooks(self):
        """Register forward hooks on all model layers."""
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
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_token_ids: List[int],
        token_positions: Optional[Dict[str, List[int]]] = None
    ) -> Tuple[Dict[str, List[Dict[int, np.ndarray]]], List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Extract activations AND compute option probabilities in one forward pass.

        Args:
            input_ids: (batch_size, seq_len) tensor of input token IDs
            attention_mask: (batch_size, seq_len) attention mask
            option_token_ids: List of token IDs for the answer options
            token_positions: Optional dict mapping position_name -> list of per-batch-item
                           token indices. If None, defaults to {"final": [-1, ...]}

        Returns:
            layer_activations: Dict[position_name, List[{layer_idx: activation}]]
            option_probs: List of probability arrays over options, one per batch item
            option_logits: List of raw logit arrays over options, one per batch item
            entropies: List of entropy values, one per batch item
        """
        self.activations = {}
        batch_size = input_ids.shape[0]

        # Default to final position for backward compatibility
        if token_positions is None:
            token_positions = {"final": [-1] * batch_size}

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        seq_len = input_ids.shape[1]

        # Extract activations at specified positions for each batch item
        # IMPORTANT: Positions are relative to unpadded sequence, but tensors are left-padded.
        # Must adjust for left-padding offset.

        # Pre-compute adjusted positions for all batch items and positions
        # This allows vectorized GPU indexing instead of per-item CPU transfers
        adjusted_positions = {}  # {pos_name: [adjusted_pos for each batch item]}
        for pos_name, positions in token_positions.items():
            adj_pos_list = []
            for batch_idx in range(batch_size):
                actual_len = int(attention_mask[batch_idx].sum())
                pad_offset = seq_len - actual_len
                pos = positions[batch_idx]

                if pos >= 0:
                    adjusted_pos = pos + pad_offset
                    if adjusted_pos >= seq_len:
                        raise ValueError(
                            f"Position {pos} (adjusted to {adjusted_pos}) out of bounds "
                            f"for sequence with {actual_len} tokens (padded to {seq_len})"
                        )
                else:
                    adjusted_pos = pos  # -1 already handles left-padding correctly
                adj_pos_list.append(adjusted_pos)
            adjusted_positions[pos_name] = adj_pos_list

        # Extract activations using vectorized indexing per layer (one GPU->CPU transfer per layer per position)
        # Instead of: num_layers * num_positions * batch_size transfers
        # Now: num_layers * num_positions transfers
        all_layer_activations = {pos_name: [{} for _ in range(batch_size)] for pos_name in token_positions}

        for layer_idx, acts in self.activations.items():
            for pos_name, adj_positions in adjusted_positions.items():
                # Gather all batch items for this position in one operation
                # acts shape: (batch_size, seq_len, hidden_dim)
                batch_indices = torch.arange(batch_size, device=acts.device)
                pos_indices = torch.tensor(adj_positions, device=acts.device)
                # Use advanced indexing to get (batch_size, hidden_dim) in one op
                extracted = acts[batch_indices, pos_indices, :].cpu().numpy()

                # Distribute to per-item dicts
                for batch_idx in range(batch_size):
                    all_layer_activations[pos_name][batch_idx][layer_idx] = extracted[batch_idx]

        # Extract logits and compute probabilities for each batch item (always at final position)
        all_probs = []
        all_logits = []
        all_entropies = []
        for batch_idx in range(batch_size):
            final_logits = outputs.logits[batch_idx, -1, :]
            option_logits = final_logits[option_token_ids]
            logits_np = option_logits.cpu().numpy()
            probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
            entropy = compute_entropy_from_probs(probs)
            all_probs.append(probs)
            all_logits.append(logits_np)
            all_entropies.append(entropy)

        return all_layer_activations, all_probs, all_logits, all_entropies

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False
