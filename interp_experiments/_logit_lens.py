"""
Apply the logit lens to a batch of saved last-token activations.

The introspection runner already extracts the residual at each layer for the
final token of every prompt and stores it as `(n_questions, hidden_dim)`. The
logit lens projection is then almost free: one `(n_questions, hidden_dim) @
(hidden_dim, vocab_size)` matmul per layer. We only persist
  - the option-token logits per layer (tracks the answer-token trajectory),
  - the top-K tokens per layer (so post-hoc analyses can see what else the
    model is "thinking" mid-stack)
which keeps the artefact under ~1 MB per (prompt-type, model, dataset).

Reuses unembed/LN extraction from `core.logit_lens` so per-question lens
projection in the introspection runner stays consistent with the helpers.
"""

from typing import Dict, List

import numpy as np
import torch

from core.logit_lens import get_final_layernorm, get_unembedding_matrix


def apply_logit_lens(
    activations: Dict[int, np.ndarray],
    model,
    option_token_ids: List[List[int]],
    *,
    top_k: int = 20,
    ln_mode: str = "final_ln",
    chunk_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Project per-layer last-token activations through the unembedding matrix.

    Args:
        activations: {layer_idx: (n_questions, hidden_dim) float array}
        model: HuggingFace causal LM. The unembed weight (and final norm,
            when ln_mode='final_ln') are read off this model.
        option_token_ids: list of token-id GROUPS, one group per option. Each
            group is a list of vocab IDs whose lensed logits should be summed
            (logsumexp) for that option — same surface-form aggregation as
            finetune.run_evaluations.run_mcq_forward_pass. So an option "A"
            with both " A" and "A" as single-token forms becomes one logit
            per layer that sums the probability mass across both.
        top_k: also store the top-K tokens at every layer (for diagnostics).
        ln_mode: 'final_ln' applies the model's final RMSNorm before unembed
            (the standard "logit lens" definition); 'none' uses raw residuals.
        chunk_size: question chunk for the matmul to bound peak GPU memory.

    Returns:
        Dict of np arrays:
          option_logits: (n_questions, n_layers, n_options) float32 — aggregated
          top_k_ids:     (n_questions, n_layers, top_k)    int32
          top_k_logits:  (n_questions, n_layers, top_k)    float32
          layer_indices: (n_layers,)                        int32
    """
    layers = sorted(activations.keys())
    n_layers = len(layers)
    n_questions = activations[layers[0]].shape[0]
    n_options = len(option_token_ids)

    W_U, b_U = get_unembedding_matrix(model)  # (d_model, vocab_size)
    final_ln = get_final_layernorm(model) if ln_mode in ("final_ln", "model_default") else None

    device = W_U.device
    dtype = W_U.dtype
    option_ids_per_opt = [
        torch.tensor(ids, device=device, dtype=torch.long) for ids in option_token_ids
    ]

    option_logits_out = np.zeros((n_questions, n_layers, n_options), dtype=np.float32)
    top_k_ids_out = np.zeros((n_questions, n_layers, top_k), dtype=np.int32)
    top_k_logits_out = np.zeros((n_questions, n_layers, top_k), dtype=np.float32)

    with torch.no_grad():
        for li, layer_idx in enumerate(layers):
            acts = activations[layer_idx]
            for start in range(0, n_questions, chunk_size):
                end = min(start + chunk_size, n_questions)
                vecs = torch.from_numpy(acts[start:end]).to(device=device, dtype=dtype)

                if final_ln is not None:
                    vecs = final_ln(vecs)

                # (chunk, d_model) @ (d_model, vocab) -> (chunk, vocab)
                logits = vecs @ W_U
                if b_U is not None:
                    logits = logits + b_U

                # Aggregate sub-token logits per option via logsumexp.
                opt_logits = torch.stack([
                    torch.logsumexp(logits.index_select(1, ids_t), dim=1)
                    for ids_t in option_ids_per_opt
                ], dim=1)  # (chunk, n_options)
                option_logits_out[start:end, li, :] = opt_logits.float().cpu().numpy()

                topk_vals, topk_ids = torch.topk(logits, top_k, dim=1)
                top_k_logits_out[start:end, li, :] = topk_vals.float().cpu().numpy()
                top_k_ids_out[start:end, li, :] = topk_ids.cpu().numpy().astype(np.int32)

    return {
        "option_logits": option_logits_out,
        "top_k_ids": top_k_ids_out,
        "top_k_logits": top_k_logits_out,
        "layer_indices": np.array(layers, dtype=np.int32),
    }
