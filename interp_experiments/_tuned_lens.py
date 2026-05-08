"""
Tuned lens (Belrose et al. 2023) — per-layer affine that maps each
intermediate residual stream to a "final-layer-equivalent" form before
applying the model's final RMSNorm + unembedding (`W_U`).

Why bother:
  Vanilla logit lens degrades at early/mid layers because `W_U` was trained
  against the FINAL residual basis, not the intermediate ones. Projecting
  layer-5 residuals through W_U typically yields gibberish — the geometry
  of the residual stream evolves through the network. Tuned lens learns
  small per-layer affines `A_L`, `b_L` so that
      W_U @ FinalLN(A_L @ resid_L + b_L)
  matches the model's actual final-layer logits in KL divergence as
  closely as possible.

Public API (drop-in compatible with `_logit_lens.apply_logit_lens`):
  - `TunedLens` — `nn.Module` holding one `nn.Linear` per layer, identity-init.
  - `train_tuned_lens(model, layer_acts, ...)` — fits the affines using
    saved residuals as the calibration set. Returns the trained module +
    per-layer training-loss curves.
  - `save_tuned_lens(lens, path)` / `load_tuned_lens(path, device=...)` —
    serialise / restore.
  - `apply_tuned_lens(activations, model, tuned_lens, option_token_ids)` —
    same signature and return shape as `_logit_lens.apply_logit_lens`.

Calibration data: by default the user passes in the saved direct + meta
residuals from a `run_collect_activations.py` run. With ~3.4k samples
(direct + meta concatenated) and weight decay keeping `A_L` close to
identity, training is fast and stable.

Memory footprint: per-layer affine is `(d_model, d_model) + d_model`. For
Llama-3-8B (d=4096, 32 layers) that's ~537M parameters; on disk fp32
~2.1 GB, fp16 ~1.1 GB. Stored once per (model, run-subfolder) and reused
across analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from _logit_lens import get_final_layernorm, get_unembedding_matrix


# =============================================================================
# Module
# =============================================================================

class TunedLens(nn.Module):
    """Per-layer affine translators, identity-initialised.

    For a model with `num_layers` transformer blocks and hidden size
    `d_model`, holds `num_layers` independent `nn.Linear(d_model, d_model)`
    modules — one per layer. At init each is `W = I, b = 0`, so an
    untrained TunedLens is exactly equivalent to vanilla logit lens.
    """

    def __init__(self, num_layers: int, d_model: int):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.translators = nn.ModuleList()
        for _ in range(num_layers):
            lin = nn.Linear(d_model, d_model, bias=True)
            with torch.no_grad():
                nn.init.eye_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.translators.append(lin)

    def forward(self, resid: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.translators[layer_idx](resid)


# =============================================================================
# Training
# =============================================================================

def train_tuned_lens(
    model,
    layer_activations: Dict[int, np.ndarray],
    *,
    n_epochs: int = 20,
    lr: float = 5e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Tuple[TunedLens, Dict[int, List[float]]]:
    """Fit per-layer affines so lensed logits match final-layer logits.

    Args:
        model: HuggingFace causal LM. Used (frozen) for `lm_head` and the
            final-norm module. We do NOT modify the model — only read its
            weights to compute targets and apply unembed during training.
        layer_activations: `{layer_idx: (n_samples, d_model) np.ndarray}` —
            saved residuals from `run_collect_activations.py`. The
            highest layer index is treated as the target ("final") basis.
        n_epochs: passes over the calibration set per layer.
        lr: Adam learning rate.
        batch_size: mini-batch size.
        weight_decay: L2 reg on affine weights — keeps `W` near identity
            given the limited (~few k) calibration samples typical here.
        device: where to place tensors (defaults to model's device).

    Returns:
        tuned_lens: trained `TunedLens` on `device`.
        losses_per_layer: `{layer_idx: list[float]}` mini-batch losses.
    """
    if device is None:
        device = next(model.parameters()).device

    layers = sorted(layer_activations.keys())
    num_layers = max(layers) + 1
    n_samples, d_model = layer_activations[layers[0]].shape

    final_norm = get_final_layernorm(model)

    # Compute the "target" final-layer log-probs once. The final layer's
    # residual stream IS what the model's unembedding was trained for, so
    # this is the gold standard each lensed layer should approximate.
    final_resid = torch.from_numpy(layer_activations[max(layers)]).to(device).float()
    with torch.no_grad():
        final_logits = model.lm_head(final_norm(final_resid))
        final_log_probs = F.log_softmax(final_logits, dim=-1).detach()
    del final_logits, final_resid

    tuned = TunedLens(num_layers, d_model).to(device).float()
    losses_per_layer: Dict[int, List[float]] = {}

    # Train each layer independently — they share no parameters with each other.
    # Iterate layers[:-1]: the last layer is identity by construction, so we
    # skip it (its translator stays at the identity init).
    pbar = tqdm(layers[:-1], desc="Training tuned lens")
    for layer_idx in pbar:
        resid = torch.from_numpy(layer_activations[layer_idx]).to(device).float()
        params = list(tuned.translators[layer_idx].parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        layer_losses = []
        for _epoch in range(n_epochs):
            perm = torch.randperm(n_samples, device=device)
            for b_start in range(0, n_samples, batch_size):
                idx = perm[b_start : b_start + batch_size]
                x = resid[idx]
                target = final_log_probs[idx]

                tuned_resid = tuned(x, layer_idx)
                tuned_logits = model.lm_head(final_norm(tuned_resid))
                pred = F.log_softmax(tuned_logits, dim=-1)

                loss = F.kl_div(pred, target, reduction="batchmean", log_target=True)
                opt.zero_grad()
                loss.backward()
                opt.step()
                layer_losses.append(loss.item())

        losses_per_layer[layer_idx] = layer_losses
        pbar.set_postfix({"L": layer_idx,
                          "init": f"{layer_losses[0]:.3f}",
                          "final": f"{layer_losses[-1]:.3f}"})
        del resid

    return tuned, losses_per_layer


# =============================================================================
# Persistence
# =============================================================================

def save_tuned_lens(lens: TunedLens, path: str) -> None:
    """Serialise the lens (state dict + shape metadata) to a single .pt file."""
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in lens.state_dict().items()},
        "num_layers": lens.num_layers,
        "d_model": lens.d_model,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_tuned_lens(path: str, device: Optional[torch.device] = None) -> TunedLens:
    """Reconstruct a TunedLens from disk."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    lens = TunedLens(payload["num_layers"], payload["d_model"])
    lens.load_state_dict(payload["state_dict"])
    if device is not None:
        lens = lens.to(device)
    lens.eval()
    return lens


# =============================================================================
# Inference (drop-in for `_logit_lens.apply_logit_lens`)
# =============================================================================

def apply_tuned_lens(
    activations: Dict[int, np.ndarray],
    model,
    tuned_lens: TunedLens,
    option_token_ids: List[List[int]],
    *,
    top_k: int = 20,
    chunk_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Drop-in replacement for `_logit_lens.apply_logit_lens` using a tuned lens.

    For each layer L's saved residuals, projects through the tuned-lens
    affine + the model's final RMSNorm + unembedding to get per-question
    per-layer logits over the vocabulary. Per-option logits are aggregated
    via `logsumexp` over each option's sub-token IDs (matches the
    eval-pipeline convention).

    Args + return shape are identical to `_logit_lens.apply_logit_lens`:
        option_logits: (n_questions, n_layers, n_options)  float32 (aggregated)
        top_k_ids:     (n_questions, n_layers, top_k)       int32
        top_k_logits:  (n_questions, n_layers, top_k)       float32
        layer_indices: (n_layers,)                          int32
    """
    layers = sorted(activations.keys())
    n_layers = len(layers)
    n_questions = activations[layers[0]].shape[0]
    n_options = len(option_token_ids)

    W_U, b_U = get_unembedding_matrix(model)
    final_norm = get_final_layernorm(model)
    device = W_U.device
    dtype = W_U.dtype
    option_ids_per_opt = [
        torch.tensor(ids, device=device, dtype=torch.long) for ids in option_token_ids
    ]

    option_logits_out = np.zeros((n_questions, n_layers, n_options), dtype=np.float32)
    top_k_ids_out = np.zeros((n_questions, n_layers, top_k), dtype=np.int32)
    top_k_logits_out = np.zeros((n_questions, n_layers, top_k), dtype=np.float32)

    tuned_lens.eval()
    with torch.no_grad():
        for li, layer_idx in enumerate(layers):
            acts = activations[layer_idx]
            for start in range(0, n_questions, chunk_size):
                end = min(start + chunk_size, n_questions)
                vecs = torch.from_numpy(acts[start:end]).to(device=device, dtype=torch.float32)

                # Tuned-lens affine in fp32 (numerical stability), then
                # downcast to the model's dtype for the final norm + unembed.
                tuned_vecs = tuned_lens(vecs, layer_idx).to(dtype)
                logits = model.lm_head(final_norm(tuned_vecs))

                # Aggregate sub-token logits per option via logsumexp
                # (matches `finetune.run_evaluations.run_mcq_forward_pass`).
                opt_logits = torch.stack(
                    [torch.logsumexp(logits.index_select(1, ids_t), dim=1)
                     for ids_t in option_ids_per_opt],
                    dim=1,
                )
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
