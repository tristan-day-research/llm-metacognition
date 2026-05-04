
# --- repo path bootstrap (so root-level imports like `prompts`,
# `finetune_config` resolve when run from anywhere) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import math
import torch


# ============================================================
# Bin specs — one per supported CONFIDENCE_FORMAT
# ============================================================
# Midpoints/widths are in PERCENTAGE units (0-100) because the Gaussian kernel
# in convert_entropy_to_soft_labels operates on a "confidence percentage"
# derived from MCQ entropy. The MSE loss uses the same midpoints, normalized
# to [0, 1].
#
# letter_8bin   — non-uniform bins matching the prompt
#                 "<5%, 5-10%, 10-20%, 20-40%, 40-60%, 60-80%, 80-90%, >90%"
# 1-5   — 5 uniform bins of 20% each
# 1-10  — 10 uniform bins of 10% each

_BIN_SPECS = {
    "letter_8bin": {
        "n_bins": 8,
        "midpoints": [2.5, 7.5, 15.0, 30.0, 50.0, 70.0, 85.0, 95.0],
        "widths":    [5.0, 5.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0],
    },
    "1-5": {
        "n_bins": 5,
        "midpoints": [10.0, 30.0, 50.0, 70.0, 90.0],
        "widths":    [20.0, 20.0, 20.0, 20.0, 20.0],
    },
    "1-10": {
        "n_bins": 10,
        "midpoints": [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0],
        "widths":    [10.0] * 10,
    },
}


def get_bin_spec(confidence_format: str) -> dict:
    """Return {n_bins, midpoints, widths} for the given format."""
    if confidence_format not in _BIN_SPECS:
        raise ValueError(
            f"Unknown confidence_format {confidence_format!r}. "
            f"Must be one of: {sorted(_BIN_SPECS)}"
        )
    return _BIN_SPECS[confidence_format]


def _bin_tensors(confidence_format: str, device, dtype):
    spec = get_bin_spec(confidence_format)
    midpoints = torch.tensor(spec["midpoints"], device=device, dtype=dtype)
    widths = torch.tensor(spec["widths"], device=device, dtype=dtype)
    return midpoints, widths, spec["n_bins"]


# ============================================================
# Entropy → soft confidence labels
# ============================================================

def compute_entropy_from_logits(logits4: torch.Tensor) -> torch.Tensor:
    """
    Given [batch, 4] logits, compute entropy of the softmax distribution.
    This is the ONLY canonical entropy function used in both training + eval.
    """
    probs4 = torch.softmax(logits4, dim=-1)
    entropy = -torch.sum(probs4 * torch.log(probs4 + 1e-12), dim=-1)
    return entropy


def compute_soft_labels(
    logits4: torch.Tensor,
    *,
    sigma: float,
    confidence_format: str = "letter_8bin",
):
    """
    Convert 4-way answer logits into a soft N-bin confidence distribution
    (where N is determined by confidence_format).

    Pipeline: logits4 → probs → entropy → soft N-bin Gaussian targets.

    Args:
        logits4: tensor of shape [..., 4] with logits for A, B, C, D
        sigma: Gaussian width in percentage space (REQUIRED)
        confidence_format: "letter_8bin" / "1-5" / "1-10".

    Returns:
        soft_targets of shape [..., N] where N matches the format's bin count.
    """
    if sigma is None:
        raise ValueError("sigma must be explicitly provided to compute_soft_labels().")
    if not isinstance(logits4, torch.Tensor):
        raise TypeError("logits4 must be a torch.Tensor.")

    device = logits4.device
    dtype = logits4.dtype

    probs4 = torch.softmax(logits4, dim=-1)
    entropy = -torch.sum(probs4 * torch.log(probs4 + 1e-12), dim=-1)

    midpoints, widths, _ = _bin_tensors(confidence_format, device, dtype)

    # entropy → "certainty" percentage
    max_entropy = math.log(4)  # 4 MCQ choices
    confidence_percent = (1.0 - entropy / max_entropy) * 100.0

    distances = (midpoints - confidence_percent.unsqueeze(-1)) ** 2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * widths

    return weights / torch.sum(weights, dim=-1, keepdim=True)


def convert_entropy_to_soft_labels(
    entropy: torch.Tensor,
    *,
    sigma: float,
    confidence_format: str = "letter_8bin",
):
    """Convert pre-computed entropy values into soft N-bin labels.

    Same Gaussian kernel as compute_soft_labels but takes entropy directly
    (used during eval where entropy was computed once and reused)."""
    if sigma is None:
        raise ValueError("sigma must be provided")
    if not isinstance(entropy, torch.Tensor):
        entropy = torch.as_tensor(entropy, dtype=torch.float32)

    device = entropy.device
    dtype = entropy.dtype

    midpoints, widths, _ = _bin_tensors(confidence_format, device, dtype)

    max_entropy = math.log(4)
    # Clamp guards against float-arithmetic drift producing values slightly
    # outside [0, 1], which would otherwise pull soft mass onto the wrong end.
    confidence_percent = ((1.0 - entropy / max_entropy).clamp(0.0, 1.0)) * 100.0

    if confidence_percent.ndim == 0:
        distances = (midpoints - confidence_percent) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * widths
    else:
        distances = (midpoints.unsqueeze(0) - confidence_percent.unsqueeze(-1)) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * widths

    return weights / torch.sum(weights, dim=-1, keepdim=True)


def build_soft_targets_from_entropy(
    entropy: torch.Tensor,
    sigma: float,
    confidence_format: str = "letter_8bin",
) -> torch.Tensor:
    """Small wrapper so both train_step and run_evaluation use the same call site."""
    return convert_entropy_to_soft_labels(
        entropy, sigma=sigma, confidence_format=confidence_format
    )


# ============================================================
# Alternative finetuning targets: top_logit and logit_gap
# ============================================================

def _confidence_percent_to_soft_labels(
    confidence_percent: torch.Tensor,
    *,
    sigma: float,
    confidence_format: str = "letter_8bin",
) -> torch.Tensor:
    """Shared Gaussian-kernel step: confidence_percent [0,100] → soft N-bin labels."""
    device = confidence_percent.device
    dtype = confidence_percent.dtype
    midpoints, widths, _ = _bin_tensors(confidence_format, device, dtype)
    if confidence_percent.ndim == 0:
        distances = (midpoints - confidence_percent) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * widths
    else:
        distances = (midpoints.unsqueeze(0) - confidence_percent.unsqueeze(-1)) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * widths
    return weights / weights.sum(dim=-1, keepdim=True)


def top_prob_to_confidence_percent(top_prob: torch.Tensor) -> torch.Tensor:
    """Top MCQ probability [0.25, 1.0] → confidence_percent [0, 100].

    0.25 = uniform over 4 choices (fully uncertain), 1.0 = certain.
    """
    return ((top_prob - 0.25) / 0.75).clamp(0.0, 1.0) * 100.0


def logit_gap_to_confidence_percent(
    top_prob: torch.Tensor, second_prob: torch.Tensor
) -> torch.Tensor:
    """Log-probability gap (≈ logit gap) [0, ∞) → confidence_percent [0, 100].

    gap = log(top_prob) - log(second_prob) = log(top_prob / second_prob).
    Mapped via tanh(gap/2): gap=0 → 0%, gap≈2 → 76%, gap→∞ → 100%.
    """
    gap = torch.log(top_prob.clamp(min=1e-12)) - torch.log(second_prob.clamp(min=1e-12))
    return torch.tanh(gap / 2.0) * 100.0


def build_soft_targets(
    finetuning_target: str,
    *,
    entropy: torch.Tensor = None,
    top_prob: torch.Tensor = None,
    second_prob: torch.Tensor = None,
    sigma: float,
    confidence_format: str = "letter_8bin",
) -> torch.Tensor:
    """Build soft N-bin targets from one of three supported signals.

    finetuning_target: "entropy" | "top_logit" | "logit_gap"
    """
    if finetuning_target == "entropy":
        if entropy is None:
            raise ValueError("entropy required for finetuning_target='entropy'")
        confidence_percent = (
            (1.0 - entropy / math.log(4)).clamp(0.0, 1.0) * 100.0
        )
    elif finetuning_target == "top_logit":
        if top_prob is None:
            raise ValueError("top_prob required for finetuning_target='top_logit'")
        confidence_percent = top_prob_to_confidence_percent(top_prob)
    elif finetuning_target == "logit_gap":
        if top_prob is None or second_prob is None:
            raise ValueError("top_prob and second_prob required for finetuning_target='logit_gap'")
        confidence_percent = logit_gap_to_confidence_percent(top_prob, second_prob)
    else:
        raise ValueError(
            f"Unknown finetuning_target {finetuning_target!r}. "
            "Must be one of: 'entropy', 'top_logit', 'logit_gap'"
        )
    return _confidence_percent_to_soft_labels(
        confidence_percent, sigma=sigma, confidence_format=confidence_format
    )


# ============================================================
# Loss computation
# ============================================================

def compute_gaussian_soft_bin_ce_loss(logits, soft_targets, reduction='mean'):
    """Cross-entropy between logits and soft target distribution.

    Bin-count agnostic: shapes must match ([B, N] for both logits and soft_targets).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)  # [B]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_scalar_confidence_mse_loss(
    logits,
    entropy,
    reduction='mean',
    confidence_format: str = "letter_8bin",
):
    """MSE between predicted scalar confidence (from logits) and target
    confidence derived from entropy. Bin midpoints come from the format.
    """
    p = torch.softmax(logits, dim=-1)  # [B, N]
    midpoints, _, _ = _bin_tensors(confidence_format, logits.device, torch.float32)
    midpoints = midpoints / 100.0  # to [0, 1]

    pred_conf = (p * midpoints).sum(dim=-1)  # [B]
    target_conf = 1.0 - entropy / math.log(4)  # [B]

    mse = (pred_conf - target_conf) ** 2  # [B]

    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_loss(
    logits,
    soft_targets=None,
    entropy=None,
    loss_type=None,
    reduction='mean',
    confidence_format: str = "letter_8bin",
):
    """Dispatch to gaussian_soft_bin_ce or scalar_confidence_mse.

    Both paths are bin-count agnostic via confidence_format.
    """
    if loss_type is None:
        raise ValueError(
            "loss_type parameter is REQUIRED for compute_loss(). "
            "Must be either 'gaussian_soft_bin_ce' or 'scalar_confidence_mse'. "
            "No default value to prevent silent training errors."
        )
    if loss_type == 'gaussian_soft_bin_ce':
        if soft_targets is None:
            raise ValueError("soft_targets required for gaussian_soft_bin_ce")
        return compute_gaussian_soft_bin_ce_loss(logits, soft_targets, reduction)

    elif loss_type == 'scalar_confidence_mse':
        if entropy is None:
            raise ValueError("entropy required for scalar_confidence_mse")
        return compute_scalar_confidence_mse_loss(
            logits, entropy, reduction, confidence_format=confidence_format
        )

    else:
        raise ValueError(f"Unknown loss_type {loss_type}")
