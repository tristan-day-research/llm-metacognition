import math
import torch


# ============================================================
# Entropy → scalar confidence → soft labels
# ============================================================

def compute_soft_labels(logits4, sigma=10.0):
    """
    Convert 4-way answer logits into soft 8-bin confidence distribution.

    Uses percentage-based Gaussian kernel to create soft labels.

    Args:
        logits4: tensor of shape [4] with logits for A, B, C, D
        sigma: Gaussian width in percentage space (default: 10)

    Returns:
        tensor of shape [8] with soft label distribution
    """
    # 1. Softmax over the 4 MCQ options
    probs = torch.softmax(logits4, dim=0)

    # 2. Entropy (natural logs)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()

    # 3. Convert entropy to "confidence percentage"
    #    confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

    # 4. Bin midpoints + widths (exact values from your colleague)
    bin_edges = torch.tensor([0, 5, 10, 20, 40, 60, 80, 90, 100],
                             dtype=torch.float32,
                             device=logits4.device)
    bin_midpoints = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95],
                                 dtype=torch.float32,
                                 device=logits4.device)
    bin_widths = bin_edges[1:] - bin_edges[:-1]   # shape [8]

    # 5. Gaussian kernel in percentage space
    distances = (bin_midpoints - confidence_percent)**2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    return weights / weights.sum()


def convert_entropy_to_soft_labels(entropy, sigma=10.0):
    """
    Convert entropy value to soft 8-bin confidence distribution.
    Handles both Tensor inputs (training) and float inputs (evaluation).
    """
    # Fix: Ensure input is a tensor so we can access .device or operate on it
    if not isinstance(entropy, torch.Tensor):
        entropy = torch.tensor(entropy, dtype=torch.float32)

    # Get device from entropy tensor (defaults to cpu if created from float)
    device = entropy.device
    
    # Convert entropy to "confidence percentage"
    # confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

    # Bin midpoints + widths
    bin_edges = torch.tensor(
        [0, 5, 10, 20, 40, 60, 80, 90, 100],
        dtype=torch.float32,
        device=device
    )
    bin_midpoints = torch.tensor(
        [2.5, 7.5, 15, 30, 50, 70, 85, 95],
        dtype=torch.float32,
        device=device
    )
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Gaussian kernel in percentage space
    # Handle broadcasting for both scalar [1] and batched [B] inputs
    if entropy.ndim > 0:
        distances = (bin_midpoints.unsqueeze(0) - confidence_percent.unsqueeze(-1)) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths.unsqueeze(0)
    else:
        # Scalar case (often hits here during simple eval loops)
        distances = (bin_midpoints - confidence_percent) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    # Normalize along the last dimension
    return weights / weights.sum(dim=-1, keepdim=True)


# ============================================================
# Loss computation
# ============================================================

def compute_gaussian_soft_bin_ce_loss(logits8, soft_targets, reduction='mean'):
    """
    Compute cross-entropy loss between logits8 and soft targets (Gaussian soft bin distribution).
    
    Args:
        logits8: Tensor of shape [B, 8] with logits for confidence bins A-H
        soft_targets: Tensor of shape [B, 8] with soft target distribution
        reduction: 'mean' to average over batch, 'none' to return per-sample losses, 
                  'sum' to sum over batch
    
    Returns:
        Loss tensor depending on reduction
    """
    log_probs = torch.log_softmax(logits8, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)  # [B]
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_scalar_confidence_mse_loss(logits8, entropy, reduction='mean'):
    """
    Compute MSE loss between:
        - predicted scalar confidence from logits8
        - target scalar confidence derived from entropy
    
    Args:
        logits8: Tensor of shape [B, 8] with logits for confidence bins A-H
        entropy: Tensor of shape [B] with entropy values
        reduction: 'mean' to average over batch, 'none' to return per-sample losses, 
                  'sum' to sum over batch
    
    Returns:
        Loss tensor depending on reduction
    """
    # softmax → distribution over 8 bins
    p = torch.softmax(logits8, dim=-1)  # [B,8]
    
    # bin midpoints (in percent), normalized to 0–1
    midpoints = torch.tensor(
        [2.5, 7.5, 15.0, 30.0, 50.0, 70.0, 85.0, 95.0],
        dtype=torch.float32,
        device=logits8.device
    ) / 100.0
    
    # predicted scalar confidence
    pred_conf = (p * midpoints).sum(dim=-1)  # [B]
    
    # target confidence from entropy
    target_conf = 1.0 - entropy / math.log(4)  # [B]
    
    # vanilla MSE
    mse = (pred_conf - target_conf) ** 2  # [B]
    
    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_loss(logits8, soft_targets=None, entropy=None,
                 loss_type='gaussian_soft_bin_ce', reduction='mean'):
    """
    Compute loss based on the specified loss type.
    
    Args:
        logits8: Tensor of shape [B, 8] with logits for confidence bins A-H
        soft_targets: Tensor of shape [B, 8] with soft target distribution (required for gaussian_soft_bin_ce)
        entropy: Tensor of shape [B] with entropy values (required for scalar_confidence_mse)
        loss_type: Type of loss to compute ('gaussian_soft_bin_ce' or 'scalar_confidence_mse')
        reduction: 'mean' to average over batch, 'none' to return per-sample losses, 
                  'sum' to sum over batch
    
    Returns:
        Loss tensor depending on reduction
    """
    if loss_type == 'gaussian_soft_bin_ce':
        if soft_targets is None:
            raise ValueError("soft_targets required for gaussian_soft_bin_ce")
        return compute_gaussian_soft_bin_ce_loss(logits8, soft_targets, reduction)
    
    elif loss_type == 'scalar_confidence_mse':
        if entropy is None:
            raise ValueError("entropy required for scalar_confidence_mse")
        return compute_scalar_confidence_mse_loss(logits8, entropy, reduction)
    
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")

