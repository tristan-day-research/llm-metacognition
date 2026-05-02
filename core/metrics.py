"""
Uncertainty metric computation for MC and next-token prediction tasks.

Metrics:
- entropy: Shannon entropy of distribution (higher = more uncertain)
- top_prob: P(argmax) (higher = more confident)
- margin: P(top) - P(second) (higher = more confident)
- logit_gap: z(top) - z(second) (linear, higher = more confident)
- top_logit: z(top) - mean(z) (linear, higher = more confident)
"""

import numpy as np
from typing import Dict, List, Optional, Union


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy from a probability distribution."""
    probs = np.asarray(probs)
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def compute_metrics_single(
    probs: np.ndarray,
    logits: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all uncertainty metrics for a single sample.

    Args:
        probs: Probability distribution (sums to 1)
        logits: Raw logits (before softmax). If None, approximated from log(probs).

    Returns:
        Dict with: entropy, top_prob, margin, logit_gap, top_logit
    """
    probs = np.asarray(probs)

    # Entropy
    entropy = compute_entropy(probs)

    # Top probability
    top_prob = float(np.max(probs))

    # Margin: P(top) - P(second)
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])

    # Logit-based metrics (linear - better for linear probes)
    if logits is None:
        logits = np.log(probs + 1e-10)
    else:
        logits = np.asarray(logits)

    sorted_logits = np.sort(logits)[::-1]

    # Logit gap: z(top) - z(second)
    logit_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else float(sorted_logits[0])

    # Top logit (centered): z(top) - mean(z)
    top_logit = float(sorted_logits[0] - np.mean(logits))

    return {
        "entropy": entropy,
        "top_prob": top_prob,
        "margin": margin,
        "logit_gap": logit_gap,
        "top_logit": top_logit,
    }


def compute_mc_metrics(
    option_probs: Union[np.ndarray, List[np.ndarray]],
    option_logits: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty metrics for multiple MC samples.

    Args:
        option_probs: (n_samples, n_options) probability matrix, or list of prob arrays
        option_logits: (n_samples, n_options) logit matrix, or list of logit arrays
        metrics: Which metrics to compute. Default: all five.

    Returns:
        Dict mapping metric name to (n_samples,) array of values
    """
    all_metrics = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
    if metrics is None:
        metrics = all_metrics

    # Handle list input
    if isinstance(option_probs, list):
        option_probs = np.array(option_probs)
    if option_logits is not None and isinstance(option_logits, list):
        option_logits = np.array(option_logits)

    n_samples = len(option_probs)
    results = {m: np.zeros(n_samples) for m in metrics}

    for i in range(n_samples):
        probs = option_probs[i]
        logits = option_logits[i] if option_logits is not None else None
        sample_metrics = compute_metrics_single(probs, logits)

        for m in metrics:
            results[m][i] = sample_metrics[m]

    return results


def compute_nexttoken_metrics(
    logits: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute metrics from full vocabulary logits for next-token prediction.

    Args:
        logits: (vocab_size,) raw logits for next token
        metrics: Which metrics to compute. Default: entropy, top_prob, top_logit.

    Returns:
        Dict with requested metrics
    """
    logits = np.asarray(logits)

    # Softmax
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum()

    return compute_metrics_single(probs, logits)


# Metric metadata
METRIC_INFO = {
    "entropy": {"higher_means": "more_uncertain", "linear": False},
    "top_prob": {"higher_means": "more_confident", "linear": False},
    "margin": {"higher_means": "more_confident", "linear": False},
    "logit_gap": {"higher_means": "more_confident", "linear": True},
    "top_logit": {"higher_means": "more_confident", "linear": True},
}


def metric_sign_for_confidence(metric_name: str) -> int:
    """
    Return +1 if higher metric = higher confidence, -1 if inverse.

    Useful for computing correlations with confidence.
    """
    if METRIC_INFO[metric_name]["higher_means"] == "more_confident":
        return 1
    else:
        return -1
