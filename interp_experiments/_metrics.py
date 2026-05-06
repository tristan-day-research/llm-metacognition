"""
Uncertainty metrics computed from option-level probability/logit vectors.

These are pure functions over numpy arrays — no model state, no module-level
config. Extracted from `run_introspection_experiment.py` so the same metric
definitions can be reused by other interp scripts (logit lens, ablation
analyses, etc.) without re-importing the giant runner module.
"""

from typing import Dict

import numpy as np


def compute_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute entropy from a probability distribution."""
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy)


def compute_uncertainty_metrics(probs: np.ndarray, logits: np.ndarray = None) -> Dict[str, float]:
    """
    Compute multiple uncertainty metrics from probability and logit distributions.

    Args:
        probs: Probability distribution over answer options (sums to 1)
        logits: Raw logits for answer options (before softmax). If None, logit-based
                metrics will be computed from log(probs) as an approximation.

    Returns:
        Dict with keys: entropy, top_prob, margin, logit_gap, top_logit
    """
    # === Prob-based metrics (nonlinear) ===

    # Entropy: -sum(p * log(p))
    entropy = compute_entropy_from_probs(probs)

    # Top probability: P(argmax)
    top_prob = float(np.max(probs))

    # Margin: P(top) - P(second)
    sorted_probs = np.sort(probs)[::-1]  # Descending
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])

    # === Logit-based metrics (linear - better for linear probes) ===

    # If logits not provided, approximate from log-probs
    # (This loses the constant offset but preserves gaps)
    if logits is None:
        logits = np.log(probs + 1e-10)

    # Sort logits descending
    sorted_logits = np.sort(logits)[::-1]

    # Logit gap: z(top) - z(second)
    # This is the cleanest linear target - invariant to temperature/scale shifts
    logit_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else float(sorted_logits[0])

    # Top logit (centered): z(top) - mean(z)
    # Subtracting mean makes it invariant to adding a constant to all logits
    top_logit = float(sorted_logits[0] - np.mean(logits))

    return {
        "entropy": entropy,
        "top_prob": top_prob,
        "margin": margin,
        "logit_gap": logit_gap,
        "top_logit": top_logit,
    }
