"""
analysis_helpers.py

Shared utilities for the analysis notebooks (analyze_performance,
analyze_activations, analyze_interventions). Anything that more than one
notebook would otherwise duplicate goes here: data loading, light wrangling,
common plotting primitives.

Goal: notebooks should focus on the *analysis*, not on parsing JSONL or
rebuilding calibration math. Keep this file conservative — only add a
helper once the same code has shown up in two notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_evaluation_jsonl(
    path: str | Path,
    type_filter: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Load a JSONL produced by ``finetune/run_evaluations.py`` (or
    ``finetune/question_difficulty_sweep.py``) into a DataFrame plus a dict
    of summary blobs.

    The JSONL contains two row kinds:
        - ``{"type": "<prefix>eval_sample", ...}`` — one per question, with
          fields like ``qid, is_correct, probs_ABCD, top_prob, prob_gap,
          entropy, expected_confidence, model_answer_position,
          correct_answer_position``.
        - ``{"type": "<prefix>eval_summary", ...}`` — aggregate metrics
          (accuracy, ECE, Brier, etc.) written once per ``evaluate_model``
          call.

    The ``<prefix>`` is set by the wrapper. ``run_evaluations.py`` uses
    ``"instruct_"`` and ``"finetuned_"`` when comparing two models in one
    file. ``question_difficulty_sweep.py`` uses dataset names like
    ``"PopMC_"``, ``"TriviaMC_"``, ``"SimpleMC_"``.

    Parameters
    ----------
    path
        Path to the JSONL file.
    type_filter
        If given, keep only sample rows whose ``type`` field contains this
        substring. Useful for picking out the finetuned half of a comparison
        file (``type_filter="finetuned"``) or one dataset from a sweep
        (``type_filter="SimpleMC"``).

    Returns
    -------
    samples_df
        One row per question.
    summaries
        ``{type_string: row_dict}`` for each summary blob in the file.
    """
    samples: list[dict] = []
    summaries: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("type", "")
            if t.endswith("eval_sample"):
                if type_filter and type_filter not in t:
                    continue
                samples.append(row)
            elif t.endswith("eval_summary"):
                summaries[t] = row
    df = pd.DataFrame(samples)
    return df, summaries


def latest_eval_log(
    log_dir: str | Path = "finetuned_evals",
    pattern: str = "*.jsonl",
) -> Optional[Path]:
    """Return the most recently modified file matching ``pattern`` in
    ``log_dir``, or None if none exists."""
    p = Path(log_dir)
    if not p.exists():
        return None
    matches = sorted(p.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_ect_results(paths: Iterable[str | Path]) -> dict[str, dict]:
    """Load multiple ECT result JSONs (from ``outputs/ECT/``) keyed by stem.

    Each file is expected to contain ``{"summary": {...}, "data": [...]}``,
    where each data row has ``stated_confidence_value, mc_probs,
    predicted_answer, stated_confidence_response, is_correct``.
    """
    out: dict[str, dict] = {}
    for p in paths:
        p = Path(p)
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


# ---------------------------------------------------------------------------
# Calibration / reliability
# ---------------------------------------------------------------------------

def calibration_table(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin predictions by ``confidence`` and report mean confidence vs.
    empirical accuracy per bin (the basic reliability-diagram table).

    Returns a DataFrame with columns ``bin_lo, bin_hi, n, mean_conf,
    accuracy``. Bins with ``n == 0`` are dropped.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Inclusive on the right edge for the final bin only.
        mask = (confidence >= lo) & (confidence < hi)
        if hi == 1.0:
            mask = mask | (confidence == 1.0)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "n": n,
            "mean_conf": float(confidence[mask].mean()),
            "accuracy": float(correct[mask].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: weighted L1 gap between bin confidence
    and bin accuracy. Lower is better (perfect = 0)."""
    tab = calibration_table(confidence, correct, n_bins=n_bins)
    if tab.empty:
        return float("nan")
    weights = tab["n"] / tab["n"].sum()
    return float((weights * (tab["mean_conf"] - tab["accuracy"]).abs()).sum())


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between confidence and 0/1 correctness."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return float(np.mean((confidence - correct) ** 2))
