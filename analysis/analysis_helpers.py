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
    log_dir: str | Path = "outputs/evaluations",
    pattern: str = "*.jsonl",
) -> Optional[Path]:
    """Return the most recently modified file matching ``pattern`` in
    ``log_dir``, or None if none exists.

    Default points at ``outputs/evaluations/`` (where ``run_evaluations.py``
    writes its JSONL logs)."""
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


# ---------------------------------------------------------------------------
# Run notebooks via papermill (parameter sweeps)
# ---------------------------------------------------------------------------
# Workflow: notebooks declare a cell tagged ``parameters`` with default values.
# Running interactively in Jupyter uses those defaults — nothing changes.
# Calling ``run_notebook(...)`` re-executes the notebook with overrides and
# saves a fully-rendered copy per config so historical runs stick around.

def run_notebook(
    notebook: str | Path,
    parameters: Optional[dict] = None,
    label: Optional[str] = None,
    output_dir: str | Path = "analysis/runs",
) -> Path:
    """Execute ``notebook`` with ``parameters`` injected and save the rendered
    copy under ``output_dir``. Returns the path to the saved notebook.

    The output filename is ``<stem>__<label>__<UTC-timestamp>.ipynb`` so
    sweeps don't collide. ``label`` defaults to a short hash of the params.

    Requires ``pip install papermill``.
    """
    import hashlib
    from datetime import datetime, timezone
    import papermill as pm

    notebook = Path(notebook)
    parameters = parameters or {}

    if label is None:
        h = hashlib.md5(json.dumps(parameters, sort_keys=True, default=str).encode()).hexdigest()[:8]
        label = f"run-{h}"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{notebook.stem}__{label}__{timestamp}.ipynb"

    pm.execute_notebook(
        str(notebook),
        str(out_path),
        parameters={**parameters, "RUN_LABEL": label},
    )
    return out_path


def run_notebook_sweep(
    notebook: str | Path,
    configs: dict[str, dict],
    output_dir: str | Path = "analysis/runs",
) -> dict[str, Path]:
    """Run ``notebook`` once per ``(label, params)`` entry in ``configs`` and
    return ``{label: output_path}``. Stops on the first failure.

    Example
    -------
    >>> run_notebook_sweep(
    ...     "analysis/analyze_performance.ipynb",
    ...     {
    ...         "base_only":  {"EVAL_PATH": "outputs/evaluations/base.jsonl"},
    ...         "ect_v1":     {"EVAL_PATH": "outputs/evaluations/ect_v1.jsonl"},
    ...     },
    ... )
    """
    out: dict[str, Path] = {}
    for label, params in configs.items():
        print(f"\n=== running '{label}' ===")
        out[label] = run_notebook(notebook, parameters=params, label=label, output_dir=output_dir)
        print(f"  → {out[label]}")
    return out


if __name__ == "__main__":
    # CLI: python analysis/analysis_helpers.py <notebook> KEY=VAL [KEY=VAL ...]
    # Values are JSON-decoded when possible, else kept as strings.
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python analysis/analysis_helpers.py <notebook.ipynb> KEY=VAL ...")
        raise SystemExit(1)
    nb = sys.argv[1]
    params: dict = {}
    for kv in sys.argv[2:]:
        if "=" not in kv:
            raise SystemExit(f"Bad arg (expected KEY=VAL): {kv!r}")
        k, v = kv.split("=", 1)
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v
    out_path = run_notebook(nb, parameters=params)
    print(f"\nSaved: {out_path}")
