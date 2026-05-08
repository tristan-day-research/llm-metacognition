"""
File I/O, path construction, and dataset loading for the interp_experiments
runners (`run_collect_activations.py` and the future `run_causal_interventions.py`).

Static config — `BASE_MODEL_NAME`, `MODEL_NAME`, `CONFIDENCE_SCALE`,
`OUTPUTS_DIR` — is read from `experiment_config.IntrospectionExperimentConfig`.
Per-iteration state — `dataset_name`, `meta_task` — lives on the small
`ctx` object below; runners mutate it once per iteration via
`set_run_context(...)` and the path-building helpers pick up the change.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiment_config import IntrospectionExperimentConfig as _C


# =============================================================================
# Per-run context — runners mutate this; helpers read from it.
# =============================================================================

class _RunCtx:
    """Per-iteration state (dataset, meta_task) consulted by path helpers.

    The runner's `run_single_experiment` sets these via `set_run_context(...)`
    before calling any helper that builds output paths. Defaults mirror
    `IntrospectionExperimentConfig.DATASET_NAME / META_TASK` so single-shot
    callers that haven't called `set_run_context` still work.
    """

    dataset_name: str = _C.DATASET_NAME
    meta_task: str = _C.META_TASK
    num_questions: int = _C.NUM_QUESTIONS


ctx = _RunCtx()


def set_run_context(*, dataset_name: Optional[str] = None,
                    meta_task: Optional[str] = None,
                    num_questions: Optional[int] = None) -> None:
    """Update per-iteration state. Pass only the fields you mean to change."""
    if dataset_name is not None:
        ctx.dataset_name = dataset_name
    if meta_task is not None:
        ctx.meta_task = meta_task
    if num_questions is not None:
        ctx.num_questions = num_questions


# =============================================================================
# Atomic file writes — survive Ctrl-C / OOM / disk-full mid-write.
# =============================================================================

def _atomic_write_json(path: str, payload: dict, *, indent: int = 2, cls=None) -> None:
    """Write a JSON file atomically: temp file + os.replace.

    A POSIX rename is atomic, so a Ctrl-C, OOM, or disk-full mid-write cannot
    leave a partial file at `path` (it'll either be the previous version or
    the new one, never a truncation).
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=indent, cls=cls) if cls else json.dump(payload, f, indent=indent)
    os.replace(tmp, path)


def _atomic_savez_compressed(path: str, **arrays) -> None:
    """np.savez_compressed wrapped in tmp+rename for atomicity."""
    final = path if path.endswith(".npz") else f"{path}.npz"
    tmp = f"{final}.tmp"
    np.savez_compressed(tmp, **arrays)
    if not os.path.exists(tmp):
        raise RuntimeError(f"Atomic NPZ write failed: temp file {tmp} not produced")
    os.replace(tmp, final)


def _question_ids_array(questions) -> np.ndarray:
    """Stable per-row question identifier for embedding in NPZs."""
    return np.array(
        [str(q.get("id", f"q_{i}")) for i, q in enumerate(questions)],
        dtype=object,
    )


def _pad_token_id_groups(groups: List[List[int]]) -> np.ndarray:
    """Convert ragged List[List[int]] to padded 2D int64 array (-1 = pad).

    Used for embedding the option_token_ids structure into NPZ files; each row
    is one option, columns are its sub-token IDs (max 2 in practice for
    Llama-3 letter / digit options).
    """
    if not groups:
        return np.zeros((0, 0), dtype=np.int64)
    max_len = max(len(g) for g in groups)
    out = np.full((len(groups), max_len), -1, dtype=np.int64)
    for i, g in enumerate(groups):
        out[i, : len(g)] = g
    return out


# =============================================================================
# Name / path helpers
# =============================================================================

def get_model_short_name(model_name: str) -> str:
    """Filesystem-safe name from a HF repo id ('org/name' → 'name')."""
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def _finetuned_short_tag(adapter: str) -> str:
    """Compact id for a finetuned adapter.

    Pulls out the run's date-time prefix and the checkpoint step. Example:
        'Tristan-Day/20260506-034609_delegate_at_..._ckpt_step_300'
        →  '20260506-034609_step_300'
    Falls back to the bare adapter basename if the pattern doesn't match.
    """
    name = adapter.split("/")[-1]
    m_prefix = re.match(r"^(\d+-\d+)", name)
    m_step = re.search(r"step_(\d+)\b", name)
    if m_prefix and m_step:
        return f"{m_prefix.group(1)}_step_{m_step.group(1)}"
    return name


def _dataset_short_name(name: str) -> str:
    """Filesystem-safe token for a dataset entry.

    Accepts either a registered dataset name (e.g. "TriviaMC") or a .jsonl
    path (e.g. "data/mixed_11931_max_balanced_test.jsonl"); for paths,
    strips the directory and the .jsonl suffix so output filenames stay
    flat and don't sprout slashes.
    """
    if name.endswith(".jsonl"):
        return Path(name).stem
    return name


def get_model_type_label() -> str:
    """Return 'base', 'instruct', or 'finetuned' based on _C config."""
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        return "finetuned"
    if "Instruct" in _C.BASE_MODEL_NAME:
        return "instruct"
    return "base"


def get_model_display_label() -> str:
    """Human-readable label for plot titles. Distinguishes base/instruct/finetuned.

    Example: 'finetuned (Llama-3.1-8B-Instruct + adapter-…_step_300)'.
    """
    mtype = get_model_type_label()
    base_short = get_model_short_name(_C.BASE_MODEL_NAME)
    if mtype == "finetuned":
        adapter_short = get_model_short_name(_C.MODEL_NAME)
        return f"{mtype} ({base_short} + adapter-{adapter_short})"
    return f"{mtype} ({base_short})"


def _run_subfolder_name() -> str:
    """Per-run output subfolder under OUTPUTS_DIR.

    Format: '8b_<model_tag>_<dataset_short>' where model_tag is
        'base'                — Llama-3.1-8B with no adapter
        'instruct'            — Llama-3.1-8B-Instruct with no adapter
        '<run-id>_step_<N>'   — Llama-3.1-8B-Instruct + LoRA adapter
    Reads dataset from `ctx.dataset_name`, model from `_C`.
    """
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        model_tag = _finetuned_short_tag(_C.MODEL_NAME)
    elif "Instruct" in _C.BASE_MODEL_NAME:
        model_tag = "instruct"
    else:
        model_tag = "base"
    return f"8b_{model_tag}_{_dataset_short_name(ctx.dataset_name)}"


def _run_dir() -> Path:
    """Concrete output directory for the current run, mkdir'd if needed."""
    d = _C.OUTPUTS_DIR / _run_subfolder_name()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_prefix(metric: Optional[str] = None) -> str:
    """File prefix for run-specific outputs (activations, paired data, lens).

    Includes the meta task in the filename so confidence and delegate runs
    on the same (model, dataset) don't collide.
    """
    model_short = get_model_short_name(_C.BASE_MODEL_NAME)
    dataset_short = _dataset_short_name(ctx.dataset_name)
    task_suffix = f"_{ctx.meta_task}" if ctx.meta_task != "confidence" else ""
    scale_suffix = f"_scale-{_C.CONFIDENCE_SCALE}" if _C.CONFIDENCE_SCALE != "letters" else ""
    metric_suffix = f"_{metric}" if metric else ""
    run_dir = _run_dir()
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        adapter_short = get_model_short_name(_C.MODEL_NAME)
        return str(run_dir / f"{model_short}_adapter-{adapter_short}_{dataset_short}_introspection{task_suffix}{scale_suffix}{metric_suffix}")
    return str(run_dir / f"{model_short}_{dataset_short}_introspection{task_suffix}{scale_suffix}{metric_suffix}")


def get_directions_prefix(metric: Optional[str] = None) -> str:
    """File prefix for direction NPZs.

    Direction files are task-independent (the meta task doesn't affect
    direction computation), so this drops the meta-task suffix that
    `get_output_prefix` includes. Scale IS preserved because the
    stated-confidence direction depends on the scale.
    """
    model_short = get_model_short_name(_C.BASE_MODEL_NAME)
    dataset_short = _dataset_short_name(ctx.dataset_name)
    scale_suffix = f"_scale-{_C.CONFIDENCE_SCALE}" if _C.CONFIDENCE_SCALE != "letters" else ""
    metric_suffix = f"_{metric}" if metric else ""
    run_dir = _run_dir()
    if _C.MODEL_NAME != _C.BASE_MODEL_NAME:
        adapter_short = get_model_short_name(_C.MODEL_NAME)
        return str(run_dir / f"{model_short}_adapter-{adapter_short}_{dataset_short}_introspection{scale_suffix}{metric_suffix}")
    return str(run_dir / f"{model_short}_{dataset_short}_introspection{scale_suffix}{metric_suffix}")


# =============================================================================
# Question loading
# =============================================================================

def load_questions(dataset_name: str, num_questions: Optional[int] = None,
                   *, seed: int = None) -> List[Dict]:
    """Load MC questions from either a registered dataset name or a .jsonl path.

    JSONL rows must contain at minimum question / options / correct_answer
    (or correct_letter); id / qid is optional and falls back to the row index.
    """
    if seed is None:
        seed = _C.SEED
    if dataset_name.endswith(".jsonl"):
        questions = _load_questions_from_jsonl(dataset_name, num_questions, seed=seed)
    else:
        from core.datasets import load_and_format_dataset
        questions = load_and_format_dataset(dataset_name, num_questions_needed=num_questions)

    if questions is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")
    return questions


def _load_questions_from_jsonl(path: str, num_questions: Optional[int] = None,
                                *, seed: int = 42) -> List[Dict]:
    """Read a generic question-per-row .jsonl and normalise to the runner schema.

    Drops rows missing question / options / correct_answer. Subsamples
    deterministically (with `seed`) when more rows are present than
    `num_questions`.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    formatted = []
    for i, row in enumerate(rows):
        question = row.get("question")
        options = row.get("options")
        correct = row.get("correct_answer") or row.get("correct_letter")
        if not question or not options or correct is None:
            continue
        qid = row.get("id") or row.get("qid") or f"{Path(path).stem}_{i}"
        formatted.append({
            "id": str(qid),
            "question": question,
            "options": options,
            "correct_answer": correct,
        })

    if num_questions is not None and len(formatted) > num_questions:
        rng = random.Random(seed)
        rng.shuffle(formatted)
        formatted = formatted[:num_questions]

    print(f"Loaded {len(formatted)} questions from {path}")
    return formatted
