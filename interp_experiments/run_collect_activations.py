"""
Run introspection experiment: Test whether models use internal uncertainty signals for meta-cognition.

This script:
1. Loads MC questions and runs them through the model in two modes:
   - Direct: Ask the MC question directly, compute uncertainty metrics over A/B/C/D
   - Meta: Ask "How confident are you that you know the answer to [Q]?"
2. Extracts activations from both prompt types
3. Computes multiple uncertainty metrics (all saved, one probed per run):
   - Prob-based (nonlinear): entropy, top_prob, margin
   - Logit-based (linear): logit_gap, top_logit
4. Trains a linear probe on direct activations → selected metric
5. Tests whether that probe transfers to meta activations → direct metric
   (If it does, the model may be "introspecting" on an internal uncertainty signal)

Key insight: If the model is truly introspecting when answering meta-questions,
it should internally access the same representations it would use for the direct case.
A probe trained on direct data should therefore transfer to meta data.

Usage:
    python run_introspection.py --metric logit_gap   # Probe logit_gap (default)
    python run_introspection.py --metric entropy     # Probe entropy
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import argparse
from collections import defaultdict
import datetime
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import os
from dotenv import load_dotenv
import random
import pickle
import uuid


# ============================================================================
# ATOMIC WRITE + ALIGNMENT-STAMPING HELPERS
# ============================================================================
# These helpers exist to prevent a class of silent corruption bugs where the
# `*_paired_data.json` and `*_activations.npz` files end up out of sync —
# e.g. a partial re-run overwrites the NPZ but not the JSON, leaving the
# notebook to pair row i of one with row i of the other (wrong question).
#
# The fixes:
#   1. Atomic writes: write to `path.tmp` then `os.replace(tmp, path)` so an
#      interrupted save can never produce a half-written file that future
#      loaders silently consume.
#   2. Every fresh save is stamped with a `run_id` (UUID) shared between the
#      paired_data.json and both NPZs of that run.
#   3. NPZs additionally store `question_ids` so the loader can verify row
#      alignment without inspecting metrics.
#
# All new fields are additive: existing files (and the notebook reading them)
# are unaffected.


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
    """np.savez_compressed wrapped in tmp+rename for atomicity.

    np.savez_compressed appends '.npz' to the path if it isn't already
    present, so we strip and re-add it carefully.
    """
    final = path if path.endswith(".npz") else f"{path}.npz"
    tmp = f"{final}.tmp"
    np.savez_compressed(tmp, **arrays)
    # np.savez_compressed always writes to exactly the path you gave it
    # when you include `.npz` in the name — confirm and rename.
    if not os.path.exists(tmp):
        # Defensive: if numpy stripped/added .npz unexpectedly, surface it.
        raise RuntimeError(f"Atomic NPZ write failed: temp file {tmp} not produced")
    os.replace(tmp, final)


def _question_ids_array(questions) -> np.ndarray:
    """Stable per-row question identifier for embedding in NPZs."""
    return np.array(
        [str(q.get("id", f"q_{i}")) for i, q in enumerate(questions)],
        dtype=object,
    )

from core.model_utils import load_model_and_tokenizer, DEVICE
from prompts import (
    # Direct MC task
    MC_SETUP_PROMPT,
    format_direct_prompt,
    format_direct_prompt_base,
    # Confidence task (letter scale S-Z)
    STATED_CONFIDENCE_SETUP,
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    STATED_CONFIDENCE_QUESTION,
    format_stated_confidence_prompt,
    format_stated_confidence_prompt_base,
    get_stated_confidence_signal,
    # Confidence task (numeric scale 1-9)
    NUMERIC_CONFIDENCE_SETUP,
    NUMERIC_CONFIDENCE_OPTIONS,
    NUMERIC_CONFIDENCE_MIDPOINTS,
    NUMERIC_CONFIDENCE_QUESTION,
    format_numeric_confidence_prompt,
    format_numeric_confidence_prompt_base,
    get_numeric_confidence_signal,
    # Other-confidence task (control: estimate human difficulty)
    OTHER_CONFIDENCE_SETUP,
    OTHER_CONFIDENCE_QUESTION,
    format_other_confidence_prompt,
    get_other_confidence_signal,
    # Delegate task
    ANSWER_OR_DELEGATE_SETUP,
    ANSWER_OR_DELEGATE_SYSPROMPT,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    format_answer_or_delegate_prompt_base,
    get_delegate_mapping,
    # Single-shot ABCDT delegate task
    ANSWER_WITH_DELEGATE_OPTIONS,
    format_answer_with_delegate_prompt,
    format_answer_with_delegate_prompt_base,
    # Unified conversion
    response_to_confidence,
)

load_dotenv()


# Configuration — edit values in experiment_config.IntrospectionExperimentConfig
from experiment_config import IntrospectionExperimentConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
MODEL_NAME = _C.MODEL_NAME
DATASETS = list(_C.DATASETS)
META_TASKS = list(_C.META_TASKS)
DATASET_NAME = _C.DATASET_NAME
META_TASK = _C.META_TASK
NUM_QUESTIONS_DEFAULT = _C.NUM_QUESTIONS_DEFAULT
NUM_QUESTIONS_BY_DATASET = dict(_C.NUM_QUESTIONS_BY_DATASET)
NUM_QUESTIONS = _C.NUM_QUESTIONS
SEED = _C.SEED
LOAD_IN_4BIT = _C.LOAD_IN_4BIT
LOAD_IN_8BIT = _C.LOAD_IN_8BIT
FEW_SHOT_MODE = _C.FEW_SHOT_MODE
BASE_DELEGATE_MODE = _C.BASE_DELEGATE_MODE
BASE_DELEGATE_POOL_SOURCE = _C.BASE_DELEGATE_POOL_SOURCE
TEAMMATE_ACCURACY = _C.TEAMMATE_ACCURACY
DELEGATE_PROMPT_DESIGN = _C.DELEGATE_PROMPT_DESIGN
REUSE_DIRECT_FROM_CONFIDENCE = _C.REUSE_DIRECT_FROM_CONFIDENCE
CONFIDENCE_SCALE = _C.CONFIDENCE_SCALE
OUTPUTS_DIR = _C.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_SPLIT = _C.TRAIN_SPLIT
PROBE_ALPHA = _C.PROBE_ALPHA
USE_PCA = _C.USE_PCA
PCA_COMPONENTS = _C.PCA_COMPONENTS
AVAILABLE_METRICS = list(_C.AVAILABLE_METRICS)
UNCERTAINTY_METRICS = set(_C.UNCERTAINTY_METRICS)
METRIC = _C.METRIC
EXTRACT_CONTRAST_DIRECTIONS = _C.EXTRACT_CONTRAST_DIRECTIONS
CONTRAST_PERCENT_ENTROPY = _C.CONTRAST_PERCENT_ENTROPY
CONTRAST_PERCENT_STATED_CONFIDENCE = _C.CONTRAST_PERCENT_STATED_CONFIDENCE
CONTRAST_SAMPLES_PER_BIN = _C.CONTRAST_SAMPLES_PER_BIN

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from a model path."""
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def _dataset_short_name(name: str) -> str:
    """Filesystem-safe token for a dataset entry.

    Accepts either a registered dataset name (e.g. "TriviaMC") or a
    .jsonl path (e.g. "data/mixed_11931_max_balanced.jsonl"); for paths,
    strips the directory and the .jsonl suffix so output filenames stay
    flat and don't sprout slashes.
    """
    if name.endswith(".jsonl"):
        return Path(name).stem
    return name


def _finetuned_short_tag(adapter: str) -> str:
    """Compact id for a finetuned adapter.

    Pulls out the run's date-time prefix and the checkpoint step. Example:
        'Tristan-Day/20260506-034609_delegate_at_..._ckpt_step_300'
        →  '20260506-034609_step_300'
    Falls back to the bare adapter basename if the pattern doesn't match.
    """
    import re
    name = adapter.split("/")[-1]
    m_prefix = re.match(r"^(\d+-\d+)", name)
    m_step = re.search(r"step_(\d+)\b", name)
    if m_prefix and m_step:
        return f"{m_prefix.group(1)}_step_{m_step.group(1)}"
    return name


def _run_subfolder_name() -> str:
    """Per-run output subfolder name under OUTPUTS_DIR.

    Format: '8b_<model_tag>_<dataset_short>' where model_tag is
        'base'                    — Llama-3.1-8B with no adapter
        'instruct'                — Llama-3.1-8B-Instruct with no adapter
        '<run-id>_step_<N>'       — Llama-3.1-8B-Instruct + LoRA adapter
    Reads BASE_MODEL_NAME / MODEL_NAME / DATASET_NAME from module globals
    so it picks up `globals()["DATASET_NAME"] = ...` mutations from
    `run_single_experiment` automatically.
    """
    if MODEL_NAME != BASE_MODEL_NAME:
        model_tag = _finetuned_short_tag(MODEL_NAME)
    elif "Instruct" in BASE_MODEL_NAME:
        model_tag = "instruct"
    else:
        model_tag = "base"
    return f"8b_{model_tag}_{_dataset_short_name(DATASET_NAME)}"


def _run_dir() -> Path:
    """Concrete output directory for the current run, mkdir'd if needed."""
    d = OUTPUTS_DIR / _run_subfolder_name()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model_type_label() -> str:
    """Return 'base', 'instruct', or 'finetuned' based on current config.

    - 'finetuned' if an adapter is loaded (MODEL_NAME != BASE_MODEL_NAME)
    - 'instruct' if BASE_MODEL_NAME contains 'Instruct' and no adapter
    - 'base'     otherwise
    """
    if MODEL_NAME != BASE_MODEL_NAME:
        return "finetuned"
    if "Instruct" in BASE_MODEL_NAME:
        return "instruct"
    return "base"


def get_model_display_label() -> str:
    """Human-readable label for plot titles that distinguishes base/instruct/finetuned.

    Example: 'finetuned (Llama-3.1-8B-Instruct + adapter-ect_20251222_215412_v0uei7y1_2000)'.
    Unambiguous even when BASE_MODEL_NAME is the same for instruct and finetuned.
    """
    mtype = get_model_type_label()
    base_short = get_model_short_name(BASE_MODEL_NAME)
    if mtype == "finetuned":
        adapter_short = get_model_short_name(MODEL_NAME)
        return f"{mtype} ({base_short} + adapter-{adapter_short})"
    return f"{mtype} ({base_short})"


def get_output_prefix(metric: str = None) -> str:
    """Generate output filename prefix based on config.

    Args:
        metric: If provided, include metric in prefix (for metric-specific outputs).
                If None, return base prefix (for shared outputs like activations).
    """
    model_short = get_model_short_name(BASE_MODEL_NAME)
    dataset_short = _dataset_short_name(DATASET_NAME)
    # Include meta task type in output prefix for clarity
    task_suffix = f"_{META_TASK}" if META_TASK != "confidence" else ""
    # Only suffix the filename if scale != 'letters' (keep letters as default for backward compat)
    scale_suffix = f"_scale-{CONFIDENCE_SCALE}" if CONFIDENCE_SCALE != "letters" else ""
    metric_suffix = f"_{metric}" if metric else ""
    run_dir = _run_dir()
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(run_dir / f"{model_short}_adapter-{adapter_short}_{dataset_short}_introspection{task_suffix}{scale_suffix}{metric_suffix}")
    return str(run_dir / f"{model_short}_{dataset_short}_introspection{task_suffix}{scale_suffix}{metric_suffix}")


def get_directions_prefix(metric: str = None) -> str:
    """Generate output filename prefix for direction files (task-independent).

    Direction files are task-independent because they're trained on direct task
    activations predicting metrics like entropy/top_logit - the meta task
    (confidence vs delegate) doesn't affect the direction computation.

    Args:
        metric: If provided, include metric in prefix.
    """
    model_short = get_model_short_name(BASE_MODEL_NAME)
    dataset_short = _dataset_short_name(DATASET_NAME)
    # NO task suffix - directions are task-independent. Scale IS relevant (stated-conf direction depends on scale)
    scale_suffix = f"_scale-{CONFIDENCE_SCALE}" if CONFIDENCE_SCALE != "letters" else ""
    metric_suffix = f"_{metric}" if metric else ""
    run_dir = _run_dir()
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(run_dir / f"{model_short}_adapter-{adapter_short}_{dataset_short}_introspection{scale_suffix}{metric_suffix}")
    return str(run_dir / f"{model_short}_{dataset_short}_introspection{scale_suffix}{metric_suffix}")


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES (now imported from tasks.py)
# Scale-aware: delegating to CONFIDENCE_SCALE so switching scales just works.
# ============================================================================


def _scale_options() -> dict:
    return NUMERIC_CONFIDENCE_OPTIONS if CONFIDENCE_SCALE == "numeric" else STATED_CONFIDENCE_OPTIONS


def _scale_midpoints() -> dict:
    return NUMERIC_CONFIDENCE_MIDPOINTS if CONFIDENCE_SCALE == "numeric" else STATED_CONFIDENCE_MIDPOINTS


def _scale_setup() -> str:
    return NUMERIC_CONFIDENCE_SETUP if CONFIDENCE_SCALE == "numeric" else STATED_CONFIDENCE_SETUP


def _scale_question() -> str:
    return NUMERIC_CONFIDENCE_QUESTION if CONFIDENCE_SCALE == "numeric" else STATED_CONFIDENCE_QUESTION




# Import utility functions from core (instead of duplicating)
from core.model_utils import is_base_model, has_chat_template


# ============================================================================
# QUESTION LOADING AND FORMATTING
# ============================================================================

def load_questions(dataset_name: str, num_questions: int = None) -> List[Dict]:
    """Load MC questions either from a registered dataset name or a .jsonl path.

    JSONL rows must contain at minimum question / options / correct_answer (or
    correct_letter); id / qid is optional and falls back to the row index.
    """
    if dataset_name.endswith(".jsonl"):
        questions = _load_questions_from_jsonl(dataset_name, num_questions)
    else:
        from core.datasets import load_and_format_dataset
        questions = load_and_format_dataset(dataset_name, num_questions_needed=num_questions)

    if questions is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    return questions


def _load_questions_from_jsonl(path: str, num_questions: int = None) -> List[Dict]:
    """Read a generic question-per-row .jsonl and normalise to the runner schema.

    Drops rows that are missing question / options / correct_answer. Subsamples
    deterministically (SEED) when more rows are present than NUM_QUESTIONS.
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
        rng = random.Random(SEED)
        rng.shuffle(formatted)
        formatted = formatted[:num_questions]

    print(f"Loaded {len(formatted)} questions from {path}")
    return formatted


# Use formatting functions from tasks.py (imported at top)
# format_direct_prompt - imported directly
# Local wrappers for meta tasks to maintain backward compatibility


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> Tuple[str, List[str]]:
    """Format a meta/confidence question for instruct/finetuned models.

    Dispatches to the numeric (1-9) or letter (S-Z) formatter based on
    CONFIDENCE_SCALE.
    """
    if CONFIDENCE_SCALE == "numeric":
        return format_numeric_confidence_prompt(question, tokenizer, use_chat_template)
    return format_stated_confidence_prompt(question, tokenizer, use_chat_template)


def format_meta_prompt_base(question: Dict, mode: str = "fixed", pool=None) -> Tuple[str, List[str]]:
    """Format a meta/confidence question for BASE (few-shot, no chat template) models."""
    if CONFIDENCE_SCALE == "numeric":
        return format_numeric_confidence_prompt_base(question, mode=mode, pool=pool)
    return format_stated_confidence_prompt_base(question, mode=mode, pool=pool)


def format_delegate_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    trial_index: int = 0,
    is_base: bool = False,
    few_shot_mode: str = "fixed",
    few_shot_pool: Optional[List[Dict]] = None,
    teammate_accuracy: float = None,
) -> Tuple[str, List[str], Optional[Dict[str, str]]]:
    """Format a delegate question using centralized tasks.py logic.

    Routes on DELEGATE_PROMPT_DESIGN:
    - "mc_integrated": single-shot A/B/C/D/T prompt (instruct/finetuned only).
      Returns (prompt, ["A","B","C","D","T"], None).
    - "two_step_digit": legacy 1-vs-2 decision with alternating mapping.
      Base models supported via few-shot.
    """
    if teammate_accuracy is None:
        teammate_accuracy = TEAMMATE_ACCURACY

    if DELEGATE_PROMPT_DESIGN == "mc_integrated":
        if is_base:
            return format_answer_with_delegate_prompt_base(
                question, mode=few_shot_mode, pool=few_shot_pool,
                teammate_accuracy=teammate_accuracy,
            )
        return format_answer_with_delegate_prompt(
            question, tokenizer,
            use_chat_template=use_chat_template,
            teammate_accuracy=teammate_accuracy,
        )

    # Legacy two-step digit design
    if is_base:
        return format_answer_or_delegate_prompt_base(
            question, trial_index=trial_index, mode=few_shot_mode, pool=few_shot_pool,
            teammate_accuracy=teammate_accuracy,
        )
    return format_answer_or_delegate_prompt(
        question, tokenizer, trial_index=trial_index,
        alternate_mapping=True, use_chat_template=use_chat_template,
        teammate_accuracy=teammate_accuracy,
    )


def get_meta_prompt_formatter():
    """Return the appropriate prompt formatter based on META_TASK setting."""
    if META_TASK == "delegate":
        return format_delegate_prompt
    else:
        return format_meta_prompt


def get_meta_options():
    """Return the meta options based on META_TASK and CONFIDENCE_SCALE."""
    if META_TASK == "delegate":
        if DELEGATE_PROMPT_DESIGN == "mc_integrated":
            return ANSWER_WITH_DELEGATE_OPTIONS
        return ANSWER_OR_DELEGATE_OPTIONS
    # Confidence task — scale-dependent
    return list(_scale_options().keys())


def _meta_task_type() -> str:
    """Map (META_TASK, CONFIDENCE_SCALE) to the task_type string used by
    tasks.response_to_confidence()."""
    if META_TASK == "delegate":
        return "delegate"
    if CONFIDENCE_SCALE == "numeric":
        return "confidence_numeric"
    return "confidence"


def local_response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None
) -> float:
    """Convert a meta response to a confidence scalar (scale-aware)."""
    return response_to_confidence(response, probs, mapping, _meta_task_type())


def get_meta_signal(probs: np.ndarray) -> float:
    """Scale-aware probability-weighted confidence signal (for confidence task)."""
    if CONFIDENCE_SCALE == "numeric":
        return get_numeric_confidence_signal(probs)
    return get_stated_confidence_signal(probs)


def save_example_prompts_and_responses_txt(
    data: dict,
    questions: list,
    tokenizer,
    is_base: bool,
    use_chat_template: bool,
    few_shot_mode: str,
    output_path: str,
    n_examples: int = 10,
    delegate_pool: Optional[list] = None,
) -> None:
    """Save a human-readable .txt showing exact prompts and model responses.

    For each of the first `n_examples` questions, includes:
      - Question, options, correct answer
      - The exact direct prompt string (including chat template tokens)
      - Direct option probabilities + argmax + correctness
      - Uncertainty metrics (entropy, logit_gap, margin, top_prob)
      - The exact meta (confidence) prompt string
      - Meta option probabilities + argmax + soft-signal confidence
    """
    import io
    buf = io.StringIO()

    n = min(n_examples, len(questions))
    direct_probs = data.get("direct_probs", [])
    meta_probs = data.get("meta_probs", [])
    meta_resp = data.get("meta_responses", [])
    meta_mappings = data.get("meta_mappings") or [None] * len(questions)
    direct_metrics = data.get("direct_metrics", {})

    # Header
    buf.write("=" * 80 + "\n")
    buf.write(f"Model: {BASE_MODEL_NAME}\n")
    if MODEL_NAME != BASE_MODEL_NAME:
        buf.write(f"Adapter: {MODEL_NAME}\n")
    buf.write(
        f"Dataset: {DATASET_NAME}   Meta task: {META_TASK}   "
        f"Scale: {CONFIDENCE_SCALE}   is_base: {is_base}   "
        f"use_chat_template: {use_chat_template}\n"
    )
    if is_base:
        buf.write(f"Few-shot mode: {few_shot_mode}\n")
    buf.write(f"Showing first {n} of {len(questions)} examples.\n")
    buf.write("=" * 80 + "\n\n")

    for i in range(n):
        q = questions[i]
        buf.write("=" * 80 + "\n")
        buf.write(f"EXAMPLE {i + 1} / {n}   (id: {q.get('id', f'q_{i}')})\n")
        buf.write("=" * 80 + "\n")
        buf.write(f"Question: {q.get('question', '')}\n")
        for key, val in q.get("options", {}).items():
            buf.write(f"  {key}: {val}\n")
        buf.write(f"Correct answer: {q.get('correct_answer', '?')}\n\n")

        # Rebuild the exact direct prompt
        if is_base:
            direct_prompt, direct_opts = format_direct_prompt_base(q, mode=few_shot_mode)
        else:
            direct_prompt, direct_opts = format_direct_prompt(q, tokenizer, use_chat_template)

        buf.write("--- DIRECT PROMPT (sent to model) ---\n")
        buf.write(direct_prompt)
        buf.write("\n\n")

        buf.write("--- DIRECT RESPONSE ---\n")
        if i < len(direct_probs) and direct_probs[i]:
            p = direct_probs[i]
            probs_str = "  ".join(f"{L}={v:.3f}" for L, v in zip(direct_opts, p))
            argmax_letter = direct_opts[int(np.argmax(p))]
            is_correct_flag = argmax_letter == q.get("correct_answer")
            buf.write(f"Option probs: {probs_str}\n")
            buf.write(
                f"Chosen (argmax): {argmax_letter}   "
                f"Correct? {'YES' if is_correct_flag else 'no'}\n"
            )
        if direct_metrics and i < len(direct_metrics.get("entropy", [])):
            e = direct_metrics["entropy"][i]
            m = direct_metrics.get("margin", [float("nan")] * len(questions))[i]
            lg = direct_metrics.get("logit_gap", [float("nan")] * len(questions))[i]
            tp = direct_metrics.get("top_prob", [float("nan")] * len(questions))[i]
            buf.write(
                f"Uncertainty: entropy={float(e):.3f}  logit_gap={float(lg):.3f}  "
                f"margin={float(m):.3f}  top_prob={float(tp):.3f}\n"
            )
        buf.write("\n")

        # Rebuild the exact meta prompt
        if META_TASK == "delegate":
            meta_prompt, meta_opts, _mapping = format_delegate_prompt(
                q, tokenizer, use_chat_template, trial_index=i,
                is_base=is_base,
                few_shot_mode=BASE_DELEGATE_MODE if is_base else "fixed",
                few_shot_pool=delegate_pool,
            )
        else:
            if is_base:
                meta_prompt, meta_opts = format_meta_prompt_base(q, mode=few_shot_mode)
            else:
                meta_prompt, meta_opts = format_meta_prompt(q, tokenizer, use_chat_template)

        buf.write("--- META PROMPT (sent to model) ---\n")
        buf.write(meta_prompt)
        buf.write("\n\n")

        buf.write("--- META RESPONSE ---\n")
        if i < len(meta_probs) and meta_probs[i]:
            p = meta_probs[i]
            probs_str = "  ".join(f"{L}={v:.3f}" for L, v in zip(meta_opts, p))
            buf.write(f"Option probs: {probs_str}\n")
        if i < len(meta_resp):
            buf.write(f"Chosen (argmax): {meta_resp[i]}\n")
        if i < len(meta_probs) and meta_probs[i]:
            soft = response_to_confidence(
                meta_resp[i] if i < len(meta_resp) else "",
                np.asarray(meta_probs[i]),
                meta_mappings[i] if i < len(meta_mappings) else None,
                _meta_task_type(),
            )
            buf.write(f"Soft confidence (probability-weighted): {float(soft):.3f}\n")
        buf.write("\n")

    buf.write("=" * 80 + "\n")
    buf.write("END\n")
    buf.write("=" * 80 + "\n")

    with open(output_path, "w") as f:
        f.write(buf.getvalue())
    print(f"Saved example prompts/responses to {output_path}")


def save_quick_summary_png(data: dict, questions: list, output_path: str) -> None:
    """Save a quick 3-panel diagnostic PNG right after collection:

      - (left)   MC chosen-answer distribution (A/B/C/D) with correct-answer
                 baseline, and overall accuracy
      - (middle) Stated-confidence distribution (scale-aware: S-Z or 1-9)
      - (right)  Scatter of MC entropy vs stated_confidence_numeric with
                 Pearson r and Spearman rho annotated

    This runs before the slow probe/introspection analysis so you can eyeball
    the results immediately.
    """
    from scipy.stats import pearsonr, spearmanr

    n = len(questions)
    # Derive direct answer choices
    direct_probs = data.get("direct_probs", [])
    options_per_q = [list(q.get("options", {}).keys()) for q in questions]
    direct_resp = [
        opts[int(np.argmax(p))] if opts and p else None
        for opts, p in zip(options_per_q, direct_probs)
    ]
    correct = [q.get("correct_answer") for q in questions]
    is_correct = [r == c for r, c in zip(direct_resp, correct)]
    acc = float(np.mean(is_correct)) if is_correct else float("nan")

    # Meta-side
    meta_probs = data.get("meta_probs", [])
    meta_resp = data.get("meta_responses", [])
    meta_mappings = data.get("meta_mappings", []) or [None] * n
    entropies = np.asarray(data.get("direct_metrics", {}).get("entropy", [np.nan] * n))

    # Delegate mode: either 2-option digit probs (legacy) or 5-option A/B/C/D/T
    # probs (mc_integrated). Confidence has an 8/10-option scale.
    # Build stated_num and the option legend accordingly so the shapes line up.
    if META_TASK == "delegate" and DELEGATE_PROMPT_DESIGN == "mc_integrated":
        confidence_options = ANSWER_WITH_DELEGATE_OPTIONS  # ["A","B","C","D","T"]
        t_idx = confidence_options.index("T")
        stated_num = np.full(n, np.nan)
        for i, p in enumerate(meta_probs):
            if not p:
                continue
            p_arr = np.asarray(p, dtype=float)
            if p_arr.shape[0] >= len(confidence_options):
                stated_num[i] = 1.0 - float(p_arr[t_idx])  # P(Answer) = 1 - P(T)
        stated_ylabel = "P(Answer)"
    elif META_TASK == "delegate":
        confidence_options = ANSWER_OR_DELEGATE_OPTIONS  # ["1", "2"]
        stated_num = np.full(n, np.nan)
        for i, p in enumerate(meta_probs):
            if not p:
                continue
            p_arr = np.asarray(p, dtype=float)
            m = meta_mappings[i] if i < len(meta_mappings) and meta_mappings[i] else None
            if m:
                tokens = sorted(m.keys())  # ['1', '2']
                ans_col = next((j for j, t in enumerate(tokens) if m.get(t) == "Answer"), None)
                if ans_col is not None and p_arr.shape[0] >= 2:
                    stated_num[i] = float(p_arr[ans_col])
            else:
                stated_num[i] = float(p_arr[0]) if p_arr.shape[0] >= 1 else np.nan
        stated_ylabel = "P(Answer)"
    else:
        confidence_options = list(_scale_options().keys())
        stated_num = np.array(
            [get_meta_signal(np.asarray(p)) if p else np.nan for p in meta_probs]
        )
        stated_ylabel = "stated_confidence_numeric (soft)"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- Panel 1: MC chosen answer vs correct distribution ---
    ax = axes[0]
    letters = ["A", "B", "C", "D"]
    chosen_counts = [direct_resp.count(L) for L in letters]
    correct_counts = [correct.count(L) for L in letters]
    x = np.arange(len(letters))
    w = 0.4
    ax.bar(x - w / 2, np.array(chosen_counts) / max(n, 1), width=w, label="chosen")
    ax.bar(x + w / 2, np.array(correct_counts) / max(n, 1), width=w, label="correct")
    ax.axhline(0.25, color="gray", linestyle=":", alpha=0.5, label="uniform")
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.set_ylabel("fraction")
    ax.set_title(f"MC answers  (acc = {acc:.1%}, n = {n})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: Meta response distribution (scale-aware) ---
    ax = axes[1]
    conf_counts = [meta_resp.count(k) for k in confidence_options]
    ax.bar(range(len(confidence_options)), np.array(conf_counts) / max(n, 1),
           color="tab:orange")
    ax.set_xticks(range(len(confidence_options)))
    ax.set_xticklabels(confidence_options, fontsize=9)
    ax.set_ylabel("fraction of responses")
    if META_TASK == "delegate" and DELEGATE_PROMPT_DESIGN == "mc_integrated":
        panel2_title = "Answer-with-delegate distribution  (A/B/C/D/T)"
    elif META_TASK == "delegate":
        panel2_title = "Delegate digit distribution  ('1' vs '2')"
    else:
        panel2_title = f"Stated confidence  ({CONFIDENCE_SCALE} scale)"
    ax.set_title(panel2_title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    # Show mean numeric value / mean P(Answer)
    valid_num = stated_num[~np.isnan(stated_num)]
    if len(valid_num):
        label = "P(Answer)" if META_TASK == "delegate" else "stated_num"
        ax.text(
            0.98, 0.95,
            f"mean {label} = {valid_num.mean():.3f}\nstd  = {valid_num.std():.3f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
        )

    # --- Panel 3: Entropy vs stated confidence / P(Answer) scatter ---
    ax = axes[2]
    mask = (~np.isnan(entropies)) & (~np.isnan(stated_num))
    if mask.sum() > 10:
        r, p_r = pearsonr(entropies[mask], stated_num[mask])
        rho, p_rho = spearmanr(entropies[mask], stated_num[mask])
    else:
        r = rho = float("nan"); p_r = p_rho = float("nan")
    ax.scatter(entropies[mask], stated_num[mask], alpha=0.35, s=12)
    ax.set_xlabel("MC entropy (nats)")
    ax.set_ylabel(stated_ylabel)
    panel3_head = (
        "Entropy  vs  P(Answer)" if META_TASK == "delegate"
        else "Entropy  vs  stated confidence"
    )
    ax.set_title(
        f"{panel3_head}\n"
        f"Pearson r = {r:+.3f}  (p={p_r:.1e})\n"
        f"Spearman ρ = {rho:+.3f}  (p={p_rho:.1e})"
    )
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{get_model_display_label()}  /  {DATASET_NAME}  /  "
        f"{META_TASK}  /  scale={CONFIDENCE_SCALE}",
        y=1.02, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Saved quick summary to {output_path}")


# Metric and extraction helpers live in sibling modules so other interp scripts
# (logit lens, ablation analyses) can reuse them without re-importing this
# 4k-line runner.
from _metrics import compute_uncertainty_metrics
from _extractor import (
    extract_cache_tensors,
    create_fresh_cache,
    BatchedExtractor,
)


# ============================================================================
# MAIN DATA COLLECTION
# ============================================================================

# Batch size for processing (adjust based on GPU memory)
BATCH_SIZE = 4  # Conservative default; increase if you have more VRAM


def find_common_prefix_length(input_ids_list: List[List[int]]) -> int:
    """Find the length of the common token prefix across all sequences."""
    if not input_ids_list:
        return 0
    ref_ids = input_ids_list[0]
    min_len = min(len(ids) for ids in input_ids_list)
    common_len = 0
    for i in range(min_len):
        if all(ids[i] == ref_ids[i] for ids in input_ids_list):
            common_len += 1
        else:
            break
    return common_len


def process_prompts_with_prefix_cache(
    prompts: List[str],
    options_list: List[List[str]],
    tokenizer,
    extractor: BatchedExtractor,
    batch_size: int,
    desc: str,
    collect_activations: bool = True,
    add_special_tokens: bool = False,
) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray], List[Dict], List[str]]:
    """
    Process prompts efficiently with prefix caching.

    When prompts share a common prefix (e.g., same system message and instructions),
    computes the KV cache for the prefix once and reuses it for all suffixes.

    Args:
        prompts: List of full prompt strings
        options_list: List of option lists (one per prompt)
        tokenizer: The tokenizer
        extractor: BatchedExtractor instance (with hooks registered)
        batch_size: Batch size for processing
        desc: Progress bar description
        collect_activations: Whether to collect layer activations
        add_special_tokens: Whether to add BOS token (True for base models)

    Returns:
        (layer_activations, probs, logits, metrics, responses)
    """
    # 1. Tokenize all prompts
    encodings = tokenizer(prompts, add_special_tokens=add_special_tokens)
    input_ids_list = encodings["input_ids"]

    # 2. Find common prefix
    common_len = find_common_prefix_length(input_ids_list)

    # 3. Compute prefix cache if prefix is substantial (>20 tokens)
    MIN_PREFIX_FOR_CACHE = 20
    if common_len > MIN_PREFIX_FOR_CACHE:
        print(f"  Found common prefix ({common_len} tokens). Computing prefix cache...")
        prefix_ids = torch.tensor([input_ids_list[0][:common_len]], device=DEVICE)
        prefix_cache = extractor.compute_prefix_cache(prefix_ids)
        suffixes = [ids[common_len:] for ids in input_ids_list]
        use_cache = True
    else:
        if common_len > 0:
            print(f"  Common prefix too short ({common_len} tokens). Using standard processing.")
        suffixes = input_ids_list
        prefix_cache = None
        use_cache = False

    # Get pad token id from tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 4. Process in batches
    results_acts = []
    results_probs = []
    results_logits = []
    results_metrics = []
    results_responses = []

    for b in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_suffixes = suffixes[b:b+batch_size]
        batch_opts = options_list[b:b+batch_size]
        actual_batch_size = len(batch_suffixes)

        # Left-pad suffixes to same length
        max_len = max(len(s) for s in batch_suffixes)
        padded = torch.full((actual_batch_size, max_len), pad_token_id, dtype=torch.long, device=DEVICE)
        for i, s in enumerate(batch_suffixes):
            padded[i, max_len-len(s):] = torch.tensor(s, dtype=torch.long, device=DEVICE)

        # Get option token IDs (assume all items in batch have same options)
        opt_ids = [tokenizer.encode(o, add_special_tokens=False)[0] for o in batch_opts[0]]

        if use_cache:
            acts, probs, logits, metrics = extractor.extract_batch_with_cache(
                padded, prefix_cache, opt_ids, pad_token_id=pad_token_id
            )
        else:
            # Build full inputs with attention mask
            mask = (padded != pad_token_id).long()
            acts, probs, logits, metrics = extractor.extract_batch(padded, mask, opt_ids)

        if collect_activations:
            results_acts.extend(acts)
        results_probs.extend(probs)
        results_logits.extend(logits)
        results_metrics.extend(metrics)

        # Determine responses based on argmax
        for i, p in enumerate(probs):
            results_responses.append(batch_opts[i][np.argmax(p)])

    return results_acts, results_probs, results_logits, results_metrics, results_responses


def _load_mc_data_for_reuse(paired_data_path: Path, acts_path: Path) -> Tuple[Dict, Dict]:
    """Load confidence-mode direct data for reuse in a delegate-only run.

    Returns (mc_data, paired) where mc_data matches the shape expected by
    collect_meta_only (direct_activations dict, metadata list of per-item
    {probabilities, logits}, and optional direct_metrics dict), and paired is
    the raw paired_data.json for sanity-checking / extracting is_correct etc.
    """
    with open(paired_data_path) as f:
        paired = json.load(f)

    with np.load(acts_path, allow_pickle=True) as f:
        direct_activations = {
            int(k.split("_")[1]): f[k].astype(np.float32)
            for k in f.files if k.startswith("layer_")
        }
        npz_run_id = str(f["run_id"]) if "run_id" in f.files else None
        npz_qids = [str(x) for x in f["question_ids"]] if "question_ids" in f.files else None

    # Cross-check: if both files carry alignment metadata, they must agree.
    pd_run_id = (paired.get("config") or {}).get("run_id")
    if npz_run_id is not None and pd_run_id is not None and npz_run_id != pd_run_id:
        raise ValueError(
            "Reuse aborted: direct_activations.npz and paired_data.json carry "
            f"different run_ids (npz={npz_run_id!r}, json={pd_run_id!r}). "
            "These files are from different runs and cannot be paired by row."
        )
    if npz_qids is not None:
        json_qids = [str((q or {}).get("id", f"q_{i}")) for i, q in enumerate(paired.get("questions", []))]
        if json_qids and json_qids != npz_qids:
            raise ValueError(
                "Reuse aborted: question_ids in direct_activations.npz do not "
                "match paired_data.json — row alignment is broken."
            )

    direct_probs = paired.get("direct_probs", []) or []
    direct_logits = paired.get("direct_logits", []) or []
    n_items = len(direct_probs) if direct_probs else len(paired.get("questions", []))
    metadata = []
    for i in range(n_items):
        metadata.append({
            "probabilities": direct_probs[i] if i < len(direct_probs) else [],
            "logits": direct_logits[i] if i < len(direct_logits) else [],
        })

    direct_metrics_raw = paired.get("direct_metrics", {}) or {}
    direct_metrics = {
        k: np.asarray(v) for k, v in direct_metrics_raw.items()
        if not isinstance(v, dict) and v is not None
    }

    mc_data = {
        "direct_activations": direct_activations,
        "metadata": metadata,
        "direct_metrics": direct_metrics,
        "_source_run_id": npz_run_id,
    }
    return mc_data, paired


def collect_paired_data(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool = True,
    batch_size: int = BATCH_SIZE,
    is_base: bool = False,
    few_shot_mode: str = "fixed",
    few_shot_pool: Optional[List[Dict]] = None,
) -> Dict:
    """
    Collect activations and uncertainty metrics for both direct and meta prompts.

    Uses batched processing with combined activation+logit extraction
    and KV cache prefix sharing for meta prompts (~2-3x overall speedup).

    Returns dict with:
        - direct_activations: {layer_idx: np.array of shape (n_questions, hidden_dim)}
        - meta_activations: {layer_idx: np.array of shape (n_questions, hidden_dim)}
        - direct_metrics: {metric_name: np.array of shape (n_questions,)}
        - direct_probs: list of prob arrays
        - direct_logits: list of logit arrays
        - meta_entropies: np.array (entropy over confidence options)
        - meta_probs: list of prob arrays over S-Z (or [P("1"), P("2")] for delegate)
        - meta_responses: list of predicted confidence letters (or "1"/"2" for delegate)
        - meta_mappings: list of mappings for delegate task (None for confidence)
        - questions: the question data
    """
    print(f"Collecting paired data for {len(questions)} questions (batch_size={batch_size})...")

    extractor = BatchedExtractor(model, num_layers, tokenizer)
    extractor.register_hooks()

    # Storage
    direct_layer_acts = {i: [] for i in range(num_layers)}
    direct_metrics_lists = defaultdict(list)
    direct_probs_list = []
    direct_logits_list = []
    errors = []  # Track per-example failures

    model.eval()

    # Pre-compute option token IDs for meta task
    meta_options = get_meta_options()

    try:
        # ============ PHASE 1: DIRECT PROMPTS ============
        print("\nProcessing direct prompts...")
        num_batches = (len(questions) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Direct prompts"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]

            direct_prompts = []
            direct_options_list = []
            for q in batch_questions:
                if is_base:
                    prompt, options = format_direct_prompt_base(q, mode=few_shot_mode)
                else:
                    prompt, options = format_direct_prompt(q, tokenizer, use_chat_template)
                direct_prompts.append(prompt)
                direct_options_list.append(options)

            # Check if all questions have same options (most MC questions do)
            first_options = direct_options_list[0]
            all_same_options = all(opts == first_options for opts in direct_options_list)

            try:
                if all_same_options:
                    # Batch process
                    direct_option_token_ids = [
                        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in first_options
                    ]

                    inputs = tokenizer(
                        direct_prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(DEVICE)

                    batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        direct_option_token_ids
                    )

                    for acts, probs, logits, metrics in zip(batch_acts, batch_probs, batch_logits, batch_metrics):
                        for layer_idx, act in acts.items():
                            direct_layer_acts[layer_idx].append(act)
                        direct_probs_list.append(probs.tolist())
                        direct_logits_list.append(logits.tolist())
                        for metric_name, metric_val in metrics.items():
                            direct_metrics_lists[metric_name].append(metric_val)

                    del inputs
                else:
                    # Fall back to per-item processing
                    for prompt, options in zip(direct_prompts, direct_options_list):
                        option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
                        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                        batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                            inputs["input_ids"],
                            inputs["attention_mask"],
                            option_token_ids
                        )

                        for layer_idx, act in batch_acts[0].items():
                            direct_layer_acts[layer_idx].append(act)
                        direct_probs_list.append(batch_probs[0].tolist())
                        direct_logits_list.append(batch_logits[0].tolist())
                        for metric_name, metric_val in batch_metrics[0].items():
                            direct_metrics_lists[metric_name].append(metric_val)

                        del inputs
            except Exception as e:
                for idx in range(start_idx, end_idx):
                    errors.append({"phase": "direct", "question_idx": idx, "error": str(e)})
                print(f"\n  Warning: batch {batch_idx} failed ({e}), skipping {end_idx - start_idx} examples")
                torch.cuda.empty_cache()
                continue

            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # ============ PHASE 2: META PROMPTS ============
        print("\nProcessing meta prompts...")

        meta_extended = defaultdict(list)

        if META_TASK == "delegate":
            # Delegate task: process individually due to alternating mapping
            meta_layer_acts = {i: [] for i in range(num_layers)}
            meta_probs_list = []
            meta_entropies = []
            meta_responses = []
            meta_mappings = []
            meta_option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in meta_options]

            for trial_idx, q in enumerate(tqdm(questions, desc="Delegate prompts")):
                prompt, _, mapping = format_delegate_prompt(
                    q, tokenizer, use_chat_template, trial_idx,
                    is_base=is_base,
                    few_shot_mode=BASE_DELEGATE_MODE if is_base else "fixed",
                    few_shot_pool=few_shot_pool,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    meta_option_token_ids
                )

                for layer_idx, act in batch_acts[0].items():
                    meta_layer_acts[layer_idx].append(act)
                meta_probs_list.append(batch_probs[0].tolist())
                meta_entropies.append(batch_metrics[0]["entropy"])
                for k in ("option_logprobs", "entropy_full_vocab", "top20_logits", "top2_margin_logprob"):
                    if k in batch_metrics[0]:
                        meta_extended[k].append(batch_metrics[0][k])
                meta_response = meta_options[np.argmax(batch_probs[0])]
                meta_responses.append(meta_response)
                meta_mappings.append(mapping)

                del inputs

                if (trial_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        else:
            # Confidence task: use prefix caching (all prompts share same prefix)
            meta_prompts = []
            meta_options_list = []
            for q in questions:
                if is_base:
                    prompt, opts = format_meta_prompt_base(q, mode=few_shot_mode)
                else:
                    prompt, opts = format_meta_prompt(q, tokenizer, use_chat_template)
                meta_prompts.append(prompt)
                meta_options_list.append(opts)

            # Use prefix caching for efficiency
            meta_acts, meta_probs_raw, _, meta_metrics_raw, meta_responses = process_prompts_with_prefix_cache(
                meta_prompts,
                meta_options_list,
                tokenizer,
                extractor,
                batch_size,
                desc="Meta prompts (with prefix cache)",
                add_special_tokens=is_base,
            )

            # Convert to expected format
            meta_layer_acts = {i: [] for i in range(num_layers)}
            for acts in meta_acts:
                for layer_idx, act in acts.items():
                    meta_layer_acts[layer_idx].append(act)

            meta_probs_list = [p.tolist() for p in meta_probs_raw]
            meta_entropies = [m["entropy"] for m in meta_metrics_raw]
            for m in meta_metrics_raw:
                for k in ("option_logprobs", "entropy_full_vocab", "top20_logits", "top2_margin_logprob"):
                    if k in m:
                        meta_extended[k].append(m[k])
            meta_mappings = [None] * len(questions)  # No mapping for confidence task

    finally:
        extractor.remove_hooks()

    # Convert to numpy arrays
    direct_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in direct_layer_acts.items()
    }
    meta_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in meta_layer_acts.items()
    }
    # Separate scalar metrics (convertible to arrays) from structured metrics
    scalar_metric_keys = AVAILABLE_METRICS + ["entropy_full_vocab", "top2_margin_logprob"]
    direct_metrics = {}
    direct_extended = {}
    for metric, values in direct_metrics_lists.items():
        if metric in scalar_metric_keys:
            direct_metrics[metric] = np.array(values)
        else:
            direct_extended[metric] = values  # option_logprobs, top20_logits stay as lists

    print(f"\nDirect activations shape (per layer): {direct_activations[0].shape}")
    print(f"Meta activations shape (per layer): {meta_activations[0].shape}")
    print(f"\nDirect uncertainty metrics:")
    for metric_name, values in direct_metrics.items():
        if isinstance(values, np.ndarray) and values.dtype.kind == 'f':
            print(f"  {metric_name}: range=[{values.min():.3f}, {values.max():.3f}], "
                  f"mean={values.mean():.3f}, std={values.std():.3f}")

    return {
        "direct_activations": direct_activations,
        "meta_activations": meta_activations,
        "direct_metrics": direct_metrics,
        "direct_extended": direct_extended,
        "direct_probs": direct_probs_list,
        "direct_logits": direct_logits_list,
        "meta_entropies": np.array(meta_entropies),
        "meta_extended": dict(meta_extended),
        "meta_probs": meta_probs_list,
        "meta_responses": meta_responses,
        "meta_mappings": meta_mappings,
        "questions": questions,
        "errors": errors,
    }


def collect_meta_only(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool,
    mc_data: Dict,
    batch_size: int = BATCH_SIZE,
    is_base: bool = False,
    few_shot_mode: str = "fixed",
    few_shot_pool: Optional[List[Dict]] = None,
) -> Dict:
    """
    Collect only meta prompt data, reusing direct activations from mc_entropy_probe.py.

    This is much faster than collect_paired_data when MC data already exists.
    Uses KV cache prefix sharing for additional speedup on confidence task.
    """
    print(f"Collecting meta data only for {len(questions)} questions (reusing direct activations)...")

    extractor = BatchedExtractor(model, num_layers, tokenizer)
    extractor.register_hooks()

    model.eval()

    # Meta options depend on META_TASK
    meta_options = get_meta_options()
    meta_extended = defaultdict(list)

    try:
        if META_TASK == "delegate":
            # Delegate task: process individually due to alternating mapping
            meta_layer_acts = {i: [] for i in range(num_layers)}
            meta_probs_list = []
            meta_entropies = []
            meta_responses = []
            meta_mappings = []
            meta_option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in meta_options]

            for trial_idx, q in enumerate(tqdm(questions, desc="Delegate prompts")):
                prompt, _, mapping = format_delegate_prompt(
                    q, tokenizer, use_chat_template, trial_idx,
                    is_base=is_base,
                    few_shot_mode=BASE_DELEGATE_MODE if is_base else "fixed",
                    few_shot_pool=few_shot_pool,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    meta_option_token_ids
                )

                for layer_idx, act in batch_acts[0].items():
                    meta_layer_acts[layer_idx].append(act)
                meta_probs_list.append(batch_probs[0].tolist())
                meta_entropies.append(batch_metrics[0]["entropy"])
                for k in ("option_logprobs", "entropy_full_vocab", "top20_logits", "top2_margin_logprob"):
                    if k in batch_metrics[0]:
                        meta_extended[k].append(batch_metrics[0][k])
                meta_response = meta_options[np.argmax(batch_probs[0])]
                meta_responses.append(meta_response)
                meta_mappings.append(mapping)

                del inputs

                if (trial_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        else:
            # Confidence task: use prefix caching (all prompts share same prefix)
            meta_prompts = []
            meta_options_list = []
            for q in questions:
                if is_base:
                    prompt, opts = format_meta_prompt_base(q, mode=few_shot_mode)
                else:
                    prompt, opts = format_meta_prompt(q, tokenizer, use_chat_template)
                meta_prompts.append(prompt)
                meta_options_list.append(opts)

            # Use prefix caching for efficiency
            meta_acts, meta_probs_raw, _, meta_metrics_raw, meta_responses = process_prompts_with_prefix_cache(
                meta_prompts,
                meta_options_list,
                tokenizer,
                extractor,
                batch_size,
                desc="Meta prompts (with prefix cache)",
                add_special_tokens=is_base,
            )

            # Convert to expected format
            meta_layer_acts = {i: [] for i in range(num_layers)}
            for acts in meta_acts:
                for layer_idx, act in acts.items():
                    meta_layer_acts[layer_idx].append(act)

            meta_probs_list = [p.tolist() for p in meta_probs_raw]
            meta_entropies = [m["entropy"] for m in meta_metrics_raw]
            for m in meta_metrics_raw:
                for k in ("option_logprobs", "entropy_full_vocab", "top20_logits", "top2_margin_logprob"):
                    if k in m:
                        meta_extended[k].append(m[k])
            meta_mappings = [None] * len(questions)

    finally:
        extractor.remove_hooks()

    meta_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in meta_layer_acts.items()
    }

    print(f"Meta activations shape (per layer): {meta_activations[0].shape}")

    # Build direct data from mc_data
    # mc_data may have "direct_metrics" dict (new format) or just "direct_entropies" (old format)
    direct_probs_list = [m.get("probabilities", []) for m in mc_data["metadata"]]
    direct_logits_list = [m.get("logits", []) for m in mc_data["metadata"]]

    # Handle both old (entropies only) and new (all metrics) mc_data formats
    if "direct_metrics" in mc_data:
        direct_metrics = mc_data["direct_metrics"]
    else:
        # Old format: only has entropies, need to compute metrics from metadata
        direct_metrics = {metric: [] for metric in AVAILABLE_METRICS}
        for m in mc_data["metadata"]:
            probs = np.array(m.get("probabilities", []))
            logits = np.array(m.get("logits", [])) if m.get("logits") else None
            if len(probs) > 0:
                item_metrics = compute_uncertainty_metrics(probs, logits)
                for metric_name, metric_val in item_metrics.items():
                    direct_metrics[metric_name].append(metric_val)
        direct_metrics = {k: np.array(v) for k, v in direct_metrics.items()}

    return {
        "direct_activations": mc_data["direct_activations"],
        "meta_activations": meta_activations,
        "direct_metrics": direct_metrics,
        "direct_extended": {},  # Not available when reusing MC data
        "direct_probs": direct_probs_list,
        "direct_logits": direct_logits_list,
        "meta_entropies": np.array(meta_entropies),
        "meta_extended": dict(meta_extended),
        "meta_probs": meta_probs_list,
        "meta_responses": meta_responses,
        "meta_mappings": meta_mappings,
        "questions": questions
    }


def collect_other_confidence(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool,
    batch_size: int = BATCH_SIZE,
    collect_activations: bool = False
) -> Dict:
    """
    Collect other-confidence (human difficulty estimation) responses.

    This is a control task: asks model to estimate what % of college-educated
    people would know the answer (instead of asking about its own confidence).

    If the model is truly introspecting on its own uncertainty, the self-confidence
    task should correlate more strongly with its actual uncertainty metrics than
    this other-confidence task.

    Uses KV cache prefix sharing for efficiency.

    Args:
        collect_activations: If True, also collect layer activations for probe analysis.

    Returns dict with:
        - other_probs: list of prob arrays over S-Z options
        - other_responses: list of predicted confidence letters
        - other_signals: list of expected confidence values (weighted avg of midpoints)
        - other_activations: (optional) dict of layer_idx -> [n_questions, hidden_dim] activations
    """
    print(f"\nCollecting other-confidence (control) data for {len(questions)} questions...")

    extractor = BatchedExtractor(model, num_layers, tokenizer)
    extractor.register_hooks()

    model.eval()

    try:
        # Format all prompts
        other_prompts = []
        other_options_list = []
        for q in questions:
            prompt, opts = format_other_confidence_prompt(q, tokenizer, use_chat_template)
            other_prompts.append(prompt)
            other_options_list.append(opts)

        # Use prefix caching for efficiency
        desc = "Other-confidence (with activations)" if collect_activations else "Other-confidence (with prefix cache)"
        other_activations_raw, other_probs_raw, _, _, other_responses = process_prompts_with_prefix_cache(
            other_prompts,
            other_options_list,
            tokenizer,
            extractor,
            batch_size,
            desc=desc,
            collect_activations=collect_activations
        )

        # Convert to expected format and compute signals
        other_probs_list = [p.tolist() for p in other_probs_raw]
        other_signals = [get_other_confidence_signal(p) for p in other_probs_raw]

    finally:
        extractor.remove_hooks()

    print(f"Other-confidence signals: mean={np.mean(other_signals):.3f}, std={np.std(other_signals):.3f}")

    result = {
        "other_probs": other_probs_list,
        "other_responses": other_responses,
        "other_signals": np.array(other_signals),
    }

    if collect_activations and other_activations_raw is not None and len(other_activations_raw) > 0:
        # other_activations_raw is a list of dicts (one per question), each dict maps layer_idx -> activation
        # Convert to dict of numpy arrays keyed by layer
        num_layers = len(other_activations_raw[0])
        other_layer_acts = {i: [] for i in range(num_layers)}
        for acts in other_activations_raw:
            for layer_idx, act in acts.items():
                other_layer_acts[layer_idx].append(act)

        other_activations = {
            layer_idx: np.array(acts)
            for layer_idx, acts in other_layer_acts.items()
        }
        result["other_activations"] = other_activations
        print(f"Collected other-confidence activations: {len(other_activations)} layers, shape={other_activations[0].shape}")

    return result


# ============================================================================
# PROBE TRAINING AND EVALUATION
# ============================================================================

def extract_direction(
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge
) -> np.ndarray:
    """
    Extract normalized direction from trained probe in original activation space.

    Maps the probe weights back through PCA (if used) and standardization
    to get the direction in the original activation space.
    """
    coef = probe.coef_

    if pca is not None:
        # Map from PCA space back to scaled space
        direction_scaled = pca.components_.T @ coef
    else:
        direction_scaled = coef

    # Undo standardization scaling
    direction_original = direction_scaled / scaler.scale_

    # Normalize to unit length
    direction_original = direction_original / np.linalg.norm(direction_original)

    return direction_original


def extract_mc_answer_direction(
    scaler: StandardScaler,
    pca: PCA,
    clf: LogisticRegression
) -> np.ndarray:
    """
    Extract a single normalized direction from a multiclass LogisticRegression classifier.

    For a 4-class classifier (A/B/C/D), clf.coef_ has shape (4, n_pca_components).
    We extract the first principal component of these 4 class vectors to get
    the dominant axis of variation that distinguishes different MC answer classes.

    Maps the direction back through PCA and standardization to original activation space.
    """
    # clf.coef_ is (n_classes, n_pca_components) - one row per class
    # Take first principal component of the class coefficient vectors
    coef_pca = PCA(n_components=1)
    coef_pca.fit(clf.coef_)
    pc1 = coef_pca.components_[0]  # (n_pca_components,)

    # Project back to scaled space
    direction_scaled = pca.components_.T @ pc1  # (hidden_dim,)

    # Undo standardization scaling
    direction_original = direction_scaled / scaler.scale_

    # Normalize to unit length
    direction_original = direction_original / np.linalg.norm(direction_original)

    return direction_original


def compute_contrast_directions(
    activations: Dict[int, np.ndarray],
    signal: np.ndarray,
    percent: float = 25.0,
    samples_per_bin: Optional[int] = None,
    seed: int = 0,
) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray]]:
    """Per-layer mean-difference direction between top and bottom groups of `signal`.

    Splits questions by percentile (default: top 25% vs bottom 25% of signal,
    middle 50% dropped) and at each layer takes
        mean(activations[high]) - mean(activations[low]),
    normalised to unit length.

    Args:
        activations: {layer_idx: (n_questions, hidden_dim)} float array per layer.
            All layers must share the same n_questions and row ordering.
        signal: (n_questions,) scalar value per question (e.g. entropy, soft
            stated confidence). NaNs must be filtered out by the caller.
        percent: tail width in percent, in (0, 50). 25 → bottom 25% vs top 25%.
        samples_per_bin: if set, randomly subsample to this many questions per
            tail before averaging (deterministic, uses `seed`). None →
            use every question that falls in the tail.
        seed: RNG seed for the subsample step.

    Returns:
        directions: {layer_idx: (hidden_dim,) unit-norm contrast vector}
        meta: small dict of scalar numpy arrays describing the split — keys
            "percent", "low_threshold", "high_threshold", "n_low", "n_high",
            "n_low_used", "n_high_used", "samples_per_bin", "seed".
            Suitable for splatting into np.savez_compressed alongside the
            per-layer direction arrays.
    """
    if not (0 < percent < 50):
        raise ValueError(f"percent must be in (0, 50), got {percent}")

    signal = np.asarray(signal, dtype=np.float64)
    quantile = percent / 100.0
    lo_thresh = float(np.quantile(signal, quantile))
    hi_thresh = float(np.quantile(signal, 1.0 - quantile))
    low_idx = np.where(signal <= lo_thresh)[0]
    high_idx = np.where(signal >= hi_thresh)[0]
    n_low = int(low_idx.size)
    n_high = int(high_idx.size)
    if n_low == 0 or n_high == 0:
        raise ValueError(
            f"Empty contrast group(s): n_low={n_low}, n_high={n_high} "
            f"(percent={percent}, signal range "
            f"[{float(signal.min()):.3f}, {float(signal.max()):.3f}])"
        )

    # Optional subsample to a fixed per-bin sample size.
    if samples_per_bin is not None and samples_per_bin > 0:
        rng = np.random.default_rng(seed)
        if low_idx.size > samples_per_bin:
            low_idx = rng.choice(low_idx, size=samples_per_bin, replace=False)
        if high_idx.size > samples_per_bin:
            high_idx = rng.choice(high_idx, size=samples_per_bin, replace=False)
    n_low_used = int(low_idx.size)
    n_high_used = int(high_idx.size)

    directions: Dict[int, np.ndarray] = {}
    for layer_idx in sorted(activations.keys()):
        acts = activations[layer_idx]
        diff = acts[high_idx].mean(axis=0) - acts[low_idx].mean(axis=0)
        norm = float(np.linalg.norm(diff))
        directions[layer_idx] = diff if norm < 1e-10 else (diff / norm)

    meta = {
        "percent": np.array(percent, dtype=np.float32),
        "low_threshold": np.array(lo_thresh, dtype=np.float32),
        "high_threshold": np.array(hi_thresh, dtype=np.float32),
        "n_low": np.array(n_low, dtype=np.int32),
        "n_high": np.array(n_high, dtype=np.int32),
        "n_low_used": np.array(n_low_used, dtype=np.int32),
        "n_high_used": np.array(n_high_used, dtype=np.int32),
        "samples_per_bin": np.array(
            -1 if samples_per_bin is None else int(samples_per_bin), dtype=np.int32
        ),
        "seed": np.array(int(seed), dtype=np.int32),
    }
    return directions, meta


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    return_components: bool = False
) -> Dict:
    """
    Train a linear probe to predict entropy from activations.

    If return_components=True, also returns the scaler, pca, and probe objects
    for applying to new data.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA if enabled
    pca = None
    if USE_PCA:
        n_components = min(PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Train Ridge regression probe
    probe = Ridge(alpha=PROBE_ALPHA)
    probe.fit(X_train_final, y_train)

    # Evaluate
    y_pred_train = probe.predict(X_train_final)
    y_pred_test = probe.predict(X_test_final)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # NEW: Correlations
    test_pearson, _ = pearsonr(y_test, y_pred_test)
    test_spearman, _ = spearmanr(y_test, y_pred_test)

    result = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_pearson": test_pearson,   # <--- Added
        "test_spearman": test_spearman, # <--- Added
        "predictions": y_pred_test,
        "pca_variance_explained": pca.explained_variance_ratio_.sum() if USE_PCA else None
    }

    if return_components:
        result["scaler"] = scaler
        result["pca"] = pca
        result["probe"] = probe

    return result


def apply_trained_probe(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge
) -> Dict:
    """Apply a pre-trained probe to new data using the original scaler."""
    X_scaled = scaler.transform(X)

    if pca is not None:
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled

    y_pred = probe.predict(X_final)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # NEW: Correlations - The "Shift Invariant" Check
    pearson, _ = pearsonr(y, y_pred)
    spearman, _ = spearmanr(y, y_pred)

    return {
        "r2": r2,
        "mae": mae,
        "pearson": pearson,     # <--- Added
        "spearman": spearman,   # <--- Added
        "predictions": y_pred
    }

def apply_probe_centering_only(
    X_meta: np.ndarray,
    y_test: np.ndarray,
    direct_scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge
) -> Dict:
    """
    Apply probe with Mean-Shift correction but Shared Variance.
    
    This fixes the 'Prompt Offset' (intercept) without distorting the 
    geometry (angles) of the activation space.
    """
    # 1. Calculate Meta Mean locally
    meta_mean = np.mean(X_meta, axis=0)
    
    # 2. Center Meta data using its OWN mean
    X_centered = X_meta - meta_mean
    
    # 3. Scale using DIRECT scaler's variance (preserve geometry)
    # direct_scaler.scale_ contains the std deviation of direct data
    X_scaled = X_centered / direct_scaler.scale_
    
    # 4. Apply Direct PCA and Probe
    if pca is not None:
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled
        
    y_pred = probe.predict(X_final)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    pearson, _ = pearsonr(y_test, y_pred)
    spearman, _ = spearmanr(y_test, y_pred)
    
    return {
        "r2": r2,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "predictions": y_pred
    }

def apply_probe_with_separate_scaling(
    X: np.ndarray,
    y: np.ndarray,
    pca: Optional[PCA],
    probe: Ridge
) -> Dict:
    """
    Apply a pre-trained probe to new data with SEPARATE standardization.
    
    Standardizes meta activations using their own statistics (Mean=0, Var=1).
    This acts as an upper bound / domain adaptation test.
    """
    # Standardize X using its own statistics (not the original scaler)
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X)

    if pca is not None:
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled

    y_pred = probe.predict(X_final)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # NEW: Correlations
    pearson, _ = pearsonr(y, y_pred)
    spearman, _ = spearmanr(y, y_pred)

    return {
        "r2": r2,
        "mae": mae,
        "pearson": pearson,     # <--- Added
        "spearman": spearman,   # <--- Added
        "predictions": y_pred
    }

def run_introspection_analysis(
    direct_activations: Dict[int, np.ndarray],
    meta_activations: Dict[int, np.ndarray],
    direct_entropies: np.ndarray,
    pretrained_probe_path: Optional[str] = None,
    extract_directions: bool = True
) -> Dict:
    """
    Run the full introspection analysis:
    1. Train probe on direct activations -> direct entropy
    2. Test on held-out direct data (Sanity Check)
    3. Test on meta activations (The Transfer Tests):
       a. Shared Scaler (Exact probe application)
       b. Centered Scaler (Mean-shift correction, Shared geometry)
       c. Separate Scaler (Domain adaptation / Upper bound)
    4. Shuffled baseline control

    Returns (results, test_idx, directions).
    """
    print(f"\nRunning introspection analysis across {len(direct_activations)} layers...")

    # Split by question index (same split for both direct and meta)
    n_questions = len(direct_entropies)
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED
    )

    print(f"Train set: {len(train_idx)} questions, Test set: {len(test_idx)} questions")

    results = {}
    directions = {} if extract_directions else None
    probe_components = {}  # Store trained probe components for reuse

    for layer_idx in tqdm(sorted(direct_activations.keys()), desc="Training probes"):
        X_direct = direct_activations[layer_idx]
        X_meta = meta_activations[layer_idx]
        y = direct_entropies

        # Split
        X_direct_train = X_direct[train_idx]
        X_direct_test = X_direct[test_idx]
        X_meta_test = X_meta[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # 1. Train on direct, test on direct (sanity check)
        # Note: train_probe now returns correlations too
        direct_results = train_probe(
            X_direct_train, y_train,
            X_direct_test, y_test,
            return_components=True
        )

        # 2a. Original approach: Shared Scaler
        # Tests: Is the uncertainty vector identical in absolute terms?
        # (Likely to fail R2 due to offset, but should have high Pearson if introspecting)
        meta_results_shared = apply_trained_probe(
            X_meta_test, y_test,
            direct_results["scaler"],
            direct_results["pca"],
            direct_results["probe"]
        )

        # 2b. NEW: Centering Only
        # Tests: Is the uncertainty vector structurally identical (same angles) 
        # but shifted in position (prompt offset)?
        meta_results_centered = apply_probe_centering_only(
            X_meta_test, y_test,
            direct_results["scaler"],
            direct_results["pca"],
            direct_results["probe"]
        )

        # 2c. Fixed approach: Separate Scaler
        # Tests: Is the uncertainty vector recoverable if we allow warping the space?
        # (Upper bound / Domain adaptation)
        meta_results_separate = apply_probe_with_separate_scaling(
            X_meta_test, y_test,
            direct_results["pca"],
            direct_results["probe"]
        )

        # 3. Shuffled baseline
        shuffled_y_train = y_train.copy()
        np.random.shuffle(shuffled_y_train)
        shuffled_results = train_probe(
            X_direct_train, shuffled_y_train,
            X_direct_test, y_test,
            return_components=False
        )

        # 4. Train on meta, test on meta (does meta have ANY signal?)
        meta_to_meta_results = train_probe(
            X_meta[train_idx], y_train,
            X_meta_test, y_test,
            return_components=False
        )

        # 5. Extract entropy direction (for steering)
        if extract_directions:
            directions[layer_idx] = extract_direction(
                direct_results["scaler"],
                direct_results["pca"],
                direct_results["probe"]
            )

        # 6. Store probe components for reuse (e.g., other-confidence transfer)
        probe_components[layer_idx] = {
            "scaler": direct_results["scaler"],
            "pca": direct_results["pca"],
            "probe": direct_results["probe"],
        }

        # Pack results
        results[layer_idx] = {
            "direct_to_direct": {
                "train_r2": direct_results["train_r2"],
                "test_r2": direct_results["test_r2"],
                "test_pearson": direct_results["test_pearson"],
                "test_mae": direct_results["test_mae"],
                "predictions": direct_results["predictions"].tolist(),
            },
            "direct_to_meta": {
                # Shared scaler (Original)
                "r2": meta_results_shared["r2"],
                "pearson": meta_results_shared["pearson"],
                "mae": meta_results_shared["mae"],
                "predictions": meta_results_shared["predictions"].tolist(),
            },
            "direct_to_meta_centered": {
                # Centered scaler (New "Goldilocks" metric)
                "r2": meta_results_centered["r2"],
                "pearson": meta_results_centered["pearson"],
                "mae": meta_results_centered["mae"],
                "predictions": meta_results_centered["predictions"].tolist(),
            },
            "direct_to_meta_fixed": {
                # Separate scaler (Separate mean+var)
                "r2": meta_results_separate["r2"],
                "pearson": meta_results_separate["pearson"],
                "mae": meta_results_separate["mae"],
                "predictions": meta_results_separate["predictions"].tolist(),
            },
            "shuffled_baseline": {
                "r2": shuffled_results["test_r2"],
                "mae": shuffled_results["test_mae"],
            },
            "meta_to_meta": {
                "train_r2": meta_to_meta_results["train_r2"],
                "test_r2": meta_to_meta_results["test_r2"],
            },
            "pca_variance_explained": direct_results["pca_variance_explained"]
        }

    return results, test_idx, directions, probe_components

def run_mc_answer_analysis(
    direct_activations: Dict[int, np.ndarray],
    meta_activations: Dict[int, np.ndarray],
    model_predicted_answer: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict, Dict, Dict[int, np.ndarray]]:
    """
    Train logistic regression probe to predict model's MC answer choice (A/B/C/D).
    Computes BOTH Separate (Upper Bound) and Centered (Rigorous) transfer metrics.

    Returns:
        results: Dict of accuracy metrics per layer
        mc_probe_components: Dict of {scaler, pca, clf} per layer for reuse
        mc_directions: Dict of normalized direction vectors per layer (for ablation)
    """
    print(f"\nRunning MC answer probe analysis across {len(direct_activations)} layers...")

    results = {}
    mc_probe_components = {}  # Store trained probe components for reuse
    mc_directions = {}  # Store extracted directions for ablation experiments

    for layer_idx in tqdm(sorted(direct_activations.keys()), desc="Training MC answer probes"):
        X_direct = direct_activations[layer_idx]
        X_meta = meta_activations[layer_idx]
        y = model_predicted_answer

        # Split
        X_direct_train = X_direct[train_idx]
        X_direct_test = X_direct[test_idx]
        X_meta_test = X_meta[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # --- Train on Direct ---
        scaler = StandardScaler()
        X_direct_train_scaled = scaler.fit_transform(X_direct_train)
        X_direct_test_scaled = scaler.transform(X_direct_test)

        # PCA
        pca = PCA(n_components=min(256, X_direct_train_scaled.shape[1], X_direct_train_scaled.shape[0]))
        X_direct_train_pca = pca.fit_transform(X_direct_train_scaled)
        X_direct_test_pca = pca.transform(X_direct_test_scaled)

        # Logistic Regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_direct_train_pca, y_train)

        # D2D Accuracy
        d2d_accuracy = clf.score(X_direct_test_pca, y_test)
        d2d_predictions = clf.predict(X_direct_test_pca)

        # --- Transfer 1: Separate Scaling (Upper Bound) ---
        meta_scaler_sep = StandardScaler()
        X_meta_test_sep = meta_scaler_sep.fit_transform(X_meta_test)
        X_meta_test_sep_pca = pca.transform(X_meta_test_sep)
        d2m_separate_accuracy = clf.score(X_meta_test_sep_pca, y_test)

        # --- Transfer 2: Centered Scaling (Rigorous) ---
        # 1. Center Meta using its own mean
        meta_mean = np.mean(X_meta_test, axis=0)
        X_meta_centered = X_meta_test - meta_mean
        # 2. Scale using DIRECT variance (scaler.scale_)
        # This ensures we aren't "re-fitting" the geometry
        X_meta_test_cen = X_meta_centered / scaler.scale_
        X_meta_test_cen_pca = pca.transform(X_meta_test_cen)
        d2m_centered_accuracy = clf.score(X_meta_test_cen_pca, y_test)

        # Shuffled baseline
        shuffled_y_train = y_train.copy()
        np.random.shuffle(shuffled_y_train)
        clf_shuffled = LogisticRegression(max_iter=1000, random_state=42)
        clf_shuffled.fit(X_direct_train_pca, shuffled_y_train)
        shuffled_accuracy = clf_shuffled.score(X_direct_test_pca, y_test)

        results[layer_idx] = {
            "d2d_accuracy": d2d_accuracy,
            "d2m_accuracy": d2m_separate_accuracy, # Legacy key
            "d2m_separate_accuracy": d2m_separate_accuracy,
            "d2m_centered_accuracy": d2m_centered_accuracy,
            "shuffled_accuracy": shuffled_accuracy,
            "d2d_predictions": d2d_predictions.tolist(),
        }

        # Store probe components for reuse (e.g., other-confidence transfer)
        mc_probe_components[layer_idx] = {
            "scaler": scaler,
            "pca": pca,
            "clf": clf,
        }

        # Extract direction for ablation experiments
        mc_directions[layer_idx] = extract_mc_answer_direction(scaler, pca, clf)

    return results, mc_probe_components, mc_directions


def apply_probes_to_other(
    other_activations: Dict[int, np.ndarray],
    entropy_probe_components: Dict[int, Dict],
    mc_probe_components: Dict[int, Dict],
    direct_entropy: np.ndarray,
    model_predicted_answer: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    """
    Apply trained probes to other-confidence activations.

    Only computes D→M(Other) - reuses trained probe components from main analysis.
    This is a control test: if transfer works equally well to other-confidence,
    the model encodes uncertainty similarly regardless of the meta task.

    Args:
        other_activations: Layer activations from other-confidence task
        entropy_probe_components: {layer: {"scaler", "pca", "probe"}} from run_introspection_analysis
        mc_probe_components: {layer: {"scaler", "pca", "clf"}} from run_mc_answer_analysis
        direct_entropy: Ground truth entropy values
        model_predicted_answer: Model's predicted MC answer (A/B/C/D as int)
        test_idx: Test set indices

    Returns:
        Dict with layer-wise D→M(Other) R² and accuracy for both probes
    """
    print(f"\nApplying trained probes to other-confidence activations...")

    y_entropy = direct_entropy[test_idx]
    y_mc = model_predicted_answer[test_idx]

    results = {}
    for layer_idx in tqdm(sorted(other_activations.keys()), desc="Testing D→M(Other) transfer"):
        X_other = other_activations[layer_idx][test_idx]

        # --- Entropy probe: centered scaling ---
        ent_comps = entropy_probe_components[layer_idx]
        other_mean = np.mean(X_other, axis=0)
        X_other_centered = X_other - other_mean
        X_other_scaled = X_other_centered / ent_comps["scaler"].scale_
        X_other_pca = ent_comps["pca"].transform(X_other_scaled)
        entropy_preds = ent_comps["probe"].predict(X_other_pca)
        entropy_r2 = r2_score(y_entropy, entropy_preds)

        # --- MC answer probe: centered scaling ---
        mc_comps = mc_probe_components[layer_idx]
        X_other_mc_centered = X_other - np.mean(X_other, axis=0)
        X_other_mc_scaled = X_other_mc_centered / mc_comps["scaler"].scale_
        X_other_mc_pca = mc_comps["pca"].transform(X_other_mc_scaled)
        mc_acc = mc_comps["clf"].score(X_other_mc_pca, y_mc)

        results[layer_idx] = {
            "d2m_other_entropy_r2": entropy_r2,
            "d2m_other_mc_accuracy": mc_acc,
        }

    return results


def compute_calibration_masks(
    stated_confidence: np.ndarray,
    direct_metric: np.ndarray,
    metric_is_uncertainty: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibrated/miscalibrated masks using median split.

    Args:
        stated_confidence: Model's stated confidence (meta-judgment)
        direct_metric: The uncertainty/confidence metric being probed
        metric_is_uncertainty: True if high metric = uncertain (entropy),
                               False if high metric = confident (logit_gap, top_prob, margin, top_logit)

    Calibrated: model's meta-judgment matches actual uncertainty
      - For uncertainty metrics (entropy): high conf + low metric, or low conf + high metric
      - For confidence metrics (logit_gap): high conf + high metric, or low conf + low metric

    Miscalibrated: model's meta-judgment opposes actual uncertainty

    Returns (calibrated_mask, miscalibrated_mask).
    Uses median split for balanced groups.
    """
    conf_median = np.median(stated_confidence)
    metric_median = np.median(direct_metric)

    if metric_is_uncertainty:
        # High metric = uncertain, so calibrated = (high conf + low metric) or (low conf + high metric)
        calibrated = (
            ((stated_confidence > conf_median) & (direct_metric < metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric > metric_median))
        )
        miscalibrated = (
            ((stated_confidence > conf_median) & (direct_metric > metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric < metric_median))
        )
    else:
        # High metric = confident, so calibrated = (high conf + high metric) or (low conf + low metric)
        calibrated = (
            ((stated_confidence > conf_median) & (direct_metric > metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric < metric_median))
        )
        miscalibrated = (
            ((stated_confidence > conf_median) & (direct_metric < metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric > metric_median))
        )

    return calibrated, miscalibrated


def split_results_by_calibration(
    results: Dict,
    y_test: np.ndarray,
    calibrated_mask: np.ndarray,
    miscalibrated_mask: np.ndarray
) -> Dict:
    """
    Recompute R² separately for calibrated/miscalibrated test trials.

    Uses predictions already stored in results dict from run_introspection_analysis.
    Also extracts shuffled baseline R² for reference.
    """
    split_results = {}
    for layer_idx, layer_data in results.items():
        d2d_preds = np.array(layer_data["direct_to_direct"]["predictions"])
        d2m_preds = np.array(layer_data["direct_to_meta_fixed"]["predictions"])
        shuffled_r2 = layer_data["shuffled_baseline"]["r2"]

        # Calibrated subset
        if calibrated_mask.sum() > 1:
            d2d_r2_cal = r2_score(y_test[calibrated_mask], d2d_preds[calibrated_mask])
            d2m_r2_cal = r2_score(y_test[calibrated_mask], d2m_preds[calibrated_mask])
        else:
            d2d_r2_cal = float('nan')
            d2m_r2_cal = float('nan')

        # Miscalibrated subset
        if miscalibrated_mask.sum() > 1:
            d2d_r2_mis = r2_score(y_test[miscalibrated_mask], d2d_preds[miscalibrated_mask])
            d2m_r2_mis = r2_score(y_test[miscalibrated_mask], d2m_preds[miscalibrated_mask])
        else:
            d2d_r2_mis = float('nan')
            d2m_r2_mis = float('nan')

        split_results[layer_idx] = {
            "calibrated": {"d2d_r2": d2d_r2_cal, "d2m_r2": d2m_r2_cal},
            "miscalibrated": {"d2d_r2": d2d_r2_mis, "d2m_r2": d2m_r2_mis},
            "shuffled_r2": shuffled_r2,
        }
    return split_results


def plot_calibration_split(
    split_results: Dict,
    n_calibrated: int,
    n_miscalibrated: int,
    output_path: str
):
    """
    Side-by-side 1x2 figure: calibrated (left) vs miscalibrated (right).

    Shows D2D and D2M R² curves for each subset to see if transfer
    is driven by calibrated or miscalibrated trials.
    """
    layers = sorted(split_results.keys())
    shuffled_r2 = [split_results[l]["shuffled_r2"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Calibrated trials
    ax1 = axes[0]
    cal_d2d = [split_results[l]["calibrated"]["d2d_r2"] for l in layers]
    cal_d2m = [split_results[l]["calibrated"]["d2m_r2"] for l in layers]
    ax1.plot(layers, cal_d2d, 'o-', label='Direct→Direct', linewidth=2, color='C0')
    ax1.plot(layers, cal_d2m, 's-', label='Direct→Meta', linewidth=2, color='C1')
    ax1.plot(layers, shuffled_r2, 'x--', label='Shuffled baseline', linewidth=1, alpha=0.5, color='gray')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'Calibrated Trials (n={n_calibrated})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Right: Miscalibrated trials
    ax2 = axes[1]
    mis_d2d = [split_results[l]["miscalibrated"]["d2d_r2"] for l in layers]
    mis_d2m = [split_results[l]["miscalibrated"]["d2m_r2"] for l in layers]
    ax2.plot(layers, mis_d2d, 'o-', label='Direct→Direct', linewidth=2, color='C0')
    ax2.plot(layers, mis_d2m, 's-', label='Direct→Meta', linewidth=2, color='C1')
    ax2.plot(layers, shuffled_r2, 'x--', label='Shuffled baseline', linewidth=1, alpha=0.5, color='gray')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('R² Score')
    ax2.set_title(f'Miscalibrated Trials (n={n_miscalibrated})')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Calibration split — {get_model_display_label()} / {DATASET_NAME}",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration split plot saved to {output_path}")


def plot_other_confidence_comparison(
    main_results: Dict,
    mc_results: Dict,
    other_results: Dict,
    output_path: str
):
    """
    Compare direct-trained probe transfer to self-confidence vs other-confidence.

    2-panel figure:
      - Left: Entropy probe (D→D, D→M Self, D→M Other, Shuffled)
      - Right: MC answer probe (D→D, D→M Self, D→M Other, Shuffled)

    If D→M(Other) transfer is as strong as D→M(Self), it suggests the model
    encodes uncertainty similarly across both meta tasks, not specifically
    during self-confidence introspection.
    """
    layers = sorted(main_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Entropy Probe ---
    ax1 = axes[0]
    d2d = [main_results[l]["direct_to_direct"]["test_r2"] for l in layers]
    d2m_self = [main_results[l]["direct_to_meta_fixed"]["r2"] for l in layers]
    d2m_other = [other_results[l]["d2m_other_entropy_r2"] for l in layers]
    shuffled = [main_results[l]["shuffled_baseline"]["r2"] for l in layers]

    ax1.plot(layers, d2d, 'o-', label='D→D', linewidth=2, color='C0')
    ax1.plot(layers, d2m_self, 's-', label='D→M (Self)', linewidth=2, color='C1')
    ax1.plot(layers, d2m_other, '^-', label='D→M (Other)', linewidth=2, color='C2')
    ax1.plot(layers, shuffled, 'x--', label='Shuffled', linewidth=1, alpha=0.5, color='gray')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Entropy Probe Transfer')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: MC Answer Probe ---
    ax2 = axes[1]
    mc_d2d = [mc_results[l]["d2d_accuracy"] for l in layers]
    mc_d2m_self = [mc_results[l]["d2m_centered_accuracy"] for l in layers]
    mc_d2m_other = [other_results[l]["d2m_other_mc_accuracy"] for l in layers]
    mc_shuffled = [mc_results[l]["shuffled_accuracy"] for l in layers]

    ax2.plot(layers, mc_d2d, 'o-', label='D→D', linewidth=2, color='C0')
    ax2.plot(layers, mc_d2m_self, 's-', label='D→M (Self)', linewidth=2, color='C1')
    ax2.plot(layers, mc_d2m_other, '^-', label='D→M (Other)', linewidth=2, color='C2')
    ax2.plot(layers, mc_shuffled, 'x--', label='Shuffled', linewidth=1, alpha=0.5, color='gray')
    ax2.axhline(y=0.25, color='black', linestyle='-', alpha=0.3)  # 4-way random chance
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('MC Answer Probe Transfer')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Self vs Other confidence transfer — {get_model_display_label()} / {DATASET_NAME}",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Other-confidence transfer comparison saved to {output_path}")


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_behavioral_introspection(
    meta_responses: List[str],
    direct_entropies: np.ndarray,
    test_idx: np.ndarray,
    meta_probs: List[List[float]] = None,
    meta_mappings: List[Dict[str, str]] = None,
    direct_probs: List[List[float]] = None,
    questions: List[Dict] = None
) -> Dict:
    """
    Analyze correlation between model's stated confidence and direct entropy.

    If the model is introspecting, its stated confidence should negatively
    correlate with direct entropy (high confidence = low entropy).

    For delegate task, confidence = P(Answer) from meta_probs.
    For confidence task, confidence = weighted average of midpoints (expected value).

    For delegate task, also computes:
    - Delegation rate: fraction of questions delegated
    - Self-answer accuracy: accuracy when model chooses to answer itself
    - Teammate accuracy when delegated (fixed at 50%)
    """
    # Convert meta responses to confidence values
    # For confidence task, use weighted average of midpoints (expected value)
    # For delegate task, use P(Answer)
    if META_TASK == "delegate":
        stated_confidence = np.array([
            local_response_to_confidence(r, np.array(p) if p else None, m)
            for r, p, m in zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses),
                meta_mappings or [None] * len(meta_responses)
            )
        ])
    else:
        # Confidence task: weighted average of midpoints (scale-aware)
        stated_confidence = np.array([
            get_meta_signal(np.array(p)) if p else 0.5
            for p in meta_probs
        ])

    # Split into test set (same split used for probe evaluation)
    test_confidence = stated_confidence[test_idx]
    test_entropy = direct_entropies[test_idx]

    # Subsample-to-m interval approach for CIs
    # This gives intervals calibrated to test size, centered on point estimates
    from scipy import stats
    n_subsamples = 200  # K iterations
    n = len(direct_entropies)
    m = len(test_idx)  # test size

    # Helper: Fisher z transform and inverse
    def fisher_z(r):
        # Clip to avoid inf at r=±1
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def fisher_z_inv(z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    # Full correlation (point estimate on all data - stable)
    full_r = np.corrcoef(stated_confidence, direct_entropies)[0, 1]
    full_z = fisher_z(full_r)

    # Test correlation (point estimate on specific test_idx)
    test_r = np.corrcoef(test_confidence, test_entropy)[0, 1]
    n_test = len(test_confidence)

    # Subsample-to-m interval for FULL correlation
    # Subsample m items, compute correlation, get deviation from full_r in z-space
    full_subsample_deviations = []
    for k in range(n_subsamples):
        rng = np.random.default_rng(k)
        idx = rng.choice(n, size=m, replace=False)
        sub_r = np.corrcoef(stated_confidence[idx], direct_entropies[idx])[0, 1]
        if not np.isnan(sub_r):
            sub_z = fisher_z(sub_r)
            full_subsample_deviations.append(sub_z - full_z)

    # Get percentiles of deviations, then map back to correlation space
    dev_lower = np.percentile(full_subsample_deviations, 2.5)
    dev_upper = np.percentile(full_subsample_deviations, 97.5)
    full_ci_lower = fisher_z_inv(full_z + dev_lower)
    full_ci_upper = fisher_z_inv(full_z + dev_upper)
    full_ci_std = np.std([fisher_z_inv(full_z + d) for d in full_subsample_deviations])

    # Subsample-to-m interval for TEST correlation (random test subsets)
    # This shows how test correlation varies across different random splits
    test_subsample_correlations = []
    for k in range(n_subsamples):
        rng = np.random.default_rng(k)
        idx = rng.choice(n, size=m, replace=False)
        sub_r = np.corrcoef(stated_confidence[idx], direct_entropies[idx])[0, 1]
        if not np.isnan(sub_r):
            test_subsample_correlations.append(sub_r)

    test_ci_mean = np.mean(test_subsample_correlations)
    test_ci_std = np.std(test_subsample_correlations)
    test_ci_lower = np.percentile(test_subsample_correlations, 2.5)
    test_ci_upper = np.percentile(test_subsample_correlations, 97.5)

    # P-value consistent with subsample-to-m CI
    # Compute from the subsample distribution: what fraction of shifted subsamples cross 0?
    # Under the null, the true correlation is 0, so we ask: if we shift our distribution
    # so full_r maps to 0, what fraction of subsamples would be on the opposite side?
    # This is equivalent to asking: does the CI include 0?
    if len(full_subsample_deviations) > 0:
        # The subsample correlations are: fisher_z_inv(full_z + deviation)
        # We want to know the probability that the true correlation is 0
        # Using the percentile method: p-value = 2 * min(fraction below 0, fraction above 0)
        subsample_correlations = [fisher_z_inv(full_z + d) for d in full_subsample_deviations]
        n_below_zero = sum(1 for r in subsample_correlations if r < 0)
        n_above_zero = sum(1 for r in subsample_correlations if r > 0)
        n_total = len(subsample_correlations)
        # Two-tailed p-value
        tail_fraction = min(n_below_zero, n_above_zero) / n_total
        full_pvalue = 2 * tail_fraction
        # Ensure p-value is at least 1/n_subsamples (can't be exactly 0 with finite samples)
        full_pvalue = max(full_pvalue, 1 / n_subsamples)
    else:
        full_pvalue = np.nan

    if abs(test_r) < 1 and n_test > 2:
        t_stat = test_r * np.sqrt((n_test - 2) / (1 - test_r**2))
        test_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_test-2))
    else:
        test_pvalue = np.nan

    # Partial correlation (always computed - equals Pearson when no controls)
    import pandas as pd
    from core.logres_helpers import partial_correlation_on_decision

    # Control variables for partial correlation (currently empty, ready for future use)
    control_series_list = []

    full_partial_result = partial_correlation_on_decision(
        dv_series=pd.Series(direct_entropies, name='entropy'),
        iv_series=pd.Series(stated_confidence, name='confidence'),
        control_series_list=control_series_list
    )

    result = {
        "full_correlation": float(full_r),
        "full_correlation_pvalue": float(full_pvalue),
        "test_correlation": float(test_r),
        "test_correlation_pvalue": float(test_pvalue),
        # Subsample-to-m interval for full: centered on full_r, width calibrated to test size
        "full_correlation_ci95": [float(full_ci_lower), float(full_ci_upper)],
        "full_correlation_ci_std": float(full_ci_std),
        # Subsample-to-m interval for test: distribution of correlations at test size
        "test_correlation_ci95": [float(test_ci_lower), float(test_ci_upper)],
        "test_correlation_ci_mean": float(test_ci_mean),
        "test_correlation_ci_std": float(test_ci_std),
        "n_subsamples": n_subsamples,
        # Partial correlation (equals Pearson when no controls)
        "partial_correlation": float(full_partial_result['correlation']),
        "partial_correlation_ci95": [float(full_partial_result['ci_lower']), float(full_partial_result['ci_upper'])],
        "partial_correlation_pvalue": float(full_partial_result['p_value']),
        "partial_correlation_controls": [s.name for s in control_series_list],
        "n_samples_full": n,
        "n_samples_test": n_test,
        "test_confidence_mean": float(test_confidence.mean()),
        "test_confidence_std": float(test_confidence.std()),
        "test_entropy_mean": float(test_entropy.mean()),
        "test_entropy_std": float(test_entropy.std()),
    }

    # Delegate-specific metrics
    if META_TASK == "delegate":
        # Determine delegation decisions: mc_integrated uses the response letter
        # directly (T = delegate); two_step_digit uses the alternating mapping.
        delegated = []
        self_answers = []
        if DELEGATE_PROMPT_DESIGN == "mc_integrated":
            for i, response in enumerate(meta_responses):
                is_delegated = (str(response) == "T")
                delegated.append(is_delegated)
                if not is_delegated:
                    self_answers.append(i)
        elif meta_mappings is not None:
            for i, (response, mapping) in enumerate(zip(meta_responses, meta_mappings)):
                if mapping is not None:
                    decision = mapping.get(response, "Unknown")
                    is_delegated = (decision == "Delegate")
                    delegated.append(is_delegated)
                    if not is_delegated:
                        self_answers.append(i)

        delegation_rate = sum(delegated) / len(delegated) if delegated else 0.0
        result["delegation_rate"] = float(delegation_rate)
        result["num_delegated"] = sum(delegated)
        result["num_self_answered"] = len(self_answers)
        # Record raw response distribution so print_results can show response-collapse warnings.
        result["response_distribution"] = {
            r: sum(1 for x in meta_responses if str(x) == r)
            for r in set(str(x) for x in meta_responses)
        }

        # Compute self-answer accuracy if we have the data
        if questions is not None and self_answers:
            self_correct = 0
            for idx in self_answers:
                if idx >= len(questions):
                    continue
                q = questions[idx]
                correct = q.get("correct_answer")
                if correct is None:
                    continue
                if DELEGATE_PROMPT_DESIGN == "mc_integrated":
                    # Under mc_integrated, the self-answer IS the meta response letter.
                    if idx < len(meta_responses) and str(meta_responses[idx]) == correct:
                        self_correct += 1
                elif direct_probs is not None and idx < len(direct_probs):
                    probs = direct_probs[idx]
                    if probs and "options" in q:
                        options = list(q["options"].keys())
                        if options[np.argmax(probs)] == correct:
                            self_correct += 1

            self_answer_accuracy = self_correct / len(self_answers)
            result["self_answer_accuracy"] = float(self_answer_accuracy)
            result["self_correct"] = self_correct

            # Teammate accuracy: reads from the same module-level constant used
            # when building the prompt, so behavioral stats match what the model saw.
            result["teammate_accuracy"] = float(TEAMMATE_ACCURACY)

            # Team score: self-answered correct + delegated * teammate_accuracy
            team_score = self_correct + sum(delegated) * TEAMMATE_ACCURACY
            result["team_score"] = float(team_score)
            result["team_score_normalized"] = float(team_score / len(delegated)) if delegated else 0.0

            # Overall MC accuracy = the "always-answer" baseline (grade every question).
            # Useful next to team_score_normalized and teammate_accuracy in the summary.
            if direct_probs is not None:
                overall_correct = 0
                overall_graded = 0
                for idx in range(min(len(direct_probs), len(questions))):
                    probs = direct_probs[idx]
                    q = questions[idx]
                    if probs and "correct_answer" in q and "options" in q:
                        options = list(q["options"].keys())
                        if options[np.argmax(probs)] == q["correct_answer"]:
                            overall_correct += 1
                        overall_graded += 1
                if overall_graded:
                    result["overall_accuracy"] = float(overall_correct / overall_graded)

    # Include stated_confidence for downstream analysis (e.g., calibration split)
    result["stated_confidence"] = stated_confidence.tolist()

    return result


def analyze_other_confidence_control(
    other_signals: np.ndarray,
    self_confidence: np.ndarray,
    direct_metric: np.ndarray,
    test_idx: np.ndarray
) -> Dict:
    """
    Analyze other-confidence (human difficulty estimation) as a control.

    Compares:
    1. Correlation of self-confidence vs direct metric (introspection)
    2. Correlation of other-confidence vs direct metric (control)
    3. Correlation between self and other confidence

    If the model is truly introspecting, self-confidence should correlate
    more strongly with its own uncertainty than other-confidence does.
    """
    from scipy import stats

    n = len(direct_metric)
    n_test = len(test_idx)

    # Helper: Fisher z transform
    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def fisher_z_inv(z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    # 1. Self-confidence vs direct metric (already computed in behavioral analysis,
    #    but we recompute for completeness and to use for comparison)
    self_r = np.corrcoef(self_confidence, direct_metric)[0, 1]
    if abs(self_r) < 1 and n > 2:
        t_stat = self_r * np.sqrt((n - 2) / (1 - self_r**2))
        self_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        self_pvalue = np.nan

    # 2. Other-confidence vs direct metric
    other_r = np.corrcoef(other_signals, direct_metric)[0, 1]
    if abs(other_r) < 1 and n > 2:
        t_stat = other_r * np.sqrt((n - 2) / (1 - other_r**2))
        other_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        other_pvalue = np.nan

    # 3. Self vs Other confidence correlation
    self_other_r = np.corrcoef(self_confidence, other_signals)[0, 1]
    if abs(self_other_r) < 1 and n > 2:
        t_stat = self_other_r * np.sqrt((n - 2) / (1 - self_other_r**2))
        self_other_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        self_other_pvalue = np.nan

    # 4. Compare correlations: is self-confidence significantly more correlated
    #    with direct metric than other-confidence?
    # Use Steiger's Z-test for comparing dependent correlations
    # (self vs metric) vs (other vs metric), where self and other are correlated
    r12 = self_r  # self vs metric
    r13 = other_r  # other vs metric
    r23 = self_other_r  # self vs other

    # Steiger's Z formula for comparing dependent correlations
    if abs(r12) < 1 and abs(r13) < 1 and abs(r23) < 1:
        # Average correlation
        r_avg = (r12 + r13) / 2

        # Hotelling-Williams t-test approximation
        f = (1 - r23) / (2 * (1 - r_avg**2)) if abs(r_avg) < 1 else 1
        det = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23

        if det > 0:
            t_stat = (r12 - r13) * np.sqrt((n - 3) * (1 + r23) / (2 * det))
            diff_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-3))
        else:
            diff_pvalue = np.nan
    else:
        diff_pvalue = np.nan

    # Bootstrap CI for the difference in correlations
    n_bootstrap = 1000
    diff_samples = []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(b)
        idx = rng.choice(n, size=n, replace=True)
        self_r_b = np.corrcoef(self_confidence[idx], direct_metric[idx])[0, 1]
        other_r_b = np.corrcoef(other_signals[idx], direct_metric[idx])[0, 1]
        if not np.isnan(self_r_b) and not np.isnan(other_r_b):
            diff_samples.append(self_r_b - other_r_b)

    if diff_samples:
        diff_mean = np.mean(diff_samples)
        diff_ci_lower = np.percentile(diff_samples, 2.5)
        diff_ci_upper = np.percentile(diff_samples, 97.5)
    else:
        diff_mean = self_r - other_r
        diff_ci_lower = diff_ci_upper = np.nan

    return {
        # Self-confidence analysis
        "self_vs_metric_r": float(self_r),
        "self_vs_metric_pvalue": float(self_pvalue),
        # Other-confidence analysis
        "other_vs_metric_r": float(other_r),
        "other_vs_metric_pvalue": float(other_pvalue),
        # Self vs Other correlation
        "self_vs_other_r": float(self_other_r),
        "self_vs_other_pvalue": float(self_other_pvalue),
        # Comparison
        "correlation_difference": float(self_r - other_r),
        "correlation_difference_ci95": [float(diff_ci_lower), float(diff_ci_upper)],
        "correlation_difference_pvalue": float(diff_pvalue) if not np.isnan(diff_pvalue) else None,
        # Descriptives
        "self_confidence_mean": float(self_confidence.mean()),
        "self_confidence_std": float(self_confidence.std()),
        "other_confidence_mean": float(other_signals.mean()),
        "other_confidence_std": float(other_signals.std()),
        "n_samples": n,
    }


def print_results(results: Dict, behavioral: Dict, other_confidence_analysis: Dict = None):
    """Print summary of results."""
    print("\n" + "=" * 100)
    print("INTROSPECTION EXPERIMENT RESULTS")
    print("=" * 100)

    print("\n--- Behavioral Analysis ---")
    n_full = behavioral.get('n_samples_full', '?')
    n_test = behavioral.get('n_samples_test', '?')
    n_subsamples = behavioral.get('n_subsamples', '?')
    print(f"Correlation (stated confidence vs direct entropy):")

    # Full dataset correlation with subsample-to-m CI
    full_p = behavioral.get('full_correlation_pvalue')
    p_str = f", p = {full_p:.2e}" if full_p is not None and not np.isnan(full_p) else ""
    full_ci = behavioral.get('full_correlation_ci95', [None, None])
    full_ci_std = behavioral.get('full_correlation_ci_std')
    if full_ci[0] is not None:
        print(f"  Full  (n={n_full}):  r = {behavioral['full_correlation']:.4f} ± {full_ci_std:.4f}  [95% CI: {full_ci[0]:.4f}, {full_ci[1]:.4f}]{p_str}")
    else:
        print(f"  Full  (n={n_full}):  r = {behavioral['full_correlation']:.4f}{p_str}")

    # Test set correlation with subsample CI
    test_p = behavioral.get('test_correlation_pvalue')
    p_str = f", p = {test_p:.2e}" if test_p is not None and not np.isnan(test_p) else ""
    test_ci = behavioral.get('test_correlation_ci95', [None, None])
    test_ci_std = behavioral.get('test_correlation_ci_std')
    if test_ci[0] is not None:
        print(f"  Test  (n={n_test}):  r = {behavioral['test_correlation']:.4f} ± {test_ci_std:.4f}  [95% CI: {test_ci[0]:.4f}, {test_ci[1]:.4f}]{p_str}")
    else:
        print(f"  Test  (n={n_test}):  r = {behavioral['test_correlation']:.4f}{p_str}")

    print(f"  (CIs from {n_subsamples} subsamples to test size, centered on point estimate)")

    # Partial correlation
    partial_r = behavioral.get('partial_correlation')
    partial_ci = behavioral.get('partial_correlation_ci95', [None, None])
    partial_p = behavioral.get('partial_correlation_pvalue')
    controls = behavioral.get('partial_correlation_controls', [])
    if partial_r is not None:
        p_str = f", p = {partial_p:.2e}" if partial_p is not None and not np.isnan(partial_p) else ""
        ctrl_str = f" (controlling for {', '.join(controls)})" if controls else ""
        print(f"  Partial{ctrl_str}: r = {partial_r:.4f}  [95% CI: {partial_ci[0]:.4f}, {partial_ci[1]:.4f}]{p_str}")

    print(f"  (Negative correlation suggests introspection; positive suggests miscalibration)")

    # Other-confidence control analysis (only for confidence task)
    if other_confidence_analysis is not None:
        print("\n--- Other-Confidence Control (Human Difficulty Estimation) ---")
        self_r = other_confidence_analysis['self_vs_metric_r']
        other_r = other_confidence_analysis['other_vs_metric_r']
        diff = other_confidence_analysis['correlation_difference']
        diff_ci = other_confidence_analysis['correlation_difference_ci95']
        diff_p = other_confidence_analysis.get('correlation_difference_pvalue')

        self_p = other_confidence_analysis['self_vs_metric_pvalue']
        other_p = other_confidence_analysis['other_vs_metric_pvalue']

        self_p_str = f", p = {self_p:.2e}" if self_p is not None and not np.isnan(self_p) else ""
        other_p_str = f", p = {other_p:.2e}" if other_p is not None and not np.isnan(other_p) else ""

        print(f"  Self-confidence vs {METRIC}:    r = {self_r:.4f}{self_p_str}")
        print(f"  Other-confidence vs {METRIC}:   r = {other_r:.4f}{other_p_str}")
        print(f"  Self vs Other confidence:       r = {other_confidence_analysis['self_vs_other_r']:.4f}")
        print(f"")
        print(f"  Difference (self - other):      Δr = {diff:.4f}  [95% CI: {diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")

        if diff_p is not None:
            print(f"  Steiger's test p-value:         p = {diff_p:.4e}")
            if diff_p < 0.05 and diff < 0:
                print(f"  → Self-confidence significantly MORE correlated with {METRIC} than other-confidence")
                print(f"    This suggests the model is introspecting on its own uncertainty,")
                print(f"    not just assessing question difficulty.")
            elif diff_p < 0.05 and diff > 0:
                print(f"  → Self-confidence significantly LESS correlated with {METRIC} than other-confidence")
                print(f"    This is unexpected - the model may be using question difficulty as a proxy.")
            else:
                print(f"  → No significant difference between self and other confidence correlations")
        else:
            print(f"  → Could not compute significance test")

    # Delegate-specific summary statistics
    if META_TASK == "delegate" and "delegation_rate" in behavioral:
        print("\n--- Delegate Task Summary ---")
        print(f"  Delegation rate:      {behavioral['delegation_rate']:.1%} ({behavioral['num_delegated']} delegated, {behavioral['num_self_answered']} self-answered)")
        if "self_answer_accuracy" in behavioral:
            print(f"  Self-answer accuracy: {behavioral['self_answer_accuracy']:.1%} ({behavioral['self_correct']}/{behavioral['num_self_answered']} correct)")
            print(f"  Teammate accuracy:    {behavioral['teammate_accuracy']:.1%} (configured)")
            print(f"  Team score:           {behavioral['team_score']:.1f} / {behavioral['num_delegated'] + behavioral['num_self_answered']} ({behavioral['team_score_normalized']:.1%})")
            # "Always delegate" baseline = teammate_accuracy, directly comparable to team_score_normalized.
            always_delegate = float(behavioral['teammate_accuracy'])
            if 'overall_accuracy' in behavioral:
                always_answer = float(behavioral['overall_accuracy'])
                print(f"  Baselines:            always-answer = {always_answer:.1%}, always-delegate = {always_delegate:.1%}")
            else:
                print(f"  Baselines:            always-delegate = {always_delegate:.1%}  (always-answer baseline: see notebook)")

        # Response-collapse diagnostic: if the model almost always picks the same
        # letter/digit, the delegate rate is a prompt-format artifact.
        dist = behavioral.get('response_distribution') or {}
        if dist:
            total = sum(dist.values()) or 1
            items = sorted(dist.items(), key=lambda kv: -kv[1])
            pretty = ", ".join(f"'{k}'={v}" for k, v in items)
            top_key, top_cnt = items[0]
            skew = top_cnt / total
            print(f"  Meta-response distribution: {pretty}  (top '{top_key}' = {skew:.1%})")
            if skew > 0.9:
                print(
                    "  \u26a0  WARNING: model almost always picks one option \u2014 the observed "
                    "delegation rate is a prompt-format artifact, not a metacognitive signal. "
                    "Raising TEAMMATE_ACCURACY or changing the prompt framing may help."
                )

    print("\n--- Probe Analysis by Layer ---")
    print(f"{'Layer':<8} {'Direct→Direct':<15} {'D→M (fixed)':<15} {'D→M (orig)':<15} {'Meta→Meta':<15} {'Shuffled':<12}")
    print(f"{'':8} {'R²':<15} {'R²':<15} {'R²':<15} {'R²':<15} {'R²':<12}")
    print("-" * 110)

    for layer_idx in sorted(results.keys()):
        res = results[layer_idx]
        d2d = res["direct_to_direct"]["test_r2"]
        d2m_fixed = res["direct_to_meta_fixed"]["r2"]
        d2m_orig = res["direct_to_meta"]["r2"]
        m2m = res["meta_to_meta"]["test_r2"]
        shuf = res["shuffled_baseline"]["r2"]
        print(f"{layer_idx:<8} {d2d:<15.4f} {d2m_fixed:<15.4f} {d2m_orig:<15.4f} {m2m:<15.4f} {shuf:<12.4f}")

    print("=" * 110)

    # Summary statistics
    layers = sorted(results.keys())

    best_d2d_layer = max(layers, key=lambda l: results[l]["direct_to_direct"]["test_r2"])
    best_d2d = results[best_d2d_layer]["direct_to_direct"]["test_r2"]

    best_d2m_fixed_layer = max(layers, key=lambda l: results[l]["direct_to_meta_fixed"]["r2"])
    best_d2m_fixed = results[best_d2m_fixed_layer]["direct_to_meta_fixed"]["r2"]

    best_m2m_layer = max(layers, key=lambda l: results[l]["meta_to_meta"]["test_r2"])
    best_m2m = results[best_m2m_layer]["meta_to_meta"]["test_r2"]

    print(f"\nBest Direct→Direct:      Layer {best_d2d_layer} (R² = {best_d2d:.4f})")
    print(f"Best Direct→Meta (fixed): Layer {best_d2m_fixed_layer} (R² = {best_d2m_fixed:.4f})")
    print(f"Best Meta→Meta:          Layer {best_m2m_layer} (R² = {best_m2m:.4f})")

    # Transfer ratio using fixed D→M
    if best_d2d > 0:
        transfer_ratio = best_d2m_fixed / best_d2d
        print(f"\nTransfer ratio (best D→M fixed / best D→D): {transfer_ratio:.2%}")
        if transfer_ratio > 0.5:
            print("  → Strong evidence for introspection!")
        elif transfer_ratio > 0.25:
            print("  → Moderate evidence for introspection")
        elif transfer_ratio > 0:
            print("  → Weak evidence for introspection")
        else:
            print("  → No evidence for introspection (negative transfer)")

def plot_results(
    results: Dict,
    behavioral: Dict,
    direct_entropies: np.ndarray,
    test_idx: np.ndarray,
    output_path: str = "introspection_results.png",
    mc_answer_results: Optional[Dict] = None
):
    """
    Create visualization of results with 4 panels.
    
    Panel 1: Separate Scaler (Upper Bound).
    Panel 2: Centered Scaler (Rigorous).
    Panel 3: Pearson Correlation.
    Panel 4: NORMALIZED Timecourse (Min-Max). Compares signal emergence timing.
    """
    layers = sorted(results.keys())

    # --- Data Extraction ---
    # 1. Entropy Metrics
    ent_d2d_r2 = [results[l]["direct_to_direct"]["test_r2"] for l in layers]
    ent_d2m_separate_r2 = [results[l]["direct_to_meta_fixed"]["r2"] for l in layers] 
    ent_d2m_centered_r2 = [results[l]["direct_to_meta_centered"]["r2"] for l in layers] 
    ent_shuffled_r2 = [results[l]["shuffled_baseline"]["r2"] for l in layers]
    ent_pearson = [results[l]["direct_to_meta"]["pearson"] for l in layers]
    ent_d2d_pearson = [results[l]["direct_to_direct"]["test_pearson"] for l in layers]
    
    # 2. MC Answer Metrics
    has_mc = mc_answer_results is not None
    if has_mc:
        mc_d2d_acc = [mc_answer_results[l]["d2d_accuracy"] for l in layers]
        # Robust check for new keys
        first_layer = mc_answer_results[layers[0]]
        if "d2m_centered_accuracy" in first_layer:
            mc_d2m_sep_acc = [mc_answer_results[l]["d2m_separate_accuracy"] for l in layers]
            mc_d2m_cen_acc = [mc_answer_results[l]["d2m_centered_accuracy"] for l in layers]
        else:
            mc_d2m_sep_acc = [mc_answer_results[l]["d2m_accuracy"] for l in layers]
            mc_d2m_cen_acc = mc_d2m_sep_acc
        
        mc_chance = 0.25
    
    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle(f'Introspection Analysis: {get_model_display_label()} on {DATASET_NAME} ({META_TASK})', fontsize=16)

    # Helper: Min-Max Normalization
    def normalize(data):
        arr = np.array(data)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    # Helper for common entropy baselines
    def plot_ent_baselines(ax, metric_d2d, metric_shuffled=None):
        ax.plot(layers, metric_d2d, 'o-', label='Entropy D→D', color='tab:blue', linewidth=2)
        if metric_shuffled is not None:
            ax.plot(layers, metric_shuffled, ':', label='Entropy Shuffled', color='gray', alpha=0.6)

    # Helper for MC lines
    def plot_mc_lines(ax, d2d, d2m, label_suffix=""):
        if has_mc:
            ax.plot(layers, d2d, 'd-', label='MC Ans D→D', color='tab:green', linewidth=2)
            ax.plot(layers, d2m, 'd-', label=f'MC Ans D→M {label_suffix}', color='tab:red', linewidth=2)
            ax.axhline(y=mc_chance, color='tab:green', linestyle=':', alpha=0.4, label='Chance')

    # =========================================================================
    # Panel 1: Method 1 - Separate Scaler (Upper Bound)
    # =========================================================================
    ax1 = axes[0, 0]
    plot_ent_baselines(ax1, ent_d2d_r2, ent_shuffled_r2)
    ax1.plot(layers, ent_d2m_separate_r2, 's-', label='Entropy D→M Sep', color='tab:orange', linewidth=2)
    if has_mc:
        plot_mc_lines(ax1, mc_d2d_acc, mc_d2m_sep_acc, "Sep ")

    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('$R^2$ / Accuracy')
    ax1.set_title('Method 1: Separate Scaler (Upper Bound)\n(Absolute Performance)')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    bottom_y = min(min(ent_shuffled_r2), min(ent_d2m_separate_r2), -0.1)
    ax1.set_ylim(bottom=max(-2.0, bottom_y - 0.1), top=1.05)


    # =========================================================================
    # Panel 2: Method 2 - Centered Scaler (Rigorous)
    # =========================================================================
    ax2 = axes[0, 1]
    plot_ent_baselines(ax2, ent_d2d_r2, ent_shuffled_r2)
    ax2.plot(layers, ent_d2m_centered_r2, 's-', label='Entropy D→M Cen', color='tab:orange', linewidth=2)
    if has_mc:
        plot_mc_lines(ax2, mc_d2d_acc, mc_d2m_cen_acc, "Cen ")

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('$R^2$ / Accuracy')
    ax2.set_title('Method 2: Centered Scaler (Rigorous)\n(Geometry Check)')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=max(-2.0, bottom_y - 0.1), top=1.05)


    # =========================================================================
    # Panel 3: Method 3 - Pearson Correlation
    # =========================================================================
    ax3 = axes[1, 0]
    ax3.plot(layers, ent_d2d_pearson, 'o-', label='Entropy D→D Pearson', color='tab:blue', linewidth=2)
    ax3.plot(layers, ent_pearson, 's-', label='Entropy D→M Pearson', color='tab:orange', linewidth=2)

    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Correlation (r)')
    ax3.set_title('Method 3: Pearson Correlation\n(Shift Invariant Signal Check)')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.05)


    # =========================================================================
    # Panel 4: Normalized Timecourse (Min-Max Scaled)
    # This visualizes the "Pointer vs Direct Access" timing
    # =========================================================================
    ax4 = axes[1, 1]
    
    # 1. Entropy D->D (Baseline Logic)
    ax4.plot(layers, normalize(ent_d2d_r2), 'o-', label='Entropy D→D', color='tab:blue', alpha=0.4)
    
    # 2. Entropy D->M Separate (Introspection Signal)
    ax4.plot(layers, normalize(ent_d2m_separate_r2), 's-', label='Entropy D→M', color='tab:orange', linewidth=2.5)
    
    if has_mc:
        # 3. MC Answer D->D (Baseline Fact)
        ax4.plot(layers, normalize(mc_d2d_acc), 'd-', label='MC Ans D→D', color='tab:green', alpha=0.4)
        
        # 4. MC Answer D->M (Transferred Fact)
        ax4.plot(layers, normalize(mc_d2m_sep_acc), 'd-', label='MC Ans D→M', color='tab:red', linewidth=2.5)

    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Normalized Score (0-1)')
    ax4.set_title('Signal Emergence (Min-Max Scaled)\nCheck: Do Red/Orange lines rise together?')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
                    


# ============================================================================
# MAIN
# ============================================================================

def run_single_experiment(
    dataset_name: str,
    meta_task: str,
    model,
    tokenizer,
    num_layers: int,
    metric: str,
    batch_size: int,
    collect_other_activations: bool = False
):
    """Run a single introspection experiment for one dataset/task combination."""
    global DATASET_NAME, META_TASK, NUM_QUESTIONS, METRIC

    # Update global variables for this run
    DATASET_NAME = dataset_name
    META_TASK = meta_task
    METRIC = metric
    NUM_QUESTIONS = NUM_QUESTIONS_BY_DATASET.get(
        dataset_name,
        NUM_QUESTIONS_BY_DATASET.get(_dataset_short_name(dataset_name), NUM_QUESTIONS_DEFAULT),
    )

    # One run_id per (model, dataset, meta_task) invocation. Stamped into the
    # paired_data.json and both NPZ files at save time so a future loader can
    # detect cases where the JSON and NPZ came from different forward passes.
    run_id = uuid.uuid4().hex

    print("\n" + "=" * 80)
    print(f"Running: {dataset_name} / {meta_task} / {metric}")
    print(f"run_id: {run_id}")
    print("=" * 80)

    # Print delegate parameters if using delegate task
    if meta_task == "delegate":
        print("\n--- Delegate Task Parameters ---")
        print(f"  prompt_design: {DELEGATE_PROMPT_DESIGN}")
        if DELEGATE_PROMPT_DESIGN == "mc_integrated":
            print("  Options: A/B/C/D (answer) or T (delegate) — single-shot")
        else:
            print("  (Matching delegate_game_from_capabilities.py)")
            print("  decision_only: True")
            print("  alternate_decision_mapping: True")
            print("  Options: 1/2 (alternating mapping per trial)")
        print("  use_phase1_summary: True")
        print("  use_phase1_history: False")
        print("  use_examples: True")
        print(f"  teammate_accuracy: {TEAMMATE_ACCURACY:.0%}")
        print("")

    # Load questions
    print(f"\nLoading {NUM_QUESTIONS} questions from {dataset_name}...")
    questions = load_questions(dataset_name, NUM_QUESTIONS)
    # Re-seed immediately before shuffle to match capabilities_test.py exactly
    random.seed(SEED)
    random.shuffle(questions)
    print(f"Loaded {len(questions)} questions")

    # Determine whether to use chat template
    is_base = is_base_model(BASE_MODEL_NAME)
    use_chat_template = has_chat_template(tokenizer) and not is_base
    print(f"Using chat template: {use_chat_template}")
    if is_base:
        print(f"Base model detected — using few-shot mode: {FEW_SHOT_MODE}")
        if meta_task == "delegate":
            print(f"Base delegate few-shot mode: {BASE_DELEGATE_MODE}")

    # Optionally load a pool of examples for balanced base-model delegate few-shot.
    few_shot_pool = None
    if is_base and meta_task == "delegate" and BASE_DELEGATE_MODE == "balanced" and BASE_DELEGATE_POOL_SOURCE:
        print(f"Loading delegate few-shot pool from: {BASE_DELEGATE_POOL_SOURCE}")
        with open(BASE_DELEGATE_POOL_SOURCE, "r") as f:
            pool_src = json.load(f)
        few_shot_pool = []
        for item in pool_src.get("data", pool_src.get("questions", [])):
            few_shot_pool.append({
                "question": item.get("question"),
                "options": item.get("options"),
                "mc_answer": item.get("predicted_answer", item.get("mc_answer")),
                "confidence": item.get("stated_confidence_response", item.get("confidence")),
            })
        print(f"  Loaded {len(few_shot_pool)} pool items")

    # Reuse direct activations from a previous confidence-mode run if requested —
    # only meaningful for META_TASK == 'delegate' (direct prompt is identical).
    reuse_direct = REUSE_DIRECT_FROM_CONFIDENCE and meta_task == "delegate"
    if reuse_direct:
        # Build the confidence-mode prefix by temporarily swapping META_TASK
        saved_meta = META_TASK
        globals()["META_TASK"] = "confidence"
        confidence_prefix = get_output_prefix()
        globals()["META_TASK"] = saved_meta
        paired_path = Path(f"{confidence_prefix}_paired_data.json")
        acts_path = Path(f"{confidence_prefix}_direct_activations.npz")
        if not paired_path.exists() or not acts_path.exists():
            raise FileNotFoundError(
                "REUSE_DIRECT_FROM_CONFIDENCE=True but confidence-mode files are missing:\n"
                f"  {paired_path}\n  {acts_path}\n"
                "Run META_TASK='confidence' for this (model, dataset) first, or set the flag to False."
            )
        print(f"\nReusing direct activations from:\n  {paired_path}\n  {acts_path}")
        mc_data, existing_paired = _load_mc_data_for_reuse(paired_path, acts_path)
        # Sanity-check that cached questions match the current ordering
        existing_ids = [q.get("id") for q in existing_paired.get("questions", [])]
        new_ids = [q.get("id") for q in questions]
        if existing_ids and new_ids and existing_ids != new_ids:
            raise ValueError(
                "Question order mismatch between cached confidence-mode data and the fresh "
                "load. Reuse requires identical (model, dataset, seed, NUM_QUESTIONS)."
            )
        data = collect_meta_only(
            questions, model, tokenizer, num_layers, use_chat_template,
            mc_data, batch_size=batch_size, is_base=is_base,
            few_shot_mode=FEW_SHOT_MODE, few_shot_pool=few_shot_pool,
        )
    else:
        # Always collect fresh paired data (direct and meta for each question)
        # to ensure consistent quantization settings between direct and meta activations
        data = collect_paired_data(
            questions, model, tokenizer, num_layers, use_chat_template,
            batch_size=batch_size, is_base=is_base, few_shot_mode=FEW_SHOT_MODE,
            few_shot_pool=few_shot_pool,
        )

    # Generate output prefixes
    # Base prefix for shared files (activations, paired data)
    base_prefix = get_output_prefix()
    # Metric-specific prefix for probe results (task-dependent)
    metric_prefix = get_output_prefix(metric)
    # Directions prefix (task-independent - directions are the same for confidence/delegate)
    directions_prefix = get_directions_prefix(metric)
    print(f"Base output prefix: {base_prefix}")
    print(f"Metric output prefix: {metric_prefix}")
    print(f"Directions prefix: {directions_prefix}")

    # Get the selected metric's values
    direct_target = data["direct_metrics"][METRIC]

    # Save activations as float16 with ALL scalar metrics.
    # Each NPZ also carries `run_id` and `question_ids` so downstream loaders
    # can detect when the NPZ and paired_data.json came from different runs
    # (which would silently corrupt row-aligned probes).
    print("\nSaving activations (float16)...")
    qid_array = _question_ids_array(data["questions"])
    run_id_array = np.array(run_id, dtype=object)

    if not reuse_direct:
        _atomic_savez_compressed(
            f"{base_prefix}_direct_activations.npz",
            **{f"layer_{i}": acts.astype(np.float16) for i, acts in data["direct_activations"].items()},
            **{k: v for k, v in data["direct_metrics"].items() if isinstance(v, np.ndarray)},
            run_id=run_id_array,
            question_ids=qid_array,
        )
    else:
        print(f"  (skipping {base_prefix}_direct_activations.npz — reused from confidence-mode run)")
    _atomic_savez_compressed(
        f"{base_prefix}_meta_activations.npz",
        **{f"layer_{i}": acts.astype(np.float16) for i, acts in data["meta_activations"].items()},
        entropy=data["meta_entropies"],  # Meta always uses entropy
        run_id=run_id_array,
        question_ids=qid_array,
    )
    print(f"Saved activations to {base_prefix}_*_activations.npz")

    # ------------------------------------------------------------------
    # Logit-lens projection on the saved last-token activations.
    # ------------------------------------------------------------------
    # We already have the residual at every layer for the final token of every
    # prompt. Projecting them through the unembedding matrix is one matmul per
    # layer (~17 GFLOPs/question for Llama-8B) — orders of magnitude cheaper
    # than the forward pass that produced them. Storing the full softmax
    # distribution would be ~16 MB/question, so we only persist the option-
    # token logits (per-layer answer-trajectory) and the top-K tokens for
    # diagnostics. The NPZ is row-aligned with `question_ids`.
    from _logit_lens import apply_logit_lens

    first_q = questions[0]
    if is_base:
        _, direct_option_strs = format_direct_prompt_base(first_q, mode=FEW_SHOT_MODE)
    else:
        _, direct_option_strs = format_direct_prompt(first_q, tokenizer, use_chat_template)
    direct_option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in direct_option_strs
    ]

    meta_option_strs = list(get_meta_options())
    meta_option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in meta_option_strs
    ]

    if not reuse_direct:
        print("\nComputing logit lens (direct)...")
        direct_lens = apply_logit_lens(
            data["direct_activations"], model, direct_option_token_ids
        )
        _atomic_savez_compressed(
            f"{base_prefix}_direct_logit_lens.npz",
            option_logits=direct_lens["option_logits"],
            top_k_ids=direct_lens["top_k_ids"],
            top_k_logits=direct_lens["top_k_logits"],
            layer_indices=direct_lens["layer_indices"],
            option_strs=np.array(direct_option_strs, dtype=object),
            option_token_ids=np.array(direct_option_token_ids, dtype=np.int64),
            run_id=run_id_array,
            question_ids=qid_array,
        )

    print("Computing logit lens (meta)...")
    meta_lens = apply_logit_lens(
        data["meta_activations"], model, meta_option_token_ids
    )
    _atomic_savez_compressed(
        f"{base_prefix}_meta_logit_lens.npz",
        option_logits=meta_lens["option_logits"],
        top_k_ids=meta_lens["top_k_ids"],
        top_k_logits=meta_lens["top_k_logits"],
        layer_indices=meta_lens["layer_indices"],
        option_strs=np.array(meta_option_strs, dtype=object),
        option_token_ids=np.array(meta_option_token_ids, dtype=np.int64),
        run_id=run_id_array,
        question_ids=qid_array,
    )
    print(f"Saved logit lens to {base_prefix}_*_logit_lens.npz")

    # Quick-look PNG: MC dist + confidence dist + entropy-vs-confidence scatter.
    # Runs BEFORE the slow probe/introspection analysis so you can sanity-check
    # distributions immediately.
    try:
        save_quick_summary_png(data, questions, f"{base_prefix}_quick_summary.png")
    except Exception as e:
        print(f"⚠ quick-summary PNG failed: {e}")

    # Plain-text dump of the first 10 exact prompts + model responses, for
    # eyeballing that prompts render correctly and the model's argmax/soft
    # outputs make sense.
    try:
        save_example_prompts_and_responses_txt(
            data, questions, tokenizer,
            is_base=is_base, use_chat_template=use_chat_template,
            few_shot_mode=FEW_SHOT_MODE,
            output_path=f"{base_prefix}_examples.txt",
            n_examples=10,
            delegate_pool=few_shot_pool,
        )
    except Exception as e:
        print(f"⚠ examples.txt dump failed: {e}")

    # Generate example prompts for verification (first 2 questions).
    # Wrapped: when reusing direct data and FEW_SHOT_MODE is a pool-requiring mode
    # (e.g. "balanced") without a loaded MC pool, direct-prompt rendering fails.
    # We fall back to "fixed" mode here — this block is purely diagnostic and does
    # not affect the saved activations.
    def _render_direct_prompt(_q):
        if not is_base:
            return format_direct_prompt(_q, tokenizer, use_chat_template)
        try:
            return format_direct_prompt_base(_q, mode=FEW_SHOT_MODE)
        except Exception as e:
            print(f"  (example-prompt fallback: direct MC '{FEW_SHOT_MODE}' mode failed ({e}); using 'fixed')")
            return format_direct_prompt_base(_q, mode="fixed")

    example_prompts = []
    for i in range(min(2, len(questions))):
        q = questions[i]
        if is_base:
            direct_prompt, direct_options = _render_direct_prompt(q)
        else:
            direct_prompt, direct_options = format_direct_prompt(q, tokenizer, use_chat_template)
        if META_TASK == "delegate":
            meta_prompt, meta_options_list, mapping = format_delegate_prompt(
                q, tokenizer, use_chat_template, trial_index=i,
                is_base=is_base,
                few_shot_mode=BASE_DELEGATE_MODE if is_base else "fixed",
                few_shot_pool=few_shot_pool,
            )
        else:
            if is_base:
                meta_prompt, meta_options_list = format_meta_prompt_base(q, mode=FEW_SHOT_MODE)
            else:
                meta_prompt, meta_options_list = format_meta_prompt(q, tokenizer, use_chat_template)
            mapping = None
        example_prompts.append({
            "question_index": i,
            "question_text": q.get("question", ""),
            "direct_prompt": direct_prompt,
            "direct_options": direct_options,
            "meta_prompt": meta_prompt,
            "meta_options": meta_options_list,
            "meta_mapping": mapping,
        })

    # Collect other-confidence control data (only for confidence task)
    # Do this before saving paired_data so we can include it
    other_data = None
    if META_TASK == "confidence":
        print("\n" + "=" * 60)
        print("Running OTHER-CONFIDENCE control task...")
        print("(Asks model to estimate what % of college-educated people would know the answer)")
        print("=" * 60)

        other_data = collect_other_confidence(
            questions, model, tokenizer, num_layers, use_chat_template,
            batch_size=batch_size, collect_activations=collect_other_activations
        )

        # Add example prompts for other-confidence
        for i in range(min(2, len(questions))):
            q = questions[i]
            other_prompt, _ = format_other_confidence_prompt(q, tokenizer, use_chat_template)
            if i < len(example_prompts):
                example_prompts[i]["other_confidence_prompt"] = other_prompt

    # Compute derived per-example fields
    options_keys = [list(q["options"].keys()) for q in questions]
    direct_responses = [
        opts[np.argmax(p)] if p else None
        for opts, p in zip(options_keys, data["direct_probs"])
    ]
    is_correct = [
        resp == q.get("correct_answer") if resp else False
        for resp, q in zip(direct_responses, questions)
    ]
    # stated_confidence_numeric — format depends on meta task + confidence scale
    meta_mappings = data.get("meta_mappings") or [None] * len(data["meta_probs"])
    meta_responses = data["meta_responses"]
    _task_type = _meta_task_type()
    stated_confidence_numeric = [
        response_to_confidence(resp, np.array(p), mapping, _task_type)
        for resp, p, mapping in zip(meta_responses, data["meta_probs"], meta_mappings)
    ]

    accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
    print(f"\nDirect accuracy: {accuracy:.1%} ({sum(is_correct)}/{len(is_correct)})")

    # In-game correctness — only meaningful under mc_integrated delegate design,
    # where the meta pass itself produces the MC letter (or T for delegate).
    game_is_correct = None
    if META_TASK == "delegate" and DELEGATE_PROMPT_DESIGN == "mc_integrated":
        game_is_correct = [
            bool((resp != "T") and (resp == q.get("correct_answer")))
            for resp, q in zip(meta_responses, data["questions"])
        ]
        n_answered = sum(1 for r in meta_responses if r != "T")
        n_correct = sum(game_is_correct)
        game_acc = (n_correct / n_answered) if n_answered else 0.0
        print(
            f"In-game accuracy (self-answered only): {game_acc:.1%} "
            f"({n_correct}/{n_answered}); delegate rate: "
            f"{(1 - n_answered / len(meta_responses)):.1%}"
        )

    # Run-level metadata
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    # Save paired data (for reproducibility and further analysis)
    paired_data = {
        # Existing fields
        "direct_metrics": {k: v.tolist() for k, v in data["direct_metrics"].items()},
        "direct_probs": data["direct_probs"],
        "direct_logits": data.get("direct_logits", []),
        "meta_entropies": data["meta_entropies"].tolist(),
        "meta_probs": data["meta_probs"],
        "meta_responses": data["meta_responses"],
        "meta_mappings": data.get("meta_mappings"),
        # New per-example fields
        "direct_responses": direct_responses,
        "is_correct": is_correct,
        "game_is_correct": game_is_correct,
        "stated_confidence_numeric": stated_confidence_numeric,
        # Extended metrics (full-vocab)
        "direct_extended": data.get("direct_extended", {}),
        "meta_extended": data.get("meta_extended", {}),
        # Questions
        "questions": [
            {
                "id": q.get("id", f"q_{i}"),
                "question": q.get("question", ""),
                "correct_answer": q.get("correct_answer", ""),
                "options": q.get("options", {})
            }
            for i, q in enumerate(data["questions"])
        ],
        "example_prompts": example_prompts,
        "config": {
            "run_id": run_id,
            # If this run reused a prior confidence run's direct activations, record
            # which run those came from so the link is auditable downstream.
            "reused_direct_from_run_id": (
                locals().get("mc_data", {}).get("_source_run_id") if reuse_direct else None
            ),
            "model_name": MODEL_NAME,
            "base_model_name": BASE_MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "num_questions": NUM_QUESTIONS,
            "seed": SEED,
            "meta_task": META_TASK,
            "metric": METRIC,
            "is_base_model": is_base,
            "few_shot_mode": FEW_SHOT_MODE if is_base else None,
            "confidence_scale": CONFIDENCE_SCALE,
            "date_run": datetime.datetime.now().isoformat(),
            "git_commit": git_commit,
            "device": str(DEVICE),
            "activation_dtype": "float16",
            "tokenizer_name": tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else BASE_MODEL_NAME,
            # Delegate task parameters (matches delegate_game_from_capabilities.py)
            "delegate_params": {
                "prompt_design": DELEGATE_PROMPT_DESIGN,
                "meta_option_keys": list(get_meta_options()),
                "decision_only": DELEGATE_PROMPT_DESIGN == "two_step_digit",
                "alternate_decision_mapping": DELEGATE_PROMPT_DESIGN == "two_step_digit",
                "use_phase1_summary": True,
                "use_phase1_history": False,
                "use_examples": True,
                "teammate_accuracy": float(TEAMMATE_ACCURACY),
            } if META_TASK == "delegate" else None,
        }
    }

    # Add other-confidence data if collected
    if other_data is not None:
        paired_data["other_confidence"] = {
            "probs": other_data["other_probs"],
            "responses": other_data["other_responses"],
            "signals": other_data["other_signals"].tolist(),
        }

    _atomic_write_json(f"{base_prefix}_paired_data.json", paired_data, indent=2)
    print(f"Saved paired data to {base_prefix}_paired_data.json (run_id={run_id})")

    # Save errors if any
    collection_errors = data.get("errors", [])
    if collection_errors:
        errors_path = f"{base_prefix}_errors.json"
        with open(errors_path, "w") as f:
            json.dump(collection_errors, f, indent=2)
        print(f"Warning: {len(collection_errors)} errors logged to {errors_path}")

    # Run introspection analysis with selected metric
    print(f"\nRunning introspection analysis with metric: {METRIC}")
    results, test_idx, directions, entropy_probe_components = run_introspection_analysis(
        data["direct_activations"],
        data["meta_activations"],
        direct_target,  # Use selected metric
        extract_directions=True
    )

    # Save directions for steering/ablation experiments (task-independent filename)
    # Directions are task-independent because they predict metrics from direct task activations
    if directions is not None:
        directions_data = {
            f"layer_{layer_idx}": direction
            for layer_idx, direction in directions.items()
        }
        directions_data["_metadata_metric"] = np.array(METRIC)
        directions_data["_metadata_dataset"] = np.array(DATASET_NAME)
        directions_data["_metadata_model"] = np.array(BASE_MODEL_NAME)
        np.savez_compressed(
            f"{directions_prefix}_directions.npz",
            **directions_data
        )
        print(f"Saved {METRIC} directions to {directions_prefix}_directions.npz")

    # Compute model's predicted MC answer (argmax over A/B/C/D probabilities)
    # This is the answer the model would give on the direct task
    direct_probs_array = np.array([p if p else [0.25, 0.25, 0.25, 0.25] for p in data["direct_probs"]])
    model_predicted_answer = np.argmax(direct_probs_array, axis=1)  # 0=A, 1=B, 2=C, 3=D
    print(f"\nModel predicted answers: {len(model_predicted_answer)} questions")
    print(f"  Answer distribution: A={np.sum(model_predicted_answer==0)}, B={np.sum(model_predicted_answer==1)}, "
          f"C={np.sum(model_predicted_answer==2)}, D={np.sum(model_predicted_answer==3)}")

    # Get train/test indices from introspection analysis (need to recompute since run_introspection_analysis
    # returns test_idx but not train_idx)
    n_questions = len(direct_target)
    indices = np.arange(n_questions)
    train_idx, _ = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)

    # Run MC answer probe analysis
    mc_answer_results, mc_probe_components, mc_directions = run_mc_answer_analysis(
        data["direct_activations"],
        data["meta_activations"],
        model_predicted_answer,
        train_idx,
        test_idx
    )

    # Print MC answer probe summary
    best_mc_d2d_layer = max(mc_answer_results.keys(), key=lambda l: mc_answer_results[l]["d2d_accuracy"])
    best_mc_d2m_layer = max(mc_answer_results.keys(), key=lambda l: mc_answer_results[l]["d2m_accuracy"])
    print(f"\nMC Answer Probe Results:")
    print(f"  Best D→D: Layer {best_mc_d2d_layer} (acc={mc_answer_results[best_mc_d2d_layer]['d2d_accuracy']:.3f})")
    print(f"  Best D→M: Layer {best_mc_d2m_layer} (acc={mc_answer_results[best_mc_d2m_layer]['d2m_accuracy']:.3f})")
    print(f"  Chance: 0.250 (4-class)")

    # Save MC answer directions for ablation experiments
    # Use metric-independent prefix since MC answer directions don't depend on which metric we're analyzing
    mc_directions_prefix = get_directions_prefix(metric=None)
    mc_directions_data = {
        f"layer_{layer_idx}": direction
        for layer_idx, direction in mc_directions.items()
    }
    mc_directions_data["_metadata_metric"] = np.array("mc_answer")
    mc_directions_data["_metadata_dataset"] = np.array(DATASET_NAME)
    mc_directions_data["_metadata_model"] = np.array(BASE_MODEL_NAME)
    mc_directions_path = f"{mc_directions_prefix}_mc_answer_directions.npz"
    np.savez_compressed(mc_directions_path, **mc_directions_data)
    print(f"Saved MC answer directions to {mc_directions_path}")

    # ------------------------------------------------------------------
    # Contrast (mean-difference) directions per layer.
    #
    # Two signals, two activation sources:
    #   entropy            → direct activations, top-vs-bottom quantile of
    #                        direct-task entropy. Always extracted.
    #   stated_confidence  → meta activations, top-vs-bottom quantile of soft
    #                        stated confidence. Only confidence-task runs
    #                        produce a stated-confidence signal, so this one
    #                        is skipped for delegate runs.
    # ------------------------------------------------------------------
    if EXTRACT_CONTRAST_DIRECTIONS:
        print("\nComputing contrast (mean-difference) directions...")
        cap_str = (
            "all available"
            if CONTRAST_SAMPLES_PER_BIN is None
            else f"capped at {CONTRAST_SAMPLES_PER_BIN}"
        )

        # Entropy contrast — DIRECT activations, signal = direct-task entropy.
        # Top CONTRAST_PERCENT_ENTROPY% (highest entropy / most uncertain) vs
        # bottom CONTRAST_PERCENT_ENTROPY% (lowest entropy / most confident).
        entropy_signal = np.asarray(data["direct_metrics"]["entropy"], dtype=np.float64)
        entropy_dirs, entropy_meta = compute_contrast_directions(
            data["direct_activations"],
            entropy_signal,
            percent=CONTRAST_PERCENT_ENTROPY,
            samples_per_bin=CONTRAST_SAMPLES_PER_BIN,
            seed=SEED,
        )
        entropy_payload = {f"layer_{i}": d for i, d in entropy_dirs.items()}
        entropy_payload.update(entropy_meta)
        entropy_payload["signal_kind"] = np.array("entropy")
        entropy_payload["activation_source"] = np.array("direct")
        entropy_payload["dataset"] = np.array(DATASET_NAME)
        entropy_payload["model"] = np.array(BASE_MODEL_NAME)
        entropy_path = f"{mc_directions_prefix}_entropy_contrast_directions.npz"
        np.savez_compressed(entropy_path, **entropy_payload)
        print(
            f"  entropy ({CONTRAST_PERCENT_ENTROPY:g}%, samples/bin {cap_str}): "
            f"low n={int(entropy_meta['n_low_used'])}/{int(entropy_meta['n_low'])} "
            f"(≤{float(entropy_meta['low_threshold']):.3f}) vs "
            f"high n={int(entropy_meta['n_high_used'])}/{int(entropy_meta['n_high'])} "
            f"(≥{float(entropy_meta['high_threshold']):.3f})"
        )
        print(f"  saved → {entropy_path}")

        # Stated-confidence contrast — META activations, signal = soft stated
        # confidence. Top CONTRAST_PERCENT_STATED_CONFIDENCE% (most confident)
        # vs bottom of the same. Only meaningful for the confidence task.
        if META_TASK == "confidence":
            sc_signal = np.asarray(stated_confidence_numeric, dtype=np.float64)
            valid_mask = ~np.isnan(sc_signal)
            min_required = int(np.ceil(200.0 / CONTRAST_PERCENT_STATED_CONFIDENCE))
            if valid_mask.sum() < min_required:
                print(
                    f"  stated_confidence: skipped — only {int(valid_mask.sum())} valid "
                    f"values, need ≥{min_required} for percent={CONTRAST_PERCENT_STATED_CONFIDENCE}"
                )
            else:
                valid_idx = np.where(valid_mask)[0]
                meta_acts_valid = {
                    l: a[valid_idx] for l, a in data["meta_activations"].items()
                }
                sc_dirs, sc_meta = compute_contrast_directions(
                    meta_acts_valid,
                    sc_signal[valid_idx],
                    percent=CONTRAST_PERCENT_STATED_CONFIDENCE,
                    samples_per_bin=CONTRAST_SAMPLES_PER_BIN,
                    seed=SEED,
                )
                sc_payload = {f"layer_{i}": d for i, d in sc_dirs.items()}
                sc_payload.update(sc_meta)
                sc_payload["signal_kind"] = np.array("stated_confidence")
                sc_payload["activation_source"] = np.array("meta")
                sc_payload["dataset"] = np.array(DATASET_NAME)
                sc_payload["model"] = np.array(BASE_MODEL_NAME)
                sc_payload["n_valid"] = np.array(int(valid_mask.sum()), dtype=np.int32)
                sc_path = f"{mc_directions_prefix}_stated_confidence_contrast_directions.npz"
                np.savez_compressed(sc_path, **sc_payload)
                print(
                    f"  stated_confidence ({CONTRAST_PERCENT_STATED_CONFIDENCE:g}%, "
                    f"samples/bin {cap_str}): "
                    f"low n={int(sc_meta['n_low_used'])}/{int(sc_meta['n_low'])} "
                    f"(≤{float(sc_meta['low_threshold']):.3f}) vs "
                    f"high n={int(sc_meta['n_high_used'])}/{int(sc_meta['n_high'])} "
                    f"(≥{float(sc_meta['high_threshold']):.3f})"
                )
                print(f"  saved → {sc_path}")
        else:
            print(
                f"  stated_confidence: skipped — META_TASK is {META_TASK!r}, "
                "no stated-confidence signal in this run"
            )

    # Behavioral analysis (uses selected METRIC for correlation with stated confidence)
    behavioral = analyze_behavioral_introspection(
        data["meta_responses"],
        data["direct_metrics"][METRIC],  # Use the selected metric for behavioral correlation
        test_idx,
        data["meta_probs"],
        data.get("meta_mappings"),
        data["direct_probs"],
        data["questions"]
    )

    # Other-confidence control analysis (only for confidence task)
    # other_data was collected earlier, now we analyze it
    other_confidence_analysis = None
    if META_TASK == "confidence" and other_data is not None:
        # Compute self-confidence signals for comparison (scale-aware)
        self_confidence = np.array([
            get_meta_signal(np.array(p)) if p else 0.5
            for p in data["meta_probs"]
        ])

        other_confidence_analysis = analyze_other_confidence_control(
            other_data["other_signals"],
            self_confidence,
            data["direct_metrics"][METRIC],
            test_idx
        )

    # Other-confidence probe analysis (only when activations were collected)
    # Uses trained probe components from main analysis - only computes D→M(Other)
    other_probe_results = None
    if META_TASK == "confidence" and other_data is not None and "other_activations" in other_data:
        print("\n" + "=" * 60)
        print("Running OTHER-CONFIDENCE TRANSFER ANALYSIS...")
        print("(Testing whether direct-trained probes transfer to other-confidence)")
        print("=" * 60)

        other_probe_results = apply_probes_to_other(
            other_data["other_activations"],
            entropy_probe_components,
            mc_probe_components,
            direct_target,
            model_predicted_answer,
            test_idx
        )

        # Print summary
        best_entropy_layer = max(other_probe_results.keys(), key=lambda l: other_probe_results[l]["d2m_other_entropy_r2"])
        best_mc_layer = max(other_probe_results.keys(), key=lambda l: other_probe_results[l]["d2m_other_mc_accuracy"])
        print(f"\nDirect Probe Transfer to Other-Confidence:")
        print(f"  Best Entropy D→M(Other): Layer {best_entropy_layer} (R²={other_probe_results[best_entropy_layer]['d2m_other_entropy_r2']:.3f})")
        print(f"  Best MC D→M(Other): Layer {best_mc_layer} (acc={other_probe_results[best_mc_layer]['d2m_other_mc_accuracy']:.3f})")

    # Calibration split analysis: split test set into calibrated vs miscalibrated trials
    # Uses stated_confidence from behavioral analysis
    stated_confidence = np.array(behavioral["stated_confidence"])
    test_confidence = stated_confidence[test_idx]
    test_metric = direct_target[test_idx]

    # Determine if the current metric indicates uncertainty (high = uncertain) or confidence (high = confident)
    metric_is_uncertainty = METRIC in UNCERTAINTY_METRICS
    calibrated_mask, miscalibrated_mask = compute_calibration_masks(
        test_confidence, test_metric, metric_is_uncertainty=metric_is_uncertainty
    )

    print(f"\nCalibration split (median) for {METRIC}:")
    if metric_is_uncertainty:
        print(f"  Calibrated: {calibrated_mask.sum()} trials (high conf + low {METRIC}, or low conf + high {METRIC})")
        print(f"  Miscalibrated: {miscalibrated_mask.sum()} trials (high conf + high {METRIC}, or low conf + low {METRIC})")
    else:
        print(f"  Calibrated: {calibrated_mask.sum()} trials (high conf + high {METRIC}, or low conf + low {METRIC})")
        print(f"  Miscalibrated: {miscalibrated_mask.sum()} trials (high conf + low {METRIC}, or low conf + high {METRIC})")

    # Split probe results by calibration
    calibration_split = split_results_by_calibration(
        results, direct_target[test_idx], calibrated_mask, miscalibrated_mask
    )

    # Print summary of calibration split
    if calibrated_mask.sum() > 1 and miscalibrated_mask.sum() > 1:
        best_cal_d2m_layer = max(calibration_split.keys(),
                                  key=lambda l: calibration_split[l]["calibrated"]["d2m_r2"])
        best_mis_d2m_layer = max(calibration_split.keys(),
                                  key=lambda l: calibration_split[l]["miscalibrated"]["d2m_r2"])
        print(f"\n  Calibrated best D→M: Layer {best_cal_d2m_layer} "
              f"(R²={calibration_split[best_cal_d2m_layer]['calibrated']['d2m_r2']:.3f})")
        print(f"  Miscalibrated best D→M: Layer {best_mis_d2m_layer} "
              f"(R²={calibration_split[best_mis_d2m_layer]['miscalibrated']['d2m_r2']:.3f})")

    # Plot calibration split
    plot_calibration_split(
        calibration_split,
        n_calibrated=int(calibrated_mask.sum()),
        n_miscalibrated=int(miscalibrated_mask.sum()),
        output_path=f"{metric_prefix}_calibration_split.png"
    )

    # Plot other-confidence probe comparison (only when available)
    if other_probe_results is not None:
        plot_other_confidence_comparison(
            results,
            mc_answer_results,
            other_probe_results,
            output_path=f"{metric_prefix}_other_confidence_transfer.png"
        )

    # Save results (metric-specific filename)
    results_to_save = {
        "config": {
            "metric": METRIC,
            "meta_task": META_TASK,
            "model": BASE_MODEL_NAME,
            "dataset": DATASET_NAME,
        },
        "metric_stats": {
            "mean": float(direct_target.mean()),
            "std": float(direct_target.std()),
            "min": float(direct_target.min()),
            "max": float(direct_target.max()),
        },
        "probe_results": {
            str(layer_idx): {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in layer_results.items()
                if not isinstance(v, dict) or k in ["direct_to_direct", "direct_to_meta", "direct_to_meta_fixed", "shuffled_baseline", "meta_to_meta"]
            }
            for layer_idx, layer_results in results.items()
        },
        "behavioral": behavioral,
        "other_confidence_analysis": other_confidence_analysis,  # None if not confidence task
        "test_indices": test_idx.tolist(),
        "mc_answer_probe": {
            str(layer_idx): {
                "d2d_accuracy": layer_results["d2d_accuracy"],
                "d2m_accuracy": layer_results["d2m_accuracy"],
                "shuffled_accuracy": layer_results["shuffled_accuracy"],
            }
            for layer_idx, layer_results in mc_answer_results.items()
        },
        "model_predicted_answer": model_predicted_answer.tolist(),
        "calibration_split": {
            str(layer_idx): layer_data
            for layer_idx, layer_data in calibration_split.items()
        },
        "calibration_counts": {
            "calibrated": int(calibrated_mask.sum()),
            "miscalibrated": int(miscalibrated_mask.sum()),
        },
    }

    # Add other-confidence probe results if available
    if other_probe_results is not None:
        results_to_save["other_confidence_transfer"] = {
            str(layer_idx): {
                "d2m_other_entropy_r2": layer_data["d2m_other_entropy_r2"],
                "d2m_other_mc_accuracy": layer_data["d2m_other_mc_accuracy"],
            }
            for layer_idx, layer_data in other_probe_results.items()
        }

    # Properly serialize nested dicts
    for layer_idx in results_to_save["probe_results"]:
        for key in ["direct_to_direct", "direct_to_meta", "direct_to_meta_fixed", "shuffled_baseline", "meta_to_meta"]:
            if key in results_to_save["probe_results"][layer_idx]:
                inner = results_to_save["probe_results"][layer_idx][key]
                for k, v in inner.items():
                    if isinstance(v, np.ndarray):
                        inner[k] = v.tolist()

    class _NumpyJSONEncoder(json.JSONEncoder):
        """Encoder that handles numpy scalars/arrays without explicit pre-conversion.

        Needed because behavioral / calibration_split / mc_answer_probe subtrees
        can contain np.float32/np.int64 scalars (especially when direct metrics
        were loaded from disk via _load_mc_data_for_reuse).
        """
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return super().default(o)

    with open(f"{metric_prefix}_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2, cls=_NumpyJSONEncoder)
    print(f"Saved results to {metric_prefix}_results.json")

    # Print and plot results. Tee stdout so the human-readable summary is
    # also saved to disk as `{metric_prefix}_results.txt`.
    import sys as _sys

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    _results_txt_path = f"{metric_prefix}_results.txt"
    with open(_results_txt_path, "w") as _fh:
        _orig_stdout = _sys.stdout
        _sys.stdout = _Tee(_orig_stdout, _fh)
        try:
            print_results(results, behavioral, other_confidence_analysis)
        finally:
            _sys.stdout = _orig_stdout
    print(f"Saved results summary to {_results_txt_path}")
    plot_results(
        results, behavioral,
        direct_target, test_idx,
        output_path=f"{metric_prefix}_results.png",
        mc_answer_results=mc_answer_results
    )

    print(f"\n✓ Introspection experiment complete! ({dataset_name} / {meta_task} / {metric})")


def main():
    parser = argparse.ArgumentParser(description="Run introspection experiment")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Uncertainty metric to probe (default: {METRIC})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for forward passes (default {BATCH_SIZE})")
    parser.add_argument("--load-in-4bit", action="store_true", default=LOAD_IN_4BIT,
                        help=f"Load model in 4-bit quantization (default: {LOAD_IN_4BIT})")
    parser.add_argument("--load-in-8bit", action="store_true", default=LOAD_IN_8BIT,
                        help=f"Load model in 8-bit quantization (default: {LOAD_IN_8BIT})")
    parser.add_argument("--collect-other-activations", action="store_true", default=False,
                        help="Collect activations during other-confidence task for probe analysis")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Metric: {args.metric}")
    print(f"Datasets to process: {DATASETS}")
    print(f"Meta-tasks to process: {META_TASKS}")
    print(f"Total combinations: {len(DATASETS) * len(META_TASKS)}")

    # Load model and tokenizer ONCE using shared utility
    print("\nLoading model (this will be shared across all experiments)...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )

    # Sanity check: every confidence option must tokenize to exactly one token.
    # If any option (especially "10" on non-Llama-3 tokenizers) is multi-token,
    # our argmax extraction will silently read the wrong probability. Fail loudly.
    if CONFIDENCE_SCALE == "numeric":
        bad = []
        for opt in NUMERIC_CONFIDENCE_OPTIONS:
            ids = tokenizer.encode(opt, add_special_tokens=False)
            if len(ids) != 1:
                bad.append((opt, ids))
        if bad:
            msg_lines = [
                "Numeric confidence scale requires every option to tokenize to exactly one token.",
                f"Tokenizer: {getattr(tokenizer, 'name_or_path', '?')}",
            ]
            for opt, ids in bad:
                decoded = [tokenizer.decode([t]) for t in ids]
                msg_lines.append(f"  '{opt}' -> {ids}  (decoded: {decoded})  [len={len(ids)}]")
            msg_lines.append(
                "Fix by either (a) switching to CONFIDENCE_SCALE='letters', or "
                "(b) narrowing the scale in tasks.py NUMERIC_CONFIDENCE_OPTIONS "
                "(e.g., drop '10' to get 1-9)."
            )
            raise RuntimeError("\n".join(msg_lines))
        print(f"✓ Numeric scale tokenizer check passed: all of {list(NUMERIC_CONFIDENCE_OPTIONS)} are single-token.")

    # Base models now have a few-shot delegate prompt (format_answer_or_delegate_prompt_base),
    # so we run delegate for all model types.
    meta_tasks = META_TASKS

    # Run all dataset/task combinations
    for dataset_name in DATASETS:
        for meta_task in meta_tasks:
            run_single_experiment(
                dataset_name=dataset_name,
                meta_task=meta_task,
                model=model,
                tokenizer=tokenizer,
                num_layers=num_layers,
                metric=args.metric,
                batch_size=args.batch_size,
                collect_other_activations=args.collect_other_activations
            )

    print("\n" + "=" * 80)
    print("✓ All experiments complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
