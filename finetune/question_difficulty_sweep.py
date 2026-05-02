"""
question_difficulty_sweep.py

Run a single model over every question in {PopMC, TriviaMC, SimpleMC} and
record per-question correctness, entropy, A-D probabilities, and stated
confidence. Output is intended for stratified sampling of an easy/moderate/
hard mix for finetuning, and for selecting an external OOD eval set.

Reuses the exact eval pipeline used during training (`evaluate_model` from
`evaluation_metrics.py`) so prompting, option-shuffling, and confidence
extraction match what the finetune sees — selecting questions with a
divergent prompt would invalidate the difficulty buckets.

Outputs (per model):
    outputs/difficulty_sweep/<model_tag>_raw.jsonl
        One row per (dataset, question) — the eval_sample log_entry,
        plus a `dataset_name` field. Includes question text, probs_ABCD,
        top_prob, prob_gap, entropy, is_correct, expected_confidence, etc.
    outputs/difficulty_sweep/<model_tag>_summary.csv
        Flat one-row-per-question CSV for pandas: dataset_name, qid,
        is_correct, top_prob, prob_gap, entropy, expected_confidence,
        p_A, p_B, p_C, p_D, model_answer_position, correct_answer_position.

Usage:
    # base instruct model, all three datasets
    python finetune/question_difficulty_sweep.py \\
        --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --sigma 0.5

    # finetuned model
    python finetune/question_difficulty_sweep.py \\
        --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --lora_repo your-org/your-lora \\
        --merge \\
        --sigma 0.5
"""

# --- repo path bootstrap (so root-level imports resolve when run from anywhere) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from core.model_utils import load_model_and_tokenizer
from data_handling import load_jsonl_dataset
from evaluation_metrics import evaluate_model
from utils import prepare_model_and_tokenizer


DEFAULT_DATASETS = [
    ("PopMC",    "data/PopMC.jsonl"),
    ("TriviaMC", "data/TriviaMC.jsonl"),
    ("SimpleMC", "data/SimpleMC.jsonl"),
]


def _model_tag(base_model: str, lora_repo: str) -> str:
    base = base_model.split("/")[-1]
    if lora_repo:
        lora = lora_repo.split("/")[-1]
        return f"{base}__{lora}"
    return base


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _summary_row(rec: dict) -> dict:
    probs = rec.get("probs_ABCD") or [None, None, None, None]
    return {
        "dataset_name": rec.get("dataset_name"),
        "qid": rec.get("qid"),
        "is_correct": rec.get("is_correct"),
        "top_prob": rec.get("top_prob"),
        "prob_gap": rec.get("prob_gap"),
        "entropy": rec.get("entropy"),
        "expected_confidence": rec.get("expected_confidence"),
        "p_A": probs[0] if len(probs) > 0 else None,
        "p_B": probs[1] if len(probs) > 1 else None,
        "p_C": probs[2] if len(probs) > 2 else None,
        "p_D": probs[3] if len(probs) > 3 else None,
        "model_answer_position": rec.get("model_answer_position"),
        "correct_answer_position": rec.get("correct_answer_position"),
    }


def run_sweep(
    base_model: str,
    lora_repo: str = "",
    merge: bool = False,
    sigma: float = 0.5,
    datasets=DEFAULT_DATASETS,
    output_dir: str = "outputs/difficulty_sweep",
    max_samples: int = None,
    compute_confidence: bool = True,
    compute_other_confidence: bool = False,
    loss_type: str = "gaussian_soft_bin_ce",
    confidence_letter_scheme: str = "A-H",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = _model_tag(base_model, lora_repo)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

    # Load model once and reuse across datasets — loading is the expensive step.
    print(f"\nLoading model: {base_model}" +
          (f" + adapter {lora_repo} (merge={merge})" if lora_repo else ""))
    model, tokenizer, _ = load_model_and_tokenizer(
        base_model_name=base_model,
        adapter_path=lora_repo or None,
        merge=merge,
    )
    model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)

    # Per-dataset log files written by evaluate_model, then merged.
    intermediate_logs = []
    for ds_name, ds_path in datasets:
        ds_path = Path(ds_path)
        if not ds_path.exists():
            print(f"  skipping {ds_name}: {ds_path} not found")
            continue

        print(f"\n=== {ds_name} ({ds_path}) ===")
        data = load_jsonl_dataset(str(ds_path), dataset_type="evaluation")
        if max_samples is not None:
            data = data[:max_samples]
        print(f"  {len(data)} questions")

        log_path = output_dir / f"{tag}__{ds_name}__{timestamp}.jsonl"
        evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=data,
            sigma=sigma,
            compute_confidence=compute_confidence,
            compute_other_confidence=compute_other_confidence,
            loss_type=loss_type,
            num_samples=max_samples,
            log_file_path=str(log_path),
            use_wandb=False,
            wandb_project=None,
            wandb_run_name=f"sweep_{tag}_{ds_name}",
            log_prefix=f"{ds_name}_",
            confidence_letter_scheme=confidence_letter_scheme,
            confidence_letter_random_seed=None,
        )
        intermediate_logs.append((ds_name, log_path))

    # Merge per-dataset logs into one raw JSONL, stamping dataset_name on each row.
    raw_path = output_dir / f"{tag}_raw.jsonl"
    summary_rows = []
    n_total = 0
    with open(raw_path, "w") as out:
        for ds_name, log_path in intermediate_logs:
            for rec in _read_jsonl(log_path):
                # Skip non-sample rows (eval_summary footer etc.)
                t = rec.get("type", "")
                if not t.endswith("eval_sample"):
                    continue
                rec["dataset_name"] = ds_name
                out.write(json.dumps(rec) + "\n")
                summary_rows.append(_summary_row(rec))
                n_total += 1

    # Flat CSV for pandas / stratified sampling.
    summary_path = output_dir / f"{tag}_summary.csv"
    fieldnames = [
        "dataset_name", "qid", "is_correct", "top_prob", "prob_gap",
        "entropy", "expected_confidence",
        "p_A", "p_B", "p_C", "p_D",
        "model_answer_position", "correct_answer_position",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {n_total} rows")
    print(f"  raw:     {raw_path}")
    print(f"  summary: {summary_path}")
    return {"raw": str(raw_path), "summary": str(summary_path), "n": n_total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-question difficulty sweep over PopMC + TriviaMC + SimpleMC")
    parser.add_argument("--base_model", required=True,
                        help="Base model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--lora_repo", default="",
                        help="Optional HF repo with LoRA weights. Empty = sweep base model.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA into base model (only used with --lora_repo)")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Gaussian width for soft-label conversion (passed to evaluate_model)")
    parser.add_argument("--output_dir", default="outputs/difficulty_sweep",
                        help="Where to write <model_tag>_raw.jsonl and _summary.csv")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit per dataset (debugging)")
    parser.add_argument("--no_confidence", action="store_true",
                        help="Skip confidence pass for speed (no expected_confidence column)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Optional dataset list as NAME=PATH pairs, e.g. PopMC=data/PopMC.jsonl. "
                             "Default: PopMC, TriviaMC, SimpleMC.")

    args = parser.parse_args()

    if args.datasets:
        datasets = []
        for entry in args.datasets:
            name, _, path = entry.partition("=")
            if not path:
                raise SystemExit(f"--datasets entries must be NAME=PATH (got {entry!r})")
            datasets.append((name, path))
    else:
        datasets = DEFAULT_DATASETS

    run_sweep(
        base_model=args.base_model,
        lora_repo=args.lora_repo,
        merge=args.merge,
        sigma=args.sigma,
        datasets=datasets,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        compute_confidence=not args.no_confidence,
        compute_other_confidence=False,
    )
