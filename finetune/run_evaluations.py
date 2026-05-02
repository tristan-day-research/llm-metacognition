"""
run_evaluations.py

Post-hoc MC evaluation on PopMC / SimpleMC / TriviaMC.

  ✓ Loads (optionally LoRA-adapted) model via core.model_utils.load_model_and_tokenizer()
  ✓ Optionally evaluates the base (un-finetuned) model first for comparison
  ✓ Runs evaluate_model() from evaluation_metrics.py for each dataset

WORKFLOW REPLICATION:
  Same workflow as run_finetuning.py for perfect replication:
  - load_jsonl_dataset() for data loading (same shuffling)
  - prepare_model_and_tokenizer() for model setup
  - evaluate_model() → run_evaluation()
  - build_multiple_choice_question_prompts() for prompts
  - run_mcq_forward_pass() / run_confidence_forward_pass() for inference

CONFIG-DRIVEN:
  No CLI flags. Edit values in finetune_config.ECTConfig and run:

      python finetune/run_evaluations.py

  Eval reads from the SAME ECTConfig that training uses. Shared knobs (sigma,
  loss_type, confidence/mcq letter scheme) come from ECTConfig directly so
  the eval pipeline cannot drift from training. Eval-only knobs are the
  EVAL_* fields on ECTConfig (EVAL_DATASETS, EVAL_LORA_REPO, EVAL_MAX_SAMPLES,
  EVAL_EVALUATE_BASE_FIRST, EVAL_LOG_DIR, …).

  ECTConfig.EVAL_DATASETS is a list — the script iterates over it, producing
  one JSONL per dataset under outputs/evaluations/.

  Per-sample row schema (in JSONL):
      qid, is_correct, probs_ABCD, top_prob, prob_gap, entropy,
      expected_confidence, model_answer_position, correct_answer_position
  Plus one eval_summary row per model with mcq_accuracy / avg_entropy / etc.
"""

# --- repo path bootstrap (so root-level imports like `finetune_prompting`,
# `finetune_config` resolve when run from anywhere) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import os
import torch
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from core.model_utils import load_model_and_tokenizer
from evaluation_metrics import evaluate_model
from data_handling import load_jsonl_dataset
from utils import prepare_model_and_tokenizer
from finetune_config import ECTConfig


def _print_letter_diagnostics(log_file_path: str, log_prefix: str = "") -> None:
    """Position-bias diagnostics computed from the per-sample JSONL log.

    Reads the rows just written by evaluate_model() and reports two checks
    that the standard summary doesn't expose:
      - chi-square test for uniformity of predicted positions (detects
        A/B/C/D answer bias independent of correctness)
      - per-position conditional accuracy P(correct | true_position=p),
        which catches the "always predicts B" failure mode

    Position-based (not display-letter-based) so it's robust to the
    randomized letter mapping used in training.
    """
    sample_type = f"{log_prefix}eval_sample" if log_prefix else "eval_sample"
    rows = []
    with open(log_file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == sample_type:
                rows.append(r)
    if not rows:
        return

    pred_pos = [r["model_answer_position"] for r in rows
                if r.get("model_answer_position") is not None]
    n = len(pred_pos)
    if n == 0:
        return

    print(f"\n--- POSITION-BIAS DIAGNOSTICS ({log_prefix or 'eval'}) ---")
    pred_dist = Counter(pred_pos)
    expected = n / 4.0
    chi2 = sum((pred_dist.get(p, 0) - expected) ** 2 / expected for p in range(4))
    print(f"  Chi-square (predicted-position uniformity): {chi2:.2f}  "
          f"(crit @ p=0.05: 7.81 → {'uniform' if chi2 < 7.81 else 'biased'})")
    print(f"  Predicted-position counts: " +
          ", ".join(f"pos{p}={pred_dist.get(p, 0)}" for p in range(4)))

    print("  Accuracy conditioned on correct position:")
    for pos in range(4):
        total = sum(1 for r in rows if r.get("correct_answer_position") == pos)
        if total == 0:
            continue
        n_correct = sum(1 for r in rows
                        if r.get("correct_answer_position") == pos
                        and r.get("is_correct"))
        print(f"    correct=pos{pos}: {n_correct}/{total} ({n_correct/total:.2%})")


###############################################################################
# Per-dataset evaluation
###############################################################################
def run_evaluation_on_dataset(
    base_model_name: str,
    lora_repo: str,
    dataset_path: str,
    sigma: float,
    merge: bool,
    max_samples,
    evaluate_base_first: bool,
    use_wandb: bool,
    wandb_project: str,
    compute_confidence: bool,
    compute_other_confidence: bool,
    loss_type: str,
    log_dir,
    confidence_letter_scheme: str,
    confidence_letter_random_seed,
):
    """Evaluate one dataset. If evaluate_base_first=True and a lora_repo is
    provided, runs the base model and the finetuned model into the same JSONL
    for side-by-side analysis. If lora_repo is empty, runs only the base model.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    dataset_name = dataset_path.stem
    base_model_safe = base_model_name.replace("/", "-").replace("_", "-")
    lora_name = lora_repo.split("/")[-1] if lora_repo else "base"

    # Load dataset using the proper loader (handles PopMC format conversion)
    print(f"\nLoading dataset from: {dataset_path}")
    data = load_jsonl_dataset(str(dataset_path), dataset_type="evaluation")
    print(f"Loaded {len(data)} samples from dataset")

    if max_samples is not None:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples for evaluation")

    # If no LoRA was provided, force a single-pass base eval.
    do_finetuned = bool(lora_repo)
    do_base_first = evaluate_base_first and do_finetuned

    suffix = "_comparison" if do_base_first else ""
    log_file = log_dir / (
        f"{timestamp}_{base_model_safe}_{lora_name}_{dataset_name}_n{len(data)}{suffix}.jsonl"
    )
    print(f"Logging to: {log_file}")

    results = {}

    if do_base_first:
        print("\n" + "=" * 60)
        print(" EVALUATING BASE (UN-FINETUNED) MODEL")
        print("=" * 60 + "\n")

        base_model_obj, base_tokenizer, _ = load_model_and_tokenizer(base_model_name)
        base_model_obj, base_tokenizer = prepare_model_and_tokenizer(
            base_model_obj, base_tokenizer
        )

        results["base"] = evaluate_model(
            model=base_model_obj,
            tokenizer=base_tokenizer,
            dataset=data,
            sigma=sigma,
            compute_confidence=compute_confidence,
            compute_other_confidence=compute_other_confidence,
            loss_type=loss_type,
            num_samples=max_samples,
            log_file_path=str(log_file),
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=f"{dataset_name}_base",
            log_prefix="instruct_",
            confidence_letter_scheme=confidence_letter_scheme,
            confidence_letter_random_seed=confidence_letter_random_seed,
        )
        _print_letter_diagnostics(str(log_file), log_prefix="instruct_")

        del base_model_obj, base_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print(" BASE MODEL EVALUATION COMPLETE")
        print("=" * 60 + "\n")

    if do_finetuned:
        print("\n" + "=" * 60)
        print(" EVALUATING FINETUNED MODEL")
        print("=" * 60 + "\n")

        model, tokenizer, _ = load_model_and_tokenizer(
            base_model_name=base_model_name,
            adapter_path=lora_repo,
            merge=merge,
        )
        model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)

        results["finetuned"] = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=data,
            sigma=sigma,
            compute_confidence=compute_confidence,
            compute_other_confidence=compute_other_confidence,
            loss_type=loss_type,
            num_samples=max_samples,
            log_file_path=str(log_file),
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=f"{dataset_name}_finetuned",
            log_prefix="finetuned_",
            confidence_letter_scheme=confidence_letter_scheme,
            confidence_letter_random_seed=confidence_letter_random_seed,
        )
        _print_letter_diagnostics(str(log_file), log_prefix="finetuned_")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Base-only path: still run a single eval pass on the base model.
        print("\n" + "=" * 60)
        print(" EVALUATING BASE MODEL (no LoRA configured)")
        print("=" * 60 + "\n")

        base_model_obj, base_tokenizer, _ = load_model_and_tokenizer(base_model_name)
        base_model_obj, base_tokenizer = prepare_model_and_tokenizer(
            base_model_obj, base_tokenizer
        )

        results["base"] = evaluate_model(
            model=base_model_obj,
            tokenizer=base_tokenizer,
            dataset=data,
            sigma=sigma,
            compute_confidence=compute_confidence,
            compute_other_confidence=compute_other_confidence,
            loss_type=loss_type,
            num_samples=max_samples,
            log_file_path=str(log_file),
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=f"{dataset_name}_base",
            log_prefix="instruct_",
            confidence_letter_scheme=confidence_letter_scheme,
            confidence_letter_random_seed=confidence_letter_random_seed,
        )
        _print_letter_diagnostics(str(log_file), log_prefix="instruct_")

        del base_model_obj, base_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"log_file": str(log_file), "metrics": results}


###############################################################################
# Entry point — reads finetune_config.ECTConfig
###############################################################################
def main():
    cfg = ECTConfig

    print("=" * 60)
    print(" run_evaluations.py — config-driven (reads ECTConfig)")
    print("=" * 60)
    print(f" base_model:        {cfg.EVAL_BASE_MODEL}")
    print(f" lora_repo:         {cfg.EVAL_LORA_REPO or '(base only)'}")
    print(f" datasets:          {list(cfg.EVAL_DATASETS)}")
    print(f" sigma:             {cfg.SIGMA}              (shared with training)")
    print(f" loss_type:         {cfg.LOSS_TYPE}  (shared with training)")
    print(f" confidence_scheme: {cfg.CONFIDENCE_LETTER_SCHEME}            (shared with training)")
    print(f" mcq_scheme:        {cfg.MCQ_LETTER_SCHEME}            (shared with training)")
    print(f" max_samples:       {cfg.EVAL_MAX_SAMPLES}")
    print(f" log_dir:           {cfg.EVAL_LOG_DIR}")
    print("=" * 60)

    all_results = {}
    for dataset_path in cfg.EVAL_DATASETS:
        out = run_evaluation_on_dataset(
            base_model_name=cfg.EVAL_BASE_MODEL,
            lora_repo=cfg.EVAL_LORA_REPO,
            dataset_path=dataset_path,
            sigma=cfg.SIGMA,
            merge=cfg.EVAL_MERGE_LORA,
            max_samples=cfg.EVAL_MAX_SAMPLES,
            evaluate_base_first=cfg.EVAL_EVALUATE_BASE_FIRST,
            use_wandb=cfg.EVAL_USE_WANDB,
            wandb_project=cfg.EVAL_WANDB_PROJECT,
            compute_confidence=cfg.EVAL_COMPUTE_CONFIDENCE,
            compute_other_confidence=cfg.EVAL_COMPUTE_OTHER_CONFIDENCE,
            loss_type=cfg.LOSS_TYPE,
            log_dir=cfg.EVAL_LOG_DIR,
            confidence_letter_scheme=cfg.CONFIDENCE_LETTER_SCHEME,
            confidence_letter_random_seed=cfg.CONFIDENCE_LETTER_RANDOM_SEED,
        )
        all_results[dataset_path] = out

    print("\n" + "=" * 60)
    print(" ALL EVALUATIONS COMPLETE")
    print("=" * 60)
    for dataset_path, out in all_results.items():
        metrics = out["metrics"]
        print(f"\n{dataset_path}")
        print(f"  log:  {out['log_file']}")
        if "base" in metrics:
            acc = metrics["base"].get("mcq_accuracy", float("nan"))
            ent = metrics["base"].get("avg_entropy", float("nan"))
            print(f"  base       — accuracy={acc:.4f}  avg_entropy={ent:.4f}")
        if "finetuned" in metrics:
            acc = metrics["finetuned"].get("mcq_accuracy", float("nan"))
            ent = metrics["finetuned"].get("avg_entropy", float("nan"))
            print(f"  finetuned  — accuracy={acc:.4f}  avg_entropy={ent:.4f}")

    return all_results


if __name__ == "__main__":
    main()
