"""
run_evaluations.py

Post-hoc MC evaluation on PopMC / SimpleMC / TriviaMC.

  ✓ Loads exactly one model — base, instruct, or finetuned (LoRA on instruct)
  ✓ Runs evaluate_model() from evaluation_metrics.py for each dataset

WORKFLOW REPLICATION:
  Same workflow as run_finetuning.py for perfect replication:
  - load_jsonl_dataset() for data loading (same shuffling)
  - prepare_model_and_tokenizer() for model setup
  - evaluate_model() → run_evaluation()
  - build_multiple_choice_question_prompts() for prompts (chat-tagged for
    instruct/finetuned, raw text for base)
  - run_mcq_forward_pass() / run_confidence_forward_pass[_numeric] for inference

CONFIG-DRIVEN:
  No CLI flags. Edit values in finetune_config.ECTConfig and run:

      python finetune/run_evaluations.py

  Eval reads from the SAME ECTConfig that training uses. Shared knobs (sigma,
  loss_type, confidence_format, letter schemes) come from ECTConfig directly
  so the eval pipeline cannot drift from training. Eval-only knobs are the
  EVAL_* fields on ECTConfig:
    EVAL_MODEL_TYPE     — "base" | "instruct" | "finetuned"
    EVAL_LORA_REPO      — adapter repo, used only for "finetuned"
    EVAL_DATASETS       — list of JSONLs to evaluate, one at a time
    EVAL_MAX_SAMPLES    — None for full dataset
    EVAL_LOG_DIR        — defaults to outputs/evaluations
    EVAL_USE_WANDB      — toggle Weights & Biases logging

OUTPUT FILES (per dataset, same stem):
  outputs/evaluations/<timestamp>_<model>_<dataset>_n<N>.jsonl
      Per-sample evaluation rows + a per-model summary row. The analysis
      notebook reads this.
  outputs/evaluations/<timestamp>_<model>_<dataset>_n<N>.txt
      Human-readable run record, written incrementally:
        - Config snapshot at start
        - One example of each prompt type (MCQ / self-conf / other-conf)
          exactly as the model sees it (chat tags for instruct/finetuned,
          raw text for base)
        - The model's actual short reply for each example prompt
        - Any warnings / exceptions raised during the run
        - Position-bias diagnostics + final summary at end
"""

# --- repo path bootstrap (so root-level imports like `finetune_prompting`,
# `finetune_config` resolve when run from anywhere) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import contextlib
import io
import json
import os
import traceback
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

# Fail-fast: this script is meaningless on CPU (8B model, single-sample loop ≈
# 13 s/iter on CPU vs ≈ 0.3 s/iter on H100). If CUDA isn't available at import
# time, abort before loading the model — silently running on CPU costs hours.
if not torch.cuda.is_available():
    raise RuntimeError(
        "run_evaluations.py: torch.cuda.is_available() is False. Refusing to "
        "run on CPU. Check the driver / torch CUDA build (nvidia-smi, then "
        "`pip install --upgrade --index-url https://download.pytorch.org/whl/cuXXX torch` "
        "matching your driver)."
    )

from core.model_utils import load_model_and_tokenizer
from data_handling import load_jsonl_dataset
from evaluation_metrics import evaluate_model
from experiment_config import LLAMA_8B_BASE, LLAMA_8B_INSTRUCT
from finetune_config import ECTConfig
from finetune_prompting import (
    build_multiple_choice_question_prompts,
    build_other_confidence_prompts,
    build_other_confidence_prompts_numeric,
    build_self_confidence_prompts,
    build_self_confidence_prompts_numeric,
    get_confidence_letter_mapping,
    get_mcq_letter_mapping,
)
from utils import prepare_model_and_tokenizer


###############################################################################
# Run logger — incremental .txt log paired with the JSONL output
###############################################################################
class RunLogger:
    """Append-only text log that flushes on every write.

    A crash mid-run leaves a partial but readable file at ``self.path``,
    which is the whole point — never wait until run-end to commit logs.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")  # truncate at start

    def write(self, *lines, blank_after: bool = False) -> None:
        with self.path.open("a") as f:
            for line in lines:
                s = str(line)
                f.write(s)
                if not s.endswith("\n"):
                    f.write("\n")
            if blank_after:
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    def section(self, title: str) -> None:
        self.write("", "=" * 70, f" {title}", "=" * 70)

    def subsection(self, title: str) -> None:
        self.write("", "-" * 70, f" {title}", "-" * 70)


def _install_warning_capture(logger: RunLogger):
    """Route Python warnings into the run log AND to the original handler."""
    original = warnings.showwarning

    def capture(message, category, filename, lineno, file=None, line=None):
        logger.write(
            f"[WARNING {category.__name__}] {message}  ({filename}:{lineno})"
        )
        original(message, category, filename, lineno, file, line)

    warnings.showwarning = capture
    return lambda: setattr(warnings, "showwarning", original)


def _dump_config_snapshot(logger: RunLogger) -> None:
    """Write every ECTConfig field to the log so the run is fully reconstructable."""
    logger.section("CONFIG SNAPSHOT (finetune_config.ECTConfig)")
    rows = []
    for name in sorted(vars(ECTConfig)):
        if name.startswith("_"):
            continue
        value = getattr(ECTConfig, name)
        if callable(value):
            continue
        rows.append((name, value))
    width = max(len(n) for n, _ in rows) if rows else 0
    for name, value in rows:
        logger.write(f"  {name.ljust(width)} = {value!r}")


def _generate_short_reply(model, tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    """Run a tiny greedy generate() so the .txt can show what the model would
    actually emit on this prompt. Special tokens kept visible — the user wants
    to see exactly what comes out of the tokenizer."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    reply_ids = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(reply_ids, skip_special_tokens=False)


def _log_sample_prompts_and_replies(
    logger: RunLogger,
    model,
    tokenizer,
    sample_row: dict,
    model_type: str,
    confidence_format: str,
    confidence_letter_mapping,
    mcq_letter_mapping,
) -> None:
    """For one sample row, dump each prompt type (MCQ / self-conf / other-conf)
    exactly as the model sees it, then run a short generate() to capture the
    model's actual reply.

    Confidence prompt variants follow CONFIDENCE_FORMAT
    ("letter_8bin" → letter prompts, "numeric_1_5" / "numeric_1_10" → numeric).
    Self- and other-confidence always use the same scheme. Prompt formatting
    follows model_type ("base" → raw text, otherwise → chat tags).
    """
    # MCQ
    mcq_prompt = build_multiple_choice_question_prompts(
        [sample_row], tokenizer, mcq_letter_mapping, model_type=model_type
    )[0]
    logger.subsection(f"[{model_type}] sample MCQ prompt (qid={sample_row.get('id')})")
    logger.write(mcq_prompt)
    logger.subsection(f"[{model_type}] sample MCQ reply (greedy, max_new_tokens=8)")
    logger.write(_generate_short_reply(model, tokenizer, mcq_prompt))

    # Self- and other-confidence (same scheme)
    if confidence_format in ("numeric_1_5", "numeric_1_10"):
        n_max = 5 if confidence_format == "numeric_1_5" else 10
        self_prompt = build_self_confidence_prompts_numeric(
            [sample_row], tokenizer, mcq_letter_mapping, n_max=n_max,
            model_type=model_type,
        )[0]
        other_prompt = build_other_confidence_prompts_numeric(
            [sample_row], tokenizer, mcq_letter_mapping, n_max=n_max,
            model_type=model_type,
        )[0]
        scheme_label = f"numeric 1-{n_max}"
    else:
        self_prompt = build_self_confidence_prompts(
            [sample_row], tokenizer, confidence_letter_mapping, mcq_letter_mapping
        )[0]
        other_prompt = build_other_confidence_prompts(
            [sample_row], tokenizer, confidence_letter_mapping, mcq_letter_mapping
        )[0]
        scheme_label = "letter A-H"

    logger.subsection(f"[{model_type}] sample self-confidence ({scheme_label}) prompt")
    logger.write(self_prompt)
    logger.subsection(f"[{model_type}] sample self-confidence ({scheme_label}) reply")
    logger.write(_generate_short_reply(model, tokenizer, self_prompt))

    logger.subsection(f"[{model_type}] sample other-confidence ({scheme_label}) prompt")
    logger.write(other_prompt)
    logger.subsection(f"[{model_type}] sample other-confidence ({scheme_label}) reply")
    logger.write(_generate_short_reply(model, tokenizer, other_prompt))


def _print_letter_diagnostics(
    log_file_path: str,
    log_prefix: str = "",
    logger: "RunLogger | None" = None,
) -> None:
    """Position-bias diagnostics computed from the per-sample JSONL log."""
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

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
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

    text = buf.getvalue()
    print(text, end="")
    if logger is not None:
        logger.subsection(f"position-bias diagnostics ({log_prefix or 'eval'})")
        logger.write(text.rstrip("\n"))


###############################################################################
# Per-dataset evaluation — single model, no comparison
###############################################################################
def _resolve_model_spec(model_type: str):
    """Map EVAL_MODEL_TYPE → (base_model_name, adapter_path_or_None)."""
    if model_type == "base":
        return LLAMA_8B_BASE, None
    if model_type == "instruct":
        return LLAMA_8B_INSTRUCT, None
    if model_type == "finetuned":
        adapter = ECTConfig.EVAL_LORA_REPO
        if not adapter:
            raise ValueError(
                "EVAL_MODEL_TYPE='finetuned' requires EVAL_LORA_REPO to be set "
                "in ECTConfig."
            )
        return LLAMA_8B_INSTRUCT, adapter
    raise ValueError(
        f"EVAL_MODEL_TYPE must be 'base', 'instruct', or 'finetuned'; got {model_type!r}"
    )


def run_evaluation_on_dataset(
    *,
    model_type: str,
    dataset_path: str,
    sigma: float,
    confidence_format: str,
    merge: bool,
    max_samples,
    use_wandb: bool,
    wandb_project: str,
    compute_confidence: bool,
    compute_other_confidence: bool,
    loss_type: str,
    log_dir,
    confidence_letter_scheme: str,
    confidence_letter_random_seed,
):
    """Evaluate one (model_type, dataset) pair into a JSONL + .txt pair."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_model_name, adapter = _resolve_model_spec(model_type)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    dataset_name = dataset_path.stem
    base_model_safe = base_model_name.replace("/", "-").replace("_", "-")
    if model_type == "finetuned":
        run_tag = f"{base_model_safe}_{adapter.split('/')[-1]}"
    else:
        run_tag = f"{base_model_safe}_{model_type}"

    print(f"\nLoading dataset from: {dataset_path}")
    data = load_jsonl_dataset(str(dataset_path), dataset_type="evaluation")
    print(f"Loaded {len(data)} samples from dataset")

    if max_samples is not None:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples for evaluation")

    stem = f"{timestamp}_{run_tag}_{dataset_name}_n{len(data)}"
    log_file = log_dir / f"{stem}.jsonl"
    txt_file = log_dir / f"{stem}.txt"
    print(f"Logging JSONL to: {log_file}")
    print(f"Logging  TXT  to: {txt_file}")

    logger = RunLogger(txt_file)
    restore_warnings = _install_warning_capture(logger)

    confidence_letter_mapping = get_confidence_letter_mapping(
        confidence_letter_scheme, seed=confidence_letter_random_seed
    )
    mcq_letter_mapping = get_mcq_letter_mapping(
        ECTConfig.MCQ_LETTER_SCHEME, seed=ECTConfig.MCQ_LETTER_RANDOM_SEED
    )

    logger.write(f"run_evaluations.py — {timestamp}Z")
    logger.write(f"model_type:        {model_type}")
    logger.write(f"base_model_name:   {base_model_name}")
    logger.write(f"adapter:           {adapter or '(none)'}")
    logger.write(f"dataset:           {dataset_path}")
    logger.write(f"jsonl out:         {log_file}")
    logger.write(f"n samples:         {len(data)}")
    logger.write(f"confidence_format: {confidence_format}")
    _dump_config_snapshot(logger)

    sample_row = data[0] if data else None
    log_prefix = f"{model_type}_"

    try:
        logger.section(f"EVALUATING {model_type.upper()} MODEL")
        print("\n" + "=" * 60)
        print(f" EVALUATING {model_type.upper()} MODEL")
        print("=" * 60 + "\n")

        if adapter is not None:
            model, tokenizer, _ = load_model_and_tokenizer(
                base_model_name=base_model_name,
                adapter_path=adapter,
                merge=merge,
            )
        else:
            model, tokenizer, _ = load_model_and_tokenizer(base_model_name)
        model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)

        if sample_row is not None:
            _log_sample_prompts_and_replies(
                logger, model, tokenizer, sample_row,
                model_type=model_type,
                confidence_format=confidence_format,
                confidence_letter_mapping=confidence_letter_mapping,
                mcq_letter_mapping=mcq_letter_mapping,
            )

        try:
            metrics = evaluate_model(
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
                wandb_run_name=f"{dataset_name}_{model_type}",
                log_prefix=log_prefix,
                confidence_letter_scheme=confidence_letter_scheme,
                confidence_letter_random_seed=confidence_letter_random_seed,
                confidence_format=confidence_format,
                model_type=model_type,
            )
        except Exception:
            logger.section(f"ERROR DURING {model_type.upper()} EVAL")
            logger.write(traceback.format_exc())
            raise

        _print_letter_diagnostics(str(log_file), log_prefix=log_prefix, logger=logger)

        logger.subsection(f"{model_type} model summary metrics")
        logger.write(json.dumps(metrics, indent=2, default=str))

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.section("DATASET COMPLETE")
        logger.write(f"jsonl: {log_file}")
        acc = metrics.get("mcq_accuracy", float("nan"))
        ent = metrics.get("avg_entropy", float("nan"))
        logger.write(f"{model_type}  — accuracy={acc:.4f}  avg_entropy={ent:.4f}")
    finally:
        restore_warnings()

    return {
        "log_file": str(log_file),
        "txt_file": str(txt_file),
        "model_type": model_type,
        "metrics": metrics,
    }


###############################################################################
# Entry point — reads finetune_config.ECTConfig
###############################################################################
def main():
    cfg = ECTConfig

    print("=" * 60)
    print(" run_evaluations.py — config-driven (reads ECTConfig)")
    print("=" * 60)
    print(f" model_type:        {cfg.EVAL_MODEL_TYPE}")
    if cfg.EVAL_MODEL_TYPE == "finetuned":
        print(f" lora_repo:         {cfg.EVAL_LORA_REPO}")
    print(f" datasets:          {list(cfg.EVAL_DATASETS)}")
    print(f" sigma:             {cfg.SIGMA}              (shared with training)")
    print(f" loss_type:         {cfg.LOSS_TYPE}  (shared with training)")
    print(f" confidence_format: {cfg.CONFIDENCE_FORMAT}  (shared with training)")
    print(f" confidence_scheme: {cfg.CONFIDENCE_LETTER_SCHEME}            (shared with training, letter_8bin only)")
    print(f" mcq_scheme:        {cfg.MCQ_LETTER_SCHEME}            (shared with training)")
    print(f" max_samples:       {cfg.EVAL_MAX_SAMPLES}")
    print(f" log_dir:           {cfg.EVAL_LOG_DIR}")
    print("=" * 60)

    all_results = {}
    for dataset_path in cfg.EVAL_DATASETS:
        out = run_evaluation_on_dataset(
            model_type=cfg.EVAL_MODEL_TYPE,
            dataset_path=dataset_path,
            sigma=cfg.SIGMA,
            confidence_format=cfg.CONFIDENCE_FORMAT,
            merge=cfg.EVAL_MERGE_LORA,
            max_samples=cfg.EVAL_MAX_SAMPLES,
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
        acc = metrics.get("mcq_accuracy", float("nan"))
        ent = metrics.get("avg_entropy", float("nan"))
        print(f"\n{dataset_path}")
        print(f"  jsonl: {out['log_file']}")
        print(f"  txt:   {out['txt_file']}")
        print(f"  {out['model_type']}  — accuracy={acc:.4f}  avg_entropy={ent:.4f}")

    return all_results


if __name__ == "__main__":
    main()
