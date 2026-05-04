"""
Shared run-logging utilities for run_finetuning.py and run_evaluations.py.

Both scripts emit a paired .txt log next to their JSONL/checkpoint output that
contains:
  - The full ECTConfig snapshot (so the run is reconstructable from the .txt)
  - One sample of each prompt the model sees (MCQ / self-conf / other-conf),
    rendered with chat tags exactly as the tokenizer emits them
  - The model's actual greedy reply for each sample prompt
  - Any warnings raised during the run

Keeping these helpers in one module guarantees the two scripts produce
byte-identical prompt/reply blocks, so a finetune .txt and an eval .txt for
the same model+config can be diffed line-for-line.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

from prompts import (
    build_multiple_choice_question_prompts,
    build_other_confidence_prompts,
    build_other_confidence_prompts_numeric,
    build_self_confidence_prompts,
    build_self_confidence_prompts_numeric,
    build_delegate_abcdt_prompts,
    build_delegate_at_prompts,
    DEFAULT_TEAMMATE_ACCURACY,
)


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


def install_warning_capture(logger: RunLogger):
    """Route Python warnings into the run log AND to the original handler."""
    original = warnings.showwarning

    def capture(message, category, filename, lineno, file=None, line=None):
        logger.write(
            f"[WARNING {category.__name__}] {message}  ({filename}:{lineno})"
        )
        original(message, category, filename, lineno, file, line)

    warnings.showwarning = capture
    return lambda: setattr(warnings, "showwarning", original)


def dump_config_snapshot(logger: RunLogger, config_class, title: str) -> None:
    """Write every public attribute of a config class to the log so the run is
    fully reconstructable. Skips dunders and callables."""
    logger.section(title)
    rows = []
    for name in sorted(vars(config_class)):
        if name.startswith("_"):
            continue
        value = getattr(config_class, name)
        if callable(value):
            continue
        rows.append((name, value))
    width = max(len(n) for n, _ in rows) if rows else 0
    for name, value in rows:
        logger.write(f"  {name.ljust(width)} = {value!r}")


def generate_short_reply(model, tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    """Greedy-decode a short reply so the .txt can show what the model would
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


def log_sample_prompts_and_replies(
    logger: RunLogger,
    model,
    tokenizer,
    sample_row: dict,
    *,
    model_type: str,
    confidence_format: str,
    confidence_letter_mapping,
    mcq_letter_mapping,
    run_mcq: bool = True,
    run_self_confidence: bool = True,
    run_other_confidence: bool = True,
    run_delegate_abcdt: bool = False,
    run_delegate_at: bool = False,
    delegate_teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> None:
    """For one sample row, dump each prompt type (MCQ / self-conf / other-conf)
    exactly as the model sees it, then run a short generate() to capture the
    model's actual reply.

    Confidence prompt variants follow CONFIDENCE_FORMAT
    ("letter_8bin" → letter prompts, "1-5" / "1-10" → numeric).
    Self- and other-confidence always use the same scheme. Prompt formatting
    follows model_type ("base" → raw text, otherwise → chat tags).
    """
    qid = sample_row.get("qid") or sample_row.get("id")

    # MCQ
    if run_mcq:
        mcq_prompt = build_multiple_choice_question_prompts(
            [sample_row], tokenizer, mcq_letter_mapping, model_type=model_type
        )[0]
        logger.subsection(f"[{model_type}] sample MCQ prompt (qid={qid})")
        logger.write(mcq_prompt)
        logger.subsection(f"[{model_type}] sample MCQ reply (greedy, max_new_tokens=8)")
        logger.write(generate_short_reply(model, tokenizer, mcq_prompt))

    # Self- and other-confidence (same scheme)
    if run_self_confidence or run_other_confidence:
        if confidence_format in ("1-5", "1-10"):
            n_max = 5 if confidence_format == "1-5" else 10
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
                [sample_row], tokenizer, confidence_letter_mapping, mcq_letter_mapping,
                model_type=model_type,
            )[0]
            other_prompt = build_other_confidence_prompts(
                [sample_row], tokenizer, confidence_letter_mapping, mcq_letter_mapping,
                model_type=model_type,
            )[0]
            scheme_label = "letter A-H"

        if run_self_confidence:
            logger.subsection(f"[{model_type}] sample self-confidence ({scheme_label}) prompt")
            logger.write(self_prompt)
            logger.subsection(f"[{model_type}] sample self-confidence ({scheme_label}) reply")
            logger.write(generate_short_reply(model, tokenizer, self_prompt))

        if run_other_confidence:
            logger.subsection(f"[{model_type}] sample other-confidence ({scheme_label}) prompt")
            logger.write(other_prompt)
            logger.subsection(f"[{model_type}] sample other-confidence ({scheme_label}) reply")
            logger.write(generate_short_reply(model, tokenizer, other_prompt))

    # Delegate-game variants. Same fenced layout as MCQ/confidence so the
    # paired prompt blocks are directly comparable in the .txt.
    if run_delegate_abcdt:
        d_prompt = build_delegate_abcdt_prompts(
            [sample_row], tokenizer, mcq_letter_mapping,
            model_type=model_type,
            teammate_accuracy=delegate_teammate_accuracy,
        )[0]
        logger.subsection(f"[{model_type}] sample delegate-game ABCDT prompt")
        logger.write(d_prompt)
        logger.subsection(f"[{model_type}] sample delegate-game ABCDT reply")
        logger.write(generate_short_reply(model, tokenizer, d_prompt))

    if run_delegate_at:
        at_prompt = build_delegate_at_prompts(
            [sample_row], tokenizer, mcq_letter_mapping,
            model_type=model_type,
            teammate_accuracy=delegate_teammate_accuracy,
        )[0]
        logger.subsection(f"[{model_type}] sample delegate-game AT prompt")
        logger.write(at_prompt)
        logger.subsection(f"[{model_type}] sample delegate-game AT reply")
        logger.write(generate_short_reply(model, tokenizer, at_prompt))
