"""
Collect activations + logit lens + probes + directions for one Llama model.

This is the orchestrator: configuration aliasing, model loading once, and a
sweep over (dataset × meta_task). The heavy lifting is in sibling helpers:

  _io          — atomic writes, JSONL loading, output-path construction.
  _collection  — KV-cache-batched forward passes (direct + meta + other-conf).
  _probes      — Ridge / LogisticRegression / contrast direction extraction.
  _analysis    — calibration / behavioral analysis + reporting + plots.

To collect for base / instruct / finetuned, run this script three times after
editing `IntrospectionExperimentConfig.BASE_MODEL_NAME` + `MODEL_NAME` in
experiment_config.py. Output filenames already include a model + adapter tag,
so the three runs don't collide on disk.

Usage:
    python run_collect_activations.py --metric logit_gap   # Probe logit_gap
    python run_collect_activations.py --metric entropy     # Probe entropy (default)
"""

# --- repo path bootstrap so root-level imports resolve from anywhere ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _preflight import check_hf_login; check_hf_login()


import argparse
import datetime
import json
import random
import sys
import uuid

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from core.model_utils import DEVICE, is_base_model, has_chat_template, load_model_and_tokenizer
from prompts import (
    NUMERIC_CONFIDENCE_OPTIONS,
    format_delegate_prompt,
    format_direct_prompt,
    format_direct_prompt_base,
    format_meta_prompt,
    format_meta_prompt_base,
    format_other_confidence_prompt,
    get_meta_options,
    get_meta_signal,
    meta_task_type,
    option_token_id_groups,
    response_to_confidence,
)
from experiment_config import IntrospectionExperimentConfig as _C

from _analysis import (
    analyze_behavioral_introspection,
    analyze_other_confidence_control,
    compute_calibration_masks,
    plot_calibration_split,
    plot_other_confidence_comparison,
    plot_results,
    print_results,
    save_example_prompts_and_responses_txt,
    save_quick_summary_png,
    split_results_by_calibration,
)
from _collection import (
    BATCH_SIZE,
    _load_mc_data_for_reuse,
    collect_meta_only,
    collect_other_confidence,
    collect_paired_data,
)
from _runio import (
    _atomic_savez_compressed,
    _atomic_write_json,
    _pad_token_id_groups,
    _question_ids_array,
    ctx as _ctx,
    get_directions_prefix,
    get_output_prefix,
    load_questions,
    set_run_context,
)
from _logit_lens import apply_logit_lens
from _probes import (
    apply_probes_to_other,
    compute_contrast_directions,
    run_introspection_analysis,
    run_mc_answer_analysis,
)


# Seed everything once at import.
np.random.seed(_C.SEED)
torch.manual_seed(_C.SEED)
random.seed(_C.SEED)

# Make sure the top-level outputs directory exists; per-run subfolders are
# created lazily by `_io._run_dir()` when the first path is requested.
_C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Single-experiment orchestrator
# ============================================================================

def run_single_experiment(
    dataset_name: str,
    meta_task: str,
    model,
    tokenizer,
    num_layers: int,
    metric: str,
    batch_size: int,
    collect_other_activations: bool = False,
):
    """Collect activations + run all analyses for one (dataset, meta_task)."""

    # Per-iteration state — path helpers and prompt routing read this.
    num_questions = _C.NUM_QUESTIONS_BY_DATASET.get(
        dataset_name,
        _C.NUM_QUESTIONS_BY_DATASET.get(_io_dataset_short(dataset_name), _C.NUM_QUESTIONS_DEFAULT),
    )
    set_run_context(
        dataset_name=dataset_name,
        meta_task=meta_task,
        num_questions=num_questions,
    )

    # One run_id per (model, dataset, meta_task) invocation. Stamped into the
    # paired_data.json and the activation NPZ so a future loader can detect
    # cases where the JSON and NPZ came from different forward passes.
    run_id = uuid.uuid4().hex

    print("\n" + "=" * 80)
    print(f"Running: {dataset_name} / {meta_task} / {metric}")
    print(f"run_id: {run_id}")
    print("=" * 80)

    if meta_task == "delegate":
        print("\n--- Delegate Task Parameters ---")
        print(f"  prompt_design: {_C.DELEGATE_PROMPT_DESIGN}")
        if _C.DELEGATE_PROMPT_DESIGN == "mc_integrated":
            print("  Options: A/B/C/D (answer) or T (delegate) — single-shot")
        else:
            print("  Options: 1/2 (alternating mapping per trial)")
        print(f"  teammate_accuracy: {_C.TEAMMATE_ACCURACY:.0%}")
        print("")

    print(f"\nLoading {num_questions} questions from {dataset_name}...")
    questions = load_questions(dataset_name, num_questions)
    # Re-seed immediately before shuffle so the order is reproducible.
    random.seed(_C.SEED)
    random.shuffle(questions)
    print(f"Loaded {len(questions)} questions")

    is_base = is_base_model(_C.BASE_MODEL_NAME)
    use_chat_template = has_chat_template(tokenizer) and not is_base
    print(f"Using chat template: {use_chat_template}")
    if is_base:
        print(f"Base model detected — using few-shot mode: {_C.FEW_SHOT_MODE}")
        if meta_task == "delegate":
            print(f"Base delegate few-shot mode: {_C.BASE_DELEGATE_MODE}")

    # Optional: pool of examples for balanced base-model delegate few-shot.
    few_shot_pool = None
    if (
        is_base and meta_task == "delegate"
        and _C.BASE_DELEGATE_MODE == "balanced"
        and _C.BASE_DELEGATE_POOL_SOURCE
    ):
        print(f"Loading delegate few-shot pool from: {_C.BASE_DELEGATE_POOL_SOURCE}")
        with open(_C.BASE_DELEGATE_POOL_SOURCE, "r") as f:
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

    # Reuse direct activations from a previous confidence-mode run if requested.
    # Only meaningful for meta_task == 'delegate' (direct prompt is identical).
    reuse_direct = _C.REUSE_DIRECT_FROM_CONFIDENCE and meta_task == "delegate"
    if reuse_direct:
        # Build the confidence-mode prefix by temporarily flipping ctx.meta_task.
        saved_meta = _ctx.meta_task
        set_run_context(meta_task="confidence")
        confidence_prefix = get_output_prefix()
        set_run_context(meta_task=saved_meta)

        paired_path = _Path(f"{confidence_prefix}_paired_data.json")
        acts_path = _Path(f"{confidence_prefix}_direct_activations.npz")
        if not paired_path.exists() or not acts_path.exists():
            raise FileNotFoundError(
                "REUSE_DIRECT_FROM_CONFIDENCE=True but confidence-mode files are missing:\n"
                f"  {paired_path}\n  {acts_path}\n"
                "Run META_TASK='confidence' for this (model, dataset) first, or set the flag to False."
            )
        print(f"\nReusing direct activations from:\n  {paired_path}\n  {acts_path}")
        mc_data, existing_paired = _load_mc_data_for_reuse(paired_path, acts_path)
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
            few_shot_mode=_C.FEW_SHOT_MODE, few_shot_pool=few_shot_pool,
        )
    else:
        data = collect_paired_data(
            questions, model, tokenizer, num_layers, use_chat_template,
            batch_size=batch_size, is_base=is_base, few_shot_mode=_C.FEW_SHOT_MODE,
            few_shot_pool=few_shot_pool,
        )

    base_prefix = get_output_prefix()
    metric_prefix = get_output_prefix(metric)
    directions_prefix = get_directions_prefix(metric)
    print(f"Base output prefix: {base_prefix}")
    print(f"Metric output prefix: {metric_prefix}")
    print(f"Directions prefix: {directions_prefix}")

    direct_target = data["direct_metrics"][metric]

    # ------------------------------------------------------------------
    # Save activations (float16) + ALL scalar direct-task metrics.
    # Each NPZ also carries `run_id` and `question_ids` for alignment audit.
    # ------------------------------------------------------------------
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
        entropy=data["meta_entropies"],
        run_id=run_id_array,
        question_ids=qid_array,
    )
    print(f"Saved activations to {base_prefix}_*_activations.npz")

    # ------------------------------------------------------------------
    # Logit-lens projection on the saved last-token activations.
    # Two flavours — vanilla and tuned — controlled by LENS_TYPE.
    # ------------------------------------------------------------------
    first_q = questions[0]
    if is_base:
        _, direct_option_strs = format_direct_prompt_base(first_q, mode=_C.FEW_SHOT_MODE)
    else:
        _, direct_option_strs = format_direct_prompt(first_q, tokenizer, use_chat_template)
    direct_option_token_ids = option_token_id_groups(tokenizer, direct_option_strs)

    meta_option_strs = list(get_meta_options(
        meta_task=_ctx.meta_task,
        scale=_C.CONFIDENCE_SCALE,
        prompt_design=_C.DELEGATE_PROMPT_DESIGN,
    ))
    meta_option_token_ids = option_token_id_groups(tokenizer, meta_option_strs)

    def _save_lens_npz(path: str, lens_out: dict, option_strs, option_token_ids):
        _atomic_savez_compressed(
            path,
            option_logits=lens_out["option_logits"],
            top_k_ids=lens_out["top_k_ids"],
            top_k_logits=lens_out["top_k_logits"],
            layer_indices=lens_out["layer_indices"],
            option_strs=np.array(option_strs, dtype=object),
            option_token_ids=_pad_token_id_groups(option_token_ids),
            run_id=run_id_array,
            question_ids=qid_array,
        )

    # ----- Vanilla logit lens ----------------------------------------
    if _C.LENS_TYPE in ("vanilla", "both"):
        if not reuse_direct:
            print("\nComputing logit lens — vanilla (direct)...")
            direct_lens = apply_logit_lens(
                data["direct_activations"], model, direct_option_token_ids,
            )
            _save_lens_npz(f"{base_prefix}_direct_logit_lens.npz",
                           direct_lens, direct_option_strs, direct_option_token_ids)

        print("Computing logit lens — vanilla (meta)...")
        meta_lens = apply_logit_lens(
            data["meta_activations"], model, meta_option_token_ids,
        )
        _save_lens_npz(f"{base_prefix}_meta_logit_lens.npz",
                       meta_lens, meta_option_strs, meta_option_token_ids)
        print(f"Saved vanilla logit lens to {base_prefix}_*_logit_lens.npz")

    # ----- Tuned lens ------------------------------------------------
    if _C.LENS_TYPE in ("tuned", "both"):
        from _tuned_lens import (
            apply_tuned_lens, load_tuned_lens, save_tuned_lens, train_tuned_lens,
        )

        tuned_lens_path = f"{base_prefix}_tuned_lens.pt"
        if Path(tuned_lens_path).exists():
            print(f"\nLoading cached tuned lens from {tuned_lens_path}...")
            tuned_lens = load_tuned_lens(tuned_lens_path, device=DEVICE)
        else:
            print("\nTraining tuned lens (per-layer affine, KL-to-final-logits)...")
            # Calibration set = direct ∪ meta residuals so the affines
            # generalize across both prompt types we'll then lens.
            combined_acts = {
                layer_idx: np.concatenate(
                    [data["direct_activations"][layer_idx],
                     data["meta_activations"][layer_idx]],
                    axis=0,
                )
                for layer_idx in data["direct_activations"]
            }
            tuned_lens, train_losses = train_tuned_lens(
                model, combined_acts,
                n_epochs=_C.TUNED_LENS_EPOCHS,
                lr=_C.TUNED_LENS_LR,
                batch_size=_C.TUNED_LENS_BATCH_SIZE,
                weight_decay=_C.TUNED_LENS_WEIGHT_DECAY,
            )
            save_tuned_lens(tuned_lens, tuned_lens_path)
            print(f"Saved tuned-lens affines → {tuned_lens_path}")
            # Quick convergence summary so it's clear training worked.
            print("Tuned-lens training (initial → final mean loss per layer):")
            for layer_idx in sorted(train_losses.keys())[::4]:
                losses = train_losses[layer_idx]
                if losses:
                    print(f"  layer {layer_idx:>2d}: {losses[0]:.4f} → {losses[-1]:.4f}")

        if not reuse_direct:
            print("Computing logit lens — tuned (direct)...")
            direct_tuned = apply_tuned_lens(
                data["direct_activations"], model, tuned_lens, direct_option_token_ids,
            )
            _save_lens_npz(f"{base_prefix}_direct_tuned_logit_lens.npz",
                           direct_tuned, direct_option_strs, direct_option_token_ids)

        print("Computing logit lens — tuned (meta)...")
        meta_tuned = apply_tuned_lens(
            data["meta_activations"], model, tuned_lens, meta_option_token_ids,
        )
        _save_lens_npz(f"{base_prefix}_meta_tuned_logit_lens.npz",
                       meta_tuned, meta_option_strs, meta_option_token_ids)
        print(f"Saved tuned logit lens to {base_prefix}_*_tuned_logit_lens.npz")

    # Quick-look PNG before the slow analysis kicks in.
    try:
        save_quick_summary_png(data, questions, f"{base_prefix}_quick_summary.png")
    except Exception as e:
        print(f"⚠ quick-summary PNG failed: {e}")

    # ------------------------------------------------------------------
    # Render two example prompts (with chat tags) for the paired_data.json
    # — small enough to embed inline; useful for the analysis notebook.
    # ------------------------------------------------------------------
    def _render_direct_prompt(_q):
        if not is_base:
            return format_direct_prompt(_q, tokenizer, use_chat_template)
        try:
            return format_direct_prompt_base(_q, mode=_C.FEW_SHOT_MODE)
        except Exception as e:
            print(f"  (example-prompt fallback: direct MC '{_C.FEW_SHOT_MODE}' mode failed ({e}); using 'fixed')")
            return format_direct_prompt_base(_q, mode="fixed")

    example_prompts = []
    for i in range(min(2, len(questions))):
        q = questions[i]
        if is_base:
            direct_prompt, direct_options = _render_direct_prompt(q)
        else:
            direct_prompt, direct_options = format_direct_prompt(q, tokenizer, use_chat_template)
        if meta_task == "delegate":
            meta_prompt, meta_options_list, mapping = format_delegate_prompt(
                q, tokenizer, use_chat_template, trial_index=i,
                is_base=is_base,
                few_shot_mode=_C.BASE_DELEGATE_MODE if is_base else "fixed",
                few_shot_pool=few_shot_pool,
                teammate_accuracy=_C.TEAMMATE_ACCURACY,
                prompt_design=_C.DELEGATE_PROMPT_DESIGN,
            )
        else:
            if is_base:
                meta_prompt, meta_options_list = format_meta_prompt_base(q, mode=_C.FEW_SHOT_MODE, scale=_C.CONFIDENCE_SCALE)
            else:
                meta_prompt, meta_options_list = format_meta_prompt(q, tokenizer, use_chat_template, scale=_C.CONFIDENCE_SCALE)
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

    # ------------------------------------------------------------------
    # Other-confidence (control task — confidence-mode runs only).
    # ------------------------------------------------------------------
    other_data = None
    if meta_task == "confidence":
        print("\n" + "=" * 60)
        print("Running OTHER-CONFIDENCE control task...")
        print("(Asks model to estimate what % of college-educated people would know the answer)")
        print("=" * 60)

        other_data = collect_other_confidence(
            questions, model, tokenizer, num_layers, use_chat_template,
            batch_size=batch_size, collect_activations=collect_other_activations,
        )

        for i in range(min(2, len(questions))):
            q = questions[i]
            other_prompt, _ = format_other_confidence_prompt(q, tokenizer, use_chat_template)
            if i < len(example_prompts):
                example_prompts[i]["other_confidence_prompt"] = other_prompt

    # ------------------------------------------------------------------
    # `_examples.txt` — exact prompts + responses + full config snapshot.
    # ------------------------------------------------------------------
    try:
        save_example_prompts_and_responses_txt(
            data, questions, tokenizer,
            is_base=is_base, use_chat_template=use_chat_template,
            few_shot_mode=_C.FEW_SHOT_MODE,
            output_path=f"{base_prefix}_examples.txt",
            n_examples=10,
            delegate_pool=few_shot_pool,
            other_data=other_data,
        )
    except Exception as e:
        print(f"⚠ examples.txt dump failed: {e}")

    # ------------------------------------------------------------------
    # Derived per-example fields (responses, correctness, soft confidence).
    # ------------------------------------------------------------------
    options_keys = [list(q["options"].keys()) for q in questions]
    direct_responses = [
        opts[np.argmax(p)] if p else None
        for opts, p in zip(options_keys, data["direct_probs"])
    ]
    is_correct = [
        resp == q.get("correct_answer") if resp else False
        for resp, q in zip(direct_responses, questions)
    ]
    meta_mappings = data.get("meta_mappings") or [None] * len(data["meta_probs"])
    meta_responses = data["meta_responses"]
    _task_type = meta_task_type(meta_task=meta_task, scale=_C.CONFIDENCE_SCALE)
    stated_confidence_numeric = [
        response_to_confidence(resp, np.array(p), mapping, _task_type)
        for resp, p, mapping in zip(meta_responses, data["meta_probs"], meta_mappings)
    ]

    accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
    print(f"\nDirect accuracy: {accuracy:.1%} ({sum(is_correct)}/{len(is_correct)})")

    game_is_correct = None
    if meta_task == "delegate" and _C.DELEGATE_PROMPT_DESIGN == "mc_integrated":
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

    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    paired_data = {
        "direct_metrics": {k: v.tolist() for k, v in data["direct_metrics"].items()},
        "direct_probs": data["direct_probs"],
        "direct_logits": data.get("direct_logits", []),
        "meta_entropies": data["meta_entropies"].tolist(),
        "meta_probs": data["meta_probs"],
        "meta_responses": data["meta_responses"],
        "meta_mappings": data.get("meta_mappings"),
        "direct_responses": direct_responses,
        "is_correct": is_correct,
        "game_is_correct": game_is_correct,
        "stated_confidence_numeric": stated_confidence_numeric,
        "direct_extended": data.get("direct_extended", {}),
        "meta_extended": data.get("meta_extended", {}),
        "questions": [
            {
                "id": q.get("id", f"q_{i}"),
                "question": q.get("question", ""),
                "correct_answer": q.get("correct_answer", ""),
                "options": q.get("options", {}),
            }
            for i, q in enumerate(data["questions"])
        ],
        "example_prompts": example_prompts,
        "config": {
            "run_id": run_id,
            "reused_direct_from_run_id": (
                locals().get("mc_data", {}).get("_source_run_id") if reuse_direct else None
            ),
            "model_name": _C.MODEL_NAME,
            "base_model_name": _C.BASE_MODEL_NAME,
            "dataset_name": dataset_name,
            "num_questions": num_questions,
            "seed": _C.SEED,
            "meta_task": meta_task,
            "metric": metric,
            "is_base_model": is_base,
            "few_shot_mode": _C.FEW_SHOT_MODE if is_base else None,
            "confidence_scale": _C.CONFIDENCE_SCALE,
            "date_run": datetime.datetime.now().isoformat(),
            "git_commit": git_commit,
            "device": str(DEVICE),
            "activation_dtype": "float16",
            "tokenizer_name": getattr(tokenizer, "name_or_path", _C.BASE_MODEL_NAME),
            "delegate_params": {
                "prompt_design": _C.DELEGATE_PROMPT_DESIGN,
                "meta_option_keys": list(get_meta_options(
                    meta_task=meta_task,
                    scale=_C.CONFIDENCE_SCALE,
                    prompt_design=_C.DELEGATE_PROMPT_DESIGN,
                )),
                "decision_only": _C.DELEGATE_PROMPT_DESIGN == "two_step_digit",
                "alternate_decision_mapping": _C.DELEGATE_PROMPT_DESIGN == "two_step_digit",
                "use_phase1_summary": True,
                "use_phase1_history": False,
                "use_examples": True,
                "teammate_accuracy": float(_C.TEAMMATE_ACCURACY),
            } if meta_task == "delegate" else None,
        },
    }

    if other_data is not None:
        paired_data["other_confidence"] = {
            "probs": other_data["other_probs"],
            "responses": other_data["other_responses"],
            "signals": other_data["other_signals"].tolist(),
        }

    _atomic_write_json(f"{base_prefix}_paired_data.json", paired_data, indent=2)
    print(f"Saved paired data to {base_prefix}_paired_data.json (run_id={run_id})")

    collection_errors = data.get("errors", [])
    if collection_errors:
        errors_path = f"{base_prefix}_errors.json"
        with open(errors_path, "w") as f:
            json.dump(collection_errors, f, indent=2)
        print(f"Warning: {len(collection_errors)} errors logged to {errors_path}")

    # ------------------------------------------------------------------
    # Probe analysis (Ridge for entropy, LogisticRegression for MC answer)
    # ------------------------------------------------------------------
    print(f"\nRunning introspection analysis with metric: {metric}")
    results, test_idx, directions, entropy_probe_components = run_introspection_analysis(
        data["direct_activations"],
        data["meta_activations"],
        direct_target,
        extract_directions=True,
    )

    if directions is not None:
        directions_data = {f"layer_{l}": d for l, d in directions.items()}
        directions_data["_metadata_metric"] = np.array(metric)
        directions_data["_metadata_dataset"] = np.array(dataset_name)
        directions_data["_metadata_model"] = np.array(_C.BASE_MODEL_NAME)
        np.savez_compressed(f"{directions_prefix}_directions.npz", **directions_data)
        print(f"Saved {metric} directions to {directions_prefix}_directions.npz")

    direct_probs_array = np.array([p if p else [0.25, 0.25, 0.25, 0.25] for p in data["direct_probs"]])
    model_predicted_answer = np.argmax(direct_probs_array, axis=1)
    print(f"\nModel predicted answers: {len(model_predicted_answer)} questions")
    print(f"  Answer distribution: A={np.sum(model_predicted_answer==0)}, B={np.sum(model_predicted_answer==1)}, "
          f"C={np.sum(model_predicted_answer==2)}, D={np.sum(model_predicted_answer==3)}")

    n_questions = len(direct_target)
    indices = np.arange(n_questions)
    train_idx, _ = train_test_split(indices, train_size=_C.TRAIN_SPLIT, random_state=_C.SEED)

    mc_answer_results, mc_probe_components, mc_directions = run_mc_answer_analysis(
        data["direct_activations"],
        data["meta_activations"],
        model_predicted_answer,
        train_idx, test_idx,
    )

    best_mc_d2d_layer = max(mc_answer_results, key=lambda l: mc_answer_results[l]["d2d_accuracy"])
    best_mc_d2m_layer = max(mc_answer_results, key=lambda l: mc_answer_results[l]["d2m_accuracy"])
    print(f"\nMC Answer Probe Results:")
    print(f"  Best D→D: Layer {best_mc_d2d_layer} (acc={mc_answer_results[best_mc_d2d_layer]['d2d_accuracy']:.3f})")
    print(f"  Best D→M: Layer {best_mc_d2m_layer} (acc={mc_answer_results[best_mc_d2m_layer]['d2m_accuracy']:.3f})")
    print(f"  Chance: 0.250 (4-class)")

    mc_directions_prefix = get_directions_prefix(metric=None)
    mc_directions_data = {f"layer_{l}": d for l, d in mc_directions.items()}
    mc_directions_data["_metadata_metric"] = np.array("mc_answer")
    mc_directions_data["_metadata_dataset"] = np.array(dataset_name)
    mc_directions_data["_metadata_model"] = np.array(_C.BASE_MODEL_NAME)
    mc_directions_path = f"{mc_directions_prefix}_mc_answer_directions.npz"
    np.savez_compressed(mc_directions_path, **mc_directions_data)
    print(f"Saved MC answer directions to {mc_directions_path}")

    # ------------------------------------------------------------------
    # Contrast (mean-difference) directions per layer.
    # ------------------------------------------------------------------
    if _C.EXTRACT_CONTRAST_DIRECTIONS:
        print("\nComputing contrast (mean-difference) directions...")
        cap_str = (
            "all available"
            if _C.CONTRAST_SAMPLES_PER_BIN is None
            else f"capped at {_C.CONTRAST_SAMPLES_PER_BIN}"
        )

        # entropy contrast — DIRECT activations
        entropy_signal = np.asarray(data["direct_metrics"]["entropy"], dtype=np.float64)
        entropy_dirs, entropy_meta = compute_contrast_directions(
            data["direct_activations"], entropy_signal,
            percent=_C.CONTRAST_PERCENT_ENTROPY,
            samples_per_bin=_C.CONTRAST_SAMPLES_PER_BIN,
            seed=_C.SEED,
        )
        entropy_payload = {f"layer_{i}": d for i, d in entropy_dirs.items()}
        entropy_payload.update(entropy_meta)
        entropy_payload["signal_kind"] = np.array("entropy")
        entropy_payload["activation_source"] = np.array("direct")
        entropy_payload["dataset"] = np.array(dataset_name)
        entropy_payload["model"] = np.array(_C.BASE_MODEL_NAME)
        entropy_path = f"{mc_directions_prefix}_entropy_contrast_directions.npz"
        np.savez_compressed(entropy_path, **entropy_payload)
        print(
            f"  entropy ({_C.CONTRAST_PERCENT_ENTROPY:g}%, samples/bin {cap_str}): "
            f"low n={int(entropy_meta['n_low_used'])}/{int(entropy_meta['n_low'])} "
            f"(≤{float(entropy_meta['low_threshold']):.3f}) vs "
            f"high n={int(entropy_meta['n_high_used'])}/{int(entropy_meta['n_high'])} "
            f"(≥{float(entropy_meta['high_threshold']):.3f})"
        )
        print(f"  saved → {entropy_path}")

        # stated-confidence contrast — META activations (confidence task only)
        if meta_task == "confidence":
            sc_signal = np.asarray(stated_confidence_numeric, dtype=np.float64)
            valid_mask = ~np.isnan(sc_signal)
            min_required = int(np.ceil(200.0 / _C.CONTRAST_PERCENT_STATED_CONFIDENCE))
            if valid_mask.sum() < min_required:
                print(
                    f"  stated_confidence: skipped — only {int(valid_mask.sum())} valid "
                    f"values, need ≥{min_required} for percent={_C.CONTRAST_PERCENT_STATED_CONFIDENCE}"
                )
            else:
                valid_idx = np.where(valid_mask)[0]
                meta_acts_valid = {
                    l: a[valid_idx] for l, a in data["meta_activations"].items()
                }
                sc_dirs, sc_meta = compute_contrast_directions(
                    meta_acts_valid, sc_signal[valid_idx],
                    percent=_C.CONTRAST_PERCENT_STATED_CONFIDENCE,
                    samples_per_bin=_C.CONTRAST_SAMPLES_PER_BIN,
                    seed=_C.SEED,
                )
                sc_payload = {f"layer_{i}": d for i, d in sc_dirs.items()}
                sc_payload.update(sc_meta)
                sc_payload["signal_kind"] = np.array("stated_confidence")
                sc_payload["activation_source"] = np.array("meta")
                sc_payload["dataset"] = np.array(dataset_name)
                sc_payload["model"] = np.array(_C.BASE_MODEL_NAME)
                sc_payload["n_valid"] = np.array(int(valid_mask.sum()), dtype=np.int32)
                sc_path = f"{mc_directions_prefix}_stated_confidence_contrast_directions.npz"
                np.savez_compressed(sc_path, **sc_payload)
                print(
                    f"  stated_confidence ({_C.CONTRAST_PERCENT_STATED_CONFIDENCE:g}%, "
                    f"samples/bin {cap_str}): "
                    f"low n={int(sc_meta['n_low_used'])}/{int(sc_meta['n_low'])} "
                    f"(≤{float(sc_meta['low_threshold']):.3f}) vs "
                    f"high n={int(sc_meta['n_high_used'])}/{int(sc_meta['n_high'])} "
                    f"(≥{float(sc_meta['high_threshold']):.3f})"
                )
                print(f"  saved → {sc_path}")
        else:
            print(
                f"  stated_confidence: skipped — meta_task is {meta_task!r}, "
                "no stated-confidence signal in this run"
            )

    # ------------------------------------------------------------------
    # Behavioral + control analyses
    # ------------------------------------------------------------------
    behavioral = analyze_behavioral_introspection(
        data["meta_responses"], data["direct_metrics"][metric], test_idx,
        data["meta_probs"], data.get("meta_mappings"),
        data["direct_probs"], data["questions"],
    )

    other_confidence_analysis = None
    if meta_task == "confidence" and other_data is not None:
        self_confidence = np.array([
            get_meta_signal(np.array(p), scale=_C.CONFIDENCE_SCALE) if p else 0.5 for p in data["meta_probs"]
        ])
        other_confidence_analysis = analyze_other_confidence_control(
            other_data["other_signals"], self_confidence,
            data["direct_metrics"][metric], test_idx,
        )

    other_probe_results = None
    if meta_task == "confidence" and other_data is not None and "other_activations" in other_data:
        print("\n" + "=" * 60)
        print("Running OTHER-CONFIDENCE TRANSFER ANALYSIS...")
        print("=" * 60)
        other_probe_results = apply_probes_to_other(
            other_data["other_activations"], entropy_probe_components, mc_probe_components,
            direct_target, model_predicted_answer, test_idx,
        )
        best_entropy_layer = max(other_probe_results, key=lambda l: other_probe_results[l]["d2m_other_entropy_r2"])
        best_mc_layer = max(other_probe_results, key=lambda l: other_probe_results[l]["d2m_other_mc_accuracy"])
        print(f"\nDirect Probe Transfer to Other-Confidence:")
        print(f"  Best Entropy D→M(Other): Layer {best_entropy_layer} "
              f"(R²={other_probe_results[best_entropy_layer]['d2m_other_entropy_r2']:.3f})")
        print(f"  Best MC D→M(Other): Layer {best_mc_layer} "
              f"(acc={other_probe_results[best_mc_layer]['d2m_other_mc_accuracy']:.3f})")

    # ------------------------------------------------------------------
    # Calibration split + plots
    # ------------------------------------------------------------------
    stated_confidence = np.array(behavioral["stated_confidence"])
    test_confidence = stated_confidence[test_idx]
    test_metric = direct_target[test_idx]

    metric_is_uncertainty = metric in _C.UNCERTAINTY_METRICS
    calibrated_mask, miscalibrated_mask = compute_calibration_masks(
        test_confidence, test_metric, metric_is_uncertainty=metric_is_uncertainty,
    )

    print(f"\nCalibration split (median) for {metric}:")
    if metric_is_uncertainty:
        print(f"  Calibrated: {calibrated_mask.sum()} trials (high conf + low {metric}, or low conf + high {metric})")
        print(f"  Miscalibrated: {miscalibrated_mask.sum()} trials (high conf + high {metric}, or low conf + low {metric})")
    else:
        print(f"  Calibrated: {calibrated_mask.sum()} trials (high conf + high {metric}, or low conf + low {metric})")
        print(f"  Miscalibrated: {miscalibrated_mask.sum()} trials (high conf + low {metric}, or low conf + high {metric})")

    calibration_split = split_results_by_calibration(
        results, direct_target[test_idx], calibrated_mask, miscalibrated_mask,
    )

    if calibrated_mask.sum() > 1 and miscalibrated_mask.sum() > 1:
        best_cal = max(calibration_split, key=lambda l: calibration_split[l]["calibrated"]["d2m_r2"])
        best_mis = max(calibration_split, key=lambda l: calibration_split[l]["miscalibrated"]["d2m_r2"])
        print(f"\n  Calibrated best D→M: Layer {best_cal} "
              f"(R²={calibration_split[best_cal]['calibrated']['d2m_r2']:.3f})")
        print(f"  Miscalibrated best D→M: Layer {best_mis} "
              f"(R²={calibration_split[best_mis]['miscalibrated']['d2m_r2']:.3f})")

    plot_calibration_split(
        calibration_split,
        n_calibrated=int(calibrated_mask.sum()),
        n_miscalibrated=int(miscalibrated_mask.sum()),
        output_path=f"{metric_prefix}_calibration_split.png",
    )

    if other_probe_results is not None:
        plot_other_confidence_comparison(
            results, mc_answer_results, other_probe_results,
            output_path=f"{metric_prefix}_other_confidence_transfer.png",
        )

    # ------------------------------------------------------------------
    # Save results JSON + plot + tee'd .txt
    # ------------------------------------------------------------------
    results_to_save = {
        "config": {
            "metric": metric,
            "meta_task": meta_task,
            "model": _C.BASE_MODEL_NAME,
            "dataset": dataset_name,
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
                if not isinstance(v, dict) or k in [
                    "direct_to_direct", "direct_to_meta", "direct_to_meta_fixed",
                    "shuffled_baseline", "meta_to_meta",
                ]
            }
            for layer_idx, layer_results in results.items()
        },
        "behavioral": behavioral,
        "other_confidence_analysis": other_confidence_analysis,
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

    if other_probe_results is not None:
        results_to_save["other_confidence_transfer"] = {
            str(layer_idx): {
                "d2m_other_entropy_r2": layer_data["d2m_other_entropy_r2"],
                "d2m_other_mc_accuracy": layer_data["d2m_other_mc_accuracy"],
            }
            for layer_idx, layer_data in other_probe_results.items()
        }

    for layer_idx in results_to_save["probe_results"]:
        for key in ("direct_to_direct", "direct_to_meta", "direct_to_meta_fixed",
                    "shuffled_baseline", "meta_to_meta"):
            if key in results_to_save["probe_results"][layer_idx]:
                inner = results_to_save["probe_results"][layer_idx][key]
                for k, v in inner.items():
                    if isinstance(v, np.ndarray):
                        inner[k] = v.tolist()

    class _NumpyJSONEncoder(json.JSONEncoder):
        """Serialise stray np scalars/arrays from the analysis subtrees."""
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    with open(f"{metric_prefix}_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2, cls=_NumpyJSONEncoder)
    print(f"Saved results to {metric_prefix}_results.json")

    # Tee print_results to a .txt next to the .json.
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
        _orig_stdout = sys.stdout
        sys.stdout = _Tee(_orig_stdout, _fh)
        try:
            print_results(results, behavioral, other_confidence_analysis, metric=metric)
        finally:
            sys.stdout = _orig_stdout
    print(f"Saved results summary to {_results_txt_path}")

    plot_results(
        results, behavioral, direct_target, test_idx,
        output_path=f"{metric_prefix}_results.png",
        mc_answer_results=mc_answer_results,
    )

    print(f"\n✓ Introspection experiment complete! ({dataset_name} / {meta_task} / {metric})")


def _io_dataset_short(name: str) -> str:
    """Local re-import — avoids circular awkwardness when typing the cache lookup."""
    from _runio import _dataset_short_name
    return _dataset_short_name(name)


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run introspection collection")
    parser.add_argument(
        "--metric", type=str, default=_C.METRIC, choices=list(_C.AVAILABLE_METRICS),
        help=f"Uncertainty metric to probe (default: {_C.METRIC})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size for forward passes (default {BATCH_SIZE})",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", default=_C.LOAD_IN_4BIT,
        help=f"Load model in 4-bit quantization (default: {_C.LOAD_IN_4BIT})",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", default=_C.LOAD_IN_8BIT,
        help=f"Load model in 8-bit quantization (default: {_C.LOAD_IN_8BIT})",
    )
    parser.add_argument(
        "--collect-other-activations", action="store_true", default=False,
        help="Collect activations during other-confidence task for probe analysis",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Metric: {args.metric}")
    print(f"Datasets to process: {list(_C.DATASETS)}")
    print(f"Meta-tasks to process: {list(_C.META_TASKS)}")
    print(f"Total combinations: {len(_C.DATASETS) * len(_C.META_TASKS)}")

    print("\nLoading model (this will be shared across all experiments)...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        _C.BASE_MODEL_NAME,
        adapter_path=_C.MODEL_NAME if _C.MODEL_NAME != _C.BASE_MODEL_NAME else None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Sanity check: every numeric confidence option must tokenize to one token.
    if _C.CONFIDENCE_SCALE == "numeric":
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

    for dataset_name in _C.DATASETS:
        for meta_task in _C.META_TASKS:
            run_single_experiment(
                dataset_name=dataset_name,
                meta_task=meta_task,
                model=model,
                tokenizer=tokenizer,
                num_layers=num_layers,
                metric=args.metric,
                batch_size=args.batch_size,
                collect_other_activations=args.collect_other_activations,
            )

    print("\n" + "=" * 80)
    print("✓ All experiments complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
