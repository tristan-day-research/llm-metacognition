
# --- repo path bootstrap (so root-level imports like `prompts`,
# `finetune_config` resolve when run from anywhere) ---
import sys as _sys
import time as _time
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


# --- Pre-flight checks (before heavy imports so failures are instant) ---
def _check_hf_login():
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return  # will fail later with a clearer error
    if HfFolder.get_token() is None and not _os.environ.get("HF_TOKEN") and not _os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit(
            "\n✗  Not logged in to Hugging Face.\n"
            "   Run:  huggingface-cli login\n"
            "   or set the HF_TOKEN environment variable.\n"
        )

def _check_wandb_login():
    try:
        import wandb as _wandb
    except ImportError:
        return
    if not _wandb.api.api_key:
        raise SystemExit(
            "\n✗  Not logged in to Weights & Biases.\n"
            "   Run:  wandb login\n"
            "   or set the WANDB_API_KEY environment variable.\n"
        )

import os as _os
_check_hf_login()
# Only check wandb if we'll actually use it (read config before heavy imports)
try:
    from finetune_config import ECTConfig as _ECTConfigPreflight
    if _ECTConfigPreflight.SAVE_WANDB_ARTIFACT:
        _check_wandb_login()
except Exception:
    pass  # if config fails to import here, the real import below will surface the error

# Immediate startup banner so the user sees activity before the slow imports
# (torch/transformers/peft/wandb take 5-10 s combined on a cold cache, and
# load_tokenizer + load_model_with_lora add several more before train() ever
# prints). Without this, `python finetune/run_finetuning.py` looks frozen.
_STARTUP_T0 = _time.monotonic()
print(f"[{_dt.now(_tz.utc).strftime('%H:%M:%SZ')}] run_finetuning.py starting — importing heavy libs (torch/peft)…", flush=True)

import math
import numpy as np
import os
import sys
import torch
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from types import SimpleNamespace
# wandb is heavy and only needed when --save_wandb_artifact is set; imported lazily below.
print(f"[+{_time.monotonic()-_STARTUP_T0:5.1f}s] heavy imports done.", flush=True)

# Defaults live in finetune_config.ECTConfig — edit there to change behavior
# globally; CLI flags still override per-run.
from finetune_config import ECTConfig as _C

# Imports from helper files
from evaluation_metrics import (
      run_evaluation,
)
from utils import (
    save_model_final,
    save_checkpoint,
    save_training_parameters,
    load_tokenizer,
    load_model_with_lora,
    prepare_model_and_tokenizer,
    validate_train_batch
)
from loss import (
    build_soft_targets_from_entropy,
    build_soft_targets,
    compute_loss
)
from prompts import (
    build_self_confidence_prompts,
    build_self_confidence_prompts_numeric,
    build_multiple_choice_question_prompts,
    run_mcq_forward_pass,
    run_confidence_forward_pass,
    run_confidence_forward_pass_numeric,
    get_confidence_letter_mapping,
    get_mcq_letter_mapping,
)
from run_logging import (
    RunLogger,
    dump_config_snapshot,
    install_warning_capture,
    log_sample_prompts_and_replies,
)
from data_handling import (
    load_mcq_results_data,
    get_batch,
    load_jsonl_dataset,
    filter_dataset_by_mcq_results,
    validate_datasets_separate
)


# ------------------------------------------------------------------
# Training Step
# ------------------------------------------------------------------

def train_step(model, tokenizer, batch, device, sigma, args, mcq_results_lookup=None, 
               train_dataset_qids=None, train_dataset_questions=None):
    """
    Train step with support for frozen teacher (pre-recorded) or dynamic teacher.
    
    CRITICAL: This function should ONLY receive batches from train_dataset.
    Never pass validation data to this function.
    
    Args:
        batch: Batch from train_dataset (list of question dicts)
        mcq_results_lookup: Dict from load_mcq_results_data() or None
        train_dataset_qids: Set of qids from train_dataset for validation (optional)
        train_dataset_questions: Set of normalized question texts from train_dataset (optional)
    
    Returns:
        loss tensor, or None if batch should be skipped
    """
    # Defensive check: Verify batch contains only questions from train_dataset
    validate_train_batch(batch, train_dataset_qids, train_dataset_questions, function_name="train_step")
    
    # Ensure model is in training mode (set it explicitly to handle cases where
    # model might be in eval mode from previous evaluation)
    model.train()

    # ----------------------------------------------
    # 1. Get entropy, either from frozen or dynamic teacher)
    # ----------------------------------------------
    
    # Generate or get MCQ letter mapping (needed for confidence prompts regardless of teacher type)
    if args.randomize_letters_per_question:
        # Generate new mapping for this batch
        mcq_letter_mapping = get_mcq_letter_mapping(
            args.mcq_letter_scheme,
            seed=None  # Don't use seed for per-question randomization
        )
    else:
        # Use pre-generated mapping
        mcq_letter_mapping = args.mcq_letter_mapping

    # ----------------------------------------------
    # If using frozen teacher, gets multiple choice answers and the output logit entroy form pre-recorded responses
    # This is selected with --no_use_recorded_responses flag when running this file
    # ----------------------------------------------

    finetuning_target = getattr(args, "finetuning_target", "entropy")

    if args.use_recorded_responses:
        if mcq_results_lookup is None:
            raise ValueError(
                "--use_recorded_responses is True but no mcq_results_data provided!"
            )

        # Look up pre-recorded signals for each question in batch
        entropies = []
        top_probs = []
        second_probs = []
        valid_indices = []  # Track which samples in batch are valid

        for i, row in enumerate(batch):
            qid = row.get("qid")
            if qid and qid in mcq_results_lookup:
                record = mcq_results_lookup[qid]
                entropies.append(record["entropy"])
                # top_prob / second_prob derived from stored probs dict (A-D order)
                probs_list = sorted(record["probs"].values(), reverse=True)
                top_probs.append(probs_list[0])
                second_probs.append(probs_list[1] if len(probs_list) > 1 else 1e-12)
                valid_indices.append(i)
            else:
                # Skip this question
                print(f"⏭️  Skipping question without pre-recorded data: qid={qid}")

        # If no valid samples in batch, skip this training step
        if len(entropies) == 0:
            print(f"⚠️  Entire batch skipped - no pre-recorded data available")
            return None

        # Filter batch to only valid samples
        if len(valid_indices) < len(batch):
            batch = [batch[i] for i in valid_indices]
            print(f"  Batch reduced from {len(valid_indices)} to {len(batch)} samples")

        entropy = torch.tensor(entropies, dtype=torch.float32, device=device)
        top_prob = torch.tensor(top_probs, dtype=torch.float32, device=device)
        second_prob = torch.tensor(second_probs, dtype=torch.float32, device=device)

    # ----------------------------------------------
    # If using dynamic teacher: compute signals live from current model
    # This is  selected with --use_recorded_responses flag when  running this file
    # ----------------------------------------------

    else:
        # Use the MCQ letter mapping already defined above.
        # model_type="instruct" is passed explicitly so this call matches
        # run_evaluations.py byte-for-byte (LoRA is always trained on the
        # instruct base model, so the prompt is always chat-templated).
        mcq_prompts = build_multiple_choice_question_prompts(
            batch, tokenizer, mcq_letter_mapping, model_type="instruct"
        )

        mcq_out = run_mcq_forward_pass(
            model=model,
            tokenizer=tokenizer,
            prompts=mcq_prompts,
            device=device,
            temperature=0.0,
            requires_grad=True,  # KEEP GRADIENTS for dynamic teacher
            mcq_letter_mapping=mcq_letter_mapping,
        )

        entropy = mcq_out["entropy"]  # [B]
        probs4 = mcq_out["probs4"]   # [B, 4]
        sorted_probs = probs4.sort(dim=-1, descending=True).values
        top_prob = sorted_probs[:, 0]
        second_prob = sorted_probs[:, 1]

    # Convert to soft labels (size depends on confidence_format)
    confidence_format = getattr(args, "confidence_format", "letter_8bin")
    soft = build_soft_targets(
        finetuning_target,
        entropy=entropy,
        top_prob=top_prob,
        second_prob=second_prob,
        sigma=sigma,
        confidence_format=confidence_format,
    )


    # ----------------------------------------------
    # 2. Confidence forward pass
    # ----------------------------------------------

    if confidence_format == "letter_8bin":
        # Generate or get confidence letter mapping
        if args.randomize_letters_per_question:
            confidence_letter_mapping = get_confidence_letter_mapping(
                args.confidence_letter_scheme,
                seed=None
            )
        else:
            confidence_letter_mapping = args.confidence_letter_mapping

        conf_prompts = build_self_confidence_prompts(
            batch, tokenizer, confidence_letter_mapping, mcq_letter_mapping,
            model_type="instruct",
        )
        conf_out = run_confidence_forward_pass(
            model=model, tokenizer=tokenizer, prompts=conf_prompts,
            device=device, temperature=args.temperature, requires_grad=True,
            confidence_letter_mapping=confidence_letter_mapping,
        )
        conf_logits = conf_out["logits8"]  # [B, 8]
    elif confidence_format in ("1-5", "1-10"):
        n_max = 5 if confidence_format == "1-5" else 10
        conf_prompts = build_self_confidence_prompts_numeric(
            batch, tokenizer, mcq_letter_mapping, n_max=n_max,
            model_type="instruct",
        )
        conf_out = run_confidence_forward_pass_numeric(
            model=model, tokenizer=tokenizer, prompts=conf_prompts,
            device=device, temperature=args.temperature, requires_grad=True,
            n_max=n_max,
        )
        conf_logits = conf_out["logits"]  # [B, n_max]
    else:
        raise ValueError(f"Unknown confidence_format: {confidence_format!r}")

    # ----------------------------------------------
    # 3. Compute loss
    # ----------------------------------------------
    high_w = float(getattr(args, "high_entropy_loss_weight", 1.0) or 1.0)
    if high_w == 1.0:
        loss = compute_loss(
            conf_logits, soft_targets=soft, entropy=entropy,
            loss_type=args.loss_type, reduction='mean',
            confidence_format=confidence_format,
        )
    else:
        # Per-sample loss, then entropy-weighted mean. High-entropy samples
        # (entropy >= 2*ln(4)/3, matching the per-bin diagnostic definition)
        # get HIGH_ENTROPY_LOSS_WEIGHT; everything else gets 1.0.
        per_sample_loss = compute_loss(
            conf_logits, soft_targets=soft, entropy=entropy,
            loss_type=args.loss_type, reduction='none',
            confidence_format=confidence_format,
        )  # [B]
        high_threshold = 2.0 * math.log(4.0) / 3.0
        sample_w = torch.where(
            entropy >= high_threshold,
            torch.tensor(high_w, device=entropy.device, dtype=per_sample_loss.dtype),
            torch.tensor(1.0,    device=entropy.device, dtype=per_sample_loss.dtype),
        )
        loss = (per_sample_loss * sample_w).sum() / sample_w.sum().clamp_min(1e-8)

    # ----------------------------------------------
    # 4. Backprop
    # ----------------------------------------------
    loss.backward()

    # ----------------------------------------------
    # 5. Per-batch diagnostics: correlation between -entropy and expected
    #    confidence over the batch. Noisy at small batch sizes (n=batch_size),
    #    but useful as a directional signal alongside loss.
    # ----------------------------------------------
    step_metrics = {
        "n": int(entropy.shape[0]),
        "self_live_corr_spearman": 0.0,
        "self_live_corr_pearson": 0.0,
    }
    try:
        ent_np = entropy.detach().to("cpu").float().numpy()
        conf_np = conf_out["expected_conf"].detach().to("cpu").float().numpy()
        if ent_np.size > 1 and float(np.std(ent_np)) > 1e-6 and float(np.std(conf_np)) > 1e-6:
            from scipy.stats import spearmanr as _sp, pearsonr as _pe
            s, _ = _sp(-ent_np, conf_np)
            p, _ = _pe(-ent_np, conf_np)
            if not np.isnan(s):
                step_metrics["self_live_corr_spearman"] = float(s)
            if not np.isnan(p):
                step_metrics["self_live_corr_pearson"] = float(p)
    except Exception:
        pass

    return loss.detach(), step_metrics

    
# ============================================================
# Validation dispatch (single or dual frozen+live)
# ============================================================

def _run_validation(args, *, base_kwargs, primary_step_name):
    """Run validation eval(s). When args.val_run_both_frozen_and_live is True,
    runs frozen FIRST under the canonical val/* namespace (preserving existing
    W&B charts), then ADDITIONALLY runs the live pass under val_live/*.
    Otherwise runs the single mode in args.val_on_frozen under val/*.

    Returns: dict with key 'primary' (single mode) OR both 'frozen' and 'live'.
    The dict's values are run_evaluation result dicts.
    """
    if not getattr(args, "val_run_both_frozen_and_live", False):
        m = run_evaluation(
            **base_kwargs,
            step_name=primary_step_name,
            val_on_frozen=args.val_on_frozen,
            log_prefix="",
            wandb_prefix="val",
        )
        return {"primary": m}

    # Frozen keeps the original val/* keys so all existing dashboards keep
    # working unchanged. The live pass adds NEW keys under val_live/*.
    frozen = run_evaluation(
        **base_kwargs,
        step_name=f"{primary_step_name}_frozen",
        val_on_frozen=True,
        log_prefix="",
        wandb_prefix="val",
    )
    live = run_evaluation(
        **base_kwargs,
        step_name=f"{primary_step_name}_live",
        val_on_frozen=False,
        log_prefix="live_",
        wandb_prefix="val_live",
    )
    return {"frozen": frozen, "live": live}


def _print_val_summary(label, m):
    print(
        f"{label:>10s}  acc={m['mcq_accuracy']:.4f}  "
        f"loss={m['avg_loss']:.4f}  ent={m['avg_entropy']:.4f}  "
        f"conf={m['avg_confidence']:.4f}  n={m['n_samples']}"
    )


# ============================================================
# Main training
# ============================================================

def train(args):
    """
    Trainer for Expected Confidence Task (ECT).
    Uses:
        - run_mcq_forward_pass()
        - run_confidence_forward_pass()
        - train_step()
        - val_step()
        - run_evaluation()
    """
    # ----- Confirm key training knobs picked up from config -----
    # Useful sanity check that edits to finetune_config.py actually reach the
    # running process (catches stale __pycache__ / unreloaded notebook kernels).
    print(
        "\n[config] "
        f"sigma={args.sigma}  "
        f"loss_type={args.loss_type}  "
        f"finetuning_target={getattr(args, 'finetuning_target', 'entropy')}  "
        f"confidence_format={args.confidence_format}  "
        f"lr={args.learning_rate}  "
        f"batch_size={args.batch_size}  "
        f"max_steps={args.max_steps}  "
        f"max_grad_norm={args.max_grad_norm}  "
        f"high_entropy_loss_weight={getattr(args, 'high_entropy_loss_weight', 1.0)}  "
        f"lora_r={args.lora_r}  "
        f"lora_target_modules={tuple(args.lora_target_modules)}",
        flush=True,
    )


    # ============================================================
    # Setup / Load model and data
    # ============================================================
    device = args.device
    print(f"[+{_time.monotonic()-_STARTUP_T0:5.1f}s] loading tokenizer ({args.model_name})…", flush=True)
    tokenizer = load_tokenizer(args)
    print(f"[+{_time.monotonic()-_STARTUP_T0:5.1f}s] loading model + LoRA adapter onto {device}…", flush=True)
    model = load_model_with_lora(args, tokenizer).to(device)
    print(f"[+{_time.monotonic()-_STARTUP_T0:5.1f}s] model ready.", flush=True)

    # Canonicalize model/tokenizer setup (fix pad_token warnings)
    model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)

    # Setup log file path early (needed for duplicate removal logging)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = str(_C.LOGS_DIR)
    os.makedirs(log_dir, exist_ok=True)
    model_name_safe = args.model_name.replace("/", "-").replace("_", "-")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{model_name_safe}_evaluation_metrics.jsonl")

    # Structured .txt run record (mirrors the one run_evaluations.py writes).
    # Same RunLogger / log_sample_prompts_and_replies helpers as eval, so the
    # config snapshot + sample-prompt block is byte-identical between
    # finetuning and eval runs.
    run_record_path = os.path.join(log_dir, f"{timestamp}_{model_name_safe}_finetune.txt")
    run_logger = RunLogger(run_record_path)
    restore_warnings = install_warning_capture(run_logger)
    run_logger.write(f"run_finetuning.py — {timestamp}Z")
    run_logger.write(f"model_name:        {args.model_name}")
    run_logger.write(f"train_data_path:   {args.train_data_path}")
    run_logger.write(f"val_data_path:     {args.val_data_path}")
    run_logger.write(f"test_data_path:    {args.test_data_path}")
    run_logger.write(f"confidence_format: {args.confidence_format}")
    run_logger.write(f"jsonl out:         {log_file_path}")
    dump_config_snapshot(run_logger, _C, "CONFIG SNAPSHOT (finetune_config.ECTConfig)")

    # Setup print log file to capture all printed output
    print_log_path = os.path.join(log_dir, f"{timestamp}_{model_name_safe}_print_output.txt")
    print_log_file = open(print_log_path, 'w', encoding='utf-8')
    
    # Write header to file immediately to ensure it's created
    header = f"Training Run Print Output Log\n"
    header += f"Started: {timestamp}\n"
    header += f"Model: {args.model_name}\n"
    header += f"{'='*80}\n\n"
    print_log_file.write(header)
    print_log_file.flush()
    
    # Create a custom print function that writes to both console and file
    import builtins
    original_print = builtins.print
    def logged_print(*args, **kwargs):
        """Print that writes to both console and log file."""
        # If file parameter is specified, use it; otherwise print to console
        file_param = kwargs.pop('file', None)
        if file_param is not None:
            # User specified a file, print to that file
            original_print(*args, file=file_param, **kwargs)
        else:
            # No file specified, print to console
            original_print(*args, **kwargs)
        
        # Always also write to log file (unless it's the log file itself to avoid recursion)
        if file_param is not print_log_file:
            original_print(*args, file=print_log_file, **kwargs)
            print_log_file.flush()  # Ensure immediate write
    
    # Replace built-in print with logged version
    builtins.print = logged_print
    
    # Print training command at the very beginning (will be logged)
    training_command = " ".join(sys.argv)
    print("\n" + "="*80)
    print("TRAINING COMMAND:")
    print("="*80)
    print(training_command)
    print("="*80 + "\n")
    
    # Print location of log file (this will also be logged)
    print(f"✓ Print output will be logged to: {print_log_path}")

    # Dataset loading ------------------------------------------------
    # CRITICAL: Load train and val datasets separately to prevent any mixing
    train_dataset = load_jsonl_dataset(args.train_data_path, dataset_type="train")
    val_dataset   = load_jsonl_dataset(args.val_data_path, dataset_type="val")

    print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")

    # Load pre-recorded MCQ results — three modes:
    #   (a) external file via --mcq_results_data (legacy path)
    #   (b) inline: rows in the loaded train/val datasets carry the recorded
    #       entropy/answer/probs/options themselves (e.g. the balanced
    #       metacognition dataset). Triggered by leaving mcq_results_data
    #       unset while use_recorded_responses or val_on_frozen is True.
    #   (c) neither: live teacher (no frozen lookup).
    mcq_results_lookup = None
    needs_frozen = args.use_recorded_responses or args.val_on_frozen
    if args.mcq_results_data is not None:
        mcq_results_lookup = load_mcq_results_data(args.mcq_results_data)
        if mcq_results_lookup is None:
            if needs_frozen:
                raise ValueError(f"Failed to load MCQ results from {args.mcq_results_data}")
            else:
                print(f"⚠️  Warning: Could not load MCQ results from {args.mcq_results_data}, continuing without pre-recorded entropy logging")

        # Filter datasets to only include questions with pre-recorded results
        if args.use_recorded_responses:
            train_dataset = filter_dataset_by_mcq_results(
                train_dataset, mcq_results_lookup, dataset_name="training"
            )
            val_dataset = filter_dataset_by_mcq_results(
                val_dataset, mcq_results_lookup, dataset_name="validation"
            )
        if args.val_on_frozen:
            val_dataset = filter_dataset_by_mcq_results(
                val_dataset, mcq_results_lookup, dataset_name="validation"
            )
    elif needs_frozen:
        # Inline mode: build the lookup from the dataset rows themselves.
        # Requires each row to carry qid + entropy + model_answer + probs_ABCD
        # + options. The balanced dataset emits all of these.
        from data_handling import build_mcq_results_lookup_from_rows
        merged = list(train_dataset) + list(val_dataset)
        mcq_results_lookup = build_mcq_results_lookup_from_rows(merged, name="train+val")
        if not mcq_results_lookup:
            raise ValueError(
                "use_recorded_responses=True / val_on_frozen=True with no "
                "MCQ_RESULTS_DATA file, but no train/val rows carried the "
                "required inline fields (qid, entropy, model_answer, "
                "probs_ABCD, options). Either supply MCQ_RESULTS_DATA or "
                "regenerate the dataset so each row has those fields."
            )

    print(f"\n✓ Training dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
    
    # Initialize validation sets (will be None if checks are disabled)
    train_dataset_qids = None
    train_dataset_questions = None
    
    # CRITICAL: Validate that train and val datasets are completely separate
    # This checks for duplicates by BOTH qid and question text
    # Location: data_handling.py:validate_datasets_separate()
    # - Line 642: Checks for overlapping qids
    # - Line 652: Checks for overlapping question text (normalized)
    if args.enable_data_leakage_checks:
        try:
            validate_datasets_separate(train_dataset, val_dataset, "train", "val")
        except ValueError as e:
            # If validation fails, offer to auto-fix by removing duplicates from train set
            error_msg = str(e)
            if "DATA LEAKAGE DETECTED" in error_msg:
                print("\n" + "="*70)
                print("DATA LEAKAGE DETECTED - Attempting automatic fix...")
                print("="*70)
                print(f"Error triggered at: data_handling.py:validate_datasets_separate()")
                print(f"  - Checks for overlapping qids (line ~642)")
                print(f"  - Checks for overlapping question text (line ~652)")
                print("="*70)
                
                from data_handling import find_and_remove_duplicates
                
                # Use the evaluation log file if available, otherwise None
                removal_summary, train_dataset_cleaned = find_and_remove_duplicates(
                    train_dataset, val_dataset, remove_from="train", log_file_path=log_file_path
                )
                
                if removal_summary["total_removed"] > 0:
                    train_dataset = train_dataset_cleaned
                    print(f"\n✓ Automatically removed {removal_summary['total_removed']} duplicate(s) from training dataset")
                    print(f"  Breakdown: {removal_summary['by_qid_only']} by qid, "
                          f"{removal_summary['by_text_only']} by text, "
                          f"{removal_summary['by_both']} by both")
                    print(f"  Training dataset: {len(train_dataset)} samples")
                    
                    # Re-validate to ensure fix worked
                    print("\nRe-validating dataset separation...")
                    validate_datasets_separate(train_dataset, val_dataset, "train", "val")
                    print("✓ Dataset separation validated after auto-fix")
                else:
                    # If auto-fix didn't work, re-raise the original error
                    raise e
            else:
                # Re-raise if it's a different ValueError
                raise e
        
        # Build sets for runtime validation (both qids and normalized question text)
        # Do this AFTER validation/auto-fix to ensure we have the final train_dataset
        from data_handling import normalize_text
        train_dataset_qids = {str(row.get("qid")) for row in train_dataset if row.get("qid")}
        train_dataset_questions = {normalize_text(row.get("question", "")) for row in train_dataset if row.get("question")}
        print(f"✓ Built train_dataset validation sets: {len(train_dataset_qids)} qids, {len(train_dataset_questions)} questions")
        print(f"✓ Runtime leakage checks ENABLED: train_step() and run_evaluation() will validate every batch/sample")
    else:
        print(f"⚠️  Data leakage checks DISABLED (not recommended)")

    if args.use_recorded_responses:
        print(f"✓ Using FROZEN TEACHER (pre-recorded responses)")
    else:
        print(f"✓ Using DYNAMIC TEACHER (current model)")
    
    if args.val_on_frozen:
        print(f"✓ Validation mode: FROZEN (pre-recorded MCQ answers and entropy)")
    else:
        print(f"✓ Validation mode: LIVE (current model's MCQ answers and entropy)")
    
    # Generate letter mappings (only if NOT randomizing per question)
    if args.randomize_letters_per_question:
        # Don't generate mappings here - will be generated per question/batch
        args.confidence_letter_mapping = None
        args.mcq_letter_mapping = None
        print(f"✓ Letter randomization mode: PER QUESTION (will randomize for each question/batch)")
    else:
        # Generate once at the beginning and reuse
        confidence_letter_mapping = get_confidence_letter_mapping(
            args.confidence_letter_scheme,
            seed=args.confidence_letter_random_seed
        )
        args.confidence_letter_mapping = confidence_letter_mapping
        display_letters = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
        print(f"✓ Confidence letter scheme: {args.confidence_letter_scheme} -> {''.join(display_letters)}")
        
        # Generate MCQ letter mapping and store in args for use in train_step
        mcq_letter_mapping = get_mcq_letter_mapping(
            args.mcq_letter_scheme,
            seed=args.mcq_letter_random_seed
        )
        args.mcq_letter_mapping = mcq_letter_mapping
        mcq_display_letters = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
        print(f"✓ MCQ letter scheme: {args.mcq_letter_scheme} -> {''.join(mcq_display_letters)}")
        print(f"✓ Letter randomization mode: ONCE AT START (same mapping for all questions)")

    # Optimizer ------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0
    )

    # Logging --------------------------------------------------------
    # Capture WandB run info for checkpoint naming
    wandb_run = None
    wandb_run_id = None
    wandb_run_name = None
    wandb_local_run_dir = None
    wandb_init_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    
    if args.save_wandb_artifact:
        import wandb
        # Auto-generate run name with current date if not provided
        if args.wandb_run_name is None:
            model_name_safe = args.model_name.replace("/", "-").replace("_", "-")
            args.wandb_run_name = f"{wandb_init_timestamp}_{model_name_safe}_ect"
        else:
            # Prepend date to provided name to ensure it's current
            args.wandb_run_name = f"{wandb_init_timestamp}_{args.wandb_run_name}"

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        wandb_run_id = wandb_run.id
        wandb_run_name = wandb_run.name
        # Capture the local run dir so we can delete it after wandb.finish()
        # (wandb_run.dir points at .../files/; the parent is the per-run dir).
        wandb_local_run_dir = os.path.dirname(wandb_run.dir) if wandb_run.dir else None
        print(f"✓ WandB run initialized: {wandb_run_name} (ID: {wandb_run_id})")

    # Output / checkpoints
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Setup checkpoint directory — use the (timestamp-prefixed) WandB run name
    # so the same identifier appears both on disk and in W&B / HF. Falls back
    # to `{timestamp}_checkpoints` when no run name is available.
    from utils import _sanitize_for_hf
    if args.wandb_run_name:
        checkpoint_dir_name = _sanitize_for_hf(args.wandb_run_name)
    else:
        checkpoint_dir_name = f"{timestamp}_checkpoints"
    checkpoint_base_dir = os.path.join(str(_C.CHECKPOINTS_DIR), checkpoint_dir_name)
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    print(f"✓ Checkpoints will be saved to: {os.path.abspath(checkpoint_base_dir)}")
    
    # Save training parameters to checkpoint directory
    save_training_parameters(args, checkpoint_base_dir)

    # ============================================================
    # Sample prompts + actual model replies — written to the structured
    # .txt run record so we can verify the model sees the exact same prompt
    # (chat tags included) that run_evaluations.py uses. Routes through the
    # shared run_logging helper, which dispatches to numeric vs letter
    # confidence prompts based on CONFIDENCE_FORMAT.
    # ============================================================
    if args.randomize_letters_per_question:
        display_mcq_mapping = get_mcq_letter_mapping(args.mcq_letter_scheme, seed=None)
        display_conf_mapping = get_confidence_letter_mapping(args.confidence_letter_scheme, seed=None)
    else:
        display_mcq_mapping = mcq_letter_mapping
        display_conf_mapping = confidence_letter_mapping

    run_logger.section("SAMPLE PROMPTS + REPLIES (pre-training)")
    log_sample_prompts_and_replies(
        run_logger, model, tokenizer, train_dataset[0],
        model_type="instruct",
        confidence_format=args.confidence_format,
        confidence_letter_mapping=display_conf_mapping,
        mcq_letter_mapping=display_mcq_mapping,
    )
    print(f"✓ Sample prompts + replies written to: {run_record_path}")

    # ============================================================
    # Baseline evaluation BEFORE training
    # ============================================================
    print("\n" + "="*60)
    print("Running baseline validation (before training)...")
    print("="*60)

    baseline_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=0,
        mcq_results_lookup=mcq_results_lookup,
        train_dataset_qids=train_dataset_qids,
        train_dataset_questions=train_dataset_questions,
        sigma=args.sigma,
    )
    baseline_results = _run_validation(args, base_kwargs=baseline_kwargs, primary_step_name="baseline")
    print()
    if "primary" in baseline_results:
        _print_val_summary("baseline", baseline_results["primary"])
        baseline_metrics = baseline_results["primary"]
    else:
        _print_val_summary("frozen", baseline_results["frozen"])
        _print_val_summary("live",   baseline_results["live"])
        # Use frozen as the canonical baseline_metrics for downstream code that
        # expects a single dict (none does, but this matches the prior behavior
        # when VAL_ON_FROZEN was True).
        baseline_metrics = baseline_results["frozen"]
    print()


    # ============================================================
    # TRAINING LOOP
    # ============================================================

    step = 0
    losses = []

    while step < args.max_steps:
        # CRITICAL: Only use train_dataset for training batches
        batch = get_batch(train_dataset, args.batch_size, is_training=True)

        # -----------------------------
        # Train step 
        # -----------------------------
        optimizer.zero_grad()

        ts_out = train_step(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            sigma=args.sigma,
            device=device,
            args=args,
            mcq_results_lookup=mcq_results_lookup,
            train_dataset_qids=train_dataset_qids,
            train_dataset_questions=train_dataset_questions,
        )

        # Skip optimizer step if batch was skipped
        if ts_out is None or ts_out[0] is None:
            continue  # Don't increment step, try next batch

        loss, step_metrics = ts_out

        # ----- Non-finite guards (loss + gradients) -----
        # CRITICAL ordering note: we must catch NaN/Inf *gradients* BEFORE
        # optimizer.step(), because:
        #   - clip_grad_norm_ does NOT remove NaN — ‖NaN‖ = NaN, so the clip
        #     rescales by NaN and preserves the corruption.
        #   - One NaN update poisons the weights permanently; every later
        #     forward pass yields NaN logits → NaN loss → unrecoverable.
        # Checking only the loss after-the-fact means the poisoning has
        # already happened.
        skip_step = False
        if not torch.isfinite(loss):
            print(f"⚠️  Non-finite loss at step {step} (loss={loss.item()}); skipping.", flush=True)
            skip_step = True
        else:
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            for p in trainable:
                if not torch.isfinite(p.grad).all():
                    print(f"⚠️  Non-finite gradient at step {step}; skipping (weights preserved).", flush=True)
                    skip_step = True
                    break

        if skip_step:
            optimizer.zero_grad()
            step += 1
            continue

        # Gradient clipping. Grads were populated by loss.backward() inside
        # train_step. Clip BEFORE optimizer.step() so the actual update uses
        # the clipped values.
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm,
            )
            step_metrics["grad_norm"] = float(grad_norm)

        # Defensive: also abort if any LoRA *parameter* is already non-finite
        # (means we got poisoned despite the guards above — only here so we
        # don't silently train on a NaN model).
        for p in model.parameters():
            if p.requires_grad and not torch.isfinite(p).all():
                raise RuntimeError(
                    f"Non-finite LoRA weights detected at step {step}. "
                    f"Training is unrecoverable; consider lowering LEARNING_RATE "
                    f"(current={args.learning_rate}) or MAX_GRAD_NORM "
                    f"(current={args.max_grad_norm})."
                )

        optimizer.step()

        losses.append(loss.item())

        # Per-train-step correlation print (also captured into print_output.txt)
        print(
            f"[train step {step}] loss={loss.item():.4f}  "
            f"corr(-entropy, self_conf) over batch n={step_metrics['n']}: "
            f"ρ={step_metrics['self_live_corr_spearman']:+.4f}  "
            f"r={step_metrics['self_live_corr_pearson']:+.4f}",
            flush=True,
        )

        # W&B logging
        if args.save_wandb_artifact:
            import wandb
            log_payload = {
                "train/loss": loss.item(),
                "train/self_live_corr_spearman": step_metrics["self_live_corr_spearman"],
                "train/self_live_corr_pearson": step_metrics["self_live_corr_pearson"],
                "train/batch_n": step_metrics["n"],
                "step": step,
            }
            if "grad_norm" in step_metrics:
                log_payload["train/grad_norm"] = step_metrics["grad_norm"]
            wandb.log(log_payload)

        # -----------------------------
        # Periodic validation (Validation Step)
        # -----------------------------
        if (step % args.val_interval) == 0 and step > 0:
            print("\n" + "="*60)
            print(f"Validation at step {step}")
            print("="*60)

            val_kwargs = dict(
                model=model,
                tokenizer=tokenizer,
                val_dataset=val_dataset,
                device=device,
                args=args,
                num_samples=args.val_num_samples,
                log_file_path=log_file_path,
                step=step,
                mcq_results_lookup=mcq_results_lookup,
                train_dataset_qids=train_dataset_qids,
                train_dataset_questions=train_dataset_questions,
                sigma=args.sigma,
            )
            val_results = _run_validation(args, base_kwargs=val_kwargs, primary_step_name="validation")
            print()
            if "primary" in val_results:
                _print_val_summary("val", val_results["primary"])
            else:
                _print_val_summary("val/frozen", val_results["frozen"])
                _print_val_summary("val/live",   val_results["live"])

        # -----------------------------
        # Periodic checkpointing
        # -----------------------------
        if (step % args.checkpoint_steps) == 0 and step > 0:
            # Generate timestamp for this checkpoint
            checkpoint_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_base_dir=checkpoint_base_dir,
                step=step,
                save_hf_checkpoints=args.save_hf_checkpoints,
                hf_checkpoint_repo=args.hf_checkpoint_repo,
                hf_checkpoint_private=args.hf_checkpoint_private,
                wandb_run_name=wandb_run_name,
                wandb_run_id=wandb_run_id,
                checkpoint_timestamp=checkpoint_timestamp,
                keep_local=args.keep_local_checkpoints,
            )

        step += 1

    # ============================================================
    # Final metrics
    # ============================================================
    print("\n" + "="*60)
    print("Final evaluation:")
    print("="*60)

    final_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=step,
        mcq_results_lookup=mcq_results_lookup,
        train_dataset_qids=train_dataset_qids,
        train_dataset_questions=train_dataset_questions,
        sigma=args.sigma,
    )
    final_results = _run_validation(args, base_kwargs=final_kwargs, primary_step_name="final")
    print()
    if "primary" in final_results:
        _print_val_summary("final", final_results["primary"])
        final_metrics = final_results["primary"]
    else:
        _print_val_summary("final/frozen", final_results["frozen"])
        _print_val_summary("final/live",   final_results["live"])
        # Downstream summary fields use a single dict; pick frozen as canonical.
        final_metrics = final_results["frozen"]

    # Generate timestamp for final model save
    final_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print("Saving model.")
    success = save_model_final(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        hf_repo=args.hf_repo if args.save_hf else None,
        hf_private=args.hf_checkpoint_private,
        save_wandb_artifact=args.save_wandb_artifact,
        wandb_run_name=wandb_run_name,
        wandb_run_id=wandb_run_id,
        step=step,
        final_timestamp=final_timestamp,
        keep_local=args.keep_local_checkpoints,
    )
    if success:
        print("✓ Model saved successfully!")
    else:
        print("❌ Model save failed! Check error messages above.")
        raise RuntimeError("Failed to save model")
    
    # Finish wandb run after model is saved (so artifact can be logged).
    # wandb.finish() blocks until all metrics + artifacts are uploaded, so
    # the local per-run directory is safe to delete afterwards.
    if args.save_wandb_artifact:
        import wandb
        import shutil
        wandb.finish()
        if wandb_local_run_dir and os.path.isdir(wandb_local_run_dir):
            try:
                shutil.rmtree(wandb_local_run_dir)
                print(f"✓ Removed local W&B run dir: {wandb_local_run_dir}")
            except Exception as _e:
                print(f"⚠️  Could not delete W&B run dir {wandb_local_run_dir}: {_e}")
    
    # Final summary into the structured .txt run record, then unhook the
    # warnings handler so we don't leak it past this run.
    run_logger.section("FINAL METRICS")
    run_logger.write(f"final_accuracy:   {final_metrics['mcq_accuracy']:.4f}")
    run_logger.write(f"final_loss:       {final_metrics['avg_loss']:.4f}")
    run_logger.write(f"final_confidence: {final_metrics['avg_confidence']:.4f}")
    restore_warnings()

    # Restore original print and close print log file
    import builtins
    builtins.print = original_print
    print_log_file.close()
    print(f"✓ Print output saved to: {print_log_path}")
    print(f"✓ Run record saved to:   {run_record_path}")
    


def build_args_from_config():
    """Build a training-args namespace directly from ECTConfig.

    No CLI: every parameter lives in finetune_config.ECTConfig. Edit there
    to change a run. Returned object is a SimpleNamespace because train()
    also assigns onto it (e.g. confidence_letter_mapping).
    """
    return SimpleNamespace(
        # Model
        model_name=_C.MODEL_NAME,
        device=_C.DEVICE,
        # Data
        train_data_path=_C.TRAIN_DATA_PATH,
        val_data_path=_C.VAL_DATA_PATH,
        test_data_path=_C.TEST_DATA_PATH,
        batch_size=_C.BATCH_SIZE,
        mcq_results_data=_C.MCQ_RESULTS_DATA,
        # LoRA
        lora_r=_C.LORA_R,
        lora_alpha=_C.LORA_ALPHA,
        lora_dropout=_C.LORA_DROPOUT,
        lora_target_modules=list(_C.LORA_TARGET_MODULES),
        # Training
        learning_rate=_C.LEARNING_RATE,
        max_steps=_C.MAX_STEPS,
        max_grad_norm=_C.MAX_GRAD_NORM,
        high_entropy_loss_weight=_C.HIGH_ENTROPY_LOSS_WEIGHT,
        log_interval=_C.LOG_INTERVAL,
        val_interval=_C.VAL_INTERVAL,
        limit_val_batches=_C.LIMIT_VAL_BATCHES,
        val_num_samples=_C.VAL_NUM_SAMPLES,
        sigma=_C.SIGMA,
        loss_type=_C.LOSS_TYPE,
        finetuning_target=_C.FINETUNING_TARGET,
        temperature=_C.TEMPERATURE,
        shuffle_options=_C.SHUFFLE_OPTIONS,
        use_recorded_responses=_C.USE_RECORDED_RESPONSES,
        enable_data_leakage_checks=_C.ENABLE_DATA_LEAKAGE_CHECKS,
        val_on_frozen=_C.VAL_ON_FROZEN,
        val_run_both_frozen_and_live=_C.VAL_RUN_BOTH_FROZEN_AND_LIVE,
        confidence_format=_C.CONFIDENCE_FORMAT,
        compute_other_confidence=_C.COMPUTE_OTHER_CONFIDENCE,
        confidence_letter_scheme=_C.CONFIDENCE_LETTER_SCHEME,
        confidence_letter_random_seed=_C.CONFIDENCE_LETTER_RANDOM_SEED,
        mcq_letter_scheme=_C.MCQ_LETTER_SCHEME,
        mcq_letter_random_seed=_C.MCQ_LETTER_RANDOM_SEED,
        randomize_letters_per_question=_C.RANDOMIZE_LETTERS_PER_QUESTION,
        # Output
        output_dir=str(_C.OUTPUT_DIR),
        save_hf=_C.SAVE_HF,
        hf_repo=_C.HF_REPO,
        save_hf_checkpoints=_C.SAVE_HF_CHECKPOINTS,
        hf_checkpoint_repo=_C.HF_CHECKPOINT_REPO,
        checkpoint_steps=_C.CHECKPOINT_STEPS,
        hf_checkpoint_private=_C.HF_CHECKPOINT_PRIVATE,
        keep_local_checkpoints=_C.KEEP_LOCAL_CHECKPOINTS,
        # Weights & Biases
        wandb_project=_C.WANDB_PROJECT,
        wandb_run_name=_C.WANDB_RUN_NAME,
        wandb_tags=_C.WANDB_TAGS,
        wandb_notes=_C.WANDB_NOTES,
        save_wandb_artifact=_C.SAVE_WANDB_ARTIFACT,
    )


if __name__ == "__main__":
    args = build_args_from_config()
    train(args)
