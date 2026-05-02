
# --- repo path bootstrap (so root-level imports like `finetune_prompting`,
# `finetune_config` resolve when run from anywhere) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np
import os
import sys
import torch
import wandb
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from types import SimpleNamespace

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
    compute_loss
)
from finetune_prompting import (
    build_self_confidence_prompts,
    build_self_confidence_prompts_numeric,
    build_multiple_choice_question_prompts,
    run_mcq_forward_pass,
    run_confidence_forward_pass,
    run_confidence_forward_pass_numeric,
    get_confidence_letter_mapping,
    get_mcq_letter_mapping,
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

    if args.use_recorded_responses:
        if mcq_results_lookup is None:
            raise ValueError(
                "--use_recorded_responses is True but no mcq_results_data provided!"
            )
        
        # Look up pre-recorded entropy for each question in batch
        entropies = []
        valid_indices = []  # Track which samples in batch are valid
        
        for i, row in enumerate(batch):
            qid = row.get("qid")
            if qid and qid in mcq_results_lookup:
                entropies.append(mcq_results_lookup[qid]["entropy"])
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

    # ----------------------------------------------
    # If using dynamic teacher: compute entropy live from current model
    # This is  selected with --use_recorded_responses flag when  running this file
    # ----------------------------------------------

    else:
        # Use the MCQ letter mapping already defined above
        mcq_prompts = build_multiple_choice_question_prompts(batch, tokenizer, mcq_letter_mapping)
        
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

    # Convert to soft labels (size depends on confidence_format)
    confidence_format = getattr(args, "confidence_format", "letter_8bin")
    soft = build_soft_targets_from_entropy(
        entropy, sigma=sigma, confidence_format=confidence_format
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
            batch, tokenizer, confidence_letter_mapping, mcq_letter_mapping
        )
        conf_out = run_confidence_forward_pass(
            model=model, tokenizer=tokenizer, prompts=conf_prompts,
            device=device, temperature=args.temperature, requires_grad=True,
            confidence_letter_mapping=confidence_letter_mapping,
        )
        conf_logits = conf_out["logits8"]  # [B, 8]
    elif confidence_format in ("numeric_1_5", "numeric_1_10"):
        n_max = 5 if confidence_format == "numeric_1_5" else 10
        conf_prompts = build_self_confidence_prompts_numeric(
            batch, tokenizer, mcq_letter_mapping, n_max=n_max
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
    loss = compute_loss(
        conf_logits, soft_targets=soft, entropy=entropy,
        loss_type=args.loss_type, reduction='mean',
        confidence_format=confidence_format,
    )

    # ----------------------------------------------
    # 4. Backprop
    # ----------------------------------------------
    loss.backward()

    return loss.detach()

    
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

    # ============================================================
    # Setup / Load model and data
    # ============================================================
    device = args.device
    tokenizer = load_tokenizer(args)
    model = load_model_with_lora(args, tokenizer).to(device)

    # Canonicalize model/tokenizer setup (fix pad_token warnings)
    model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)

    # Setup log file path early (needed for duplicate removal logging)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = str(_C.LOGS_DIR)
    os.makedirs(log_dir, exist_ok=True)
    model_name_safe = args.model_name.replace("/", "-").replace("_", "-")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{model_name_safe}_evaluation_metrics.jsonl")
    
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

    # Load pre-recorded MCQ results if using frozen teacher OR if provided for evaluation
    mcq_results_lookup = None
    if args.mcq_results_data is not None:
        mcq_results_lookup = load_mcq_results_data(args.mcq_results_data)
        if mcq_results_lookup is None:
            if args.use_recorded_responses:
                raise ValueError(f"Failed to load MCQ results from {args.mcq_results_data}")
            else:
                print(f"⚠️  Warning: Could not load MCQ results from {args.mcq_results_data}, continuing without pre-recorded entropy logging")
        
        # Filter datasets to only include questions with pre-recorded results (if using frozen teacher)
        if args.use_recorded_responses:
            train_dataset = filter_dataset_by_mcq_results(
                train_dataset, mcq_results_lookup, dataset_name="training"
            )
            val_dataset = filter_dataset_by_mcq_results(
                val_dataset, mcq_results_lookup, dataset_name="validation"
            )
        
        # Filter validation dataset if using frozen validation
        if args.val_on_frozen:
            val_dataset = filter_dataset_by_mcq_results(
                val_dataset, mcq_results_lookup, dataset_name="validation"
            )
    elif args.use_recorded_responses:
        raise ValueError(
            "--use_recorded_responses requires --mcq_results_data to be specified"
        )
    
    # Validate val_on_frozen requirements
    if args.val_on_frozen:
        if args.mcq_results_data is None:
            raise ValueError(
                "--val_on_frozen requires --mcq_results_data to be specified"
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
    wandb_init_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    
    if args.save_wandb_artifact:
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
        print(f"✓ WandB run initialized: {wandb_run_name} (ID: {wandb_run_id})")

    # Output / checkpoints
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Setup checkpoint directory with datetime
    checkpoint_base_dir = os.path.join(str(_C.CHECKPOINTS_DIR), f"{timestamp}_checkpoints")
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    print(f"✓ Checkpoints will be saved to: {os.path.abspath(checkpoint_base_dir)}")
    
    # Save training parameters to checkpoint directory
    save_training_parameters(args, checkpoint_base_dir)

    # ============================================================
    # Print first MCQ and confidence prompts
    # ============================================================
    print("\n" + "="*80)
    print("First Multiple Choice Question Prompt:")
    print("="*80)
    first_batch = train_dataset[0:1]
    # Generate mapping for display (use sample mapping if per-question randomization)
    if args.randomize_letters_per_question:
        display_mcq_mapping = get_mcq_letter_mapping(args.mcq_letter_scheme, seed=None)
    else:
        display_mcq_mapping = mcq_letter_mapping
    mcq_prompts = build_multiple_choice_question_prompts(first_batch, tokenizer, display_mcq_mapping)
    print(mcq_prompts[0])
    print("="*80 + "\n")
    
    print("="*80)
    print("First Confidence Question Prompt:")
    print("="*80)
    # Generate mapping for display (use sample mapping if per-question randomization)
    if args.randomize_letters_per_question:
        display_conf_mapping = get_confidence_letter_mapping(args.confidence_letter_scheme, seed=None)
    else:
        display_conf_mapping = confidence_letter_mapping
    conf_prompts = build_self_confidence_prompts(first_batch, tokenizer, display_conf_mapping, display_mcq_mapping)
    print(conf_prompts[0])
    print("="*80 + "\n")

    # ============================================================
    # Baseline evaluation BEFORE training
    # ============================================================
    print("\n" + "="*60)
    print("Running baseline validation (before training)...")
    print("="*60)

    baseline_metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        step_name="baseline",
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=0,
        mcq_results_lookup=mcq_results_lookup,
        train_dataset_qids=train_dataset_qids,
        train_dataset_questions=train_dataset_questions,
        sigma=args.sigma,
        val_on_frozen=args.val_on_frozen,
    )

    print(f"\nBaseline Accuracy: {baseline_metrics['mcq_accuracy']:.4f}")
    print(f"Baseline Avg Entropy: {baseline_metrics['avg_entropy']:.4f}")
    print(f"Baseline Avg Confidence: {baseline_metrics['avg_confidence']:.4f}")
    print(f"- samples: {baseline_metrics['n_samples']}\n")


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

        loss = train_step(
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
        if loss is None:
            continue  # Don't increment step, try next batch
        
        optimizer.step()

        losses.append(loss.item())

        # W&B logging
        if args.save_wandb_artifact:
            wandb.log({"train/loss": loss.item(), "step": step})

        # -----------------------------
        # Periodic validation (Validation Step)
        # -----------------------------
        if (step % args.val_interval) == 0 and step > 0:
            print("\n" + "="*60)
            print(f"Validation at step {step}")
            print("="*60)

            val_metrics = run_evaluation(
                model=model,
                tokenizer=tokenizer,
                val_dataset=val_dataset,
                device=device,
                args=args,
                step_name="validation",
                num_samples=args.val_num_samples,
                log_file_path=log_file_path,
                step=step,
                mcq_results_lookup=mcq_results_lookup,
                train_dataset_qids=train_dataset_qids,
                train_dataset_questions=train_dataset_questions,
                sigma=args.sigma,
                val_on_frozen=args.val_on_frozen,
            )

            print(f"Val Accuracy: {val_metrics['mcq_accuracy']:.4f}")
            print(f"Val Loss:     {val_metrics['avg_loss']:.4f}")
            print(f"Val Entropy:  {val_metrics['avg_entropy']:.4f}")
            print(f"Val Conf:     {val_metrics['avg_confidence']:.4f}")
            print(f"Samples:      {val_metrics['n_samples']}")

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
                checkpoint_timestamp=checkpoint_timestamp
            )

        step += 1

    # ============================================================
    # Final metrics
    # ============================================================
    print("\n" + "="*60)
    print("Final evaluation:")
    print("="*60)

    final_metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        step_name="final",
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=step,
        mcq_results_lookup=mcq_results_lookup,
        train_dataset_qids=train_dataset_qids,
        train_dataset_questions=train_dataset_questions,
        sigma=args.sigma,
        val_on_frozen=args.val_on_frozen,
    )

    print(f"\nFinal Accuracy:  {final_metrics['mcq_accuracy']:.4f}")
    print(f"Final Loss:      {final_metrics['avg_loss']:.4f}")
    print(f"Final Confidence:{final_metrics['avg_confidence']:.4f}")

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
        final_timestamp=final_timestamp
    )
    if success:
        print("✓ Model saved successfully!")
    else:
        print("❌ Model save failed! Check error messages above.")
        raise RuntimeError("Failed to save model")
    
    # Finish wandb run after model is saved (so artifact can be logged)
    if args.save_wandb_artifact:
        wandb.finish()
    
    # Restore original print and close print log file
    import builtins
    builtins.print = original_print
    print_log_file.close()
    print(f"✓ Print output saved to: {print_log_path}")
    


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
        log_interval=_C.LOG_INTERVAL,
        val_interval=_C.VAL_INTERVAL,
        limit_val_batches=_C.LIMIT_VAL_BATCHES,
        val_num_samples=_C.VAL_NUM_SAMPLES,
        sigma=_C.SIGMA,
        loss_type=_C.LOSS_TYPE,
        temperature=_C.TEMPERATURE,
        shuffle_options=_C.SHUFFLE_OPTIONS,
        use_recorded_responses=_C.USE_RECORDED_RESPONSES,
        enable_data_leakage_checks=_C.ENABLE_DATA_LEAKAGE_CHECKS,
        val_on_frozen=_C.VAL_ON_FROZEN,
        confidence_format=_C.CONFIDENCE_FORMAT,
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
