import argparse
import math
import numpy as np
import os
import torch
from datetime import datetime, timezone
from torch.utils.data import DataLoader, Subset
import random
from peft import LoraConfig, get_peft_model

# Imports from helper files
from finetune_diagnostics import compute_metrics_for_wandb, assess_mcq_accuracy
from finetune_utils import (
    write_log,
    MCQDataset,
    get_single_token_id,
    load_mcq_results_data,
    verify_and_resolve_options,
    init_wandb,
    log_wandb_metrics,
    log_wandb_config,
    log_device_info,
    save_hf_checkpoint,
    save_model_final,
    finish_wandb,
    build_self_confidence_prompts,
    build_multiple_choice_question_prompts,
    _get_log_file_path,
    check_and_clear_gpu_memory,
    load_model_with_error_handling,
    setup_tokenizer,
    validate_and_load_dataset,
    log_prompts_and_responses,
    validate_training_files,
    compute_ABCD_entropy,
    shuffle_options_and_update_correct_letter,
)


def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries."""
    return batch


def log_answer_distributions(log_file_path, step_type, step_number, 
                             predicted_letter_counts, correct_letter_counts, 
                             total_questions, accuracy=None, avg_entropy=None,
                             answer_variety=None, answer_entropy_std=None):
    """
    Log answer distribution information to a dedicated log file.
    
    Args:
        log_file_path: Path to the answer distributions log file
        step_type: "val" or "test"
        step_number: Step number (for validation) or None (for test)
        predicted_letter_counts: dict with counts for A, B, C, D
        correct_letter_counts: dict with counts for A, B, C, D
        total_questions: Total number of questions
        accuracy: Optional accuracy value
        avg_entropy: Optional average entropy value
        answer_variety: Optional answer variety score (0 = always same, 1 = 25% each)
        answer_entropy_std: Optional standard deviation of answer entropy
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Calculate percentages
    predicted_percentages = {
        letter: (count / total_questions * 100) if total_questions > 0 else 0.0
        for letter, count in predicted_letter_counts.items()
    }
    correct_percentages = {
        letter: (count / total_questions * 100) if total_questions > 0 else 0.0
        for letter, count in correct_letter_counts.items()
    }
    
    log_entry = {
        "type": "answer_distribution",
        "timestamp": timestamp,
        "step_type": step_type,  # "val" or "test"
        "step_number": step_number,  # None for test
        "total_questions": total_questions,
        "accuracy": accuracy,
        "avg_entropy": avg_entropy,
        "answer_entropy_std": answer_entropy_std,
        "answer_variety": answer_variety,
        "predicted_letter_distribution": {
            letter: {
                "count": predicted_letter_counts.get(letter, 0),
                "percentage": predicted_percentages.get(letter, 0.0)
            }
            for letter in "ABCD"
        },
        "correct_letter_distribution": {
            letter: {
                "count": correct_letter_counts.get(letter, 0),
                "percentage": correct_percentages.get(letter, 0.0)
            }
            for letter in "ABCD"
        }
    }
    
    write_log(log_file_path, log_entry)

# ============================================================
# Entropy → scalar confidence → soft labels
# ============================================================

def compute_soft_labels(logits4, sigma=10.0):
    """
    Convert 4-way answer logits into soft 8-bin confidence distribution.

    Uses percentage-based Gaussian kernel to create soft labels.

    Args:
        logits4: tensor of shape [4] with logits for A, B, C, D
        sigma: Gaussian width in percentage space (default: 10)

    Returns:
        tensor of shape [8] with soft label distribution
    """
    # 1. Softmax over the 4 MCQ options
    probs = torch.softmax(logits4, dim=0)

    # 2. Entropy (natural logs)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()

    # 3. Convert entropy to "confidence percentage"
    #    confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

    # 4. Bin midpoints + widths (exact values from your colleague)
    bin_edges = torch.tensor([0, 5, 10, 20, 40, 60, 80, 90, 100],
                             dtype=torch.float32,
                             device=logits4.device)
    bin_midpoints = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95],
                                 dtype=torch.float32,
                                 device=logits4.device)
    bin_widths = bin_edges[1:] - bin_edges[:-1]   # shape [8]

    # 5. Gaussian kernel in percentage space
    distances = (bin_midpoints - confidence_percent)**2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    return weights / weights.sum()


def convert_entropy_to_soft_labels(entropy, sigma=10.0):
    """
    Convert entropy value to soft 8-bin confidence distribution.

    Args:
        entropy: scalar entropy value
        sigma: Gaussian width in percentage space (default: 10)

    Returns:
        tensor of shape [8] with soft label distribution
    """
    # Convert entropy to "confidence percentage"
    # confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

    # Bin midpoints + widths (same as compute_soft_labels)
    bin_edges = torch.tensor(
        [0, 5, 10, 20, 40, 60, 80, 90, 100],
        dtype=torch.float32
    )
    bin_midpoints = torch.tensor(
        [2.5, 7.5, 15, 30, 50, 70, 85, 95],
        dtype=torch.float32
    )
    bin_widths = bin_edges[1:] - bin_edges[:-1]  # shape [8]

    # Gaussian kernel in percentage space
    distances = (bin_midpoints - confidence_percent) ** 2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    return weights / weights.sum()


# ------------------------------------------------------------------
# Training Step
# ------------------------------------------------------------------

def val_step(model, tokenizer, batch, sigma=10.0, device="cuda",
             mcq_results_lookup=None, log_file_path=None, args=None, temperature=0.0):
    """
    Single validation step for Explicit Confidence Task.
    
    Similar to train_step but runs in eval mode with no gradients.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        batch: Batch of validation examples
        sigma: Gaussian width for soft labels
        device: Device to run on
        mcq_results_lookup: Lookup dict for recorded MCQ results
        log_file_path: Path for logging
        args: Training arguments
        
    Returns:
        dict with keys:
            - "loss": loss tensor (detached)
            - "correct": tensor of correctness (1.0 or 0.0) for each sample [B]
            - "entropy": tensor of entropy values for each sample [B]
            - "verbal_conf": tensor of verbal confidence scores for each sample [B]
    """
    model.eval()
    
    resolved_results = []
    
    # 1. Verify question/choices and resolve options
    for row in batch:
        result_data, opts = verify_and_resolve_options(
            row, mcq_results_lookup, log_file_path
        )
        row["options"] = opts
        resolved_results.append(result_data)
    
    # 1.5. Shuffle options to prevent position bias
    for row in batch:
        shuffle_options_and_update_correct_letter(row)
    
    # 2. Build MCQ prompts
    answer_prompts = build_multiple_choice_question_prompts(batch)
    
    # 3. First forward pass (NO GRAD) - MCQ answer prediction
    enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        try:
            out = model(**enc, use_cache=False)
        except TypeError:
            out = model(**enc)
        
        final_logits = out.logits[:, -1, :]
        abcd_ids = torch.tensor(
            [get_single_token_id(tokenizer, c) for c in "ABCD"],
            device=device,
            dtype=torch.long
        )
        answer_logits4 = final_logits[:, abcd_ids]
    
    # 4. Compute soft targets
    soft_targets_list = []
    
    for i, row in enumerate(batch):
        if args.use_recorded_responses:
            qid = str(row.get("qid"))
            rd = resolved_results[i]
            
            if rd is None:
                soft_target = compute_soft_labels(
                    answer_logits4[i], sigma=sigma
                ).to(device)
            else:
                teacher_probs = (
                    rd.get("predicted_probs") or
                    rd.get("answer_probs") or
                    rd.get("probs")
                )
                if teacher_probs is None:
                    soft_target = compute_soft_labels(
                        answer_logits4[i], sigma=sigma
                    ).to(device)
                else:
                    entropy = compute_ABCD_entropy(teacher_probs)
                    soft_target = convert_entropy_to_soft_labels(
                        entropy
                    ).to(device)
        else:
            soft_target = compute_soft_labels(
                answer_logits4[i], sigma=sigma
            ).to(device)
        
        soft_targets_list.append(soft_target)
    
    soft_targets = torch.stack(soft_targets_list)
    
    # 5. Build confidence prompt and run second forward pass
    confidence_prompts = build_self_confidence_prompts(batch)
    enc2 = tokenizer(confidence_prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        try:
            out2 = model(**enc2, use_cache=False)
        except TypeError:
            out2 = model(**enc2)
        
        final_logits2 = out2.logits[:, -1, :]
        bins_ids = torch.tensor(
            [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
            device=device,
            dtype=torch.long
        )
        conf_logits8 = final_logits2[:, bins_ids]
    
    # 6. Compute loss
    log_probs = torch.log_softmax(conf_logits8, dim=-1)
    loss = (-soft_targets * log_probs).sum(dim=1).mean()
    
    # --- NEW: Calculate Raw Data for Diagnostics ---
    # 1. Get Pass 1 Entropy (use temperature-scaled probabilities)
    if temperature > 0:
        scaled_logits = answer_logits4 / temperature
        probs_mcq = torch.softmax(scaled_logits, dim=-1)
    else:
        probs_mcq = torch.softmax(answer_logits4, dim=-1)
    entropies = -(probs_mcq * torch.log(probs_mcq + 1e-12)).sum(dim=-1)
    
    # 2. Get Pass 1 Correctness (sample with temperature if > 0, else argmax)
    if temperature > 0:
        # Sample from the distribution
        preds = torch.multinomial(probs_mcq, num_samples=1).squeeze(-1)
    else:
        # Deterministic: use argmax
        preds = probs_mcq.argmax(dim=-1)
    # Convert batch rows to check correctness
    is_correct_list = []
    for j, r in enumerate(batch):
        pred_char = "ABCD"[preds[j].item()]
        is_correct_list.append(1.0 if pred_char == r['correct_letter'] else 0.0)
    is_correct = torch.tensor(is_correct_list, device=device, dtype=torch.float32)
    
    # 3. Get Pass 2 Verbal Confidence
    conf_probs = torch.softmax(conf_logits8, dim=-1)
    bin_midpoints = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95], 
                                 device=device, dtype=torch.float32)
    verbal_conf = (conf_probs * bin_midpoints).sum(dim=-1)
    
    # 4. Get Pass 2 Confidence Entropy (entropy of confidence distribution)
    conf_entropies = -(conf_probs * torch.log(conf_probs + 1e-12)).sum(dim=-1)
    
    return {
        "loss": loss.detach(),
        "correct": is_correct,      # [B]
        "entropy": entropies.detach(), # [B] - MCQ answer entropy
        "verbal_conf": verbal_conf.detach(), # [B] - Expected confidence value
        "conf_entropy": conf_entropies.detach() # [B] - Entropy of confidence distribution
    }


def train_step(model, tokenizer, batch, sigma=10.0, device="cuda",
               mcq_results_lookup=None, log_file_path=None, args=None,
               step=None, prompt_log_file_path=None, temperature=0.0):
    """
    Single training step for Explicit Confidence Task.

    Supports two modes:
        - args.use_recorded_responses == True:
            Use recorded MCQ results (frozen teacher) as target.
        - args.use_recorded_responses == False:
            Use live logits from current model (dynamic teacher).

    Args:
        model: Language model
        tokenizer: Tokenizer
        batch: Batch of training examples
        sigma: Gaussian width for soft labels
        device: Device to run on
        mcq_results_lookup: Lookup dict for recorded MCQ results
        log_file_path: Path for logging
        args: Training arguments
        step: Current training step (for prompt logging)
        prompt_log_file_path: Path for logging prompts (for first 5 steps)

    Returns:
        loss tensor
    """

    resolved_results = []  # store recorded MCQ results for training targets

    # ------------------------------------------------------------------
    # 1. Verify question/choices and resolve options
    # ------------------------------------------------------------------
    for row in batch:
        result_data, opts = verify_and_resolve_options(
            row, mcq_results_lookup, log_file_path
        )
        row["options"] = opts
        resolved_results.append(result_data)  # save for later target lookup

    # ------------------------------------------------------------------
    # 1.5. Shuffle options to prevent position bias
    # ------------------------------------------------------------------
    for row in batch:
        shuffle_options_and_update_correct_letter(row)

    # ------------------------------------------------------------------
    # 2. Build MCQ prompts using the utility function
    # ------------------------------------------------------------------
    answer_prompts = build_multiple_choice_question_prompts(batch)

    # ------------------------------------------------------------------
    # 3. First forward pass (NO GRAD) - COMPLETELY SEPARATE CONTEXT
    # ------------------------------------------------------------------
    enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)

    # Ensure completely separate context - no cached states
    # Set model to eval mode temporarily to avoid any training-specific state leakage
    model_was_training = model.training
    model.eval()
    
    with torch.no_grad():
        # Explicitly disable any potential caching mechanisms
        # Note: use_cache=False ensures no state is cached between forward passes
        try:
            out = model(**enc, use_cache=False)
        except TypeError:
            # Fallback if model doesn't support use_cache parameter
            out = model(**enc)

        # last token logits
        final_logits = out.logits[:, -1, :]  # [B, vocab]

        # extract A/B/C/D logits
        abcd_ids = torch.tensor(
            [get_single_token_id(tokenizer, c) for c in "ABCD"],
            device=device,
            dtype=torch.long
        )

        answer_logits4 = final_logits[:, abcd_ids]  # [B, 4]
    
    # Restore training mode if it was enabled
    if model_was_training:
        model.train()

    # ------------------------------------------------------------------
    # 3. Compute soft targets (frozen teacher or dynamic)
    # ------------------------------------------------------------------
    soft_targets_list = []

    for i, row in enumerate(batch):

        if args.use_recorded_responses:
            # ------------------------------
            # FROZEN TEACHER MODE (default)
            # ------------------------------
            qid = str(row.get("qid"))
            rd = resolved_results[i]

            if rd is None:
                # Fallback to dynamic teacher mode when recorded data missing
                error_msg = (
                    f"No recorded MCQ results for qid: {qid}, "
                    "using dynamic teacher fallback"
                )
                print(error_msg)
                if log_file_path:
                    write_log(log_file_path, {"error": error_msg, "qid": qid})
                # Use current model's logits as fallback
                soft_target = compute_soft_labels(
                    answer_logits4[i], sigma=sigma
                ).to(device)
            else:
                # Pre-recorded file may store probabilities in different formats:
                # - `predicted_probs` or `answer_probs` (list or dict)
                # - `probs` (dict with keys "A", "B", "C", "D")
                teacher_probs = (
                    rd.get("predicted_probs") or
                    rd.get("answer_probs") or
                    rd.get("probs")
                )
                if teacher_probs is None:
                    # Fallback when probability data is missing
                    error_msg = (
                        f"Recorded MCQ results missing probability data for "
                        f"qid {qid}, using dynamic teacher fallback"
                    )
                    print(error_msg)
                    if log_file_path:
                        write_log(log_file_path,
                                  {"error": error_msg, "qid": qid})
                    # Use current model's logits as fallback
                    soft_target = compute_soft_labels(
                        answer_logits4[i], sigma=sigma
                    ).to(device)
                else:
                    entropy = compute_ABCD_entropy(teacher_probs)
                    soft_target = convert_entropy_to_soft_labels(
                        entropy
                    ).to(device)

        else:
            # ------------------------------
            # DYNAMIC TEACHER MODE (old)
            # ------------------------------
            soft_target = compute_soft_labels(
                answer_logits4[i], sigma=sigma
            ).to(device)

        soft_targets_list.append(soft_target)

    soft_targets = torch.stack(soft_targets_list)  # [B, 8]

    # ------------------------------------------------------------------
    # 4. Build explicit-confidence prompt and run second forward pass
    #    COMPLETELY SEPARATE CONTEXT - model has no knowledge of first prompt
    # ------------------------------------------------------------------
    confidence_prompts = build_self_confidence_prompts(batch)
    
    enc2 = tokenizer(confidence_prompts, return_tensors="pt", padding=True).to(device)
    
    # Ensure completely separate context - no cached states from first pass
    # The model should have no knowledge of the first prompt or its response
    try:
        out2 = model(**enc2, use_cache=False)
    except TypeError:
        # Fallback if model doesn't support use_cache parameter
        out2 = model(**enc2)

    # Extract logits for confidence bins A–H
    final_logits2 = out2.logits[:, -1, :]
    bins_ids = torch.tensor(
        [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
        device=device,
        dtype=torch.long
    )
    conf_logits8 = final_logits2[:, bins_ids]  # [B,8]

    # ------------------------------------------------------------------
    # 5. Log prompts and responses for first 2 steps
    # ------------------------------------------------------------------
    # if step is not None and step < 3 and prompt_log_file_path:
    #     log_prompts_and_responses(
    #         step, prompt_log_file_path, answer_logits4, conf_logits8,
    #         batch, answer_prompts, confidence_prompts, soft_targets
    #     )

    # ------------------------------------------------------------------
    # 6. Compute loss
    # ------------------------------------------------------------------
    log_probs = torch.log_softmax(conf_logits8, dim=-1)  # [B,8]
    loss = (-soft_targets * log_probs).sum(dim=1).mean()

    return loss


# ============================================================
# Validation and Test Evaluation Functions
# ============================================================

def run_evaluation(model, tokenizer, device, args, mcq_results_lookup, log_file_path, step,
                   mcq_accuracy_log_file_path, answer_distributions_log_file_path,
                   eval_type, val_dataloader=None, val_dataset=None, data_path=None,
                   val_metrics_log_file_path=None, limit_val_batches=None, temperature=0.0):
    """
    Unified function to run evaluation on validation or test datasets.
    
    This function handles:
    - Baseline validation (before training, step=0)
    - Regular validation during training (step=step)
    - Final test evaluation (after training)
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        args: Training arguments
        mcq_results_lookup: Lookup dict for recorded MCQ results
        log_file_path: Path for question comparison logging
        step: Step number (0 for baseline, training step number, or final step for test)
        mcq_accuracy_log_file_path: Path for MCQ accuracy assessment logging
        answer_distributions_log_file_path: Path for answer distributions logging
        eval_type: One of "baseline", "validation", or "test"
        val_dataloader: DataLoader for validation data (required for baseline/validation)
        val_dataset: Dataset for MCQ accuracy assessment (required for baseline/validation)
        data_path: Path to dataset JSONL file (required for test, optional for others)
        val_metrics_log_file_path: Path for validation metrics logging (only for validation)
        limit_val_batches: Optional limit on number of batches to process (only for validation)
        
    Returns:
        dict with evaluation metrics
    """
    if eval_type not in ["baseline", "validation", "test"]:
        raise ValueError(f"eval_type must be one of 'baseline', 'validation', or 'test', got '{eval_type}'")
    
    # Determine prefix and step_type based on eval_type
    if eval_type == "test":
        prefix = "test"
        step_type = "test"
        step_number_for_dist = None
        print_header = True
    else:  # baseline or validation
        prefix = "val"
        step_type = "val"
        step_number_for_dist = step
        print_header = False
    
    # Load dataset if needed (for test evaluation)
    dataloader = val_dataloader
    dataset = val_dataset
    
    if eval_type == "test":
        if data_path is None:
            raise ValueError("data_path must be provided for test evaluation")
        if print_header:
            print("\n" + "="*80)
            print("Running final test evaluation...")
            print("="*80)
        
        dataset = validate_and_load_dataset(data_path, "test")
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn
        )
        print(f"✓ Test dataset loaded: {len(dataset)} samples")
        num_questions_for_mcq = min(500, len(dataset))
    else:
        # For baseline and validation: sample 500 random questions
        if dataloader is None or dataset is None:
            raise ValueError(f"val_dataloader and val_dataset must be provided for {eval_type} evaluation")
        
        # Sample 500 random indices from the dataset
        dataset_size = len(dataset)
        num_questions_for_mcq = min(500, dataset_size)
        # Use seed=42 for baseline (to match capabilities_test.py), step for validation steps
        if eval_type == "baseline":
            random.seed(42)  # Match capabilities_test.py seed for baseline
        else:
            random.seed(step if step >= 0 else 0)  # Use step as seed for reproducibility
        sampled_indices = random.sample(range(dataset_size), num_questions_for_mcq)
        
        # Create subset with sampled indices
        subset_dataset = Subset(dataset, sampled_indices)
        dataloader = DataLoader(
            subset_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn
        )
        dataset = subset_dataset  # Use subset for MCQ accuracy assessment too
    
    # Print evaluation start message
    eval_name = {
        "baseline": "baseline validation",
        "validation": "validation",
        "test": "test evaluation"
    }[eval_type]
    
    total_samples = len(dataset) if dataset else len(dataloader.dataset) if hasattr(dataloader, 'dataset') else "unknown"
    
    if eval_type != "test":
        print(f"\nStarting {eval_name} on {num_questions_for_mcq} random questions...")
    else:
        print(f"\nStarting {eval_name} on {total_samples} questions (MCQ accuracy will be assessed on {num_questions_for_mcq} questions)...")
    
    if limit_val_batches and eval_type != "test":
        print(f"  (Limited to {limit_val_batches} batches)")
    
    model.eval()
    losses = []
    all_correct = []
    all_entropies = []
    all_verbal_conf = []
    all_conf_entropies = []  # Confidence entropy
    
    batches_processed = 0
    
    with torch.no_grad():
        for batch in dataloader:
            out_metrics = val_step(
                model, tokenizer, batch, device=device,
                sigma=args.sigma, mcq_results_lookup=mcq_results_lookup,
                log_file_path=log_file_path, args=args, temperature=temperature
            )
            
            losses.append(out_metrics["loss"].item())
            all_correct.extend(out_metrics["correct"].cpu().tolist())
            all_entropies.extend(out_metrics["entropy"].cpu().tolist())
            all_verbal_conf.extend(out_metrics["verbal_conf"].cpu().tolist())
            all_conf_entropies.extend(out_metrics["conf_entropy"].cpu().tolist())
            
            batches_processed += 1
            
            # Limit validation batches if specified (only for baseline/validation)
            if eval_type != "test" and limit_val_batches and batches_processed >= limit_val_batches:
                break
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    
    # Compute confidence entropy and variance metrics
    conf_entropy_arr = np.array(all_conf_entropies)
    avg_conf_entropy = float(np.mean(conf_entropy_arr)) if len(conf_entropy_arr) > 0 else 0.0
    conf_entropy_variance = float(np.var(conf_entropy_arr)) if len(conf_entropy_arr) > 0 else 0.0
    
    # Compute metrics
    metrics = compute_metrics_for_wandb(
        all_correct, all_entropies, all_verbal_conf
    )
    # Rename keys based on prefix
    wandb_metrics = {}
    for key, value in metrics.items():
        if key.startswith("val/"):
            wandb_metrics[key.replace("val/", f"{prefix}/")] = value
        else:
            wandb_metrics[key] = value
    wandb_metrics[f"{prefix}/loss"] = avg_loss
    wandb_metrics[f"{prefix}/batches_processed"] = batches_processed
    wandb_metrics[f"{prefix}/avg_conf_entropy"] = avg_conf_entropy
    wandb_metrics[f"{prefix}/conf_entropy_variance"] = conf_entropy_variance
    
    # Assess MCQ accuracy
    mcq_accuracy = None
    mcq_entropy = None
    mcq_entropy_std = None
    answer_variety = None
    if dataset:
        # For baseline/validation, we already have a subset of 500 questions
        # For test, assess_mcq_accuracy will sample internally
        if eval_type == "test":
            print(f"Assessing MCQ accuracy on {num_questions_for_mcq} questions...")
            # For test, use seed=42 to match capabilities_test.py
            mcq_results = assess_mcq_accuracy(
                model, tokenizer, dataset, device=device,
                validation_step=step, log_file_path=mcq_accuracy_log_file_path,
                num_questions=num_questions_for_mcq, temperature=temperature, seed=42
            )
        else:
            # For baseline/validation, use the same subset we already processed
            # assess_mcq_accuracy will use all questions in the subset (no seed needed)
            print(f"Assessing MCQ accuracy on {len(dataset)} questions...")
            mcq_results = assess_mcq_accuracy(
                model, tokenizer, dataset, device=device,
                validation_step=step, log_file_path=mcq_accuracy_log_file_path,
                num_questions=len(dataset), temperature=temperature, seed=None  # Use all questions in the subset
            )
        mcq_accuracy = mcq_results["accuracy"]
        mcq_entropy = mcq_results["avg_entropy"]
        mcq_entropy_std = mcq_results.get("std_entropy", 0.0)
        wandb_metrics[f"{prefix}/accuracy"] = mcq_accuracy
        wandb_metrics[f"{prefix}/answer_entropy"] = mcq_entropy
        wandb_metrics[f"{prefix}/answer_entropy_std"] = mcq_entropy_std
        
        # Calculate answer variety (normalized measure of answer distribution)
        # 0 = always picking same answer, 1 = picking 25% each (perfect variety)
        predicted_letter_counts = mcq_results["predicted_letter_counts"]
        total_questions = mcq_results["total_questions"]
        if total_questions > 0:
            # Get proportions for each answer choice
            proportions = np.array([
                predicted_letter_counts.get("A", 0) / total_questions,
                predicted_letter_counts.get("B", 0) / total_questions,
                predicted_letter_counts.get("C", 0) / total_questions,
                predicted_letter_counts.get("D", 0) / total_questions,
            ])
            # Calculate standard deviation of proportions
            # Perfect balance (25% each) = std = 0.0
            # All one answer (e.g., 100% A) = std ≈ 0.433
            std = float(np.std(proportions))
            max_std = np.sqrt(0.1875)  # Maximum std when all answers are the same (≈ 0.433)
            
            # Normalize: 0 = always same answer, 1 = perfect balance
            # answer_variety = 1 - (std / max_std)
            answer_variety = float(1.0 - (std / max_std)) if max_std > 0 else 1.0
            # Clamp to [0, 1] in case of floating point issues
            answer_variety = max(0.0, min(1.0, answer_variety))
            wandb_metrics[f"{prefix}/answer_variety"] = answer_variety
        else:
            answer_variety = 0.0
            wandb_metrics[f"{prefix}/answer_variety"] = answer_variety
        
        # Log answer distributions
        log_answer_distributions(
            answer_distributions_log_file_path,
            step_type=step_type,
            step_number=step_number_for_dist,
            predicted_letter_counts=mcq_results["predicted_letter_counts"],
            correct_letter_counts=mcq_results["correct_letter_counts"],
            total_questions=mcq_results["total_questions"],
            accuracy=mcq_accuracy,
            avg_entropy=mcq_entropy,
            answer_variety=answer_variety,
            answer_entropy_std=mcq_entropy_std
        )
    
    # Log to WandB
    log_wandb_metrics(wandb_metrics, step=step)
    
    # Log to validation metrics file (only for regular validation, not baseline or test)
    if eval_type == "validation" and step >= 0 and val_metrics_log_file_path:
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "type": "validation_metrics",
            "timestamp": timestamp,
            "step": step,
            "metrics": {
                "loss": avg_loss,
                "accuracy": mcq_accuracy if mcq_accuracy is not None else None,
                "answer_entropy": mcq_entropy if mcq_entropy is not None else None,
                "answer_entropy_std": mcq_entropy_std if mcq_entropy_std is not None else None,
                "avg_verbal_conf": wandb_metrics[f"{prefix}/avg_verbal_conf"],
                "std_verbal_conf": wandb_metrics[f"{prefix}/std_verbal_conf"],
                "alignment_corr": wandb_metrics[f"{prefix}/alignment_corr"],
                "calibration_corr": wandb_metrics[f"{prefix}/calibration_corr"],
                "avg_conf_entropy": avg_conf_entropy,
                "conf_entropy_variance": conf_entropy_variance,
                "answer_variety": answer_variety if dataset else None,
                "batches_processed": batches_processed,
            }
        }
        write_log(val_metrics_log_file_path, log_entry)
    
    # Print results
    if eval_type == "baseline":
        info_prefix = "Baseline"
    elif eval_type == "test":
        info_prefix = "Test"
    else:  # validation
        info_prefix = f"Step {step}"
    
    info = f"{info_prefix} | {'Val' if prefix == 'val' else 'Test'} Loss: {avg_loss:.4f} | "
    if mcq_accuracy is not None:
        info += f"MCQ Acc: {mcq_accuracy:.2%} | "
    info += f"Align: {wandb_metrics.get(f'{prefix}/alignment_corr', 0.0):.3f}"
    if eval_type != "test" and limit_val_batches:
        info += f" (over {batches_processed} batches)"
    print(info)
    
    if eval_type == "test":
        print("="*80 + "\n")
    
    return {
        "loss": avg_loss,
        "accuracy": mcq_accuracy,
        "entropy": mcq_entropy,
        "wandb_metrics": wandb_metrics,
    }


# ============================================================
# Main training
# ============================================================


def train(args):
    """Main training function."""
    # Set up parameters log file immediately at the start
    log_dir = "fine_tune_logs"
    params_log_file_path = _get_log_file_path(
        log_dir, args.model_name, "parameters"
    )

    # Log all parameters (including defaults) before any other operations
    all_params = vars(args)
    write_log(params_log_file_path, {
        "type": "script_parameters",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": all_params
    })
    print(f"All parameters logged to: {params_log_file_path}")

    # ============================================================
    # VALIDATE ALL FILES FIRST (before loading model)
    # ============================================================
    validate_training_files(args)

    # Generate meaningful run name if not provided
    if args.wandb_run_name is None:
        # Extract model name (last part after /)
        model_short = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
        # Extract dataset name from train_data_path
        dataset_name = os.path.basename(args.train_data_path).replace(".jsonl", "").replace("_train", "")
        # Create run name
        args.wandb_run_name = f"{model_short}_{dataset_name}_lr{args.learning_rate}_r{args.lora_r}"

    # Initialize Weights & Biases
    init_wandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        script_path=__file__
    )

    # Handle device selection with CPU fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # Check GPU memory before loading model
    check_and_clear_gpu_memory(device)

    # Log device info
    log_device_info(device)
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(args.model_name)

    # Load model with error handling
    model = load_model_with_error_handling(args.model_name, device)

    # Ensure model config also has pad_token_id set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA configuration
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )
    model = get_peft_model(model, lora).to(device)

    # Load and validate training dataset
    train_dataset = validate_and_load_dataset(args.train_data_path, "training")
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
    
    # Load and validate validation dataset
    val_dataset = None
    val_dataloader = None
    if args.val_data_path:
        val_dataset = validate_and_load_dataset(args.val_data_path, "validation")
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn
        )
        print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
        print(f"  Validation will run every {args.val_interval} steps")
    else:
        print("⚠️  WARNING: No validation dataset provided (--val_data_path not set).")
        print("   Training will proceed WITHOUT validation. No validation metrics will be logged to W&B.")
        print("   To enable validation, provide --val_data_path when running the script.")
    
    # Load MCQ results data if provided
    mcq_results_lookup = None
    # Set up logging file for question matches/mismatches
    log_file_path = None
    if args.mcq_results_data:
        log_file_path = _get_log_file_path(
            log_dir, args.model_name, "question_comparison"
        )
        print(f"Loading pre-recorded Multiple Choice Results data from: "
              f"{args.mcq_results_data}")
        mcq_results_lookup = load_mcq_results_data(
            args.mcq_results_data, log_file_path
        )
        
        # Validate that MCQ results were actually loaded (not empty)
        if mcq_results_lookup is None or len(mcq_results_lookup) == 0:
            raise ValueError(
                f"MCQ results file is empty or could not be loaded: {args.mcq_results_data}\n"
                f"The file exists but contains no valid data. Please check your MCQ results file."
            )
        print(f"✓ MCQ results loaded: {len(mcq_results_lookup)} entries")
    else:
        print("No pre-recorded Multiple Choice Results data has been loaded")

    # Log file path already created above
    if log_file_path:
        print(f"Question comparison log will be written to: {log_file_path}")

    # Set up prompt logging file for first 5 steps
    prompt_log_file_path = _get_log_file_path(
        log_dir, args.model_name, "prompt_pairs"
    )
    print(f"Prompt pairs log will be written to: {prompt_log_file_path}")
    
    # Set up validation metrics logging file
    val_metrics_log_file_path = _get_log_file_path(
        log_dir, args.model_name, "validation_metrics"
    )
    if val_dataloader:
        print(f"Validation metrics log will be written to: {val_metrics_log_file_path}")
    
    # Set up MCQ accuracy assessment logging file
    mcq_accuracy_log_file_path = _get_log_file_path(
        log_dir, args.model_name, "mcq_accuracy_assessment"
    )
    if val_dataset:
        print(f"MCQ accuracy assessment log will be written to: {mcq_accuracy_log_file_path}")
    
    # Set up answer distributions logging file
    answer_distributions_log_file_path = os.path.join(
        log_dir, f"answer_distributions_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}.jsonl"
    )
    print(f"Answer distributions log will be written to: {answer_distributions_log_file_path}")

    # Log dataset info
    log_wandb_config({
        "train_dataset_size": len(train_dataset),
        "train_num_batches": len(train_dataloader),
        "val_dataset_size": len(val_dataset) if val_dataset else 0,
        "val_num_batches": len(val_dataloader) if val_dataloader else 0
    })

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )

    # Log optimizer config
    log_wandb_config({
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate
    })

    # ============================================================
    # BASELINE VALIDATION (before any training)
    # ============================================================
    if val_dataloader:
        print("\n" + "="*80)
        print("Running baseline validation (before training)...")
        print("="*80)
        
        run_evaluation(
            model, tokenizer, device, args, mcq_results_lookup, log_file_path, step=0,
            mcq_accuracy_log_file_path=mcq_accuracy_log_file_path,
            answer_distributions_log_file_path=answer_distributions_log_file_path,
            eval_type="baseline",
            val_dataloader=val_dataloader,
            val_dataset=val_dataset,
            val_metrics_log_file_path=None,  # Don't log baseline to val_metrics file
            limit_val_batches=args.limit_val_batches,
            temperature=args.temperature
        )
        
        print("="*80 + "\n")
        model.train()

    step = 0
    while step < args.max_steps:
        # Training loop
        for batch in train_dataloader:
            if step >= args.max_steps:
                break

            loss = train_step(
                model, tokenizer, batch, device=device, sigma=args.sigma,
                mcq_results_lookup=mcq_results_lookup,
                log_file_path=log_file_path, args=args,
                step=step, prompt_log_file_path=prompt_log_file_path,
                temperature=args.temperature
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training metrics to wandb
            log_wandb_metrics({
                "train/loss": loss.item(),
                "train/step": step,
                "train/learning_rate": args.learning_rate
            }, step=step)

            if step % args.log_interval == 0:
                print(f"Step {step} | Train Loss: {loss.item():.4f}")

            # Run validation
            if val_dataloader and (step + 1) % args.val_interval == 0:
                run_evaluation(
                    model, tokenizer, device, args, mcq_results_lookup, log_file_path, step=step,
                    mcq_accuracy_log_file_path=mcq_accuracy_log_file_path,
                    answer_distributions_log_file_path=answer_distributions_log_file_path,
                    eval_type="validation",
                    val_dataloader=val_dataloader,
                    val_dataset=val_dataset,
                    val_metrics_log_file_path=val_metrics_log_file_path,
                    limit_val_batches=args.limit_val_batches,
                    temperature=args.temperature
                )
                model.train()

            # Save checkpoint to HuggingFace Hub
            if args.save_hf_checkpoints and args.hf_checkpoint_repo:
                if (step + 1) % args.checkpoint_steps == 0:
                    save_hf_checkpoint(
                        model, tokenizer, args.hf_checkpoint_repo, step + 1,
                        private=args.hf_checkpoint_private
                    )
                    print(f"Checkpoint saved to HuggingFace Hub: "
                          f"{args.hf_checkpoint_repo}-step-{step+1}")

            step += 1

    # Save model locally and optionally to HuggingFace Hub
    if args.save_hf and args.hf_repo is None:
        raise ValueError("--hf_repo must be provided when --save_hf is set")

    save_model_final(
        model, tokenizer, args.output_dir,
        hf_repo=args.hf_repo if args.save_hf else None,
        hf_private=args.hf_checkpoint_private,
        save_wandb_artifact=args.save_wandb_artifact
    )

    # ============================================================
    # FINAL TEST EVALUATION (after training)
    # ============================================================
    if args.test_data_path:
        run_evaluation(
            model, tokenizer, device, args, mcq_results_lookup, log_file_path, step,
            mcq_accuracy_log_file_path=mcq_accuracy_log_file_path,
            answer_distributions_log_file_path=answer_distributions_log_file_path,
            eval_type="test",
            data_path=args.test_data_path,
            temperature=args.temperature
        )

    finish_wandb()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train dynamic metacognition model "
                    "(Explicit Confidence Task)"
    )

    # -----------------------
    # Model
    # -----------------------
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HF model name or path")
    parser.add_argument("--device", type=str,
                        default="cuda",
                        choices=["cuda", "cpu"],
                        help="Compute device")

    # -----------------------
    # Data
    # -----------------------
    parser.add_argument("--train_data_path", type=str,
                        required=True,
                        help="Path to JSONL training dataset")
    
    parser.add_argument("--val_data_path", type=str,
                        default=None,
                        help="Path to JSONL validation dataset (optional)")
    
    parser.add_argument("--test_data_path", type=str,
                        default=None,
                        help="Path to JSONL test dataset (optional, will be evaluated after training)")

    parser.add_argument("--batch_size", type=int,
                        default=4,
                        help="Training batch size")

    parser.add_argument("--mcq_results_data", type=str,
                        default=None,
                        help="Path to JSON/JSONL file with previous MCQ results for verification")

    # -----------------------
    # LoRA
    # -----------------------
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Modules to apply LoRA to")

    # -----------------------
    # Training
    # -----------------------
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000000000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Run validation every N training steps")
    parser.add_argument("--limit_val_batches", type=int, default=None,
                        help="Limit validation to N batches (None = use all validation data)")
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Sigma parameter for soft label distribution")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for sampling predictions (0.0 = deterministic/argmax, >0 = sampling)")
    parser.add_argument(
        "--use_recorded_responses", action="store_true", default=True,
        help=("Use recorded MCQ responses (frozen teacher) as training "
              "targets instead of recomputing logits. (Default: True)")
    )
    parser.add_argument(
        "--no_use_recorded_responses", dest="use_recorded_responses",
        action="store_false",
        help=("Disable using recorded responses, use dynamic teacher "
              "(current model logits) instead.")
    )


    # -----------------------
    # Output
    # -----------------------
    parser.add_argument("--output_dir", type=str,
                        default="dynamic_ect_lora",
                        help="Directory to save final model")

    parser.add_argument("--save_hf", action="store_true",
                        help="If set, push LoRA model to HuggingFace Hub")

    parser.add_argument("--hf_repo", type=str,
                        default=None,
                        help="HF repo name if pushing to Hub")

    parser.add_argument("--save_hf_checkpoints", action="store_true",
                        help="If set, save checkpoints to HuggingFace Hub during training")

    parser.add_argument("--hf_checkpoint_repo", type=str,
                        default=None,
                        help="Base HF repo name for checkpoints (e.g., 'username/model-name')")

    parser.add_argument("--checkpoint_steps", type=int,
                        default=500,
                        help="Save checkpoint every N steps")

    parser.add_argument("--hf_checkpoint_private", action="store_true",
                        help="If set, make checkpoint repos private")

    # -----------------------
    # Weights & Biases
    # -----------------------
    parser.add_argument("--wandb_project", type=str,
                        default="llm-metacognition-ect",
                        help="W&B project name")
    
    parser.add_argument("--wandb_run_name", type=str,
                        default=None,
                        help="W&B run name (auto-generated if not provided)")
    
    parser.add_argument("--wandb_tags", type=str, nargs="+",
                        default=None,
                        help="Tags for W&B run")
    
    parser.add_argument("--wandb_notes", type=str,
                        default=None,
                        help="Notes/description for W&B run")
    
    parser.add_argument("--save_wandb_artifact", action="store_true",
                        help="Save model as W&B artifact for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
