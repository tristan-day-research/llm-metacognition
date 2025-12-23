import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # <--- ADDED FOR LORA
import random
import wandb

from finetune_utils import (
    write_log,
    prepare_model_and_tokenizer,
    validate_eval_dataset
)
from finetune_loss import (
    convert_entropy_to_soft_labels,
    compute_loss,
    build_soft_targets_from_entropy
)
from finetune_prompting import (
    build_multiple_choice_question_prompts,
    build_self_confidence_prompts,
    build_other_confidence_prompts,
    run_mcq_forward_pass,
    run_confidence_forward_pass
)
from finetune_data_handling import (
    verify_and_resolve_options,
    write_jsonl,
    load_mcq_results_data,
)
from finetune_data_handling import collate_fn


# ============================================================
# Validation and Test Evaluation Functions
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    dataset,
    sigma,
    compute_confidence=True,
    compute_other_confidence=True,
    loss_type="gaussian_soft_bin_ce",
    temperature=0.0,
    num_samples=None,
    log_file_path=None,
    use_wandb=False,
    wandb_project=None,
    wandb_run_name=None,
    log_prefix="",
    confidence_letter_scheme="A-H",
    confidence_letter_random_seed=None,
    mcq_letter_scheme="A-D",
    mcq_letter_random_seed=None,
):
    """
    Simplified evaluation function for arbitrary datasets.
    
    Wrapper around run_evaluation() that provides a simpler interface
    without requiring a full args object or training-specific parameters.
    
    This is the standalone evaluation function for use outside of training loops.
    For training-time evaluation, use run_evaluation() instead.
    
    WORKFLOW REPLICATION:
    This function ensures perfect replication of the evaluation workflow from
    finetune_ECT.py by:
    - Using the same run_evaluation() function
    - Using the same prompt building functions (build_multiple_choice_question_prompts, etc.)
    - Using the same forward pass functions (run_mcq_forward_pass, run_confidence_forward_pass)
    - Using the same default parameters (temperature=0.0, loss_type defaults, etc.)
    - Creating a SimpleArgs object that matches the args structure used in training
    
    Args:
        model: Loaded model (can be base model or LoRA-finetuned)
        tokenizer: Tokenizer for the model
        dataset: List of dicts with 'question', 'options' (or 'choices'), and 'correct_letter'
        sigma: Gaussian width parameter for soft label conversion (REQUIRED - no default to prevent silent errors)
        compute_confidence: If True, compute self-confidence predictions
        compute_other_confidence: If True, compute other-confidence predictions
        loss_type: Loss type for evaluation ('gaussian_soft_bin_ce' or 'scalar_confidence_mse')
        temperature: Temperature for confidence forward pass
        num_samples: Optional limit on number of samples to evaluate
        log_file_path: Optional path to log file for per-sample results
        use_wandb: If True, log results to Weights & Biases (default: False)
        wandb_project: WandB project name (only used if use_wandb=True)
        wandb_run_name: WandB run name (only used if use_wandb=True)
        log_prefix: Prefix to add to log entry "type" field (e.g., "base_" or "finetuned_")
        confidence_letter_scheme: Letter scheme for confidence bins ('A-H', 'S-Z', or 'random', default: 'A-H')
        confidence_letter_random_seed: Random seed for 'random' scheme (optional)
        mcq_letter_scheme: Letter scheme for MCQ answers ('A-D', 'E-H', etc., or 'random', default: 'A-D')
        mcq_letter_random_seed: Random seed for 'random' mcq_letter_scheme (optional)
    
    Returns:
        results: dict with evaluation metrics (mcq_accuracy, avg_entropy, etc.)
    """
    # Auto-detect device
    device = next(model.parameters()).device
    
    # Initialize WandB if requested
    if use_wandb:
        try:
            import wandb
            # Check if wandb is already initialized
            if wandb.run is None:
                wandb.init(
                    project=wandb_project or "llm-evaluation",
                    name=wandb_run_name,
                    job_type="evaluation"
                )
        except (ImportError, AttributeError) as e:
            print(f"Warning: WandB requested but not available: {e}")
            use_wandb = False
    
    # Create a simple namespace object to pass to run_evaluation()
    class SimpleArgs:
        def __init__(self):
            self.loss_type = loss_type
            self.temperature = temperature
            self.save_wandb_artifact = use_wandb
            self.shuffle_options = False  # Not used in evaluation, but needed for args
            self.compute_confidence = compute_confidence
            self.compute_other_confidence = compute_other_confidence
            self.confidence_letter_scheme = confidence_letter_scheme
            self.confidence_letter_random_seed = confidence_letter_random_seed
            self.mcq_letter_scheme = mcq_letter_scheme
            self.mcq_letter_random_seed = mcq_letter_random_seed
    
    args = SimpleArgs()
    
    # Generate confidence letter mapping
    from finetune_prompting import get_confidence_letter_mapping, get_mcq_letter_mapping
    confidence_letter_mapping = get_confidence_letter_mapping(
        confidence_letter_scheme,
        seed=confidence_letter_random_seed
    )
    mcq_letter_mapping = get_mcq_letter_mapping(
        mcq_letter_scheme,
        seed=mcq_letter_random_seed
    )
    args.mcq_letter_mapping = mcq_letter_mapping
    
    # Note: We skip train_dataset validation checks since this is for arbitrary datasets
    # Pass None for train_dataset_qids and train_dataset_questions
    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        val_dataset=dataset,
        device=device,
        args=args,
        step_name="evaluation",
        num_samples=num_samples,
        log_file_path=log_file_path,
        step=None,
        mcq_results_lookup=None,
        train_dataset_qids=None,  # Skip train dataset validation
        train_dataset_questions=None,  # Skip train dataset validation
        log_prefix=log_prefix,
        sigma=sigma,
        confidence_letter_mapping=confidence_letter_mapping,
        mcq_letter_mapping=mcq_letter_mapping,
    )
    
    return results


def run_evaluation(
    model,
    tokenizer,
    val_dataset,
    device,
    args,
    step_name=None,
    num_samples=None,
    log_file_path=None,
    step=None,
    mcq_results_lookup=None,
    train_dataset_qids=None,
    train_dataset_questions=None,
    log_prefix="",
    sigma=None,
    val_on_frozen=False,
    confidence_letter_mapping=None,
    mcq_letter_mapping=None,
):
    """
    Evaluation loop:
    
    CRITICAL: This function should ONLY receive val_dataset (or test_dataset).
    Never pass train_dataset to this function.

    For each sample:
        1. Run MCQ pass (extract predicted letter + entropy) OR use frozen pre-recorded data
        2. Run confidence pass (A–H distribution)
        3. Compute soft targets from entropy
        4. Compute loss
        5. Log per-sample results

    After loop:
        - Compute MCQ accuracy
        - Compute average entropy
        - Compute average expected confidence
        - Compute average loss
        - Compute model answer distribution
        - Compute correct answer distribution
    
    Args:
        val_dataset: Validation or test dataset (NEVER train_dataset)
        train_dataset_qids: Set of qids from train_dataset for validation (optional)
        train_dataset_questions: Set of normalized question texts from train_dataset (optional)
        sigma: Gaussian width parameter for soft label conversion (REQUIRED - no default to prevent silent errors)
        val_on_frozen: If True, use pre-recorded MCQ answers and entropy from mcq_results_lookup instead of live model
        confidence_letter_mapping: Optional dict mapping A-H (internal) to display letters.
                                  If None, will try to get from args.confidence_letter_mapping or
                                  generate from args.confidence_letter_scheme.
    """
    # Defensive check: ensure we're in eval mode
    model.eval()
    
    # Additional defensive check: verify dataset is not empty
    if len(val_dataset) == 0:
        raise ValueError("run_evaluation() received empty dataset - this should not happen")
    
    # CRITICAL: Require sigma parameter (no default to prevent silent errors)
    if sigma is None:
        # Try to get from args if available
        if hasattr(args, 'sigma') and args.sigma is not None:
            sigma = args.sigma
        else:
            raise ValueError(
                "sigma parameter is REQUIRED for run_evaluation(). "
                "Either pass sigma directly or ensure args.sigma is set. "
                "No default value to prevent silent training errors."
            )
    
    # Get confidence letter mapping from args if not provided
    if confidence_letter_mapping is None:
        if hasattr(args, 'confidence_letter_mapping') and args.confidence_letter_mapping is not None:
            confidence_letter_mapping = args.confidence_letter_mapping
        elif hasattr(args, 'confidence_letter_scheme') and args.confidence_letter_scheme is not None:
            from finetune_prompting import get_confidence_letter_mapping
            confidence_letter_mapping = get_confidence_letter_mapping(
                args.confidence_letter_scheme,
                seed=getattr(args, 'confidence_letter_random_seed', None)
            )
        else:
            # Default to A-H if nothing specified
            from finetune_prompting import get_confidence_letter_mapping
            confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    
    # Check if per-question randomization is enabled
    randomize_per_question = getattr(args, 'randomize_letters_per_question', False)
    
    # Get MCQ letter mapping from args if not provided (only if NOT randomizing per question)
    if randomize_per_question:
        # Will generate per sample in the loop
        mcq_letter_mapping = None
    elif mcq_letter_mapping is None:
        if hasattr(args, 'mcq_letter_mapping') and args.mcq_letter_mapping is not None:
            mcq_letter_mapping = args.mcq_letter_mapping
        elif hasattr(args, 'mcq_letter_scheme') and args.mcq_letter_scheme is not None:
            from finetune_prompting import get_mcq_letter_mapping
            mcq_letter_mapping = get_mcq_letter_mapping(
                args.mcq_letter_scheme,
                seed=getattr(args, 'mcq_letter_random_seed', None)
            )
        else:
            # Default to A-D if nothing specified
            from finetune_prompting import get_mcq_letter_mapping
            mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    
    # Get confidence letter mapping (only if NOT randomizing per question)
    if randomize_per_question:
        # Will generate per sample in the loop
        confidence_letter_mapping = None
    elif confidence_letter_mapping is None:
        if hasattr(args, 'confidence_letter_mapping') and args.confidence_letter_mapping is not None:
            confidence_letter_mapping = args.confidence_letter_mapping
        elif hasattr(args, 'confidence_letter_scheme') and args.confidence_letter_scheme is not None:
            from finetune_prompting import get_confidence_letter_mapping
            confidence_letter_mapping = get_confidence_letter_mapping(
                args.confidence_letter_scheme,
                seed=getattr(args, 'confidence_letter_random_seed', None)
            )
        else:
            # Default to A-H if nothing specified
            from finetune_prompting import get_confidence_letter_mapping
            confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    
    # Validate val_on_frozen requirements
    if val_on_frozen:
        if mcq_results_lookup is None:
            raise ValueError(
                "val_on_frozen=True requires mcq_results_lookup to be provided. "
                "Please provide --mcq_results_data when using --val_on_frozen."
            )
    
    # Defensive check: Verify val_dataset doesn't contain any train_dataset questions
    validate_eval_dataset(val_dataset, train_dataset_qids, train_dataset_questions, function_name="run_evaluation")

    # ===========================================================
    # Select samples
    # ===========================================================
    # Use seeded RNG for reproducible sample selection
    # CRITICAL: Use a different seed (999) than training RNG (42) to ensure
    # complete independence between train and eval sampling
    # This ensures self_live_corr is computed on the same samples across runs
    eval_rng = np.random.default_rng(999)
    if num_samples is not None:
        idxs = eval_rng.choice(len(val_dataset), size=num_samples, replace=False)
    else:
        idxs = np.arange(len(val_dataset))

    # # At the start of run_evaluation()
    # print("DEBUG: First question:")
    # print(f"Question: {val_dataset[0]['question'][:100]}")
    # print(f"Options: {val_dataset[0].get('options', {})}")
    # print(f"Correct: {val_dataset[0].get('correct_letter')}")
    # print(f"Shuffle setting: {args.shuffle_options}")

    # ----- Accumulators -----
    correctness_flags = []
    entropy_values = []
    expected_conf_values = []
    expected_other_conf_values = []  # Other confidence (college-educated people)
    loss_values = []
    predicted_letters = []  # Display letters (e.g., E, F, G, H)
    predicted_positions = []  # Position indices (0-3, corresponding to A-D internally)
    correct_letters = []
    correct_positions = []  # Position indices for correct answers
    predicted_confidence_letters = []  # Display letter confidence predictions (self) (e.g., S-Z)
    predicted_confidence_bin_indices = []  # Bin indices (0-7, corresponding to A-H internally)
    predicted_other_confidence_letters = []  # Display letter confidence predictions (other) (e.g., S-Z)
    predicted_other_confidence_bin_indices = []  # Bin indices for other confidence
    prerecorded_entropy_values = []  # Pre-recorded entropy from mcq_results_data

    # ==================== MAIN LOOP ==========================
    total_samples = len(idxs)
    for idx_in_loop, i in enumerate(idxs):
        # Progress marker every 100 questions
        if (idx_in_loop + 1) % 100 == 0:
            print(f"  Progress: {idx_in_loop + 1}/{total_samples} questions evaluated ({100.0 * (idx_in_loop + 1) / total_samples:.1f}%)")
        
        batch = val_dataset[i:i+1]    # single-sample batch (list of 1 dict)

        # Generate letter mappings for this sample if per-question randomization is enabled
        if randomize_per_question:
            from finetune_prompting import get_mcq_letter_mapping, get_confidence_letter_mapping
            # Generate new mappings for this sample
            sample_mcq_letter_mapping = get_mcq_letter_mapping(
                args.mcq_letter_scheme,
                seed=None  # Don't use seed for per-question randomization
            )
            sample_confidence_letter_mapping = get_confidence_letter_mapping(
                args.confidence_letter_scheme,
                seed=None  # Don't use seed for per-question randomization
            )
        else:
            # Use pre-generated mappings
            sample_mcq_letter_mapping = mcq_letter_mapping
            sample_confidence_letter_mapping = confidence_letter_mapping

        # 0. Look up pre-recorded entropy if available (before any option changes)
        # We do this BEFORE verify_and_resolve_options to avoid option mismatches
        prerecorded_entropy = None
        if mcq_results_lookup is not None:
            qid = batch[0].get("qid")
            if qid and str(qid) in mcq_results_lookup:
                prerecorded_entropy = mcq_results_lookup[str(qid)].get("entropy")
                if prerecorded_entropy is not None:
                    prerecorded_entropy_values.append(prerecorded_entropy)
                else:
                    prerecorded_entropy_values.append(None)
            else:
                prerecorded_entropy_values.append(None)
        else:
            prerecorded_entropy_values.append(None)
        
        # 0.5. Resolve options (but don't change them for evaluation - keep shuffled options)
        # We pass None to avoid replacing options with unshuffled recorded options
        # which would break the correct_letter mapping
        resolved_row, options = verify_and_resolve_options(
            batch[0],
            mcq_results_lookup=None,  # Don't replace options in evaluation
            log_file_path=None
        )
        # Keep original batch options - don't replace with recorded options
        # batch[0]["options"] = options  # COMMENTED OUT - keep shuffled options

        # # Optional shuffle
        # if args.shuffle_options:
        #     shuffle_options_and_update_correct_letter(batch[0])


        # ==================================================
        # 1. MCQ pass (live or frozen)
        # ==================================================
        correct_answer_letter = batch[0]["correct_letter"]
        
        if val_on_frozen:
            # Use pre-recorded frozen data from mcq_results_lookup
            qid = batch[0].get("qid")
            if qid and str(qid) in mcq_results_lookup:
                frozen_data = mcq_results_lookup[str(qid)]
                frozen_subject_answer_letter = frozen_data.get("subject_answer")
                entropy_value = frozen_data.get("entropy")
                frozen_options = frozen_data.get("options", {})
                
                # Validate that we have the required data
                if frozen_subject_answer_letter is None:
                    raise ValueError(
                        f"val_on_frozen=True but missing 'subject_answer' for qid={qid} in mcq_results_lookup. "
                        f"This should not happen if dataset was properly filtered."
                    )
                if entropy_value is None:
                    raise ValueError(
                        f"val_on_frozen=True but missing 'entropy' for qid={qid} in mcq_results_lookup. "
                        f"This should not happen if dataset was properly filtered."
                    )
                
                # CRITICAL: Map frozen answer letter to current option order
                # The frozen subject_answer is a letter (A-D) that refers to the option order
                # from when the data was recorded. We need to:
                # 1. Get the actual option text that the frozen letter refers to
                # 2. Find which letter that option text corresponds to in the current dataset's option order
                if frozen_subject_answer_letter in frozen_options:
                    frozen_answer_text = frozen_options[frozen_subject_answer_letter]
                    # Find which letter this text corresponds to in the current batch's options
                    current_options = batch[0]["options"]
                    predicted_answer_letter = None
                    for letter, option_text in current_options.items():
                        if option_text == frozen_answer_text:
                            predicted_answer_letter = letter
                            break
                    
                    if predicted_answer_letter is None:
                        # Fallback: if exact text match fails, try normalized comparison
                        from finetune_data_handling import normalize_text
                        frozen_answer_normalized = normalize_text(frozen_answer_text)
                        for letter, option_text in current_options.items():
                            if normalize_text(option_text) == frozen_answer_normalized:
                                predicted_answer_letter = letter
                                break
                    
                    if predicted_answer_letter is None:
                        raise ValueError(
                            f"val_on_frozen=True: Could not map frozen answer '{frozen_answer_text}' "
                            f"(from letter '{frozen_subject_answer_letter}') to current options for qid={qid}. "
                            f"Current options: {list(current_options.values())}"
                        )
                else:
                    raise ValueError(
                        f"val_on_frozen=True: Frozen subject_answer '{frozen_subject_answer_letter}' "
                        f"not found in frozen_options for qid={qid}. Frozen options keys: {list(frozen_options.keys())}"
                    )
            else:
                # This should not happen if dataset was properly filtered upfront
                raise ValueError(
                    f"val_on_frozen=True but no frozen data found for qid={qid} in mcq_results_lookup. "
                    f"Dataset should have been filtered to only include questions with pre-recorded results."
                )
        else:
            # Use live model (current behavior)
            mcq_prompts = build_multiple_choice_question_prompts(batch, tokenizer, sample_mcq_letter_mapping)

            # print("DEBUG mcq_prompts", mcq_prompts)

            mcq_out = run_mcq_forward_pass(
                model=model,
                tokenizer=tokenizer,
                prompts=mcq_prompts,
                device=device,
                temperature=0.0,
                mcq_letter_mapping=sample_mcq_letter_mapping,
            )

            predicted_answer_letter = mcq_out["pred_letters"][0]  # Display letter
            predicted_answer_position = mcq_out["pred_positions"][0]  # Position index (0-3)
            entropy_value = mcq_out["entropy"][0].item()
        
        # Map correct_answer_letter to position index (0-3 for A-D)
        correct_answer_position = ord(correct_answer_letter) - ord('A') if correct_answer_letter in "ABCD" else None
        
        # Map correct_answer_letter to display letter for logging
        if correct_answer_position is not None and correct_answer_position < 4:
            correct_answer_display_letter = sample_mcq_letter_mapping[chr(ord('A') + correct_answer_position)]
        else:
            correct_answer_display_letter = correct_answer_letter  # Fallback if mapping fails
        
        # For frozen validation, we need to map the predicted_answer_letter back to position
        if val_on_frozen:
            # predicted_answer_letter is in A-D space (internal), map to position
            predicted_answer_position = ord(predicted_answer_letter) - ord('A') if predicted_answer_letter in "ABCD" else None
            # Map to display letter using sample_mcq_letter_mapping
            if predicted_answer_position is not None and predicted_answer_position < 4:
                predicted_answer_letter = sample_mcq_letter_mapping[chr(ord('A') + predicted_answer_position)]
            else:
                predicted_answer_letter = None

        predicted_letters.append(predicted_answer_letter)
        predicted_positions.append(predicted_answer_position)
        correct_letters.append(correct_answer_display_letter)  # Use display letter for logging
        correct_positions.append(correct_answer_position)

        # Compare positions for correctness (more robust than comparing display letters)
        if predicted_answer_position is not None and correct_answer_position is not None:
            correctness_flags.append(
                1.0 if predicted_answer_position == correct_answer_position else 0.0
            )
        else:
            # Fallback: compare display letters if positions are unavailable
            correctness_flags.append(
                1.0 if predicted_answer_letter == correct_answer_display_letter else 0.0
            )
        entropy_values.append(entropy_value)

        # ==================================================
        # 2. Soft targets
        # ==================================================
        # entropy_value is a scalar float from MCQ evaluation or prerecorded data
        # Convert to tensor and use the resolved sigma (from parameter or args.sigma)
        entropy_tensor = torch.tensor([entropy_value], dtype=torch.float32, device=device)
        soft_targets = build_soft_targets_from_entropy(entropy_tensor, sigma=sigma)
 


        # ==================================================
        # 3. Self Confidence pass (optional)
        # ==================================================
        compute_confidence = getattr(args, 'compute_confidence', True)
        compute_other_confidence = getattr(args, 'compute_other_confidence', True)
        
        if compute_confidence:
            conf_prompts = build_self_confidence_prompts(batch, tokenizer, sample_confidence_letter_mapping, sample_mcq_letter_mapping)

            conf_out = run_confidence_forward_pass(
                model=model,
                tokenizer=tokenizer,
                prompts=conf_prompts,
                device=device,
                temperature=args.temperature,
                confidence_letter_mapping=sample_confidence_letter_mapping,
            )

            logits8 = conf_out["logits8"][0]
            expected_confidence_value = conf_out["expected_conf"][0].item()
            predicted_confidence_letter = conf_out["pred_bins"][0]  # Display letter (e.g., S-Z)
            predicted_confidence_bin_index = conf_out["pred_bin_indices"][0]  # Bin index (0-7 for A-H)
        else:
            # Skip confidence computation - use dummy values
            logits8 = None
            expected_confidence_value = 0.0
            predicted_confidence_letter = None
            predicted_confidence_bin_index = None

        expected_conf_values.append(expected_confidence_value)
        predicted_confidence_letters.append(predicted_confidence_letter)
        predicted_confidence_bin_indices.append(predicted_confidence_bin_index)

        # ==================================================
        # 3.5. Other confidence pass (optional)
        # ==================================================
        if compute_other_confidence:
            other_conf_prompts = build_other_confidence_prompts(batch, tokenizer, sample_confidence_letter_mapping, sample_mcq_letter_mapping)

            other_conf_out = run_confidence_forward_pass(
                model=model,
                tokenizer=tokenizer,
                prompts=other_conf_prompts,
                device=device,
                temperature=args.temperature,
                confidence_letter_mapping=sample_confidence_letter_mapping,
            )

            expected_other_confidence_value = other_conf_out["expected_conf"][0].item()
            predicted_other_confidence_letter = other_conf_out["pred_bins"][0]  # Display letter (e.g., S-Z)
            predicted_other_confidence_bin_index = other_conf_out["pred_bin_indices"][0]  # Bin index (0-7 for A-H)
        else:
            # Skip other confidence computation - use dummy values
            expected_other_confidence_value = 0.0
            predicted_other_confidence_letter = None
            predicted_other_confidence_bin_index = None
            
        expected_other_conf_values.append(expected_other_confidence_value)
        predicted_other_confidence_letters.append(predicted_other_confidence_letter)
        predicted_other_confidence_bin_indices.append(predicted_other_confidence_bin_index)

        # ==================================================
        # 4. Loss (only if confidence was computed)
        # ==================================================
        if compute_confidence and logits8 is not None:
            # Prepare inputs based on loss type
            if args.loss_type == 'scalar_confidence_mse':
                # For scalar_confidence_mse, pass entropy directly
                # logits8 is [8], need to unsqueeze to [1, 8] for batch dimension
                logits8_batch = logits8.unsqueeze(0) if logits8.ndim == 1 else logits8
                entropy_tensor = torch.tensor([entropy_value], dtype=torch.float32, device=device)
                loss = compute_loss(logits8_batch, entropy=entropy_tensor, loss_type=args.loss_type, reduction='mean')
                loss_value = loss.item()
            else:
                # For gaussian_soft_bin_ce, pass soft_targets
                # logits8 is [8], need to unsqueeze to [1, 8] for batch dimension
                logits8_batch = logits8.unsqueeze(0) if logits8.ndim == 1 else logits8
                soft_targets_batch = soft_targets.unsqueeze(0) if soft_targets.ndim == 1 else soft_targets
                loss = compute_loss(logits8_batch, soft_targets=soft_targets_batch, loss_type=args.loss_type, reduction='mean')
                loss_value = loss.item()
            loss_values.append(loss_value)
        else:
            # Skip loss computation if confidence was not computed
            loss_values.append(0.0)

        # ==================================================
        # 5. Optional per-sample logging
        # ==================================================
        if log_file_path is not None:
            log_entry = {
                "type": f"{log_prefix}eval_sample" if log_prefix else "eval_sample",
                "qid": batch[0].get("qid"),
                "question": batch[0]["question"],
                "model_answer": predicted_answer_letter,  # Display letter (e.g., E, F, G, H)
                "model_answer_position": predicted_answer_position,  # Position index (0-3 for A-D)
                "correct_answer": correct_answer_letter,  # Display letter
                "correct_answer_position": correct_answer_position,  # Position index (0-3 for A-D)
                "predicted_confidence_letter": predicted_confidence_letter,  # Display letter (e.g., S-Z)
                "predicted_confidence_bin_index": predicted_confidence_bin_index,  # Bin index (0-7 for A-H)
                "predicted_other_confidence_letter": predicted_other_confidence_letter,  # Display letter
                "predicted_other_confidence_bin_index": predicted_other_confidence_bin_index,  # Bin index
                "entropy": entropy_value,
                "expected_confidence": expected_confidence_value,
                "expected_other_confidence": expected_other_confidence_value,
                "loss": loss_value,
            }
            # Add pre-recorded entropy if available
            if prerecorded_entropy is not None:
                log_entry["prerecorded_entropy"] = prerecorded_entropy
            write_jsonl(log_file_path, log_entry)

    # ===========================================================
    # DISTRIBUTION STATS - MCQ Answers
    # Track both letter distributions and position distributions
    # ===========================================================

    def count_dist(values, letters):
        return {letter: values.count(letter) for letter in letters}
    
    def count_dist_indices(values, max_index):
        """Count distribution of indices (0 to max_index)."""
        dist = {i: 0 for i in range(max_index + 1)}
        for v in values:
            if v is not None and 0 <= v <= max_index:
                dist[v] = dist.get(v, 0) + 1
        return dist

    n = len(predicted_letters)
    
    # Get MCQ display letters for distribution stats
    if randomize_per_question:
        # When randomizing per question, collect all unique letters that appeared
        all_mcq_letters = set([l for l in predicted_letters + correct_letters if l is not None])
        mcq_display_letters_str = ''.join(sorted(all_mcq_letters))
    else:
        # Use the fixed mapping
        mcq_display_letters = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
        mcq_display_letters_str = ''.join(mcq_display_letters)
    
    # Count by display letters
    pred_dist_letters = count_dist([l for l in predicted_letters if l is not None], mcq_display_letters_str)
    gold_dist_letters = count_dist([l for l in correct_letters if l is not None], mcq_display_letters_str)
    pred_dist_letters_pct = {k: (v / n) * 100.0 for k, v in pred_dist_letters.items()}
    gold_dist_letters_pct = {k: (v / n) * 100.0 for k, v in gold_dist_letters.items()}
    
    # Count by position (0-3)
    pred_dist_positions = count_dist_indices([p for p in predicted_positions if p is not None], 3)
    gold_dist_positions = count_dist_indices([p for p in correct_positions if p is not None], 3)
    pred_dist_positions_pct = {k: (v / n) * 100.0 for k, v in pred_dist_positions.items()}
    gold_dist_positions_pct = {k: (v / n) * 100.0 for k, v in gold_dist_positions.items()}

    # Pretty print - Letter distributions
    print("\n============================================================")
    print(f"{step_name.upper()} — MCQ Answer Distributions (by Letter)")
    print("============================================================")
    print(f"Correct (Ground Truth) Distribution ({mcq_display_letters_str}):")
    for k in mcq_display_letters_str:
        print(f"  {k}: {gold_dist_letters[k]:4d}  ({gold_dist_letters_pct[k]:6.2f}%)")

    print(f"\nModel Prediction Distribution ({mcq_display_letters_str}):")
    for k in mcq_display_letters_str:
        print(f"  {k}: {pred_dist_letters[k]:4d}  ({pred_dist_letters_pct[k]:6.2f}%)")
    
    # Pretty print - Position distributions
    print("\n============================================================")
    print(f"{step_name.upper()} — MCQ Answer Distributions (by Position)")
    print("============================================================")
    print("Correct (Ground Truth) Distribution (positions 0-3, corresponding to A-D):")
    for pos in range(4):
        letter = chr(ord('A') + pos)
        print(f"  Position {pos} ({letter}): {gold_dist_positions[pos]:4d}  ({gold_dist_positions_pct[pos]:6.2f}%)")

    print("\nModel Prediction Distribution (positions 0-3, corresponding to A-D):")
    for pos in range(4):
        letter = chr(ord('A') + pos)
        print(f"  Position {pos} ({letter}): {pred_dist_positions[pos]:4d}  ({pred_dist_positions_pct[pos]:6.2f}%)")

    # ===========================================================
    # CONFIDENCE DISTRIBUTION STATS - only if computed
    # Use display letters from confidence_letter_mapping
    # ===========================================================
    compute_confidence = getattr(args, 'compute_confidence', True)
    compute_other_confidence = getattr(args, 'compute_other_confidence', True)
    
    # Get display letters from confidence_letter_mapping for distribution stats
    if randomize_per_question:
        # When randomizing per question, collect all unique letters that appeared
        all_conf_letters = set([l for l in predicted_confidence_letters + predicted_other_confidence_letters if l is not None])
        display_letters_str = ''.join(sorted(all_conf_letters))
    else:
        # Use the fixed mapping
        display_letters = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
        display_letters_str = ''.join(display_letters)
    
    # Confidence bin labels (A-H correspond to confidence levels)
    confidence_bin_labels = {
        0: "<5%", 1: "5-10%", 2: "10-20%", 3: "20-40%",
        4: "40-60%", 5: "60-80%", 6: "80-90%", 7: ">90%"
    }
    
    if compute_confidence:
        # Count using display letters (which are what's actually stored)
        conf_dist_letters = count_dist([l for l in predicted_confidence_letters if l is not None], display_letters_str)
        conf_dist_letters_pct = {k: (v / n) * 100.0 for k, v in conf_dist_letters.items()}
        
        # Count by bin index (0-7 for A-H)
        conf_dist_bins = count_dist_indices([b for b in predicted_confidence_bin_indices if b is not None], 7)
        conf_dist_bins_pct = {k: (v / n) * 100.0 for k, v in conf_dist_bins.items()}

        # Pretty print confidence distribution using display letters
        print("\n============================================================")
        print(f"{step_name.upper()} — Self Confidence Prediction Distributions (by Letter)")
        print("============================================================")
        print(f"Model Self Confidence Prediction Distribution ({display_letters_str}):")
        for k in display_letters_str:
            print(f"  {k}: {conf_dist_letters[k]:4d}  ({conf_dist_letters_pct[k]:6.2f}%)")
        
        # Pretty print confidence distribution by bin
        print("\n============================================================")
        print(f"{step_name.upper()} — Self Confidence Prediction Distributions (by Bin)")
        print("============================================================")
        print("Model Self Confidence Prediction Distribution (bins 0-7, corresponding to A-H):")
        for bin_idx in range(8):
            bin_letter = chr(ord('A') + bin_idx)
            bin_label = confidence_bin_labels[bin_idx]
            print(f"  Bin {bin_idx} ({bin_letter}, {bin_label}): {conf_dist_bins[bin_idx]:4d}  ({conf_dist_bins_pct[bin_idx]:6.2f}%)")
    else:
        conf_dist_letters = {k: 0 for k in display_letters_str}
        conf_dist_letters_pct = {k: 0.0 for k in display_letters_str}
        conf_dist_bins = {i: 0 for i in range(8)}
        conf_dist_bins_pct = {i: 0.0 for i in range(8)}

    # ===========================================================
    # OTHER CONFIDENCE DISTRIBUTION STATS - only if computed
    # Use display letters from confidence_letter_mapping
    # ===========================================================
    if compute_other_confidence:
        # Count using display letters (which are what's actually stored)
        other_conf_dist_letters = count_dist([l for l in predicted_other_confidence_letters if l is not None], display_letters_str)
        other_conf_dist_letters_pct = {k: (v / n) * 100.0 for k, v in other_conf_dist_letters.items()}
        
        # Count by bin index (0-7 for A-H)
        other_conf_dist_bins = count_dist_indices([b for b in predicted_other_confidence_bin_indices if b is not None], 7)
        other_conf_dist_bins_pct = {k: (v / n) * 100.0 for k, v in other_conf_dist_bins.items()}

        # Pretty print other confidence distribution using display letters
        print("\n============================================================")
        print(f"{step_name.upper()} — Other Confidence Prediction Distributions (by Letter)")
        print("============================================================")
        print(f"Model Other Confidence Prediction Distribution ({display_letters_str}):")
        for k in display_letters_str:
            print(f"  {k}: {other_conf_dist_letters[k]:4d}  ({other_conf_dist_letters_pct[k]:6.2f}%)")
        
        # Pretty print other confidence distribution by bin
        print("\n============================================================")
        print(f"{step_name.upper()} — Other Confidence Prediction Distributions (by Bin)")
        print("============================================================")
        print("Model Other Confidence Prediction Distribution (bins 0-7, corresponding to A-H):")
        for bin_idx in range(8):
            bin_letter = chr(ord('A') + bin_idx)
            bin_label = confidence_bin_labels[bin_idx]
            print(f"  Bin {bin_idx} ({bin_letter}, {bin_label}): {other_conf_dist_bins[bin_idx]:4d}  ({other_conf_dist_bins_pct[bin_idx]:6.2f}%)")
    else:
        other_conf_dist_letters = {k: 0 for k in display_letters_str}
        other_conf_dist_letters_pct = {k: 0.0 for k in display_letters_str}
        other_conf_dist_bins = {i: 0 for i in range(8)}
        other_conf_dist_bins_pct = {i: 0.0 for i in range(8)}

    # ===========================================================
    # FINAL METRICS
    # ===========================================================
    results = {
        "mcq_accuracy": float(np.mean(correctness_flags)),
        "avg_entropy": float(np.mean(entropy_values)),
        "avg_confidence": float(np.mean(expected_conf_values)),
        "avg_other_confidence": float(np.mean(expected_other_conf_values)),
        "avg_loss": float(np.mean(loss_values)),
        "loss_type": args.loss_type,  # Log the loss type used for evaluation
        "n_samples": n,
        # MCQ distributions by letter
        "correct_answer_distribution_by_letter_raw": gold_dist_letters,
        "correct_answer_distribution_by_letter_pct": gold_dist_letters_pct,
        "predicted_answer_distribution_by_letter_raw": pred_dist_letters,
        "predicted_answer_distribution_by_letter_pct": pred_dist_letters_pct,
        # MCQ distributions by position
        "correct_answer_distribution_by_position_raw": gold_dist_positions,
        "correct_answer_distribution_by_position_pct": gold_dist_positions_pct,
        "predicted_answer_distribution_by_position_raw": pred_dist_positions,
        "predicted_answer_distribution_by_position_pct": pred_dist_positions_pct,
        # Confidence distributions by letter
        "predicted_confidence_distribution_by_letter_raw": conf_dist_letters,
        "predicted_confidence_distribution_by_letter_pct": conf_dist_letters_pct,
        "predicted_other_confidence_distribution_by_letter_raw": other_conf_dist_letters,
        "predicted_other_confidence_distribution_by_letter_pct": other_conf_dist_letters_pct,
        # Confidence distributions by bin
        "predicted_confidence_distribution_by_bin_raw": conf_dist_bins,
        "predicted_confidence_distribution_by_bin_pct": conf_dist_bins_pct,
        "predicted_other_confidence_distribution_by_bin_raw": other_conf_dist_bins,
        "predicted_other_confidence_distribution_by_bin_pct": other_conf_dist_bins_pct,
    }

    # Compute additional metrics for wandb
    correctness_arr = np.array(correctness_flags)
    entropy_arr = np.array(entropy_values)
    
    # Only compute confidence metrics if confidence was computed
    compute_confidence = getattr(args, 'compute_confidence', True)
    compute_other_confidence = getattr(args, 'compute_other_confidence', True)
    
    if compute_confidence:
        conf_arr = np.array(expected_conf_values)
        std_conf = float(np.std(conf_arr)) if len(conf_arr) > 1 else 0.0
    else:
        conf_arr = np.array([0.0] * len(expected_conf_values))
        std_conf = 0.0
    
    if compute_other_confidence:
        other_conf_arr = np.array(expected_other_conf_values)
        std_other_conf = float(np.std(other_conf_arr)) if len(other_conf_arr) > 1 else 0.0
    else:
        other_conf_arr = np.array([0.0] * len(expected_other_conf_values))
        std_other_conf = 0.0
    
    std_entropy = float(np.std(entropy_arr)) if len(entropy_arr) > 1 else 0.0
    
    # Self live correlation: entropy from live model's output logits correlated with 
    # live model's prediction of its own confidence
    self_live_corr = 0.0
    if compute_confidence and len(conf_arr) > 1 and std_conf > 0.001:
        try:
            self_live_corr, _ = pearsonr(entropy_arr, conf_arr)
            self_live_corr = float(self_live_corr)
        except Exception:
            self_live_corr = 0.0
    
    # Other live correlation: entropy from live model's output logits correlated with 
    # live model's prediction of 'other's' accuracy on the question
    other_live_corr = 0.0
    if compute_other_confidence and len(other_conf_arr) > 1 and std_other_conf > 0.001:
        try:
            other_live_corr, _ = pearsonr(entropy_arr, other_conf_arr)
            other_live_corr = float(other_live_corr)
        except Exception:
            other_live_corr = 0.0
    
    # Self frozen correlation: entropy from pre-recorded output logits correlated with 
    # live model's prediction of its own confidence
    # Other frozen correlation: entropy from pre-recorded output logits correlated with 
    # live model's prediction of 'other's' accuracy on the question
    self_frozen_corr = 0.0
    other_frozen_corr = 0.0
    avg_prerecorded_entropy = None
    std_prerecorded_entropy = None
    if mcq_results_lookup is not None and len(prerecorded_entropy_values) > 0:
        # Filter out None values and align with confidence array
        valid_prerecorded = []
        valid_conf_for_prerecorded = []
        valid_other_conf_for_prerecorded = []
        for i, (prerec_ent, conf_val, other_conf_val) in enumerate(zip(prerecorded_entropy_values, expected_conf_values, expected_other_conf_values)):
            if prerec_ent is not None:
                valid_prerecorded.append(prerec_ent)
                if compute_confidence:
                    valid_conf_for_prerecorded.append(conf_val)
                if compute_other_confidence:
                    valid_other_conf_for_prerecorded.append(other_conf_val)
        
        if len(valid_prerecorded) > 1:
            prerecorded_arr = np.array(valid_prerecorded)
            avg_prerecorded_entropy = float(np.mean(prerecorded_arr))
            std_prerecorded_entropy = float(np.std(prerecorded_arr)) if len(prerecorded_arr) > 1 else 0.0
            
            if compute_confidence and len(valid_conf_for_prerecorded) > 1:
                conf_arr_prerecorded = np.array(valid_conf_for_prerecorded)
                std_conf_prerecorded = float(np.std(conf_arr_prerecorded)) if len(conf_arr_prerecorded) > 1 else 0.0
                
                if std_prerecorded_entropy > 0.001 and std_conf_prerecorded > 0.001 and len(conf_arr_prerecorded) > 1:
                    try:
                        self_frozen_corr, _ = pearsonr(prerecorded_arr, conf_arr_prerecorded)
                        self_frozen_corr = float(self_frozen_corr)
                    except Exception:
                        self_frozen_corr = 0.0
            
            # Correlation between pre-recorded entropy and other confidence
            if compute_other_confidence and len(valid_other_conf_for_prerecorded) > 1:
                other_conf_arr_prerecorded = np.array(valid_other_conf_for_prerecorded)
                std_other_conf_prerecorded = float(np.std(other_conf_arr_prerecorded)) if len(other_conf_arr_prerecorded) > 1 else 0.0
                
                if std_prerecorded_entropy > 0.001 and std_other_conf_prerecorded > 0.001 and len(other_conf_arr_prerecorded) > 1:
                    try:
                        other_frozen_corr, _ = pearsonr(prerecorded_arr, other_conf_arr_prerecorded)
                        other_frozen_corr = float(other_frozen_corr)
                    except Exception:
                        other_frozen_corr = 0.0
    
    # Calibration: correlation between confidence and correctness
    calibration_corr = 0.0
    if compute_confidence and len(conf_arr) > 1 and std_conf > 0.001:
        try:
            calibration_corr, _ = pearsonr(conf_arr, correctness_arr)
            calibration_corr = float(calibration_corr)
        except Exception:
            calibration_corr = 0.0

    # Log the summary as one blob
    if log_file_path is not None:
        write_jsonl(log_file_path, {
            "type": f"{log_prefix}eval_summary" if log_prefix else "eval_summary",
            **results
        })

    # ===========================================================
    # WANDB LOGGING
    # ===========================================================
    if args.save_wandb_artifact:
        try:
            
            prefix = "val" 
            
            wandb_metrics = {
                f"{prefix}/accuracy": results["mcq_accuracy"],
                f"{prefix}/loss": results["avg_loss"],
                f"{prefix}/entropy": results["avg_entropy"],
                f"{prefix}/confidence": results["avg_confidence"],
                f"{prefix}/other_confidence": results["avg_other_confidence"],
                f"{prefix}/std_confidence": std_conf,
                f"{prefix}/std_other_confidence": std_other_conf,
                f"{prefix}/std_entropy": std_entropy,
                f"{prefix}/self_live_corr": self_live_corr,
                f"{prefix}/other_live_corr": other_live_corr,
                f"{prefix}/calibration_corr": calibration_corr,
                f"{prefix}/n_samples": n,
            }
            
            # Add pre-recorded entropy metrics if available
            if avg_prerecorded_entropy is not None:
                wandb_metrics[f"{prefix}/prerecorded_entropy"] = avg_prerecorded_entropy
                wandb_metrics[f"{prefix}/std_prerecorded_entropy"] = std_prerecorded_entropy
                wandb_metrics[f"{prefix}/self_frozen_corr"] = self_frozen_corr
                wandb_metrics[f"{prefix}/other_frozen_corr"] = other_frozen_corr
            
            # Add answer distribution percentages by letter
            for letter in mcq_display_letters_str:
                wandb_metrics[f"{prefix}/pred_dist_letter_{letter}_pct"] = pred_dist_letters_pct[letter]
                wandb_metrics[f"{prefix}/correct_dist_letter_{letter}_pct"] = gold_dist_letters_pct[letter]
            
            # Add answer distribution percentages by position
            for pos in range(4):
                letter = chr(ord('A') + pos)
                wandb_metrics[f"{prefix}/pred_dist_position_{pos}_{letter}_pct"] = pred_dist_positions_pct[pos]
                wandb_metrics[f"{prefix}/correct_dist_position_{pos}_{letter}_pct"] = gold_dist_positions_pct[pos]
            
            # Add self confidence distribution percentages by letter (using display letters)
            for letter in display_letters_str:
                wandb_metrics[f"{prefix}/self_conf_letter_{letter}_pct"] = conf_dist_letters_pct[letter]
            
            # Add self confidence distribution percentages by bin
            for bin_idx in range(8):
                bin_letter = chr(ord('A') + bin_idx)
                bin_label = confidence_bin_labels[bin_idx].replace('%', 'pct').replace('-', '_').replace('<', 'lt').replace('>', 'gt')
                wandb_metrics[f"{prefix}/self_conf_bin_{bin_idx}_{bin_letter}_{bin_label}_pct"] = conf_dist_bins_pct[bin_idx]
            
            # Add other confidence distribution percentages by letter (using display letters)
            for letter in display_letters_str:
                wandb_metrics[f"{prefix}/other_conf_letter_{letter}_pct"] = other_conf_dist_letters_pct[letter]
            
            # Add other confidence distribution percentages by bin
            for bin_idx in range(8):
                bin_letter = chr(ord('A') + bin_idx)
                bin_label = confidence_bin_labels[bin_idx].replace('%', 'pct').replace('-', '_').replace('<', 'lt').replace('>', 'gt')
                wandb_metrics[f"{prefix}/other_conf_bin_{bin_idx}_{bin_letter}_{bin_label}_pct"] = other_conf_dist_bins_pct[bin_idx]
            
            # Add step if provided
            if step is not None:
                wandb_metrics["step"] = step
            
            wandb.log(wandb_metrics)
        except (ImportError, AttributeError):
            pass  # Silently fail if wandb not available

    return results


def compute_calibration_metrics(correctness, confidence, n_bins=10):
    """
    Compute calibration metrics: ECE, Brier score, and decomposition.
    """
    # Align and drop NaNs
    data = pd.DataFrame({
        'correct': correctness,
        'prob': confidence
    }).dropna()
    
    if len(data) == 0:
        return {
            'ece': np.nan,
            'brier': np.nan,
            'reliability': np.nan,
            'resolution': np.nan,
            'uncertainty': np.nan,
        
        }
    
    n_samples = len(data)
    base_rate = data['correct'].mean()
    
    # 1. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (data['prob'] > bin_lower) & (data['prob'] <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            bin_acc = data.loc[in_bin, 'correct'].mean()
            bin_conf = data.loc[in_bin, 'prob'].mean()
            ece += prop_in_bin * abs(bin_acc - bin_conf)
    
    # 2. Brier Score and Decomposition
    brier = ((data['prob'] - data['correct']) ** 2).mean()
    
    # Reliability (calibration)
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        bin_mask = (data['prob'] >= bin_boundaries[i]) & (data['prob'] < bin_boundaries[i+1])
        n_bin = bin_mask.sum()
        
        if n_bin > 0:
            bin_prob = data.loc[bin_mask, 'prob'].mean()
            bin_freq = data.loc[bin_mask, 'correct'].mean()
            bin_weight = n_bin / len(data)
            
            reliability += bin_weight * (bin_prob - bin_freq) ** 2
            resolution += bin_weight * (bin_freq - base_rate) ** 2
    
    uncertainty = base_rate * (1 - base_rate)

    
    return {
        'ece': float(ece),
        'brier': float(brier),
        'reliability': float(reliability),
        'resolution': float(resolution),
        'uncertainty': float(uncertainty),

        'n_samples': n_samples,
    }


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

