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
    compute_loss
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
):
    """
    Evaluation loop:
    
    CRITICAL: This function should ONLY receive val_dataset (or test_dataset).
    Never pass train_dataset to this function.

    For each sample:
        1. Run MCQ pass (extract predicted letter + entropy)
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
    """
    # Defensive check: ensure we're in eval mode
    model.eval()
    
    # Additional defensive check: verify dataset is not empty
    if len(val_dataset) == 0:
        raise ValueError("run_evaluation() received empty dataset - this should not happen")
    
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
    predicted_letters = []
    correct_letters = []
    predicted_confidence_letters = []  # A-H confidence predictions (self)
    predicted_other_confidence_letters = []  # A-H confidence predictions (other)
    prerecorded_entropy_values = []  # Pre-recorded entropy from mcq_results_data

    # ==================== MAIN LOOP ==========================
    for i in idxs:
        batch = val_dataset[i:i+1]    # single-sample batch (list of 1 dict)

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
        # 1. MCQ pass
        # ==================================================
        mcq_prompts = build_multiple_choice_question_prompts(batch, tokenizer)

        # print("DEBUG mcq_prompts", mcq_prompts)

        mcq_out = run_mcq_forward_pass(
            model=model,
            tokenizer=tokenizer,
            prompts=mcq_prompts,
            device=device,
            temperature=0.0,
        )

        predicted_answer_letter = mcq_out["pred_letters"][0]
        entropy_value = mcq_out["entropy"][0].item()

        correct_answer_letter = batch[0]["correct_letter"]

        predicted_letters.append(predicted_answer_letter)
        correct_letters.append(correct_answer_letter)

        correctness_flags.append(
            1.0 if predicted_answer_letter == correct_answer_letter else 0.0
        )
        entropy_values.append(entropy_value)

        # ==================================================
        # 2. Soft targets
        # ==================================================
        soft_targets = convert_entropy_to_soft_labels(entropy_value).to(device)

        # ==================================================
        # 3. Self Confidence pass
        # ==================================================
        conf_prompts = build_self_confidence_prompts(batch, tokenizer)

        conf_out = run_confidence_forward_pass(
            model=model,
            tokenizer=tokenizer,
            prompts=conf_prompts,
            device=device,
            temperature=args.temperature,
        )

        logits8 = conf_out["logits8"][0]
        expected_confidence_value = conf_out["expected_conf"][0].item()
        predicted_confidence_letter = conf_out["pred_bins"][0]  # A-H

        expected_conf_values.append(expected_confidence_value)
        predicted_confidence_letters.append(predicted_confidence_letter)

        # ==================================================
        # 3.5. Other confidence pass
        # ==================================================
        other_conf_prompts = build_other_confidence_prompts(batch, tokenizer)

        other_conf_out = run_confidence_forward_pass(
            model=model,
            tokenizer=tokenizer,
            prompts=other_conf_prompts,
            device=device,
            temperature=args.temperature,
        )

        expected_other_confidence_value = other_conf_out["expected_conf"][0].item()
        predicted_other_confidence_letter = other_conf_out["pred_bins"][0]  # A-H
        expected_other_conf_values.append(expected_other_confidence_value)
        predicted_other_confidence_letters.append(predicted_other_confidence_letter)

        # ==================================================
        # 4. Loss
        # ==================================================
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

        # ==================================================
        # 5. Optional per-sample logging
        # ==================================================
        if log_file_path is not None:
            log_entry = {
                "type": "eval_sample",
                "qid": batch[0].get("qid"),
                "question": batch[0]["question"],
                "model_answer": predicted_answer_letter,
                "correct_answer": correct_answer_letter,
                "predicted_confidence_letter": predicted_confidence_letter,  # A-H
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
    # DISTRIBUTION STATS (A–D)
    # ===========================================================

    def count_dist(values, letters):
        return {letter: values.count(letter) for letter in letters}

    pred_dist = count_dist(predicted_letters, "ABCD")
    gold_dist = count_dist(correct_letters, "ABCD")

    n = len(predicted_letters)

    pred_dist_pct = {k: (v / n) * 100.0 for k, v in pred_dist.items()}
    gold_dist_pct = {k: (v / n) * 100.0 for k, v in gold_dist.items()}

    # Pretty print
    print("\n============================================================")
    print(f"{step_name.upper()} — MCQ Answer Distributions")
    print("============================================================")
    print("Correct (Ground Truth) Distribution:")
    for k in "ABCD":
        print(f"  {k}: {gold_dist[k]:4d}  ({gold_dist_pct[k]:6.2f}%)")

    print("\nModel Prediction Distribution:")
    for k in "ABCD":
        print(f"  {k}: {pred_dist[k]:4d}  ({pred_dist_pct[k]:6.2f}%)")

    # ===========================================================
    # CONFIDENCE DISTRIBUTION STATS (A–H)
    # ===========================================================
    conf_dist = count_dist(predicted_confidence_letters, "ABCDEFGH")
    conf_dist_pct = {k: (v / n) * 100.0 for k, v in conf_dist.items()}

    # Pretty print confidence distribution
    print("\n============================================================")
    print(f"{step_name.upper()} — Self Confidence Prediction Distributions")
    print("============================================================")
    print("Model Self Confidence Prediction Distribution (A-H):")
    for k in "ABCDEFGH":
        print(f"  {k}: {conf_dist[k]:4d}  ({conf_dist_pct[k]:6.2f}%)")

    # ===========================================================
    # OTHER CONFIDENCE DISTRIBUTION STATS (A–H)
    # ===========================================================
    other_conf_dist = count_dist(predicted_other_confidence_letters, "ABCDEFGH")
    other_conf_dist_pct = {k: (v / n) * 100.0 for k, v in other_conf_dist.items()}

    # Pretty print other confidence distribution
    print("\n============================================================")
    print(f"{step_name.upper()} — Other Confidence Prediction Distributions")
    print("============================================================")
    print("Model Other Confidence Prediction Distribution (A-H):")
    for k in "ABCDEFGH":
        print(f"  {k}: {other_conf_dist[k]:4d}  ({other_conf_dist_pct[k]:6.2f}%)")

    # ===========================================================
    # FINAL METRICS
    # ===========================================================
    results = {
        "mcq_accuracy": float(np.mean(correctness_flags)),
        "avg_entropy": float(np.mean(entropy_values)),
        "avg_confidence": float(np.mean(expected_conf_values)),
        "avg_other_confidence": float(np.mean(expected_other_conf_values)),
        "avg_loss": float(np.mean(loss_values)),
        "n_samples": n,
        "correct_answer_distribution_raw": gold_dist,
        "correct_answer_distribution_pct": gold_dist_pct,
        "predicted_answer_distribution_raw": pred_dist,
        "predicted_answer_distribution_pct": pred_dist_pct,
        "predicted_confidence_distribution_raw": conf_dist,
        "predicted_confidence_distribution_pct": conf_dist_pct,
        "predicted_other_confidence_distribution_raw": other_conf_dist,
        "predicted_other_confidence_distribution_pct": other_conf_dist_pct,
    }

    # Compute additional metrics for wandb
    correctness_arr = np.array(correctness_flags)
    entropy_arr = np.array(entropy_values)
    conf_arr = np.array(expected_conf_values)
    other_conf_arr = np.array(expected_other_conf_values)
    
    # Standard deviation of confidence (mode collapse check)
    std_conf = float(np.std(conf_arr)) if len(conf_arr) > 1 else 0.0
    std_other_conf = float(np.std(other_conf_arr)) if len(other_conf_arr) > 1 else 0.0
    std_entropy = float(np.std(entropy_arr)) if len(entropy_arr) > 1 else 0.0
    
    # Self live correlation: entropy from live model's output logits correlated with 
    # live model's prediction of its own confidence
    self_live_corr = 0.0
    if len(conf_arr) > 1 and std_conf > 0.001:
        try:
            self_live_corr, _ = pearsonr(entropy_arr, conf_arr)
            self_live_corr = float(self_live_corr)
        except Exception:
            self_live_corr = 0.0
    
    # Other live correlation: entropy from live model's output logits correlated with 
    # live model's prediction of 'other's' accuracy on the question
    other_live_corr = 0.0
    if len(other_conf_arr) > 1 and std_other_conf > 0.001:
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
                valid_conf_for_prerecorded.append(conf_val)
                valid_other_conf_for_prerecorded.append(other_conf_val)
        
        if len(valid_prerecorded) > 1:
            prerecorded_arr = np.array(valid_prerecorded)
            conf_arr_prerecorded = np.array(valid_conf_for_prerecorded)
            other_conf_arr_prerecorded = np.array(valid_other_conf_for_prerecorded)
            avg_prerecorded_entropy = float(np.mean(prerecorded_arr))
            std_prerecorded_entropy = float(np.std(prerecorded_arr)) if len(prerecorded_arr) > 1 else 0.0
            std_conf_prerecorded = float(np.std(conf_arr_prerecorded)) if len(conf_arr_prerecorded) > 1 else 0.0
            std_other_conf_prerecorded = float(np.std(other_conf_arr_prerecorded)) if len(other_conf_arr_prerecorded) > 1 else 0.0
            
            if std_prerecorded_entropy > 0.001 and std_conf_prerecorded > 0.001 and len(conf_arr_prerecorded) > 1:
                try:
                    self_frozen_corr, _ = pearsonr(prerecorded_arr, conf_arr_prerecorded)
                    self_frozen_corr = float(self_frozen_corr)
                except Exception:
                    self_frozen_corr = 0.0
            
            # Correlation between pre-recorded entropy and other confidence
            if std_prerecorded_entropy > 0.001 and std_other_conf_prerecorded > 0.001 and len(other_conf_arr_prerecorded) > 1:
                try:
                    other_frozen_corr, _ = pearsonr(prerecorded_arr, other_conf_arr_prerecorded)
                    other_frozen_corr = float(other_frozen_corr)
                except Exception:
                    other_frozen_corr = 0.0
    
    # Calibration: correlation between confidence and correctness
    calibration_corr = 0.0
    if len(conf_arr) > 1 and std_conf > 0.001:
        try:
            calibration_corr, _ = pearsonr(conf_arr, correctness_arr)
            calibration_corr = float(calibration_corr)
        except Exception:
            calibration_corr = 0.0

    # Log the summary as one blob
    if log_file_path is not None:
        write_jsonl(log_file_path, {
            "type": "eval_summary",
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
            
            # Add answer distribution percentages
            for letter in "ABCD":
                wandb_metrics[f"{prefix}/pred_dist_{letter}_pct"] = pred_dist_pct[letter]
                wandb_metrics[f"{prefix}/correct_dist_{letter}_pct"] = gold_dist_pct[letter]
            
            # Add self confidence distribution percentages (A-H)
            for letter in "ABCDEFGH":
                wandb_metrics[f"{prefix}/self_conf_{letter}"] = conf_dist_pct[letter]
            
            # Add other confidence distribution percentages (A-H)
            for letter in "ABCDEFGH":
                wandb_metrics[f"{prefix}/other_conf_{letter}"] = other_conf_dist_pct[letter]
            
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

