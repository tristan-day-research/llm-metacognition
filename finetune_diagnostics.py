import argparse
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # <--- ADDED FOR LORA

from finetune_utils import (
    write_log,
    MCQDataset,
    get_single_token_id,
    build_multiple_choice_question_prompts,
    build_self_confidence_prompts,
    setup_tokenizer,
    load_model_with_error_handling,
    check_and_clear_gpu_memory,
    compute_ABCD_entropy,
    normalize_text,
    shuffle_options_and_update_correct_letter,
)


def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries."""
    return batch


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


def run_inference(model, tokenizer, dataset, device="cuda", batch_size=4):
    """Run inference to get verbalized confidence scores."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_predictions = []
    all_verbal_confidences = [] # Renamed for clarity
    all_correctness = []
    all_qids = []
    all_entropies = []
    
    print(f"Running inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            # --- PASS 1: ANSWER ---
            answer_prompts = build_multiple_choice_question_prompts(batch)
            enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)
            
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
            
            # Answer Probs & Entropy
            answer_probs = torch.softmax(answer_logits4, dim=-1)
            predicted_indices = answer_probs.argmax(dim=-1)
            predicted_letters = ["ABCD"[idx.item()] for idx in predicted_indices]
            
            entropies = -(answer_probs * torch.log(answer_probs + 1e-12)).sum(dim=-1)
            
            # --- PASS 2: CONFIDENCE ---
            confidence_prompts = build_self_confidence_prompts(batch)
            enc2 = tokenizer(confidence_prompts, return_tensors="pt", padding=True).to(device)
            
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
            conf_probs = torch.softmax(conf_logits8, dim=-1)
            
            # Map confidence bins to percentages (midpoints)
            bin_midpoints = np.array([2.5, 7.5, 15, 30, 50, 70, 85, 95]) / 100.0
            # Expected confidence (Verbal)
            expected_conf = (conf_probs * torch.tensor(bin_midpoints, device=device)).sum(dim=-1)
            
            for i, row in enumerate(batch):
                correct_letter = row["correct_letter"]
                predicted_letter = predicted_letters[i]
                is_correct = 1 if predicted_letter == correct_letter else 0
                
                # CRITICAL CHANGE: Use VERBAL expected confidence, not internal softmax max
                verbal_confidence = expected_conf[i].item()
                entropy = entropies[i].item()
                
                all_predictions.append(predicted_letter)
                all_verbal_confidences.append(verbal_confidence)
                all_correctness.append(is_correct)
                all_qids.append(row.get("qid", f"batch_{batch_idx}_item_{i}"))
                all_entropies.append(entropy)
    
    return {
        'predictions': all_predictions,
        'confidences': np.array(all_verbal_confidences), # Now this is verbal confidence
        'correctness': np.array(all_correctness),
        'qids': all_qids,
        'entropies': np.array(all_entropies),
    }


def run_diagnostics(base_model, data_path, checkpoint_path=None, device="cuda", 
                    batch_size=4, log_file="diagnostics_log.jsonl", output_csv=None):
    
    print("=" * 80)
    print("Running Model Diagnostics")
    print("=" * 80)
    print(f"Base Model: {base_model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {data_path}")
    print()
    
    check_and_clear_gpu_memory(device)
    
    print("Loading tokenizer...")
    tokenizer = setup_tokenizer(base_model)
    
    print("Loading base model...")
    model = load_model_with_error_handling(base_model, device)
    
    if checkpoint_path:
        print(f"Loading LoRA adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload() # Merge for efficiency
        print("Adapter loaded and merged.")
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    dataset = MCQDataset(data_path)
    print(f"Loaded {len(dataset)} samples")
    
    results = run_inference(model, tokenizer, dataset, device=device, batch_size=batch_size)
    
    print("\nComputing metrics...")
    calibration_metrics = compute_calibration_metrics(
        results['correctness'],
        results['confidences']
    )
    
    # --- METRICS CALCULATIONS ---
    
    # 1. Mode Collapse (Std Dev of Verbal Confidence)
    std_verbal_conf = np.std(results['confidences'])
    avg_verbal_conf = np.mean(results['confidences'])
    
    # 2. Alignment (Entropy vs Verbal Confidence)
    # Ideally NEGATIVE correlation (High Entropy = Low Confidence)
    try:
        corr_align_p, p_align_p = pearsonr(results['entropies'], results['confidences'])
    except:
        corr_align_p, p_align_p = np.nan, np.nan
        
    # 3. Usefulness (Confidence vs Correctness)
    try:
        corr_calib_p, p_calib_p = pearsonr(results['confidences'], results['correctness'])
    except:
        corr_calib_p, p_calib_p = np.nan, np.nan

    # Print results
    print("\n" + "=" * 80)
    print("DIAGNOSTICS RESULTS")
    print("=" * 80)
  
    print("-" * 40)
    print("MODE COLLAPSE CHECK:")
    print(f"Avg Verbal Conf:    {avg_verbal_conf:.4f}")
    print(f"Std Verbal Conf:    {std_verbal_conf:.4f}  (If < 0.02, model is likely collapsed)")
    print("-" * 40)
    print("ALIGNMENT CHECK (Metacognition):")
    print(f"Corr(Entropy, Conf): {corr_align_p:.4f}  (Target: Negative, e.g., -0.6)")
    print("-" * 40)
    print("CALIBRATION CHECK:")
    print(f"ECE:                {calibration_metrics['ece']:.4f}")
    print(f"Brier Score:        {calibration_metrics['brier']:.4f}")
    print(f"Corr(Conf, Correct): {corr_calib_p:.4f}")
    print("=" * 80)
    
    # Save raw CSV
    if output_csv:
        df = pd.DataFrame({
            'qid': results['qids'],
            'prediction': results['predictions'],
            'verbal_confidence': results['confidences'],
            'correctness': results['correctness'],
            'internal_entropy': results['entropies'],
        })
        df.to_csv(output_csv, index=False)
        print(f"Saved raw data to {output_csv}")
    
    # Log entry
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = {
        "type": "diagnostics_summary",
        "timestamp": timestamp,
        "base_model": base_model,
        "checkpoint": checkpoint_path,
        "dataset": data_path,
        "metrics": {

            "ece": calibration_metrics['ece'],
            "brier": calibration_metrics['brier'],
            "std_verbal_conf": float(std_verbal_conf),
            "avg_verbal_conf": float(avg_verbal_conf),
            "alignment_entropy_conf_pearson": float(corr_align_p) if not np.isnan(corr_align_p) else None,
            "calibration_conf_correct_pearson": float(corr_calib_p) if not np.isnan(corr_calib_p) else None,
        }
    }
    
    if log_file:
        write_log(log_file, log_entry)
        print(f"Logged summary to {log_file}")



def assess_mcq_accuracy(model, tokenizer, val_dataset, device="cuda", 
                       validation_step=0, log_file_path=None, num_questions=500, temperature=0.0, seed=None):
    """
    Assess model accuracy on multiple choice questions from validation set.
    
    Randomly samples questions from validation dataset, runs inference,
    and logs results to file and W&B.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        val_dataset: MCQDataset instance
        device: Device to run on
        validation_step: Current validation step number (e.g., 100, 200, etc.)
        log_file_path: Path to log file for detailed question-level logs
        num_questions: Number of random questions to sample (default: 500)
        temperature: Temperature for sampling predictions (0.0 = deterministic)
        seed: Random seed for sampling (None = use current random state)
        
    Returns:
        dict with keys:
            - "accuracy": average accuracy (0-1)
            - "avg_entropy": average entropy across all questions
    """
    import random
    import torch
    
    model.eval()
    
    # Check if we should sample or use all questions
    dataset_size = len(val_dataset)
    num_questions = min(num_questions, dataset_size)
    
    # If num_questions equals dataset_size, use all questions (no sampling needed)
    # This happens when a pre-sampled subset is passed
    if num_questions == dataset_size:
        # Use all questions in order (no random sampling)
        sampled_indices = list(range(dataset_size))
        print(f"Assessing MCQ accuracy on {num_questions} questions...")
    else:
        # Set seed if provided (for reproducibility)
        if seed is not None:
            random.seed(seed)
        # Randomly sample questions from validation dataset
        sampled_indices = random.sample(range(dataset_size), num_questions)
        print(f"Assessing MCQ accuracy on {num_questions} random questions from validation set...")
    
    all_correct = []
    all_entropies = []
    all_confidence_predictions = []  # For confidence assessment
    predicted_letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    correct_letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    # Process in batches for efficiency
    batch_size = 4
    batch = []
    batch_indices = []
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(sampled_indices):
            # Get question from dataset
            row = val_dataset[sample_idx]
            batch.append(row)
            batch_indices.append(sample_idx)
            
            # Process batch when full or at end
            if len(batch) == batch_size or idx == len(sampled_indices) - 1:
                # Shuffle options to prevent position bias
                for row in batch:
                    shuffle_options_and_update_correct_letter(row)
                
                # Build prompts using the utility function
                answer_prompts = build_multiple_choice_question_prompts(batch)
                
                # Tokenize and run inference
                enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)
                
                try:
                    out = model(**enc, use_cache=False)
                except TypeError:
                    out = model(**enc)
                
                # Extract logits for A, B, C, D
                final_logits = out.logits[:, -1, :]
                abcd_ids = torch.tensor(
                    [get_single_token_id(tokenizer, c) for c in "ABCD"],
                    device=device,
                    dtype=torch.long
                )
                answer_logits4 = final_logits[:, abcd_ids]
                
                # Compute probabilities and predictions for MCQ (with temperature)
                if temperature > 0:
                    scaled_logits = answer_logits4 / temperature
                    answer_probs = torch.softmax(scaled_logits, dim=-1)
                    # Sample from the distribution
                    predicted_indices = torch.multinomial(answer_probs, num_samples=1).squeeze(-1)
                else:
                    answer_probs = torch.softmax(answer_logits4, dim=-1)
                    # Deterministic: use argmax
                    predicted_indices = answer_probs.argmax(dim=-1)
                
                # Now run confidence assessment on the same batch
                confidence_prompts = build_self_confidence_prompts(batch)
                enc2 = tokenizer(confidence_prompts, return_tensors="pt", padding=True).to(device)
                
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
                conf_probs = torch.softmax(conf_logits8, dim=-1)
                
                # Map confidence bins to percentages (midpoints)
                bin_midpoints = np.array([2.5, 7.5, 15, 30, 50, 70, 85, 95]) / 100.0
                # Expected confidence (Verbal)
                expected_conf = (conf_probs * torch.tensor(bin_midpoints, device=device)).sum(dim=-1)
                predicted_conf_bins = conf_probs.argmax(dim=-1)
                
                # Process each question in batch - compute and log both MCQ and confidence
                for i, row in enumerate(batch):
                    # === MCQ Assessment ===
                    # Get predicted answer letter
                    pred_idx = predicted_indices[i].item()
                    predicted_letter = "ABCD"[pred_idx]
                    
                    # Get correct answer letter
                    correct_letter = row["correct_letter"]
                    
                    # Track letter distributions for debugging
                    predicted_letter_counts[predicted_letter] = predicted_letter_counts.get(predicted_letter, 0) + 1
                    if correct_letter:
                        correct_letter_counts[correct_letter] = correct_letter_counts.get(correct_letter, 0) + 1
                    
                    # Get the actual text for both answers from options
                    options = row.get("options", {})
                    predicted_answer_text = options.get(predicted_letter, "")
                    correct_answer_text = options.get(correct_letter, "")
                    
                    # Compare normalized text instead of just letters
                    # This ensures we catch any issues with answer matching
                    predicted_normalized = normalize_text(predicted_answer_text)
                    correct_normalized = normalize_text(correct_answer_text)
                    
                    # Check if correct by comparing normalized text
                    is_correct_by_text = 1 if predicted_normalized == correct_normalized else 0
                    
                    # Also check by letter for logging/debugging
                    is_correct_by_letter = 1 if predicted_letter == correct_letter else 0
                    
                    # Use letter-based comparison for accuracy calculation (matches capabilities_test.py)
                    # This ensures consistency: capabilities_test.py uses subject_decision == question["correct_answer"]
                    all_correct.append(is_correct_by_letter)
                    
                    # Debug: Warn if text and letter comparisons don't match
                    if is_correct_by_text != is_correct_by_letter:
                        print(f"WARNING: Mismatch for qid {row.get('qid')}: text_match={is_correct_by_text}, letter_match={is_correct_by_letter}")
                        print(f"  Predicted: {predicted_letter}='{predicted_answer_text}' (norm: '{predicted_normalized}')")
                        print(f"  Correct: {correct_letter}='{correct_answer_text}' (norm: '{correct_normalized}')")
                        print(f"  Note: Using letter_match for accuracy (matching capabilities_test.py behavior)")
                    
                    # Calculate entropy for this answer distribution
                    probs_for_entropy = answer_probs[i].cpu()
                    entropy = compute_ABCD_entropy(probs_for_entropy).item()
                    all_entropies.append(entropy)
                    
                    # === Confidence Assessment ===
                    conf_pred_letter = "ABCDEFGH"[predicted_conf_bins[i].item()]
                    verbal_confidence = expected_conf[i].item()
                    # Compute confidence entropy (entropy of confidence distribution)
                    conf_entropy = -(conf_probs[i] * torch.log(conf_probs[i] + 1e-12)).sum().item()
                    all_confidence_predictions.append({
                        "qid": row.get("qid", f"sample_{batch_indices[i]}"),
                        "confidence_letter": conf_pred_letter,
                        "verbal_confidence": verbal_confidence,
                        "confidence_probs": conf_probs[i].cpu().tolist(),
                        "confidence_entropy": conf_entropy,
                    })
                    
                    # Log both MCQ and confidence in the same log file
                    if log_file_path:
                        # Log MCQ assessment
                        mcq_log_entry = {
                            "type": "mcq_accuracy_assessment",
                            "validation_step": validation_step,
                            "qid": row.get("qid", f"sample_{batch_indices[i]}"),
                            "question": row.get("question", ""),
                            "correct_answer_letter": correct_letter,
                            "correct_answer_text": correct_answer_text,
                            "model_answer_letter": predicted_letter,
                            "model_answer_text": predicted_answer_text,
                            "is_correct_by_text": bool(is_correct_by_text),
                            "is_correct_by_letter": bool(is_correct_by_letter),
                            "entropy": float(entropy),
                            "probabilities": {
                                "A": float(answer_probs[i][0].item()),
                                "B": float(answer_probs[i][1].item()),
                                "C": float(answer_probs[i][2].item()),
                                "D": float(answer_probs[i][3].item()),
                            },
                            "all_options": {
                                "A": options.get("A", ""),
                                "B": options.get("B", ""),
                                "C": options.get("C", ""),
                                "D": options.get("D", ""),
                            },
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        write_log(log_file_path, mcq_log_entry)
                        
                        # Log confidence assessment
                        conf_log_entry = {
                            "type": "confidence_assessment",
                            "validation_step": validation_step,
                            "qid": row.get("qid", f"sample_{batch_indices[i]}"),
                            "question": row.get("question", ""),
                            "confidence_predicted_letter": conf_pred_letter,
                            "verbal_confidence": float(verbal_confidence),
                            "confidence_entropy": float(conf_entropy),
                            "confidence_probabilities": {
                                "A": float(conf_probs[i][0].item()),
                                "B": float(conf_probs[i][1].item()),
                                "C": float(conf_probs[i][2].item()),
                                "D": float(conf_probs[i][3].item()),
                                "E": float(conf_probs[i][4].item()),
                                "F": float(conf_probs[i][5].item()),
                                "G": float(conf_probs[i][6].item()),
                                "H": float(conf_probs[i][7].item()),
                            },
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        write_log(log_file_path, conf_log_entry)
                
                # Clear batch
                batch = []
                batch_indices = []
    
    # Calculate aggregate metrics
    accuracy = np.mean(all_correct) if all_correct else 0.0
    avg_entropy = np.mean(all_entropies) if all_entropies else 0.0
    std_entropy = np.std(all_entropies) if all_entropies else 0.0
    
    print(f"\n{'='*80}")
    print(f"MCQ Accuracy Assessment Results:")
    print(f"{'='*80}")
    print(f"  Total questions: {len(all_correct)}")
    print(f"  Correct answers: {sum(all_correct)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Expected random accuracy: 0.2500 (25.00%)")
    print(f"  Average Entropy: {avg_entropy:.4f}")
    print(f"\n  Model's predicted letter distribution:")
    for letter in "ABCD":
        count = predicted_letter_counts.get(letter, 0)
        pct = (count / len(all_correct) * 100) if all_correct else 0
        print(f"    {letter}: {count:4d} ({pct:5.2f}%)")
    print(f"\n  Correct answer letter distribution:")
    for letter in "ABCD":
        count = correct_letter_counts.get(letter, 0)
        pct = (count / len(all_correct) * 100) if all_correct else 0
        print(f"    {letter}: {count:4d} ({pct:5.2f}%)")
    print(f"{'='*80}\n")
    
    # Log to W&B
    try:
        import wandb
        wandb.log({
            "val/accuracy": accuracy,
            "val/answer_entropy": avg_entropy,
        }, step=validation_step)
    except (ImportError, AttributeError):
        pass  # Silently fail if wandb not available
    
    return {
        "accuracy": accuracy,
        "avg_entropy": avg_entropy,
        "std_entropy": std_entropy,
        "predicted_letter_counts": predicted_letter_counts,
        "correct_letter_counts": correct_letter_counts,
        "total_questions": len(all_correct),
    }


def compute_metrics_for_wandb(all_correct, all_entropies, all_verbal_conf):
    """
    Aggregates validation results into metrics for WandB.
    
    Args:
        all_correct: list of 1s (correct) and 0s (incorrect)
        all_entropies: list of internal entropy values
        all_verbal_conf: list of predicted verbal confidence scores (0-100)
        
    Returns:
        dict: Metrics ready for wandb.log()
    """
    # Convert to numpy for easy math
    correct_arr = np.array(all_correct)
    entropies_arr = np.array(all_entropies)
    conf_arr = np.array(all_verbal_conf)
    
    # 1. Capability: Is it still answering questions correctly?

    
    # 2. Mode Collapse: Is it outputting the same confidence everywhere?
    avg_conf = np.mean(conf_arr)
    std_conf = np.std(conf_arr)
    
    # 3. Alignment: Does uncertainty (entropy) match verbal report?
    # We expect NEGATIVE correlation (High Entropy = Low Confidence)
    try:
        if len(conf_arr) > 1 and std_conf > 0.001:
            align_corr, _ = pearsonr(entropies_arr, conf_arr)
        else:
            align_corr = 0.0
    except Exception:
        align_corr = 0.0
        
    # 4. Calibration: Is the confidence useful?
    try:
        if len(conf_arr) > 1 and std_conf > 0.001:
            calib_corr, _ = pearsonr(conf_arr, correct_arr)
        else:
            calib_corr = 0.0
    except Exception:
        calib_corr = 0.0

    return {
   
        "val/avg_verbal_conf": avg_conf,
        "val/std_verbal_conf": std_conf,  # < 2.0 = Collapse
        "val/alignment_corr": align_corr, # The most important metric
        "val/calibration_corr": calib_corr
    }


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base HF model (e.g. Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to LoRA adapter checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--log_file", type=str, default="diagnostics_log.jsonl")
    parser.add_argument("--output_csv", type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostics(
        base_model=args.base_model,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        batch_size=args.batch_size,
        log_file=args.log_file,
        output_csv=args.output_csv
    )