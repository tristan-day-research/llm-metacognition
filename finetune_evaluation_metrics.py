import argparse
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
    validate_and_load_dataset,
    log_wandb_metrics,
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
    
    # DIAGNOSTIC: Track logit statistics to detect token bias
    logit_sums = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    logit_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    # VERIFICATION: Check that MCQ and confidence tokens are what we expect
    # (They should overlap for A-D, but we need to ensure we're using the right logits)
    mcq_tokens = {letter: get_single_token_id(tokenizer, letter) for letter in "ABCD"}
    conf_tokens = {letter: get_single_token_id(tokenizer, letter) for letter in "ABCDEFGH"}
    
    # Verify MCQ tokens A-D match confidence tokens A-D (expected behavior)
    tokens_match = all(mcq_tokens[letter] == conf_tokens[letter] for letter in "ABCD")
    if not tokens_match:
        print(f"⚠️  WARNING: MCQ tokens A-D don't match confidence tokens A-D!")
        print(f"   MCQ tokens: {mcq_tokens}")
        conf_tokens_abcd = {k: v for k, v in list(conf_tokens.items())[:4]}
        print(f"   Conf tokens A-D: {conf_tokens_abcd}")
    else:
        if validation_step == 0 or validation_step % 100 == 0:  # Only print occasionally
            print(f"✓ Verified: MCQ tokens (A-D) match confidence tokens (A-D) as expected")
            print(f"   MCQ/Conf tokens A-D: {mcq_tokens}")
            conf_tokens_efgh = {'E': conf_tokens['E'], 'F': conf_tokens['F'], 'G': conf_tokens['G'], 'H': conf_tokens['H']}
            print(f"   Confidence tokens E-H: {conf_tokens_efgh}")
    
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
                
                # Extract logits for A, B, C, D (MCQ answers only)
                # CRITICAL: We use answer_logits4 (4 logits) for MCQ accuracy, NOT conf_logits8 (8 logits)
                final_logits = out.logits[:, -1, :]
                abcd_ids = torch.tensor(
                    [mcq_tokens[c] for c in "ABCD"],  # Use pre-computed tokens
                    device=device,
                    dtype=torch.long
                )
                answer_logits4 = final_logits[:, abcd_ids]  # [B, 4] - MCQ logits only
                
                # DIAGNOSTIC: Track raw logit values to detect bias
                # Accumulate logits across all batches
                batch_logits = answer_logits4.mean(dim=0).cpu().numpy()
                for i, letter in enumerate("ABCD"):
                    logit_sums[letter] += batch_logits[i]
                    logit_counts[letter] += 1
                
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
                    [conf_tokens[c] for c in "ABCDEFGH"],  # Use pre-computed tokens
                    device=device,
                    dtype=torch.long
                )
                conf_logits8 = final_logits2[:, bins_ids]  # [B, 8] - Confidence logits (separate from MCQ!)
                
                # Compute confidence probabilities (with temperature if > 0)
                if temperature > 0:
                    scaled_conf_logits = conf_logits8 / temperature
                    conf_probs = torch.softmax(scaled_conf_logits, dim=-1)
                else:
                    conf_probs = torch.softmax(conf_logits8, dim=-1)
                
                # Map confidence bins to percentages (midpoints)
                bin_midpoints = np.array([2.5, 7.5, 15, 30, 50, 70, 85, 95]) / 100.0
                # Expected confidence (Verbal)
                expected_conf = (conf_probs * torch.tensor(bin_midpoints, device=device)).sum(dim=-1)
                predicted_conf_bins = conf_probs.argmax(dim=-1)
                
                # Process each question in batch - compute and log both MCQ and confidence
                for i, row in enumerate(batch):
                    # === MCQ Assessment ===
                    # Get predicted answer letter from MCQ logits (A-D only, NOT confidence A-H)
                    # CRITICAL: predicted_indices comes from answer_probs which comes from answer_logits4 (4 logits)
                    # We are NOT using conf_logits8 (8 logits) for accuracy calculation
                    pred_idx = predicted_indices[i].item()
                    predicted_letter = "ABCD"[pred_idx]
                    
                    # CRITICAL: Ensure we're using MCQ answer (A-D), not confidence prediction (A-H)
                    assert predicted_letter in "ABCD", f"MCQ prediction must be A-D, got {predicted_letter}"
                    assert pred_idx < 4, f"MCQ prediction index must be 0-3 (A-D), got {pred_idx}"
                    
                    # Get correct answer letter
                    correct_letter = row["correct_letter"]
                    assert correct_letter in "ABCD", f"Correct answer must be A-D, got {correct_letter}"
                    
                    # Track letter distributions for debugging (MCQ answers A-D only)
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
                    # NOTE: This is separate from MCQ assessment above
                    # Confidence uses A-H scale, but we do NOT use this for accuracy calculation
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
    
    # DIAGNOSTIC: Report average logit values to detect token bias
    print(f"\n  DIAGNOSTIC: Average raw logits per token (across all batches):")
    avg_logits = {}
    for letter in "ABCD":
        if logit_counts[letter] > 0:
            avg_logits[letter] = logit_sums[letter] / logit_counts[letter]
            print(f"    {letter} (token {mcq_tokens[letter]}): {avg_logits[letter]:.4f}")
    
    # Check if there's a significant logit bias
    if avg_logits:
        logit_values = list(avg_logits.values())
        max_logit_letter = max(avg_logits, key=avg_logits.get)
        min_logit_letter = min(avg_logits, key=avg_logits.get)
        logit_range = max(logit_values) - min(logit_values)
        if logit_range > 1.0:  # More than 1.0 logit difference suggests bias
            print(f"  ⚠️  WARNING: Significant logit bias detected (range={logit_range:.4f})")
            print(f"     Highest: {max_logit_letter} ({avg_logits[max_logit_letter]:.4f}), "
                  f"Lowest: {min_logit_letter} ({avg_logits[min_logit_letter]:.4f})")
            print(f"     This may explain why the model favors certain answers.")
        else:
            print(f"  ✓ Logit values are relatively balanced (range={logit_range:.4f})")
    
    # Calculate average logits for each token (diagnostic for bias detection)
    avg_logits = {}
    for letter in "ABCD":
        if logit_counts[letter] > 0:
            avg_logits[letter] = logit_sums[letter] / logit_counts[letter]
        else:
            avg_logits[letter] = 0.0
    
    print(f"\n{'='*80}")
    print(f"MCQ Accuracy Assessment Results:")
    print(f"{'='*80}")
    print(f"  Total questions: {len(all_correct)}")
    print(f"  Correct answers: {sum(all_correct)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Expected random accuracy: 0.2500 (25.00%)")
    print(f"  Average Entropy: {avg_entropy:.4f}")
    print(f"\n  Average Logits (diagnostic for token bias):")
    for letter in "ABCD":
        print(f"    {letter}: {avg_logits[letter]:.4f}")
    print(f"\n  Model's predicted MCQ answer distribution (A-D only, NOT confidence A-H):")
    for letter in "ABCD":
        count = predicted_letter_counts.get(letter, 0)
        pct = (count / len(all_correct) * 100) if all_correct else 0
        expected_pct = 25.0  # Expected if uniform
        diff = pct - expected_pct
        print(f"    {letter}: {count:4d} ({pct:5.2f}%) [expected ~25%, diff: {diff:+.2f}%]")
    print(f"\n  Correct answer letter distribution (A-D only):")
    print(f"    (After shuffling, should be ~25% each if shuffling works correctly)")
    for letter in "ABCD":
        count = correct_letter_counts.get(letter, 0)
        pct = (count / len(all_correct) * 100) if all_correct else 0
        expected_pct = 25.0  # Expected if shuffling is uniform
        diff = pct - expected_pct
        print(f"    {letter}: {count:4d} ({pct:5.2f}%) [expected ~25%, diff: {diff:+.2f}%]")
    
    # DIAGNOSTIC: Check if shuffling is working (correct answers should be evenly distributed)
    correct_dist_std = np.std([correct_letter_counts.get(letter, 0) / len(all_correct) * 100 
                                for letter in "ABCD"]) if all_correct else 0.0
    if correct_dist_std > 5.0:  # More than 5% standard deviation suggests shuffling issue
        print(f"\n  ⚠️  WARNING: Correct answer distribution has high variance (std={correct_dist_std:.2f}%)")
        print(f"     This suggests shuffling may not be working correctly or dataset has bias.")
    else:
        print(f"\n  ✓ Correct answer distribution is uniform (std={correct_dist_std:.2f}%), shuffling appears to work.")
    
    # DIAGNOSTIC: Check if model predictions are biased
    pred_dist_std = np.std([predicted_letter_counts.get(letter, 0) / len(all_correct) * 100 
                             for letter in "ABCD"]) if all_correct else 0.0
    if pred_dist_std > 10.0:  # More than 10% standard deviation suggests model bias
        print(f"  ⚠️  WARNING: Model predictions have high variance (std={pred_dist_std:.2f}%)")
        print(f"     This suggests the model has learned a position bias or token bias.")
    else:
        print(f"  ✓ Model predictions are relatively uniform (std={pred_dist_std:.2f}%).")
    
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
    # Import val_step here to avoid circular import
    from finetune_ECT import val_step
    
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