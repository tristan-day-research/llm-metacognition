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
            'accuracy': np.nan,
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
    accuracy = data['correct'].mean()
    
    return {
        'ece': float(ece),
        'brier': float(brier),
        'reliability': float(reliability),
        'resolution': float(resolution),
        'uncertainty': float(uncertainty),
        'accuracy': float(accuracy),
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
    print(f"Pass 1 Accuracy:    {calibration_metrics['accuracy']:.4f}")
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
            "accuracy": calibration_metrics['accuracy'],
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
    accuracy = np.mean(correct_arr)
    
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
        "val/accuracy": accuracy,
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