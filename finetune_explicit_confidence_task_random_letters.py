from finetune_load_finetuned_model import load_base_model, load_finetuned_model
from finetune_data_handling import load_jsonl_dataset
from finetune_prompting import (
    get_letter_token_ids,
    run_confidence_forward_pass,
)
import torch
import argparse
from collections import Counter
import random
import math
import os
import sys
import json
from datetime import datetime
from pathlib import Path


class Tee:
    """Context manager that writes to both console and file"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self  # Also capture stderr
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()


def generate_output_filename(args, dataset_path, suffix="summary"):
    """Generate output filename based on test parameters"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_model_safe = args.base_model.replace("/", "-").replace("_", "-")
    
    # Get dataset name
    dataset_name = Path(dataset_path).stem
    
    # Build filename components
    parts = [timestamp, base_model_safe]
    
    if args.lora_repo:
        lora_name = args.lora_repo.split("/")[-1]
        parts.append(lora_name)
    else:
        parts.append("base")
    
    parts.append("random_letters")
    parts.append(dataset_name)
    
    if args.test_type == "all":
        parts.append("all_tests")
    elif args.test_type == "normal":
        parts.append("normal")
    elif args.test_type == "mcq_random":
        parts.append("mcq_random")
    elif args.test_type == "conf_random":
        parts.append("conf_random")
    elif args.test_type == "both_random":
        parts.append("both_random")
    elif args.test_type == "baseline":
        parts.append("baseline")
    
    if args.num_questions:
        parts.append(f"n{args.num_questions}")
    
    if suffix == "summary":
        filename = "_".join(parts) + ".txt"
    else:
        filename = "_".join(parts) + f"_{suffix}.jsonl"
    
    return os.path.join("finetune_evals", filename)


def get_letter_token_ids_simple(tokenizer, letter):
    """Simple version for backward compatibility"""
    token_ids = []
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        token_ids.append(ids[0])
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1 and ids[0] not in token_ids:
        token_ids.append(ids[0])
    return token_ids


def create_random_letter_mapping(num_letters=4, seed=None):
    """
    Create a random mapping to N random letters.
    
    Args:
        num_letters: 4 for MCQ (A/B/C/D), 8 for confidence (A-H)
        seed: Random seed
    
    Returns:
        letter_mapping: dict mapping original letters to random letters
    """
    if seed is not None:
        random.seed(seed)
    
    if num_letters == 4:
        original = ['A', 'B', 'C', 'D']
        # Exclude A/B/C/D to avoid confusion
        available_letters = [c for c in 'EFGHIJKLMNOPQRSTUVWXYZ']
    elif num_letters == 8:
        original = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        # Exclude A-H to avoid confusion
        available_letters = [c for c in 'IJKLMNOPQRSTUVWXYZ']
    else:
        raise ValueError(f"num_letters must be 4 or 8, got {num_letters}")
    
    # Choose random letters
    random_letters = random.sample(available_letters, num_letters)
    
    # Shuffle positions
    positions = list(range(num_letters))
    random.shuffle(positions)
    
    # Create mapping
    letter_mapping = {}
    for i, orig_letter in enumerate(original):
        new_position = positions[i]
        letter_mapping[orig_letter] = random_letters[new_position]
    
    return letter_mapping


def build_mcq_prompts(batch, tokenizer, letter_mapping=None):
    """
    Build MCQ prompts with optional randomized letters.
    
    Args:
        batch: List of question dicts
        tokenizer: Tokenizer
        letter_mapping: Dict mapping A/B/C/D to random letters, or None for normal A/B/C/D
    
    Returns:
        prompts: List of formatted prompts
    """
    prompts = []
    
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})
        
        if letter_mapping is None:
            # Normal A/B/C/D format
            q_text = f"Question: {question}\n\n"
            for letter in "ABCD":
                q_text += f"{letter}) {options[letter]}\n"
            q_text += f"\nYour choice (A/B/C/D): "
        else:
            # Randomized letters
            reverse_map = {v: k for k, v in letter_mapping.items()}
            display_letters = sorted(letter_mapping.values())
            
            q_text = f"Question: {question}\n\n"
            for display_letter in display_letters:
                orig_letter = reverse_map[display_letter]
                q_text += f"{display_letter}) {options[orig_letter]}\n"
            q_text += f"\nYour choice ({'/'.join(display_letters)}): "
        
        messages = [{"role": "user", "content": q_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompts.append(prompt)
    
    return prompts


def build_confidence_prompts(batch, tokenizer, mcq_letter_mapping=None, conf_letter_mapping=None):
    """
    Build confidence prompts with optional randomized letters for both MCQ and confidence bins.
    
    Args:
        batch: List of question dicts
        tokenizer: Tokenizer
        mcq_letter_mapping: Dict mapping A/B/C/D to random letters for MCQ options, or None
        conf_letter_mapping: Dict mapping A-H to random letters for confidence bins, or None
    
    Returns:
        prompts: List of formatted prompts
    """
    setup_prompt = (
        "I'm going to show you a multiple-choice question, and I want you to tell me "
        "your level of confidence that you would get the question right. Respond only "
        "with the letter corresponding to the percentage range you choose; do NOT output "
        "any other text."
    )

    confidence_bins = {
        "A": "<5%",
        "B": "5-10%",
        "C": "10-20%",
        "D": "20-40%",
        "E": "40-60%",
        "F": "60-80%",
        "G": "80-90%",
        "H": ">90%",
    }

    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})

        q_lines = [
            "------------------------------",
            "How confident are you that you would get this question right?",
            "----------",
            f"Question: {question}",
        ]
        
        # Show MCQ options (with or without randomization)
        if mcq_letter_mapping is None:
            # Normal A/B/C/D
            for letter in "ABCD":
                q_lines.append(f"{letter}: {options[letter]}")
        else:
            # Randomized MCQ letters
            reverse_map = {v: k for k, v in mcq_letter_mapping.items()}
            display_letters = sorted(mcq_letter_mapping.values())
            for display_letter in display_letters:
                orig_letter = reverse_map[display_letter]
                q_lines.append(f"{display_letter}: {options[orig_letter]}")
        
        q_lines.extend([
            "----------",
            "Confidence options:",
        ])

        # Show confidence bins (with or without randomization)
        if conf_letter_mapping is None:
            # Normal A-H
            for letter in "ABCDEFGH":
                q_lines.append(f"{letter}: {confidence_bins[letter]}")
            response_prompt = "Your choice (A, B, C, D, E, F, G, or H):"
        else:
            # Randomized confidence letters
            reverse_conf_map = {v: k for k, v in conf_letter_mapping.items()}
            conf_display_letters = sorted(conf_letter_mapping.values())
            for display_letter in conf_display_letters:
                orig_letter = reverse_conf_map[display_letter]
                q_lines.append(f"{display_letter}: {confidence_bins[orig_letter]}")
            response_prompt = f"Your choice ({', '.join(conf_display_letters)}):"

        q_lines.extend([
            "------------------------------",
            response_prompt
        ])

        user_content = setup_prompt + "\n\n" + "\n".join(q_lines)
        
        # Manual format to match training
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompts.append(prompt)

    return prompts


def run_mcq_forward_pass(model, tokenizer, prompts, letter_mapping=None, device="cuda"):
    """
    Run MCQ forward pass with optional random letters.
    
    Returns logits, probs, entropy in the ORIGINAL A/B/C/D order.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model(**enc, use_cache=False)

    final_logits = out.logits[:, -1, :]  # [B, vocab]

    if letter_mapping is None:
        # Normal A/B/C/D
        test_letters = ['A', 'B', 'C', 'D']
        reverse_mapping = {l: l for l in test_letters}
    else:
        # Random letters
        reverse_mapping = {v: k for k, v in letter_mapping.items()}
        test_letters = list(letter_mapping.values())
    
    letter_token_ids = {
        letter: get_letter_token_ids(tokenizer, letter) 
        for letter in test_letters
    }
    
    # Get logits for each letter
    logits_dict = {}
    for test_letter in test_letters:
        token_ids = letter_token_ids[test_letter]
        letter_logits = final_logits[:, token_ids]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)
        
        # Map back to original A/B/C/D
        orig_letter = reverse_mapping[test_letter]
        logits_dict[orig_letter] = aggregated_logit
    
    # Stack in A/B/C/D order
    logits4 = torch.stack([logits_dict['A'], logits_dict['B'], 
                           logits_dict['C'], logits_dict['D']], dim=-1)
    
    probs4 = torch.softmax(logits4, dim=-1)
    log_probs4 = torch.log_softmax(logits4, dim=-1)
    entropy = -(probs4 * torch.log(probs4 + 1e-12)).sum(dim=-1)

    # Predicted answer in original A/B/C/D space
    idx = logits4.argmax(dim=-1).tolist()
    pred_letters_orig = ["ABCD"[i] for i in idx]
    
    # What model actually output (display letters)
    if letter_mapping is None:
        pred_letters_display = pred_letters_orig
    else:
        logits_display_order = torch.stack([logits_dict[reverse_mapping[l]] 
                                            for l in sorted(test_letters)], dim=-1)
        idx_display = logits_display_order.argmax(dim=-1).tolist()
        pred_letters_display = [sorted(test_letters)[i] for i in idx_display]

    return {
        "pred_letters": pred_letters_orig,
        "pred_letters_display": pred_letters_display,
        "logits4": logits4,
        "probs4": probs4,
        "log_probs4": log_probs4,
        "entropy": entropy,
    }


def generate_text_output(model, tokenizer, prompt, max_new_tokens=10, device="cuda"):
    """
    Generate actual text output from the model for a given prompt.
    
    Returns the generated text (just the new tokens, not including the prompt).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get only the newly generated tokens
    input_length = enc.input_ids.shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return generated_text


def run_confidence_forward_pass_custom(model, tokenizer, prompts, conf_letter_mapping=None, device="cuda"):
    """
    Run confidence forward pass with optional random letters for A-H bins.
    
    Returns confidence in original A-H space.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model(**enc, use_cache=False)

    final_logits = out.logits[:, -1, :]  # [B, vocab]

    if conf_letter_mapping is None:
        # Normal A-H
        test_letters = list("ABCDEFGH")
        reverse_mapping = {l: l for l in test_letters}
    else:
        # Random letters
        reverse_mapping = {v: k for k, v in conf_letter_mapping.items()}
        test_letters = list(conf_letter_mapping.values())
    
    bin_token_ids = {
        letter: get_letter_token_ids(tokenizer, letter) 
        for letter in test_letters
    }
    
    # Get logits for each letter
    logits_dict = {}
    for test_letter in test_letters:
        token_ids = bin_token_ids[test_letter]
        letter_logits = final_logits[:, token_ids]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)
        
        # Map back to original A-H
        orig_letter = reverse_mapping[test_letter]
        logits_dict[orig_letter] = aggregated_logit
    
    # Stack in A-H order
    logits8 = torch.stack([logits_dict[l] for l in "ABCDEFGH"], dim=-1)
    probs8 = torch.softmax(logits8, dim=-1)
    log_probs8 = torch.log_softmax(logits8, dim=-1)

    # Expected confidence (midpoints of bins)
    mids = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95], dtype=torch.float32, device=device)
    expected_conf = (probs8 * mids).sum(dim=-1)

    # Argmax prediction
    idx = logits8.argmax(dim=-1).tolist()
    pred_bins = ["ABCDEFGH"[i] for i in idx]

    return {
        "logits8": logits8,
        "probs8": probs8,
        "log_probs8": log_probs8,
        "expected_conf": expected_conf,
        "pred_bins": pred_bins,
    }


def test_confidence_calibration(model, tokenizer, dataset, device="cuda", num_questions=200, 
                                mcq_randomized=False, conf_randomized=False, log_file=None, model_type=None):
    """
    Test confidence calibration with optional randomization.
    
    Args:
        mcq_randomized: If True, randomize MCQ letters (A/B/C/D -> random)
        conf_randomized: If True, randomize confidence letters (A-H -> random)
        log_file: File handle for detailed logging (JSONL format)
        model_type: String indicating model type (e.g., "base" or "finetuned")
    """
    
    results = []
    
    # Use subset for speed
    test_dataset = dataset[:num_questions]
    
    mode_str = f"MCQ: {'Random' if mcq_randomized else 'ABCD'}, Confidence: {'Random' if conf_randomized else 'A-H'}"
    # print(f"\nTesting {len(test_dataset)} questions - {mode_str}")
    
    # Determine condition name
    if not mcq_randomized and not conf_randomized:
        condition = "Both Normal"
    elif mcq_randomized and not conf_randomized:
        condition = "MCQ Random, Confidence Normal"
    elif not mcq_randomized and conf_randomized:
        condition = "MCQ Normal, Confidence Random"
    else:
        condition = "Both Random"
    
    for i, row in enumerate(test_dataset):
        # Create random mappings if needed
        mcq_letter_mapping = create_random_letter_mapping(4, seed=42 + i) if mcq_randomized else None
        conf_letter_mapping = create_random_letter_mapping(8, seed=1000 + i) if conf_randomized else None
        
        # 1. MCQ question
        mcq_prompts = build_mcq_prompts([row], tokenizer, mcq_letter_mapping)
        mcq_result = run_mcq_forward_pass(model, tokenizer, mcq_prompts, mcq_letter_mapping, device)
        
        # 2. Confidence question
        conf_prompts = build_confidence_prompts([row], tokenizer, mcq_letter_mapping, conf_letter_mapping)
        conf_result = run_confidence_forward_pass_custom(model, tokenizer, conf_prompts, conf_letter_mapping, device)
        
        # Get results
        pred_letter = mcq_result["pred_letters"][0]
        pred_letter_display = mcq_result["pred_letters_display"][0]
        is_correct = (pred_letter == row["correct_letter"])
        
        mcq_entropy = mcq_result["entropy"][0].item()
        mcq_confidence = (1 - mcq_entropy / math.log(4)) * 100
        
        predicted_confidence = conf_result["expected_conf"][0].item()
        predicted_bin = conf_result["pred_bins"][0]
        
        # Print first question's prompts and answers for this condition
        # if i == 0:
        #     print("\n" + "="*80)
        #     print(f"FIRST QUESTION EXAMPLE - {condition}")
        #     print("="*80)
        #     print("\n--- MCQ QUESTION PROMPT (exactly as model sees it) ---")
        #     print(mcq_prompts[0])
        #     print("\n--- MODEL'S MCQ ANSWER (from logits) ---")
        #     print(f"Predicted letter (display): {pred_letter_display}")
        #     print(f"Predicted letter (original space): {pred_letter}")
        #     print(f"Correct answer: {row['correct_letter']}")
        #     print(f"Is correct: {is_correct}")
        #     # Generate actual text output
        #     mcq_generated = generate_text_output(model, tokenizer, mcq_prompts[0], max_new_tokens=5, device=device)
        #     print(f"\n--- MODEL'S MCQ ANSWER (actual generated text) ---")
        #     print(f"Generated text: {repr(mcq_generated)}")
        #     print(f"Generated text (display): {mcq_generated}")
        #     print("\n--- CONFIDENCE QUESTION PROMPT (exactly as model sees it) ---")
        #     print(conf_prompts[0])
        #     print("\n--- MODEL'S CONFIDENCE ANSWER (from logits) ---")
        #     print(f"Predicted confidence bin: {predicted_bin}")
        #     print(f"Predicted confidence (expected value): {predicted_confidence:.1f}%")
        #     # Generate actual text output
        #     conf_generated = generate_text_output(model, tokenizer, conf_prompts[0], max_new_tokens=5, device=device)
        #     print(f"\n--- MODEL'S CONFIDENCE ANSWER (actual generated text) ---")
        #     print(f"Generated text: {repr(conf_generated)}")
        #     print(f"Generated text (display): {conf_generated}")
        #     print("="*80 + "\n")
        
        # Extract log probabilities
        mcq_log_probs = mcq_result["log_probs4"][0].cpu().tolist()  # [A, B, C, D]
        conf_log_probs = conf_result["log_probs8"][0].cpu().tolist()  # [A, B, C, D, E, F, G, H]
        
        # Create detailed log entry
        if log_file:
            log_entry = {
                "model_type": model_type,
                "condition": condition,
                "qid": row.get("qid", ""),
                "question": row["question"],
                "options": row.get("options", {}),
                "correct_letter": row["correct_letter"],
                "mcq": {
                    "predicted_letter": pred_letter,
                    "predicted_letter_display": pred_letter_display,
                    "is_correct": is_correct,
                    "log_probs": {
                        "A": mcq_log_probs[0],
                        "B": mcq_log_probs[1],
                        "C": mcq_log_probs[2],
                        "D": mcq_log_probs[3],
                    }
                },
                "confidence": {
                    "predicted_bin": predicted_bin,
                    "predicted_confidence": predicted_confidence,
                    "log_probs": {
                        "A": conf_log_probs[0],
                        "B": conf_log_probs[1],
                        "C": conf_log_probs[2],
                        "D": conf_log_probs[3],
                        "E": conf_log_probs[4],
                        "F": conf_log_probs[5],
                        "G": conf_log_probs[6],
                        "H": conf_log_probs[7],
                    }
                }
            }
            
            if mcq_randomized:
                log_entry["mcq_letter_mapping"] = mcq_letter_mapping
            
            if conf_randomized:
                log_entry["conf_letter_mapping"] = conf_letter_mapping
            
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
        
        result_dict = {
            "qid": row.get("qid", ""),
            "question": row["question"],
            "pred_letter_orig": pred_letter,
            "pred_letter_display": pred_letter_display,
            "correct_letter": row["correct_letter"],
            "is_correct": is_correct,
            "mcq_entropy": mcq_entropy,
            "mcq_confidence": mcq_confidence,
            "predicted_confidence": predicted_confidence,
            "predicted_bin": predicted_bin,
        }
        
        if mcq_randomized:
            result_dict["mcq_letter_mapping"] = mcq_letter_mapping
            result_dict["mcq_random_letters"] = sorted(mcq_letter_mapping.values())
        
        if conf_randomized:
            result_dict["conf_letter_mapping"] = conf_letter_mapping
            result_dict["conf_random_letters"] = sorted(conf_letter_mapping.values())
        
        results.append(result_dict)
    
    return results


def analyze_results(results, model_name="Model", mode_str=""):
    """Analyze and print results"""
    
    print("\n" + "="*80)
    print(f"CONFIDENCE CALIBRATION TEST - {model_name}")
    print(f"Mode: {mode_str}")
    print("="*80)
    
    # Basic stats
    total = len(results)
    correct = sum(r["is_correct"] for r in results)
    accuracy = correct / total
    
    print(f"\n--- BASIC STATS ---")
    print(f"Total questions: {total}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.2%})")
    
    # Show example mappings if randomized
    if "mcq_letter_mapping" in results[0]:
        print(f"\n--- EXAMPLE MCQ LETTER MAPPINGS ---")
        for i in range(min(3, len(results))):
            mapping = results[i]["mcq_letter_mapping"]
            letters = results[i]["mcq_random_letters"]
            print(f"Q{i+1}: A→{mapping['A']}, B→{mapping['B']}, C→{mapping['C']}, D→{mapping['D']} = {letters}")
    
    if "conf_letter_mapping" in results[0]:
        print(f"\n--- EXAMPLE CONFIDENCE LETTER MAPPINGS ---")
        for i in range(min(2, len(results))):
            mapping = results[i]["conf_letter_mapping"]
            letters = results[i]["conf_random_letters"]
            print(f"Q{i+1}: {', '.join(f'{k}→{v}' for k,v in list(mapping.items())[:4])}...")
    
    # Answer distribution
    pred_dist = Counter(r["pred_letter_orig"] for r in results)
    correct_dist = Counter(r["correct_letter"] for r in results)
    
    print(f"\n--- ANSWER DISTRIBUTION (A/B/C/D space) ---")
    print("Model predictions:")
    for letter in "ABCD":
        count = pred_dist.get(letter, 0)
        pct = count / total * 100
        print(f"  {letter}: {count:4d} ({pct:5.1f}%)")
    
    print("\nCorrect answers:")
    for letter in "ABCD":
        count = correct_dist.get(letter, 0)
        pct = count / total * 100
        print(f"  {letter}: {count:4d} ({pct:5.1f}%)")
    
    # Chi-square
    expected = total / 4
    chi2 = sum((pred_dist.get(l, 0) - expected)**2 / expected for l in "ABCD")
    print(f"\nChi-square: {chi2:.2f} (critical value: 7.81)")
    if chi2 < 7.81:
        print("  ✓ Answers look uniformly distributed")
    else:
        print("  ⚠️  Model shows answer bias")
    
    # Confidence calibration
    print(f"\n--- CONFIDENCE CALIBRATION ---")
    
    correct_results = [r for r in results if r["is_correct"]]
    wrong_results = [r for r in results if not r["is_correct"]]
    
    discrimination_mcq = 0
    discrimination_pred = 0
    
    if correct_results:
        avg_conf_correct_mcq = sum(r["mcq_confidence"] for r in correct_results) / len(correct_results)
        avg_conf_correct_pred = sum(r["predicted_confidence"] for r in correct_results) / len(correct_results)
        print(f"\nWhen CORRECT ({len(correct_results)} questions):")
        print(f"  MCQ entropy-based confidence: {avg_conf_correct_mcq:.1f}%")
        print(f"  Predicted confidence (bins):  {avg_conf_correct_pred:.1f}%")
    
    if wrong_results:
        avg_conf_wrong_mcq = sum(r["mcq_confidence"] for r in wrong_results) / len(wrong_results)
        avg_conf_wrong_pred = sum(r["predicted_confidence"] for r in wrong_results) / len(wrong_results)
        print(f"\nWhen WRONG ({len(wrong_results)} questions):")
        print(f"  MCQ entropy-based confidence: {avg_conf_wrong_mcq:.1f}%")
        print(f"  Predicted confidence (bins):  {avg_conf_wrong_pred:.1f}%")
    
    if correct_results and wrong_results:
        discrimination_mcq = avg_conf_correct_mcq - avg_conf_wrong_mcq
        discrimination_pred = avg_conf_correct_pred - avg_conf_wrong_pred
        print(f"\nDiscrimination (correct - wrong):")
        print(f"  MCQ entropy-based: {discrimination_mcq:+.1f}%")
        print(f"  Predicted (bins):  {discrimination_pred:+.1f}%")
    
    # Correlation
    mcq_confs = [r["mcq_confidence"] for r in results]
    pred_confs = [r["predicted_confidence"] for r in results]
    
    from scipy.stats import spearmanr, pearsonr
    pearson_corr, _ = pearsonr(mcq_confs, pred_confs)
    spearman_corr, _ = spearmanr(mcq_confs, pred_confs)
    
    print(f"\n--- CORRELATION: MCQ Entropy vs Predicted Confidence ---")
    print(f"Pearson:  {pearson_corr:.3f}")
    print(f"Spearman: {spearman_corr:.3f}")
    
    # Calibration by bins
    print(f"\n--- CALIBRATION BY PREDICTED CONFIDENCE BIN ---")
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    print(f"{'Bin':^12} {'Count':>6} {'Accuracy':>9} {'Avg Pred':>9} {'Gap':>7}")
    print("-" * 55)
    
    for bin_min, bin_max in bins:
        bin_results = [r for r in results 
                      if bin_min <= r["predicted_confidence"] < bin_max]
        if bin_results:
            bin_acc = sum(r["is_correct"] for r in bin_results) / len(bin_results) * 100
            bin_pred = sum(r["predicted_confidence"] for r in bin_results) / len(bin_results)
            gap = abs(bin_acc - bin_pred)
            print(f"{bin_min:3d}-{bin_max:3d}%  {len(bin_results):6d}  {bin_acc:8.1f}%  {bin_pred:8.1f}%  {gap:6.1f}%")
    
    # ECE and Brier
    try:
        from finetune_evaluation_metrics import compute_expected_calibration_error, compute_brier_score
        
        confidences = torch.tensor([r["predicted_confidence"] / 100 for r in results])
        correctness = torch.tensor([float(r["is_correct"]) for r in results])
        
        ece = compute_expected_calibration_error(confidences, correctness, n_bins=10)
        brier = compute_brier_score(confidences, correctness)
        
        print(f"\nECE: {ece:.4f}, Brier: {brier:.4f}")
    except:
        pass
    
    print("="*80)
    
    return {
        "accuracy": accuracy,
        "discrimination_pred": discrimination_pred,
        "pearson_corr": pearson_corr,
        "chi2": chi2,
    }


def test_baseline(model, tokenizer, dataset, device="cuda"):
    """Original baseline test"""
    correct = 0
    predictions = []
    correct_answers = []
    
    for row in dataset:
        question = row["question"]
        options = row["options"]
        
        prompt = (
            f"{question}\n"
            f"A: {options['A']}\n"
            f"B: {options['B']}\n"
            f"C: {options['C']}\n"
            f"D: {options['D']}\n"
            "Answer:"
        )
        
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        enc = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        
        logits = out.logits[0, -1, :]
        
        letter_logits = []
        for letter in "ABCD":
            ids = get_letter_token_ids_simple(tokenizer, letter)
            if ids:
                letter_logits.append(max(logits[i].item() for i in ids))
            else:
                letter_logits.append(float('-inf'))
        
        pred = "ABCD"[letter_logits.index(max(letter_logits))]
        predictions.append(pred)
        correct_answers.append(row['correct_letter'])
        
        if pred == row['correct_letter']:
            correct += 1
    
    accuracy = correct / len(dataset)
    
    print("\n" + "="*60)
    print("BASELINE ACCURACY TEST")
    print("="*60)
    print(f"Accuracy: {correct}/{len(dataset)} ({accuracy:.2%})")
    
    pred_dist = Counter(predictions)
    print("\nAnswer distribution:")
    for letter in 'ABCD':
        count = pred_dist.get(letter, 0)
        pct = count/len(predictions)*100
        print(f"  {letter}: {count:4d} ({pct:5.1f}%)")
    
    expected = len(predictions) / 4
    chi2 = sum((pred_dist.get(l, 0) - expected)**2 / expected for l in 'ABCD')
    print(f"\nChi-square: {chi2:.2f}")
    print("="*60)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_repo", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--test_type", type=str, default="all",
                       choices=["baseline", "all", "normal", "mcq_random", "conf_random", "both_random"],
                       help="Type of test to run")
    parser.add_argument("--num_questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate output filenames
    summary_file = generate_output_filename(args, args.dataset_path, "summary")
    detailed_log_file = generate_output_filename(args, args.dataset_path, "detailed")
    
    # Create output directory
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    # Set up file output for summary
    with Tee(summary_file):
        print("="*80)
        print(f"OUTPUT FILE: {summary_file}")
        print(f"DETAILED LOG FILE: {detailed_log_file}")
        print("="*80)
        print()
        
        print("Loading dataset...")
        dataset = load_jsonl_dataset(args.dataset_path)
    
    # Define test modes
    if args.test_type == "all":
        test_modes = [
            (False, False, "Both Normal (ABCD, A-H)"),
            (True, False, "MCQ Random, Confidence Normal"),
            (False, True, "MCQ Normal, Confidence Random"),
            (True, True, "Both Random"),
        ]
    elif args.test_type == "normal":
        test_modes = [(False, False, "Both Normal (ABCD, A-H)")]
    elif args.test_type == "mcq_random":
        test_modes = [(True, False, "MCQ Random, Confidence Normal")]
    elif args.test_type == "conf_random":
        test_modes = [(False, True, "MCQ Normal, Confidence Random")]
    elif args.test_type == "both_random":
        test_modes = [(True, True, "Both Random")]
    elif args.test_type == "baseline":
        test_modes = []
    
    # Test base model
    if args.test_type != "baseline":
        print("\n" + "="*80)
        print("TESTING BASE MODEL")
        print("="*80)
        
        print("Loading base model...")
        base_model, tokenizer = load_base_model(args.base_model)
        base_model.to(device)
        
        base_results_all = {}
        # Open detailed log file for writing (will append for finetuned model later)
        log_file = open(detailed_log_file, 'w', encoding='utf-8')
        try:
            for mcq_rand, conf_rand, mode_name in test_modes:
                results = test_confidence_calibration(
                    base_model, tokenizer, dataset, device, args.num_questions,
                    mcq_randomized=mcq_rand, conf_randomized=conf_rand,
                    log_file=log_file, model_type="base"
                )
                summary = analyze_results(results, "Base Model", mode_name)
                base_results_all[mode_name] = summary
        finally:
            log_file.close()
    
    # Test finetuned model
    if args.lora_repo and args.test_type != "baseline":
        print("\n" + "="*80)
        print("TESTING FINETUNED MODEL")
        print("="*80)
        
        print("Loading finetuned model...")
        ft_model, tokenizer = load_finetuned_model(args.base_model, args.lora_repo)
        ft_model.to(device)
        
        # Append to detailed log file for finetuned model
        with open(detailed_log_file, 'a', encoding='utf-8') as log_file:
            ft_results_all = {}
            for mcq_rand, conf_rand, mode_name in test_modes:
                results = test_confidence_calibration(
                    ft_model, tokenizer, dataset, device, args.num_questions,
                    mcq_randomized=mcq_rand, conf_randomized=conf_rand,
                    log_file=log_file, model_type="finetuned"
                )
                summary = analyze_results(results, "Finetuned Model", mode_name)
                ft_results_all[mode_name] = summary
        
        # Comparison across all modes
        print("\n" + "="*80)
        print("COMPARISON: BASE vs FINETUNED (ALL MODES)")
        print("="*80)
        
        for mode_name in [m[2] for m in test_modes]:
            base_sum = base_results_all[mode_name]
            ft_sum = ft_results_all[mode_name]
            
            print(f"\n{mode_name}:")
            acc_diff = (ft_sum['accuracy'] - base_sum['accuracy']) * 100
            print(f"  Accuracy: {base_sum['accuracy']:.2%} → {ft_sum['accuracy']:.2%} ({acc_diff:+.1f}%)")
            
            disc_diff = ft_sum['discrimination_pred'] - base_sum['discrimination_pred']
            print(f"  Discrimination: {base_sum['discrimination_pred']:.1f}% → {ft_sum['discrimination_pred']:.1f}% ({disc_diff:+.1f}%)")
            
            corr_diff = ft_sum['pearson_corr'] - base_sum['pearson_corr']
            print(f"  Correlation: {base_sum['pearson_corr']:.3f} → {ft_sum['pearson_corr']:.3f} ({corr_diff:+.3f})")
            
            chi2_diff = ft_sum['chi2'] - base_sum['chi2']
            print(f"  Chi-square: {base_sum['chi2']:.1f} → {ft_sum['chi2']:.1f} ({chi2_diff:+.1f})")
        
        # Final interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        
        normal_mode = "Both Normal (ABCD, A-H)"
        both_random_mode = "Both Random"
        
        if normal_mode in ft_results_all and both_random_mode in ft_results_all:
            normal_acc_adv = (ft_results_all[normal_mode]['accuracy'] - base_results_all[normal_mode]['accuracy']) * 100
            random_acc_adv = (ft_results_all[both_random_mode]['accuracy'] - base_results_all[both_random_mode]['accuracy']) * 100
            
            print(f"\nAccuracy advantage:")
            print(f"  Normal (ABCD, A-H):  +{normal_acc_adv:.1f}%")
            print(f"  Both Random:         +{random_acc_adv:.1f}%")
            print(f"  Retention:           {random_acc_adv/normal_acc_adv*100:.0f}%")
            
            if random_acc_adv < normal_acc_adv * 0.3:
                print("\n❌ TOKEN-SPECIFIC LEARNING DETECTED!")
                print("   Advantage mostly disappeared with random letters")
            elif random_acc_adv > normal_acc_adv * 0.7:
                print("\n✅ TRUE METACOGNITION DETECTED!")
                print("   Advantage maintained with random letters")
            else:
                print("\n⚠️  MIXED SIGNAL")
                print("   Partial advantage retained")
        
        print("="*80)
    
        # Baseline test
        if args.test_type == "baseline":
            print("Loading base model...")
            model, tokenizer = load_base_model(args.base_model)
            model.to(device)
            test_baseline(model, tokenizer, dataset, device)
        
        print()
        print("="*80)
        print(f"Summary output saved to: {summary_file}")
        print(f"Detailed log saved to: {detailed_log_file}")
        print("="*80)


if __name__ == "__main__":
    main()