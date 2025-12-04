# finetune_delegate_game_simple.py
import argparse
import json
import torch
import os
import random
import numpy as np
from tqdm import tqdm

# Import your existing helper functions for loading models/data
from finetune_load_finetuned_model import load_finetuned_model, load_base_model
from finetune_data_handling import load_jsonl_dataset

# ==========================================
# 1. Prompting Logic (Simplified - No Phase 1)
# ==========================================

def get_letter_token_ids(tokenizer, letter: str) -> list:
    """
    Get all single-token IDs that represent a letter (with and without space).
    Essential for checking logits of 'A', 'B', 'C', 'D', 'T'.
    """
    token_ids = []
    # Try with leading space (Llama 3 often tokenizes " A" distinct from "A")
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        token_ids.append(ids[0])
    # Try bare letter
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1 and ids[0] not in token_ids:
        token_ids.append(ids[0])
    return token_ids

# def build_simple_delegate_prompts(tokenizer, teammate_accuracy, batch_rows):
#     """
#     Simplified version: Just tell model the teammate accuracy.
#     No Phase 1 history needed - tests pure metacognition.
#     """
#     prompts = []
    
#     teammate_pct = int(teammate_accuracy * 100)
    
#     for row in batch_rows:
#         question = row["question"].strip()
#         options = row.get("options", {})
        
#         content = (
#             f"You are playing a question-answering game with a teammate.\n\n"
#             f"Your teammate typically gets {teammate_pct}% of questions correct.\n\n"
#             f"For each question, you can either:\n"
#             f"- Answer yourself by choosing A, B, C, or D\n"
#             f"- Delegate to your teammate by choosing T\n\n"
#             f"Your goal is to maximize the number of correct answers.\n\n"
#             f"Question: {question}\n\n"
#             f"A. {options.get('A', '')}\n"
#             f"B. {options.get('B', '')}\n"
#             f"C. {options.get('C', '')}\n"
#             f"D. {options.get('D', '')}\n"
#             f"T. Delegate to teammate\n\n"
#             f"Your choice (A/B/C/D/T):"
#         )
        
#         messages = [{"role": "user", "content": content}]
#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         prompts.append(prompt)
    
#     return prompts

# With raw text
def build_simple_delegate_prompts(tokenizer, teammate_accuracy, batch_rows):
    """
    Raw text format - NO chat template
    """
    prompts = []
    
    teammate_pct = int(teammate_accuracy * 100)
    
    for row in batch_rows:
        question = row["question"].strip()
        options = row.get("options", {})
        
        # Raw text - no chat template!
        prompt = (
            f"You're playing a game with a teammate who gets {teammate_pct}% correct.\n\n"
            f"Question: {question}\n\n"
            f"A) {options.get('A', '')}\n"
            f"B) {options.get('B', '')}\n"
            f"C) {options.get('C', '')}\n"
            f"D) {options.get('D', '')}\n\n"
            f"Answer (A/B/C/D) or delegate (T): "
        )
        
        prompts.append(prompt)  # ← No apply_chat_template!
    
    return prompts

# Original version
# def run_delegate_forward_pass(model, tokenizer, prompts, device="cuda", debug=False):
#     """
#     Runs inference and extracts logits for tokens: A, B, C, D, T
#     Uses greedy decoding (argmax) - equivalent to Temperature 0.
#     """
#     # DEBUG: Print first prompt to see what model sees
#     if debug and len(prompts) > 0:
#         print("\n" + "="*80)
#         print("DEBUG: ACTUAL PROMPT SENT TO MODEL (last 1000 chars)")
#         print("="*80)
#         print(prompts[0][-1000:])
#         print("="*80)
#         print("\nPress Enter to continue...")
#         input()
    
#     enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
#     with torch.no_grad():
#         out = model(**enc, use_cache=False)
        
#     # Get logits of the last token
#     final_logits = out.logits[:, -1, :]  # [B, vocab]
    
#     # Identify token IDs for our targets (A, B, C, D, T)
#     target_letters = "ABCDT"
#     letter_ids = {k: get_letter_token_ids(tokenizer, k) for k in target_letters}
    
#     logits_list = []
#     for letter in target_letters:
#         ids = letter_ids[letter]
#         if not ids:
#             # Safety: if token not found, give -inf
#             logits_list.append(torch.full((final_logits.shape[0],), float('-inf'), device=device))
#         else:
#             # If multiple tokens represent 'A' (e.g. " A" and "A"), logsumexp them
#             sub_logits = final_logits[:, ids]
#             logits_list.append(torch.logsumexp(sub_logits, dim=-1))
            
#     # Stack -> [B, 5] corresponding to A, B, C, D, T
#     logits5 = torch.stack(logits_list, dim=-1)
#     probs5 = torch.softmax(logits5, dim=-1)
    
#     # DEBUG: Print token analysis (after variables are computed)
#     if debug and len(prompts) > 0:
#         print("\n" + "="*80)
#         print("TOKEN EXTRACTION DEBUG")
#         print("="*80)
        
#         # Show what tokens we're looking for
#         for letter in target_letters:
#             ids = letter_ids[letter]
#             print(f"\nLetter '{letter}':")
#             print(f"  Token IDs: {ids}")
#             if ids:
#                 tokens_decoded = [tokenizer.decode([id]) for id in ids]
#                 print(f"  Decoded: {tokens_decoded}")
        
#         # Show actual logits for first example
#         print(f"\nFirst example logits:")
#         for letter in target_letters:
#             ids = letter_ids[letter]
#             if ids:
#                 logit_val = torch.logsumexp(final_logits[0, ids], dim=-1).item()
#                 print(f"  {letter}: {logit_val:.4f}")
        
#         # Show actual probabilities
#         print(f"\nFirst example probabilities:")
#         for i, letter in enumerate(target_letters):
#             print(f"  {letter}: {probs5[0, i].item():.4f}")
        
#         # Show top 10 tokens by probability
#         print(f"\nTop 10 most likely tokens:")
#         top_logits, top_indices = torch.topk(final_logits[0], 10)
#         for logit, idx in zip(top_logits, top_indices):
#             token = tokenizer.decode([idx.item()])
#             print(f"  '{token}' (id={idx.item()}): {logit.item():.4f}")
        
#         print("="*80)
    
#     # Prediction (Argmax / Greedy Decoding)
#     idx = logits5.argmax(dim=-1).tolist()
#     pred_choices = [target_letters[i] for i in idx]
    
#     return {
#         "pred_choices": pred_choices,
#         "probs5": probs5
#     }

# Constrained version
def run_delegate_forward_pass(model, tokenizer, prompts, device="cuda", debug=False):
    """
    Use logit biasing to force model to only consider A/B/C/D/T tokens
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    
    # Get logits of the last token
    final_logits = out.logits[:, -1, :]  # [B, vocab]
    
    # Get token IDs for A, B, C, D, T
    target_letters = "ABCDT"
    letter_ids = {k: get_letter_token_ids(tokenizer, k) for k in target_letters}
    
    # Create a mask: -inf for all tokens except our target letters
    constrained_logits = torch.full_like(final_logits, float('-inf'))
    
    for letter in target_letters:
        ids = letter_ids[letter]
        for id in ids:
            constrained_logits[:, id] = final_logits[:, id]
    
    # Now softmax over only valid tokens
    probs = torch.softmax(constrained_logits, dim=-1)
    
    # Get the most likely token
    pred_token_ids = constrained_logits.argmax(dim=-1)
    
    # Map back to letters
    token_to_letter = {}
    for letter in target_letters:
        for id in letter_ids[letter]:
            token_to_letter[id] = letter
    
    pred_choices = [token_to_letter.get(tid.item(), "D") for tid in pred_token_ids]
    
    # Extract probabilities for each letter (sum over both token variants)
    logits_list = []
    for letter in target_letters:
        ids = letter_ids[letter]
        if not ids:
            logits_list.append(torch.full((final_logits.shape[0],), float('-inf'), device=device))
        else:
            sub_logits = final_logits[:, ids]
            logits_list.append(torch.logsumexp(sub_logits, dim=-1))
    
    logits5 = torch.stack(logits_list, dim=-1)
    probs5 = torch.softmax(logits5, dim=-1)
    
    return {
        "pred_choices": pred_choices,
        "probs5": probs5
    }


# ==========================================
# 2. Main Execution
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_output_filename(args):
    """
    Automatically generate output filename based on run parameters.
    Format: DG_simple_{base_model}_{checkpoint}_{dataset}_teamacc_{teammate_accuracy}.jsonl
    (checkpoint is omitted if no --lora_repo is provided)
    """
    # Extract base model name (last part after /)
    base_model_name = args.base_model.split('/')[-1]
    
    # Extract checkpoint name if using LoRA
    # Only include checkpoint if lora_repo is provided, not None, not empty, and not a placeholder
    if (args.lora_repo is not None and 
        args.lora_repo.strip() and 
        args.lora_repo.strip().lower() not in ['your-checkpoint', 'checkpoint', 'none', '']):
        checkpoint_name = args.lora_repo.split('/')[-1]
        # Construct filename with checkpoint
        checkpoint_part = f"_{checkpoint_name}"
    else:
        # No checkpoint - omit from filename
        checkpoint_part = ""
    
    # Extract dataset name (filename without extension)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    
    # Format teammate accuracy (remove decimal point for cleaner filename)
    teammate_acc_str = f"{int(args.teammate_accuracy * 100)}"
    
    # Construct filename
    filename = f"DG_simple_{base_model_name}{checkpoint_part}_{dataset_name}_teamacc_{teammate_acc_str}.jsonl"
    
    return filename

def main():
    parser = argparse.ArgumentParser(description="Run the Delegate Game (Simplified - No Phase 1)")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_repo", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path. If not provided, will auto-generate based on run parameters.")
    parser.add_argument("--output_dir", type=str, default="finetune_evals",
                        help="Directory for output files (default: finetune_evals)")
    
    # Game parameters
    parser.add_argument("--teammate_accuracy", type=float, default=0.6, 
                        help="Accuracy of simulated teammate (default: 0.6)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true",
                        help="Print first prompt for debugging")
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # EARLY VALIDATION - Setup output path
    # ==========================================
    print("--- Setting Up Output Path ---")
    
    # Auto-generate filename if not provided
    if args.output_file is None:
        filename = generate_output_filename(args)
        args.output_file = os.path.join(args.output_dir, filename)
        print(f"Auto-generated filename: {filename}")
    
    # Check if output_file is actually a directory
    if os.path.isdir(args.output_file):
        raise ValueError(
            f"Error: --output_file is a directory, not a file path.\n"
            f"You provided: {args.output_file}\n"
            f"Example correct usage: --output_file finetune_evals/delegate_game_results.jsonl"
        )
    
    # Get the directory path
    output_dir = os.path.dirname(args.output_file)
    
    # If there's a directory component, check/create it
    if output_dir:
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create output directory '{output_dir}': {e}")
        elif not os.path.isdir(output_dir):
            raise ValueError(f"Output directory path exists but is not a directory: {output_dir}")
    
    # Check if we can write to the output file (test write)
    try:
        with open(args.output_file, 'w') as f:
            pass  # Just test if we can open it for writing
        print(f"✓ Output file path is valid: {args.output_file}")
    except Exception as e:
        raise ValueError(f"Cannot write to output file '{args.output_file}': {e}")
    
    print()
    
    # ==========================================
    # Now proceed with the rest
    # ==========================================
    print(f"--- Delegate Game Setup (Simplified Version) ---")
    print(f"Model: {args.lora_repo if args.lora_repo else args.base_model}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Teammate Accuracy: {args.teammate_accuracy}")
    print(f"Output: {args.output_file}")
    print(f"No Phase 1 - Direct delegation test")
    print()

    # 1. Load Model
    print("Loading Model...")
    if args.lora_repo:
        print(f"Loading Base + LoRA: {args.lora_repo}")
        model, tokenizer = load_finetuned_model(args.base_model, args.lora_repo)
    else:
        print(f"Loading Base: {args.base_model}")
        model, tokenizer = load_base_model(args.base_model)
    model.to(device)

    # 2. Load Data
    print("Loading data...")
    dataset = load_jsonl_dataset(args.dataset_path)
    print(f"Total questions: {len(dataset)}")

    # 3. Run Game (No Phase 1!)
    results = []
    print("Starting Delegate Game (Simplified)...")
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i : i + args.batch_size]
        
        # Build Prompts (Simple version - no history)
        prompts = build_simple_delegate_prompts(
            tokenizer, 
            args.teammate_accuracy, 
            batch
        )
        
        # Forward Pass (Argmax / Greedy Decoding)
        outputs = run_delegate_forward_pass(
            model, 
            tokenizer, 
            prompts, 
            device=device,
            debug=(args.debug and i == 0)  # Only debug first batch
        )
        
        # Extract Data
        preds = outputs["pred_choices"]
        probs = outputs["probs5"].cpu().tolist()
        
        for j, row in enumerate(batch):
            choice = preds[j]
            row_probs = probs[j]
            
            # Game Logic: Determine correctness
            did_delegate = (choice == 'T')
            correct_letter = row["correct_letter"]
            
            if did_delegate:
                # Simulate Teammate (Randomized based on accuracy setting)
                is_correct = (random.random() < args.teammate_accuracy)
                source = "teammate"
            else:
                # Model Answer
                is_correct = (choice == correct_letter)
                source = "model"
                
            results.append({
                "qid": row.get("qid"),
                "question": row["question"],
                "correct_letter": correct_letter,
                "model_choice": choice,
                "did_delegate": did_delegate,
                "is_correct": is_correct,
                "source": source,
                "probs": {
                    "A": row_probs[0], "B": row_probs[1], 
                    "C": row_probs[2], "D": row_probs[3], 
                    "T": row_probs[4]
                }
            })

    # 4. Summary & Save
    score = sum(1 for r in results if r['is_correct'])
    delegations = sum(1 for r in results if r['did_delegate'])
    total = len(results)
    
    # Calculate breakdown by source
    model_answered = [r for r in results if not r['did_delegate']]
    model_correct = sum(1 for r in model_answered if r['is_correct'])
    teammate_answered = [r for r in results if r['did_delegate']]
    teammate_correct = sum(1 for r in teammate_answered if r['is_correct'])
    
    accuracy = score/total if total > 0 else 0
    del_rate = delegations/total if total > 0 else 0
    model_acc = model_correct/len(model_answered) if model_answered else 0
    teammate_acc = teammate_correct/len(teammate_answered) if teammate_answered else 0
    
    print("\n" + "="*50)
    print("--- Final Results ---")
    print("="*50)
    print(f"Model: {args.lora_repo if args.lora_repo else args.base_model}")
    print(f"\nOverall Performance:")
    print(f"  Total Accuracy: {score}/{total} ({accuracy:.2%})")
    print(f"  Delegation Rate: {delegations}/{total} ({del_rate:.2%})")
    print(f"\nBreakdown by Source:")
    print(f"  Model answered: {len(model_answered)} questions → {model_correct} correct ({model_acc:.2%})")
    print(f"  Teammate answered: {len(teammate_answered)} questions → {teammate_correct} correct ({teammate_acc:.2%})")
    print(f"\nExpected Baselines:")
    print(f"  Delegate all → {args.teammate_accuracy:.1%} accuracy")
    print(f"  Answer all → Check your model's baseline accuracy")
    print("="*50)
    
    # Save results
    with open(args.output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n✓ Saved detailed results to {args.output_file}")

if __name__ == "__main__":
    main()