import argparse
import json
import torch
import os
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Import your existing helper functions for loading models/data
from finetune_load_finetuned_model import load_finetuned_model, load_base_model
from finetune_data_handling import load_jsonl_dataset


# ==========================================
# 1. Prompting Logic (Paper Figure 4 Implementation)
# ==========================================

# def construct_phase_1_messages(history_data, baseline_answers, teammate_accuracy=0.6, 
#                                 include_model_feedback=False, seed=42):
#     """
#     Constructs the conversational history for Phase 1 (50 questions).
#     Matches the turn-based structure in Figure 4 of the paper.
    
#     Args:
#         history_data: List of question dicts for Phase 1
#         baseline_answers: Dict mapping question index to model's baseline answer
#         teammate_accuracy: Simulated teammate accuracy (default 0.6)
#         include_model_feedback: Whether to show feedback on model's answers (optional per paper)
#         seed: Random seed for reproducibility
    
#     Structure:
#     1. User: Intro + Question 1
#     2. Assistant: [Model's Answer]
#     3. User: Teammate Feedback + Question 2
#     4. Assistant: [Model's Answer]
#     ...
#     Returns: List of message dicts AND the final teammate feedback string 
#              (which must be prepended to Phase 2).
#     """
#     rng = random.Random(seed)
    
#     # Intro Text from Figure 4 [cite: 398-408]
#     intro_text = (
#         "Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.\n"
#         "The game consists of two phases:\n"
#         "1. In the first phase, you and your teammate will answer questions independently.\n"
#         "2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.\n"
#         "The goal is to see how well you can work with your teammate and make decisions together.\n"
#         "In phase 1, you will receive feedback on your teammate's performance.\n"
#         "You want to maximize your total score. In phase 1, that means answering as best you can. "
#         "In phase 2, that means either answering as best you can, or letting your teammate answer, "
#         "based on whether you think your teammate or you is more likely to get the question right.\n"
#         "----------\n"
#         "Starting Phase 1: Answer the Questions\n"
#         "----------"
#     )
    
#     messages = []
#     total_q = len(history_data)
#     last_feedback = ""

#     for i, row in enumerate(history_data):
#         q_num = i + 1
#         question = row["question"].strip()
#         options = row.get("options", {})
        
#         # Format Question Block [cite: 410-416]
#         q_text = (
#             f"Question {q_num}/{total_q}:\n{question}\n"
#             f"A: {options.get('A', '')}\n"
#             f"B: {options.get('B', '')}\n"
#             f"C: {options.get('C', '')}\n"
#             f"D: {options.get('D', '')}\n"
#             "Your answer (A, B, C, D):"
#         )
        
#         if i == 0:
#             # First turn: Intro + Q1
#             content = intro_text + "\n\n" + q_text
#             messages.append({"role": "user", "content": content})
#         else:
#             # Subsequent turns: Previous Feedback + Q_i
#             # [cite: 419-421] User: Teammate's answer... Question X...
#             content = f"{last_feedback}\n\n{q_text}"
#             messages.append({"role": "user", "content": content})
        
#         # Use Model's ACTUAL baseline answer (not simulated random)
#         # This is critical - Phase 1 should show what the model actually answered at baseline
#         model_choice = baseline_answers.get(i, rng.choice(["A", "B", "C", "D"]))
#         messages.append({"role": "assistant", "content": model_choice})
        
#         # Generate Feedback for THIS question to be used in NEXT turn
#         is_teammate_correct = rng.random() < teammate_accuracy
#         outcome = "Correct" if is_teammate_correct else "Incorrect"
        
#         # Build feedback string
#         # Paper shows: "Teammate's answer: (Correct)" - no letter shown
#         feedback_parts = [f"Teammate's answer: ({outcome})"]
        
#         # Optional: Include model feedback if requested
#         if include_model_feedback:
#             correct_letter = row.get("correct_letter", "")
#             is_model_correct = (model_choice == correct_letter)
#             model_outcome = "Correct" if is_model_correct else "Incorrect"
#             feedback_parts.append(f"Your answer: ({model_outcome})")
        
#         last_feedback = "\n".join(feedback_parts)

#     return messages, last_feedback

def construct_phase_1_messages(history_data, baseline_answers, teammate_accuracy=0.6, 
                                include_model_feedback=False, seed=42):
    """
    Constructs the conversational history for Phase 1 (20 questions).
    CRITICAL FIX: Show T option in Phase 1 so format matches Phase 2
    """
    rng = random.Random(seed)
    
    intro_text = (
        "Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.\n"
        "The game consists of two phases:\n"
        "1. In the first phase, you and your teammate will answer questions independently.\n"
        "2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.\n"
        "The goal is to see how well you can work with your teammate and make decisions together.\n"
        "In phase 1, you will receive feedback on your teammate's performance.\n"
        "You want to maximize your total score. In phase 1, that means answering as best you can. "
        "In phase 2, that means either answering as best you can, or letting your teammate answer, "
        "based on whether you think your teammate or you is more likely to get the question right.\n"
        "----------\n"
        "Starting Phase 1: Answer the Questions\n"
        "----------"
    )
    
    messages = []
    total_q = len(history_data)
    last_feedback = ""

    for i, row in enumerate(history_data):
        q_num = i + 1
        question = row["question"].strip()
        options = row.get("options", {})
        
        # CRITICAL FIX: Include T option in Phase 1 too!
        q_text = (
            f"Question {q_num}/{total_q}:\n{question}\n"
            f"A: {options.get('A', '')}\n"
            f"B: {options.get('B', '')}\n"
            f"C: {options.get('C', '')}\n"
            f"D: {options.get('D', '')}\n"
            f"T: Delegate to teammate\n"  # ← ADD THIS
            "Your choice:"  # ← Changed from "Your answer" to "Your choice"
        )
        
        if i == 0:
            content = intro_text + "\n\n" + q_text
            messages.append({"role": "user", "content": content})
        else:
            content = f"{last_feedback}\n\n{q_text}"
            messages.append({"role": "user", "content": content})
        
        # Use Model's ACTUAL baseline answer (never T in Phase 1)
        model_choice = baseline_answers.get(i, rng.choice(["A", "B", "C", "D"]))
        messages.append({"role": "assistant", "content": model_choice})
        
        # Generate Feedback
        is_teammate_correct = rng.random() < teammate_accuracy
        outcome = "Correct" if is_teammate_correct else "Incorrect"
        
        feedback_parts = [f"Teammate's answer: ({outcome})"]
        
        if include_model_feedback:
            correct_letter = row.get("correct_letter", "")
            is_model_correct = (model_choice == correct_letter)
            model_outcome = "Correct" if is_model_correct else "Incorrect"
            feedback_parts.append(f"Your answer: ({model_outcome})")
        
        last_feedback = "\n".join(feedback_parts)

    return messages, last_feedback


def build_phase_2_prompts(tokenizer, history_messages, last_feedback, batch_rows):
    """
    Appends the Phase 2 instruction and current question to the history.
    Uses tokenizer.apply_chat_template to format it for the specific model.
    """
    
    # Text from Figure 4 [cite: 422-423]
    phase_2_intro = (
        "----------\n"
        "Phase 2: Answer or Delegate\n"
        "----------\n"
        "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer."
    )
    
    prompts = []
    for row in batch_rows:
        question = row["question"].strip()
        options = row.get("options", {})
        
        # Format Phase 2 Question [cite: 424-435]
        # q_text = (
        #     f"Question:\n{question}\n"
        #     f"A: {options.get('A', '')}\n"
        #     f"B: {options.get('B', '')}\n"
        #     f"C: {options.get('C', '')}\n"
        #     f"D: {options.get('D', '')}\n"
        #     "Your choice (A, B, C, D, or T): "
        # )

        # Change just the prompt format
        q_text = (
            f"Question:\n{question}\n"
            f"A: {options.get('A', '')}\n"
            f"B: {options.get('B', '')}\n"
            f"C: {options.get('C', '')}\n"
            f"D: {options.get('D', '')}\n"
            f"T: Delegate to teammate\n"  # Make T symmetric
            "Your choice:"
        )
                
        # Combine: [Last Feedback from Phase 1] + [Phase 2 Intro] + [Current Q]
        final_user_content = f"{last_feedback}\n\n{phase_2_intro}\n\n{q_text}"
        
        # Copy history and append final message
        current_messages = history_messages.copy()
        current_messages.append({"role": "user", "content": final_user_content})
        
        # Apply Chat Template
        # add_generation_prompt=True ensures it ends with "<|start_header_id|>assistant<|end_header_id|>"
        prompt = tokenizer.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        
    return prompts


def run_delegate_forward_pass(model, tokenizer, prompts, device="cuda"):
    """
    Runs inference and extracts logits for tokens: A, B, C, D, T
    Uses greedy decoding (argmax) - equivalent to Temperature 0.
    
    Note: Paper mentions "temperature 1.0" but describes taking "highest-probability output"
    which is argmax (temp 0). This implementation uses argmax as that matches the description.
    """

    # DEBUG: Print first prompt to see what model sees
    # if len(prompts) > 0:
    #     print("\n" + "="*80)
    #     print("DEBUG: ACTUAL PROMPT SENT TO MODEL (last 1000 chars)")
    #     print("="*80)
    #     print(prompts[0][-1000:])
    #     print("="*80)
        # print("\nPress Enter to continue...")
        # input()

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        out = model(**enc, use_cache=False)
        
    # Get logits of the last token
    final_logits = out.logits[:, -1, :]  # [B, vocab]
    
    # Identify token IDs for our targets (A, B, C, D, T)
    target_letters = "ABCDT"
    letter_ids = {k: get_letter_token_ids(tokenizer, k) for k in target_letters}
    
    logits_list = []
    for letter in target_letters:
        ids = letter_ids[letter]
        if not ids:
            # Safety: if token not found, give -inf
            logits_list.append(torch.full((final_logits.shape[0],), float('-inf'), device=device))
        else:
            # If multiple tokens represent 'A' (e.g. " A" and "A"), logsumexp them
            sub_logits = final_logits[:, ids]
            logits_list.append(torch.logsumexp(sub_logits, dim=-1))
            
    # Stack -> [B, 5] corresponding to A, B, C, D, T
    logits5 = torch.stack(logits_list, dim=-1)
    probs5 = torch.softmax(logits5, dim=-1)
    
    # Prediction (Argmax / Greedy Decoding)
    idx = logits5.argmax(dim=-1).tolist()
    pred_choices = [target_letters[i] for i in idx]
    
    return {
        "pred_choices": pred_choices,
        "probs5": probs5
    }


def get_baseline_answers(dataset, model, tokenizer, device="cuda", batch_size=4):
    """
    Get baseline answers for the dataset to use in Phase 1 construction.
    This runs the model on each question individually to get its default answer.
    
    Returns: Dict mapping question index -> answer letter
    """
    baseline_answers = {}
    
    print("Collecting baseline answers for Phase 1 history construction...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        
        # Simple baseline prompts (no game context)
        prompts = []
        for row in batch:
            question = row["question"].strip()
            options = row.get("options", {})
            
            prompt_text = (
                f"Answer this multiple choice question.\n\n"
                f"Question: {question}\n"
                f"A: {options.get('A', '')}\n"
                f"B: {options.get('B', '')}\n"
                f"C: {options.get('C', '')}\n"
                f"D: {options.get('D', '')}\n\n"
                f"Your answer (A, B, C, D):"
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        # Get answers
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            out = model(**enc, use_cache=False)
        
        final_logits = out.logits[:, -1, :]
        
        # Extract A, B, C, D logits
        target_letters = "ABCD"
        letter_ids = {k: get_letter_token_ids(tokenizer, k) for k in target_letters}
        
        logits_list = []
        for letter in target_letters:
            ids = letter_ids[letter]
            if not ids:
                logits_list.append(torch.full((final_logits.shape[0],), float('-inf'), device=device))
            else:
                sub_logits = final_logits[:, ids]
                logits_list.append(torch.logsumexp(sub_logits, dim=-1))
        
        logits4 = torch.stack(logits_list, dim=-1)
        idx = logits4.argmax(dim=-1).tolist()
        pred_choices = [target_letters[j] for j in idx]
        
        # Store answers
        for j, choice in enumerate(pred_choices):
            baseline_answers[i + j] = choice
    
    return baseline_answers


# ==========================================
# Delegate Game utils
# ==========================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_output_filename(args):
    """
    Automatically generate output filename based on run parameters.
    Format: DG_{base_model}_{checkpoint}_{dataset}_teamacc_{teammate_accuracy}_{timestamp}.jsonl
    """
    # Extract base model name (last part after /)
    base_model_name = args.base_model.split('/')[-1]
    
    # Extract checkpoint name if using LoRA
    if args.lora_repo:
        checkpoint_name = args.lora_repo.split('/')[-1]
    else:
        checkpoint_name = "base"
    
    # Extract dataset name (filename without extension)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    
    # Format teammate accuracy (remove decimal point for cleaner filename)
    teammate_acc_str = f"{int(args.teammate_accuracy * 100)}"
    
    # Generate timestamp (format: YYYYMMDD-HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Construct filename
    filename = f"DG_{base_model_name}_{checkpoint_name}_{dataset_name}_teamacc_{teammate_acc_str}_{timestamp}.jsonl"
    
    return filename


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


# ==========================================
# Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Run the Delegate Game (Paper Implementation)")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_repo", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path. If not provided, will auto-generate based on run parameters.")
    parser.add_argument("--output_dir", type=str, default="finetune_evals",
                        help="Directory for output files (default: finetune_evals)")
    
    # Paper parameters
    parser.add_argument("--teammate_accuracy", type=float, default=0.6, help="Accuracy of simulated teammate")
    parser.add_argument("--phase_1_size", type=int, default=50, help="Number of questions in Phase 1 history (Paper uses 50)")
    parser.add_argument("--include_model_feedback", action="store_true", 
                        help="Include feedback on model's Phase 1 answers (optional per paper)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
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
    print(f"--- Delegate Game Setup ---")
    print(f"Model: {args.lora_repo if args.lora_repo else args.base_model}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Teammate Accuracy: {args.teammate_accuracy}")
    print(f"Phase 1 History Size: {args.phase_1_size}")
    print(f"Include Model Feedback: {args.include_model_feedback}")
    print(f"Output: {args.output_file}")
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

    # 2. Load & Split Data
    print("Loading data...")
    full_dataset = load_jsonl_dataset(args.dataset_path)
    
    if len(full_dataset) < args.phase_1_size + 1:
        raise ValueError(f"Dataset too small ({len(full_dataset)}) for requested history size ({args.phase_1_size}).")
        
    # Split: First N questions are history, rest are test
    history_data = full_dataset[:args.phase_1_size]
    test_data = full_dataset[args.phase_1_size:]
    print(f"Data Split: {len(history_data)} History / {len(test_data)} Test")

    # 3. Get Baseline Answers for Phase 1 History
    # CRITICAL: Phase 1 must show model's actual baseline answers, not random simulation
    baseline_answers = get_baseline_answers(
        history_data, 
        model, 
        tokenizer, 
        device=device,
        batch_size=args.batch_size
    )

    # 4. Construct Phase 1 History
    # We build the message list once. This ensures consistent "history" for the whole run.
    print("Constructing Phase 1 conversation history...")
    history_messages, last_feedback = construct_phase_1_messages(
        history_data,
        baseline_answers,
        teammate_accuracy=args.teammate_accuracy,
        include_model_feedback=args.include_model_feedback,
        seed=args.seed
    )

    # 5. Run Game
    results = []
    print("Starting Phase 2 Inference...")
    
    # Process in batches
    for i in tqdm(range(0, len(test_data), args.batch_size)):
        batch = test_data[i : i + args.batch_size]
        
        # Build Prompts
        prompts = build_phase_2_prompts(tokenizer, history_messages, last_feedback, batch)
        
        # Forward Pass (Argmax / Greedy Decoding)
        outputs = run_delegate_forward_pass(model, tokenizer, prompts, device=device)
        
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

    # 6. Summary & Save
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