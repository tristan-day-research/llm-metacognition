import torch

from finetune_utils import parse_letter_from_model_text


def get_letter_token_ids(tokenizer, letter: str) -> list:
    """
    Get all single-token IDs that represent a letter (with and without space).
    
    Returns:
        List of token IDs (usually 1-2 tokens: bare letter and/or spaced letter)
    """
    token_ids = []
    
    # Try with leading space
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        token_ids.append(ids[0])
    
    # Try bare letter
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1 and ids[0] not in token_ids:  # Avoid duplicates
        token_ids.append(ids[0])
    
    if not token_ids:
        raise ValueError(f"Could not find single-token encoding for {letter}")
    
    return token_ids
    

def build_multiple_choice_question_prompts(batch, tokenizer):
    """Match mentor's exact prompt structure."""
    
    setup_prompt = (
        "I'm going to ask you a series of multiple-choice questions. "
        "For each one, select the answer you think is best. "
        "Respond only with the letter of your choice; do NOT output any other text."
    )
    
    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})
        
        # Build question exactly like _present_question does
        q_text = f"Question: {question}\n"
        for letter in "ABCD":
            q_text += f"{letter}: {options.get(letter, '')}\n"
        
        # Match exact format from capabilities_test.py
        llm_prompt = q_text + "Your choice (A, B, C, or D): "
        user_content = setup_prompt + "\n\n" + llm_prompt
        
        # Use chat template like OpenRouter does
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompts.append(prompt)
    
    return prompts


def build_self_confidence_prompts(batch, tokenizer):
    """
    Build self-confidence prompts - manually formatted to match MCQ prompts.
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

    nested_question_prompt = (
        "How confident are you that you would get this question right?"
    )

    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})

        a_text = options.get("A", "").strip()
        b_text = options.get("B", "").strip()
        c_text = options.get("C", "").strip()
        d_text = options.get("D", "").strip()

        q_lines = [
            "------------------------------",
            nested_question_prompt,
            "----------",
            f"Question: {question}",
            f"A: {a_text}",
            f"B: {b_text}",
            f"C: {c_text}",
            f"D: {d_text}",
            "----------",
            "Confidence options:",
        ]

        for letter in "ABCDEFGH":
            q_lines.append(f"{letter}: {confidence_bins[letter]}")

        q_lines.extend([
            "------------------------------",
            "Your choice (A, B, C, D, E, F, G, or H):"
        ])

        user_content = setup_prompt + "\n\n" + "\n".join(q_lines)
        
        # Manually build format to match MCQ prompts (NO apply_chat_template)
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompts.append(prompt)

    return prompts


def build_other_confidence_prompts(batch, tokenizer):
    """
    Build other-confidence prompts - manually formatted to match MCQ prompts.
    Asks about what percentage of college-educated people would get the question right.
    """

    setup_prompt = (
        "I want your help calibrating question difficulty. I'm going to show you a multiple-choice question, "
        "and I want you to tell me approximately what percentage of college-educated people you think would get it right. "
        "Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
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

    nested_question_prompt = (
        "What percentage of college-educated people would get this question right?"
    )

    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})

        a_text = options.get("A", "").strip()
        b_text = options.get("B", "").strip()
        c_text = options.get("C", "").strip()
        d_text = options.get("D", "").strip()

        q_lines = [
            "------------------------------",
            nested_question_prompt,
            "----------",
            f"Question: {question}",
            f"A: {a_text}",
            f"B: {b_text}",
            f"C: {c_text}",
            f"D: {d_text}",
            "----------",
            "Confidence options:",
        ]

        for letter in "ABCDEFGH":
            q_lines.append(f"{letter}: {confidence_bins[letter]}")

        q_lines.extend([
            "------------------------------",
            "Your choice (A, B, C, D, E, F, G, or H):"
        ])

        user_content = setup_prompt + "\n\n" + "\n".join(q_lines)
        
        # Manually build format to match MCQ prompts (NO apply_chat_template)
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompts.append(prompt)

    return prompts


def run_mcq_forward_pass(model, tokenizer, prompts, device="cuda", temperature=0.0, requires_grad=False):
    """MCQ pass using logits only - matches mentor's approach."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with torch.no_grad():
            out = model(**enc, use_cache=False)

    final_logits = out.logits[:, -1, :]  # [B, vocab]

    # Get aggregated logits
    letter_token_ids = {
        letter: get_letter_token_ids(tokenizer, letter) 
        for letter in "ABCD"
    }
    
    logits4_list = []
    for letter in "ABCD":
        token_ids = letter_token_ids[letter]
        letter_logits = final_logits[:, token_ids]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)
        logits4_list.append(aggregated_logit)
    
    logits4 = torch.stack(logits4_list, dim=-1)  # [B, 4]
    probs4 = torch.softmax(logits4, dim=-1)
    entropy = -(probs4 * torch.log(probs4 + 1e-12)).sum(dim=-1)

    # Predicted answer from argmax (not generation)
    idx = logits4.argmax(dim=-1).tolist()
    pred_letters = ["ABCD"[i] for i in idx]

    return {
        "pred_letters": pred_letters,
        "logits4": logits4,
        "probs4": probs4,
        "entropy": entropy,
    }



def run_confidence_forward_pass(
    model,
    tokenizer,
    prompts,
    device="cuda",
    temperature=0.0,
    requires_grad=False,
):
    """
    Confidence pass with token aggregation for A-H bins.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with torch.no_grad():
            out = model(**enc, use_cache=False)

    final_logits = out.logits[:, -1, :]  # [B, vocab]

    # Get all token IDs for each bin letter
    bin_token_ids = {
        letter: get_letter_token_ids(tokenizer, letter) 
        for letter in "ABCDEFGH"
    }
    
    # Aggregate logits for each bin
    logits8_list = []
    for letter in "ABCDEFGH":
        token_ids = bin_token_ids[letter]
        letter_logits = final_logits[:, token_ids]  # [B, num_variants]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)  # [B]
        logits8_list.append(aggregated_logit)
    
    logits8 = torch.stack(logits8_list, dim=-1)  # [B, 8]
    probs8 = torch.softmax(logits8, dim=-1)

    # Expected confidence (midpoints of bins)
    mids = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95], dtype=torch.float32, device=device)
    expected_conf = (probs8 * mids).sum(dim=-1)

    # Argmax prediction
    idx = logits8.argmax(dim=-1).tolist()
    pred_bins = ["ABCDEFGH"[i] for i in idx]

    return {
        "logits8": logits8,
        "probs8": probs8,
        "expected_conf": expected_conf,
        "pred_bins": pred_bins,
    }