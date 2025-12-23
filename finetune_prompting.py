import torch
import random
from finetune_utils import parse_letter_from_model_text


def get_mcq_letter_mapping(scheme: str, seed: int = None):
    """
    Generate a mapping from A-D (internal position labels) to display letters.
    
    Args:
        scheme: One of "A-D", "E-H", "I-L", "M-P", "Q-T", "U-X", "Y-Z" (wraps), or "random"
        seed: Optional random seed for "random" scheme (for reproducibility)
    
    Returns:
        dict: Mapping from A-D to display letters (e.g., {"A": "A", "B": "B", ...} or {"A": "E", "B": "F", ...})
    """
    if scheme == "A-D":
        return {chr(ord('A') + i): chr(ord('A') + i) for i in range(4)}
    elif scheme in ["E-H", "I-L", "M-P", "Q-T", "U-X"]:
        start_letter = scheme[0]
        return {chr(ord('A') + i): chr(ord(start_letter) + i) for i in range(4)}
    elif scheme == "Y-Z":
        # Wraps around: Y, Z, then A, B
        return {"A": "Y", "B": "Z", "C": "A", "D": "B"}
    elif scheme == "random":
        if seed is not None:
            random.seed(seed)
        # Use all letters except those that might conflict with confidence letters
        # For now, use any 4 letters from the alphabet
        available_letters = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        selected = random.sample(available_letters, 4)
        return {chr(ord('A') + i): selected[i] for i in range(4)}
    else:
        raise ValueError(f"Unknown mcq_letter_scheme: {scheme}. Must be one of: 'A-D', 'E-H', 'I-L', 'M-P', 'Q-T', 'U-X', 'Y-Z', or 'random'")


def get_confidence_letter_mapping(scheme: str, seed: int = None):
    """
    Generate a mapping from A-H (internal bin labels) to display letters.
    
    Args:
        scheme: One of "A-H", "S-Z", or "random"
        seed: Optional random seed for "random" scheme (for reproducibility)
    
    Returns:
        dict: Mapping from A-H to display letters (e.g., {"A": "A", "B": "B", ...} or {"A": "S", "B": "T", ...})
    """
    if scheme == "A-H":
        return {chr(ord('A') + i): chr(ord('A') + i) for i in range(8)}
    elif scheme == "S-Z":
        return {chr(ord('A') + i): chr(ord('S') + i) for i in range(8)}
    elif scheme == "random":
        if seed is not None:
            random.seed(seed)
        # Use all letters except A-D (which are used for MCQ answers)
        available_letters = [c for c in 'EFGHIJKLMNOPQRSTUVWXYZ']
        selected = random.sample(available_letters, 8)
        return {chr(ord('A') + i): selected[i] for i in range(8)}
    else:
        raise ValueError(f"Unknown confidence_letter_scheme: {scheme}. Must be one of: 'A-H', 'S-Z', 'random'")


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
    

def build_multiple_choice_question_prompts(batch, tokenizer, mcq_letter_mapping=None):
    """
    Build multiple choice question prompts with optional letter mapping.
    
    Args:
        batch: List of question dicts
        tokenizer: Tokenizer
        mcq_letter_mapping: Optional dict mapping A-D to display letters.
                           If None, uses A-D.
    
    Returns:
        List of prompt strings
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    
    setup_prompt = (
        "I'm going to ask you a series of multiple-choice questions. "
        "For each one, select the answer you think is best. "
        "Respond only with the letter of your choice; do NOT output any other text."
    )
    
    # Get display letters in order
    display_letters = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    display_letters_str = ", ".join(display_letters)
    
    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})
        
        # Build question using display letters
        q_text = f"Question: {question}\n"
        for i, internal_letter in enumerate("ABCD"):
            display_letter = display_letters[i]
            q_text += f"{display_letter}: {options.get(internal_letter, '')}\n"
        
        # Use display letters in prompt
        llm_prompt = q_text + f"Your choice ({display_letters_str}): "
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


def build_self_confidence_prompts(batch, tokenizer, confidence_letter_mapping=None, mcq_letter_mapping=None):
    """
    Build self-confidence prompts - manually formatted to match MCQ prompts.
    
    Args:
        batch: List of question dicts
        tokenizer: Tokenizer (unused but kept for API consistency)
        confidence_letter_mapping: Optional dict mapping A-H to display letters.
                                  If None, uses A-H.
        mcq_letter_mapping: Optional dict mapping A-D to display letters for the nested MCQ question.
                           If None, uses A-D.
    """
    if confidence_letter_mapping is None:
        confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")

    setup_prompt = (
        "I'm going to show you a multiple-choice question, and I want you to tell me "
        "your level of confidence that you would get the question right. Respond only "
        "with the letter corresponding to the percentage range you choose; do NOT output "
        "any other text."
    )

    # Confidence bins in A-H order (internal)
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

        # Get MCQ display letters for the nested question
        mcq_display_letters = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
        
        a_text = options.get("A", "").strip()
        b_text = options.get("B", "").strip()
        c_text = options.get("C", "").strip()
        d_text = options.get("D", "").strip()

        q_lines = [
            "------------------------------",
            nested_question_prompt,
            "----------",
            f"Question: {question}",
            f"{mcq_display_letters[0]}: {a_text}",
            f"{mcq_display_letters[1]}: {b_text}",
            f"{mcq_display_letters[2]}: {c_text}",
            f"{mcq_display_letters[3]}: {d_text}",
            "----------",
            "Confidence options:",
        ]

        # Use display letters from mapping
        display_letters = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
        for i, internal_letter in enumerate("ABCDEFGH"):
            display_letter = display_letters[i]
            q_lines.append(f"{display_letter}: {confidence_bins[internal_letter]}")

        # Build choice prompt with display letters
        choice_list = ", ".join(display_letters)
        q_lines.extend([
            "------------------------------",
            f"Your choice ({choice_list}):"
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


def build_other_confidence_prompts(batch, tokenizer, confidence_letter_mapping=None, mcq_letter_mapping=None):
    """
    Build other-confidence prompts - manually formatted to match MCQ prompts.
    Asks about what percentage of college-educated people would get the question right.
    
    Args:
        batch: List of question dicts
        tokenizer: Tokenizer (unused but kept for API consistency)
        confidence_letter_mapping: Optional dict mapping A-H to display letters.
                                  If None, uses A-H.
        mcq_letter_mapping: Optional dict mapping A-D to display letters for the nested MCQ question.
                           If None, uses A-D.
    """
    if confidence_letter_mapping is None:
        confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")

    setup_prompt = (
        "I want your help calibrating question difficulty. I'm going to show you a multiple-choice question, "
        "and I want you to tell me approximately what percentage of college-educated people you think would get it right. "
        "Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
    )

    # Confidence bins in A-H order (internal)
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

        # Get MCQ display letters for the nested question
        mcq_display_letters = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
        
        a_text = options.get("A", "").strip()
        b_text = options.get("B", "").strip()
        c_text = options.get("C", "").strip()
        d_text = options.get("D", "").strip()

        q_lines = [
            "------------------------------",
            nested_question_prompt,
            "----------",
            f"Question: {question}",
            f"{mcq_display_letters[0]}: {a_text}",
            f"{mcq_display_letters[1]}: {b_text}",
            f"{mcq_display_letters[2]}: {c_text}",
            f"{mcq_display_letters[3]}: {d_text}",
            "----------",
            "Confidence options:",
        ]

        # Use display letters from mapping
        display_letters = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
        for i, internal_letter in enumerate("ABCDEFGH"):
            display_letter = display_letters[i]
            q_lines.append(f"{display_letter}: {confidence_bins[internal_letter]}")

        # Build choice prompt with display letters
        choice_list = ", ".join(display_letters)
        q_lines.extend([
            "------------------------------",
            f"Your choice ({choice_list}):"
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


def run_mcq_forward_pass(model, tokenizer, prompts, device="cuda", temperature=0.0, requires_grad=False, mcq_letter_mapping=None):
    """
    Run MCQ forward pass with optional letter mapping.
    
    Args:
        model: Model to run forward pass on
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device to run on
        temperature: Temperature for sampling (unused, kept for API consistency)
        requires_grad: Whether to compute gradients
        mcq_letter_mapping: Optional dict mapping A-D (internal) to display letters.
                          If None, uses A-D.
    
    Returns:
        dict with:
            - pred_letters: [B] predicted display letters (e.g., E, F, G, H)
            - pred_positions: [B] predicted position indices (0-3, corresponding to A-D internally)
            - logits4: [B, 4] logits in A-D order (for loss calculation)
            - probs4: [B, 4] probabilities in A-D order
            - entropy: [B] entropy values
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    
    # Create reverse mapping: display_letter -> internal_letter (A-D)
    reverse_mapping = {v: k for k, v in mcq_letter_mapping.items()}
    
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with torch.no_grad():
            out = model(**enc, use_cache=False)

    final_logits = out.logits[:, -1, :]  # [B, vocab]

    # Get display letters (the letters shown in prompts)
    display_letters = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    
    # Get token IDs for display letters
    letter_token_ids = {
        display_letter: get_letter_token_ids(tokenizer, display_letter) 
        for display_letter in display_letters
    }
    
    # Aggregate logits for each display letter, then map back to A-D order
    logits_dict = {}
    for display_letter in display_letters:
        token_ids = letter_token_ids[display_letter]
        if not token_ids:
            raise ValueError(f"Could not find token IDs for display letter '{display_letter}'. "
                           f"This is required for MCQ token aggregation.")
        letter_logits = final_logits[:, token_ids]  # [B, num_variants]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)  # [B]
        
        # Map back to internal A-D letter
        internal_letter = reverse_mapping[display_letter]
        logits_dict[internal_letter] = aggregated_logit
    
    # Stack in A-D order (for consistent loss calculation)
    logits4_list = [logits_dict[chr(ord('A') + i)] for i in range(4)]
    logits4 = torch.stack(logits4_list, dim=-1)  # [B, 4]
    probs4 = torch.softmax(logits4, dim=-1)
    entropy = -(probs4 * torch.log(probs4 + 1e-12)).sum(dim=-1)

    # Argmax prediction - get position index (0-3) and display letter
    idx = logits4.argmax(dim=-1).tolist()
    pred_positions = idx  # [B] list of 0-3 indices
    pred_letters = [display_letters[i] for i in idx]  # [B] display letters

    return {
        "pred_letters": pred_letters,
        "pred_positions": pred_positions,
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
    confidence_letter_mapping=None,
):
    """
    Confidence pass with token aggregation for confidence bins.
    
    Args:
        model: Model to run forward pass on
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device to run on
        temperature: Temperature for sampling (unused, kept for API consistency)
        requires_grad: Whether to compute gradients
        confidence_letter_mapping: Optional dict mapping A-H (internal) to display letters.
                                  If None, uses A-H. The function extracts logits for display
                                  letters and maps them back to A-H space internally.
    
    Returns:
        dict with:
            - logits8: [B, 8] logits in A-H order (for loss calculation)
            - probs8: [B, 8] probabilities in A-H order
            - expected_conf: [B] expected confidence values
            - pred_bins: [B] predicted bin letters in display letter space (e.g., S-Z if using S-Z scheme)
            - pred_bin_indices: [B] predicted bin indices (0-7, corresponding to A-H internally)
    """
    if confidence_letter_mapping is None:
        confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    
    # Create reverse mapping: display_letter -> internal_letter (A-H)
    reverse_mapping = {v: k for k, v in confidence_letter_mapping.items()}
    
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with torch.no_grad():
            out = model(**enc, use_cache=False)

    final_logits = out.logits[:, -1, :]  # [B, vocab]

    # Get token IDs for display letters (the letters shown in prompts)
    display_letters = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
    bin_token_ids = {
        display_letter: get_letter_token_ids(tokenizer, display_letter) 
        for display_letter in display_letters
    }
    
    # Aggregate logits for each display letter, then map back to A-H order
    logits_dict = {}
    for display_letter in display_letters:
        token_ids = bin_token_ids[display_letter]
        if not token_ids:
            raise ValueError(f"Could not find token IDs for display letter '{display_letter}'. "
                           f"This is required for confidence bin token aggregation.")
        letter_logits = final_logits[:, token_ids]  # [B, num_variants]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)  # [B]
        
        # Map back to internal A-H letter
        internal_letter = reverse_mapping[display_letter]
        logits_dict[internal_letter] = aggregated_logit
    
    # Stack in A-H order (for consistent loss calculation)
    logits8_list = [logits_dict[chr(ord('A') + i)] for i in range(8)]
    logits8 = torch.stack(logits8_list, dim=-1)  # [B, 8]
    probs8 = torch.softmax(logits8, dim=-1)

    # Expected confidence (midpoints of bins) - always in A-H order
    mids = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95], dtype=torch.float32, device=device)
    expected_conf = (probs8 * mids).sum(dim=-1)

    # Argmax prediction - get bin index (0-7 for A-H) and display letter
    idx = logits8.argmax(dim=-1).tolist()
    pred_bin_indices = idx  # [B] list of 0-7 indices (A-H internally)
    pred_bins = [display_letters[i] for i in idx]  # [B] display letters

    return {
        "logits8": logits8,
        "probs8": probs8,
        "expected_conf": expected_conf,
        "pred_bins": pred_bins,
        "pred_bin_indices": pred_bin_indices,
    }



