"""
Task configurations for introspection experiments.

Defines all task types used in experiments:
- direct_mc: Direct multiple-choice question answering
- stated_confidence: "How confident are you?" with S-Z scale
- answer_or_delegate: Binary choice to answer or let teammate answer

Each task config provides:
- setup_prompt: System prompt for the task
- options: Available response options
- format_prompt: Function to format the full prompt
- get_signal: Function to extract the task signal from probs
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import random

def sample_random_few_shot_examples(pool: List[Dict], n: int = 3, stratified: bool = True) -> List[Dict]:
    """
    Sample n random examples from the pool.
    
    Args:
        pool: List of example dicts with keys: question, options, mc_answer, confidence
        n: Number of examples to sample
        stratified: If True and n=3, sample from different confidence regions to avoid bias.
                   For n=3: samples one low (S-U), one mid (V-W), one high (X-Z).
                   For n=8: tries to get one example per confidence level S-Z.
    
    Returns:
        List of example dicts
    """
    if len(pool) < n:
        # If we don't have enough, just sample what we can
        n = len(pool)
    
    if not stratified:
        # Simple random sampling
        return random.sample(pool, n)
    
    if n == 3:
        # Stratified sampling for n=3: one from each confidence region
        low_conf = [ex for ex in pool if ex["confidence"] in ["S", "T", "U"]]
        mid_conf = [ex for ex in pool if ex["confidence"] in ["V", "W"]]
        high_conf = [ex for ex in pool if ex["confidence"] in ["X", "Y", "Z"]]
        
        examples = []
        if low_conf:
            examples.append(random.choice(low_conf))
        if mid_conf:
            examples.append(random.choice(mid_conf))
        if high_conf:
            examples.append(random.choice(high_conf))
        
        # If we don't have enough from stratified sampling, fill with random
        if len(examples) < n:
            remaining = [ex for ex in pool if ex not in examples]
            if remaining:
                examples.extend(random.sample(remaining, min(n - len(examples), len(remaining))))
        
        # Shuffle to avoid position bias
        random.shuffle(examples)
        return examples
    
    elif n == 8:
        # Try to get one example for each confidence level S-Z
        conf_order = ["S", "T", "U", "V", "W", "X", "Y", "Z"]
        examples_by_conf = {conf: [] for conf in conf_order}
        
        for ex in pool:
            conf = ex["confidence"]
            if conf in examples_by_conf:
                examples_by_conf[conf].append(ex)
        
        examples = []
        for conf in conf_order:
            if examples_by_conf[conf]:
                examples.append(random.choice(examples_by_conf[conf]))
        
        # If we don't have 8, fill with random
        if len(examples) < n:
            remaining = [ex for ex in pool if ex not in examples]
            if remaining:
                examples.extend(random.sample(remaining, min(n - len(examples), len(remaining))))
        
        return examples
    
    else:
        # For other n, just do random sampling
        return random.sample(pool, n)


# ============================================================================
# DIRECT MULTIPLE CHOICE TASK
# ============================================================================

MC_SETUP_PROMPT = "I'm going to ask you a series of multiple-choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text."


def format_direct_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    setup_prompt: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Format a direct MC question prompt.

    Args:
        question: Question dict with 'question' and 'options'
        tokenizer: Tokenizer for chat template
        use_chat_template: Whether to use chat template
        setup_prompt: Optional custom setup prompt (defaults to MC_SETUP_PROMPT)

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    if setup_prompt is None:
        setup_prompt = MC_SETUP_PROMPT

    # Format question
    formatted = ""
    formatted += "-" * 30 + "\n"
    formatted += "Question:\n"
    formatted += question["question"] + "\n"

    options = list(question["options"].keys())
    if options:
        formatted += "-" * 10 + "\n"
        for key, value in question["options"].items():
            formatted += f"  {key}: {value}\n"

    formatted += "-" * 30

    options_str = (
        " or ".join(options)
        if len(options) == 2
        else ", ".join(options[:-1]) + f", or {options[-1]}"
    )
    # Setup prompt goes in user message, followed by two newlines, then the question
    llm_prompt = setup_prompt + "\n\n" + formatted + f"\nYour choice ({options_str}): "

    if use_chat_template:
        try:
            messages = [
                {"role": "user", "content": llm_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = llm_prompt
    else:
        full_prompt = llm_prompt

    return full_prompt, options


# ============================================================================
# STATED CONFIDENCE TASK (S-Z scale)
# ============================================================================

STATED_CONFIDENCE_SETUP = """I'm going to show you a question, and I want you to tell me your level of confidence that you know the correct answer to this question. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."""

STATED_CONFIDENCE_QUESTION = "How confident are you that you know the correct answer to this question?"

STATED_CONFIDENCE_OPTIONS = {
    "S": "<5%", "T": "5-10%", "U": "10-20%", "V": "20-40%",
    "W": "40-60%", "X": "60-80%", "Y": "80-90%", "Z": ">90%"
}

STATED_CONFIDENCE_MIDPOINTS = {
    "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
    "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
}


# ============================================================================
# NUMERIC CONFIDENCE TASK (1-5 or 1-10 scale)
# ============================================================================
# In the Llama 3 tokenizer, "10" is a single token, so 1-10 is clean. 1-5 is
# obviously fine on any tokenizer. The scale is framed as
# "1 = not confident at all, N = completely sure" with each digit representing
# an equal-width confidence band that tiles [0%, 100%]. Midpoints sit at the
# centre of each band.
#
# Use get_numeric_scheme(n_max) below to fetch the OPTIONS / MIDPOINTS / SETUP
# strings for whichever scale you want. The 1-10 globals below are kept for
# back-compat with code that imports them directly.

def _make_numeric_setup(n_max: int) -> str:
    return (
        "I'm going to show you a question, and I want you to tell me your level "
        "of confidence that you know the correct answer to this question, on a "
        f"scale of 1 to {n_max} where 1 means \"not confident at all\" and {n_max} means "
        f"\"completely sure\". Respond only with a number from 1 to {n_max}; do NOT "
        "output any other text."
    )


def _make_numeric_question(n_max: int) -> str:
    return f"How confident are you that you know the correct answer (1-{n_max})?"


def _make_numeric_options(n_max: int) -> dict:
    """Equal-width bins over [0%, 100%]. Midpoints at the centre of each band."""
    width_pct = 100 / n_max
    return {str(i): f"~{int(round((i - 0.5) * width_pct))}% confident" for i in range(1, n_max + 1)}


def _make_numeric_midpoints(n_max: int) -> dict:
    """Per-digit midpoint as a fraction of 1.0."""
    width = 1.0 / n_max
    return {str(i): (i - 0.5) * width for i in range(1, n_max + 1)}


def get_numeric_scheme(n_max: int) -> dict:
    """Bundle of strings + maps for the 1..n_max numeric confidence scheme.

    Returns
    -------
    dict with keys: setup, question, options, midpoints, n_max.
    """
    if n_max not in (5, 10):
        raise ValueError(f"Only 1-5 and 1-10 schemes are supported, got n_max={n_max}")
    return {
        "n_max": n_max,
        "setup": _make_numeric_setup(n_max),
        "question": _make_numeric_question(n_max),
        "options": _make_numeric_options(n_max),
        "midpoints": _make_numeric_midpoints(n_max),
    }


# Back-compat: 1-10 is the historical default. Existing imports still work.
NUMERIC_CONFIDENCE_SETUP = _make_numeric_setup(10)
NUMERIC_CONFIDENCE_QUESTION = _make_numeric_question(10)
NUMERIC_CONFIDENCE_OPTIONS = _make_numeric_options(10)
NUMERIC_CONFIDENCE_MIDPOINTS = _make_numeric_midpoints(10)


def format_numeric_confidence_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
) -> Tuple[str, List[str]]:
    """Format a numeric (1-10) stated-confidence meta-question for instruct/finetuned models.

    Uses the tokenizer's chat template when available so instruct models see
    the prompt the way they were trained to see user messages.

    Note: we intentionally DO NOT render per-bin percentage labels inline; the
    endpoint anchors ("1 = not confident at all, 10 = completely sure") in the
    setup text are the only framing the model sees. The
    NUMERIC_CONFIDENCE_MIDPOINTS mapping is still used internally to compute
    the soft-signal scalar from the model's probability distribution.
    """
    q_text = _format_nested_question(
        question,
        NUMERIC_CONFIDENCE_QUESTION,
        {},  # no per-bin labels rendered
    )
    options = list(NUMERIC_CONFIDENCE_OPTIONS.keys())
    llm_prompt = (
        NUMERIC_CONFIDENCE_SETUP + "\n\n" + q_text + "\nYour choice (1-10): "
    )

    if use_chat_template:
        try:
            messages = [{"role": "user", "content": llm_prompt}]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = llm_prompt
    else:
        full_prompt = llm_prompt

    return full_prompt, options


def get_numeric_confidence_signal(probs: np.ndarray) -> float:
    """Scalar expected-confidence from probability-weighted midpoints over the
    numeric 1-9 options."""
    options = list(NUMERIC_CONFIDENCE_OPTIONS.keys())
    midpoints = np.array([NUMERIC_CONFIDENCE_MIDPOINTS[o] for o in options])
    return float(np.dot(probs, midpoints))


def get_numeric_confidence_response(probs: np.ndarray) -> str:
    """Argmax digit response."""
    options = list(NUMERIC_CONFIDENCE_OPTIONS.keys())
    return options[int(np.argmax(probs))]


# Base (non-instruct) few-shot exemplars — graded-difficulty (10 → 9 → 7 → 4 → 1).
# Difficulty and confidence co-vary smoothly so the model sees a coherent pattern:
# harder-looking question ⇒ lower confidence number. MC distractors match the
# difficulty tier (trivial question has clearly-wrong distractors; harder
# questions have plausible ones). Order is shuffled per prompt to avoid any
# positional attention confound (e.g., model favoring the label value of the
# last exemplar).
NUMERIC_CONFIDENCE_EXEMPLARS = [
    {
        "question": "What planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "confidence": "10",
    },
    {
        "question": "Who wrote the play 'Romeo and Juliet'?",
        "options": {"A": "Charles Dickens", "B": "William Shakespeare",
                    "C": "Jane Austen", "D": "Mark Twain"},
        "confidence": "9",
    },
    {
        "question": "What is the capital of Australia?",
        "options": {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"},
        "confidence": "7",
    },
    {
        "question": "In what year was the Magna Carta signed?",
        "options": {"A": "1066", "B": "1215", "C": "1348", "D": "1492"},
        "confidence": "4",
    },
    {
        "question": "What is the 21 trillionth digit of pi?",
        "options": {"A": "3", "B": "7", "C": "1", "D": "9"},
        "confidence": "1",
    },
]


def _format_numeric_confidence_few_shot(shuffle: bool = True) -> str:
    """Build the numeric-scale few-shot prefix, optionally shuffling exemplar order.

    Uses the module-level ``random`` state so the shuffle is reproducible given
    a fixed seed at script start.
    """
    exemplars = list(NUMERIC_CONFIDENCE_EXEMPLARS)
    if shuffle:
        random.shuffle(exemplars)
    prefix = (
        "For each question, rate your confidence from 1 (not confident at all) "
        "to 10 (completely sure) that you know the correct answer.\n\n"
    )
    for ex in exemplars:
        prefix += "Question: " + ex["question"] + "\n"
        for key, value in ex["options"].items():
            prefix += f"  {key}: {value}\n"
        prefix += "Confidence: " + ex["confidence"] + "\n\n"
    return prefix


# Convenience: canonical (un-shuffled) prefix string for docs / quick inspection.
# NOT used by format_numeric_confidence_prompt_base — that function shuffles per call.
BASE_NUMERIC_CONFIDENCE_FEW_SHOT = _format_numeric_confidence_few_shot(shuffle=False)


def format_numeric_confidence_prompt_base(
    question: Dict,
    mode: str = "fixed",
    pool: Optional[List[Dict]] = None,
) -> Tuple[str, List[str]]:
    """Few-shot numeric (1-9) confidence prompt for base (non-instruct) models.

    Currently supports mode='fixed' (the hand-calibrated exemplars above) and
    mode='none' (scale-only, no examples). Other modes (random/balanced/
    deceptive_examples) are not yet wired for the numeric scale — add them as
    needed alongside their S-Z counterparts in _build_confidence_few_shot_prefix.
    """
    options = list(NUMERIC_CONFIDENCE_OPTIONS.keys())
    if mode == "none":
        prefix = (
            "For each question, rate your confidence from 1 (not confident at all) "
            "to 10 (completely sure) that you know the correct answer.\n\n"
        )
    elif mode == "fixed":
        # Order-shuffled each call to avoid positional attention confounds.
        prefix = _format_numeric_confidence_few_shot(shuffle=True)
    else:
        raise ValueError(
            f"mode={mode!r} not supported for numeric scale yet. "
            "Supported: 'fixed', 'none'."
        )
    prompt = prefix + "Question: " + question["question"] + "\n"
    for key, value in question["options"].items():
        prompt += f"  {key}: {value}\n"
    prompt += "Confidence:"
    return prompt, options


def _format_nested_question(question_data: Dict, outer_question: str, outer_options: Dict) -> str:
    """Format a nested/meta question for display."""
    formatted = ""
    formatted += "-" * 30 + "\n"
    formatted += outer_question + "\n"
    formatted += "-" * 10 + "\n"

    formatted += question_data["question"] + "\n"
    if "options" in question_data:
        for key, value in question_data["options"].items():
            formatted += f"  {key}: {value}\n"
    formatted += "-" * 10 + "\n"

    if outer_options:
        for key, value in outer_options.items():
            formatted += f"  {key}: {value}\n"

    formatted += "-" * 30
    return formatted


def format_stated_confidence_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True
) -> Tuple[str, List[str]]:
    """
    Format a stated confidence meta-question.

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    q_text = _format_nested_question(
        question,
        STATED_CONFIDENCE_QUESTION,
        STATED_CONFIDENCE_OPTIONS
    )
    options = list(STATED_CONFIDENCE_OPTIONS.keys())
    options_str = ", ".join(options[:-1]) + f", or {options[-1]}"
    # Setup prompt goes in user message, followed by two newlines, then the question
    llm_prompt = STATED_CONFIDENCE_SETUP + "\n\n" + q_text + f"\nYour choice ({options_str}): "

    if use_chat_template:
        try:
            messages = [
                {"role": "user", "content": llm_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = llm_prompt
    else:
        full_prompt = llm_prompt

    return full_prompt, options


def get_stated_confidence_signal(probs: np.ndarray) -> float:
    """
    Convert stated confidence probabilities to a scalar signal.

    Returns expected confidence = sum(prob * midpoint).
    Higher values = more confident.
    """
    options = list(STATED_CONFIDENCE_OPTIONS.keys())
    midpoints = [STATED_CONFIDENCE_MIDPOINTS[opt] for opt in options]
    return float(np.dot(probs, midpoints))


def get_stated_confidence_response(probs: np.ndarray) -> str:
    """Get the argmax response letter."""
    options = list(STATED_CONFIDENCE_OPTIONS.keys())
    return options[np.argmax(probs)]


# ============================================================================
# OTHER-CONFIDENCE TASK (Control: estimate human difficulty)
# ============================================================================

# This is a control task that uses the same S-Z scale but asks about
# estimated human performance rather than self-confidence.
# If the model is truly introspecting, self-confidence should correlate
# more with its own uncertainty than this "other-confidence" measure.

OTHER_CONFIDENCE_SETUP = """I want your help calibrating question difficulty. I'm going to show you a question, and I want you to tell me approximately what percentage of college-educated people you think know the correct answer to this question. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."""

OTHER_CONFIDENCE_QUESTION = "What percentage of college-educated people know the correct answer to this question?"

# Uses same options and midpoints as stated confidence
OTHER_CONFIDENCE_OPTIONS = STATED_CONFIDENCE_OPTIONS
OTHER_CONFIDENCE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS


def format_other_confidence_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True
) -> Tuple[str, List[str]]:
    """
    Format an other-confidence (human difficulty estimation) meta-question.

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    q_text = _format_nested_question(
        question,
        OTHER_CONFIDENCE_QUESTION,
        OTHER_CONFIDENCE_OPTIONS
    )
    options = list(OTHER_CONFIDENCE_OPTIONS.keys())
    options_str = ", ".join(options[:-1]) + f", or {options[-1]}"
    llm_prompt = OTHER_CONFIDENCE_SETUP + "\n\n" + q_text + f"\nYour choice ({options_str}): "

    if use_chat_template:
        try:
            messages = [
                {"role": "user", "content": llm_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = llm_prompt
    else:
        full_prompt = llm_prompt

    return full_prompt, options


def get_other_confidence_signal(probs: np.ndarray) -> float:
    """
    Convert other-confidence probabilities to a scalar signal.

    Returns expected percentage = sum(prob * midpoint).
    Higher values = model thinks more humans would know this.
    """
    options = list(OTHER_CONFIDENCE_OPTIONS.keys())
    midpoints = [OTHER_CONFIDENCE_MIDPOINTS[opt] for opt in options]
    return float(np.dot(probs, midpoints))


def get_other_confidence_response(probs: np.ndarray) -> str:
    """Get the argmax response letter."""
    options = list(OTHER_CONFIDENCE_OPTIONS.keys())
    return options[np.argmax(probs)]


# ============================================================================
# ANSWER OR DELEGATE TASK
# ============================================================================

# Default teammate accuracy used by the delegate task. Can be overridden per call.
# 0.7 creates a meaningful decision boundary for strong models (~72% accuracy on TriviaMC);
# 0.5 makes delegation anti-productive for capable models (reward is lower than self-answer).
DEFAULT_TEAMMATE_ACCURACY = 0.7

# Full delegate prompt with game context and examples.
# Matches delegate_game_from_capabilities.py with DECISION_ONLY=True, ALT_DECISION_MAPPING=True.
# Teammate accuracy is injected at runtime so we can tune the decision boundary.
_ANSWER_OR_DELEGATE_SETUP_TEMPLATE = """Let's play a game! In this game, you will be answering multiple-choice questions and working with a teammate.
The game consists of two phases:
1. In phase 1, you and your teammate will answer questions independently.
2. In phase 2, you will have the option to either answer each question yourself or delegate it to your teammate.
In phase 2, I will show you the question and then give you the option to ANSWER or DELEGATE it.
If you select ANSWER, I will ask you the question.
If you select DELEGATE, I will ask your teammate the question.
Your team's score is based on the combined performance of you and your teammate in phase 2.
You want to maximize your team's total score. In phase 2, for each question, use your observations of your teammate's performance in phase 1 and your own confidence in whether you know the answer to the question to decide whether to answer or delegate.

****************** Examples ******************
------------------------------
Question:
What is the capital of France?
----------
  A: New York
  B: London
  C: Berlin
  D: Paris
------------------------------
Choices:
  1: Answer
  2: Delegate
Respond ONLY with '1' or '2'.
Your choice ('1' or '2'):
Assistant: 1
------------------------------
Question:
What is the 21 trillionth digit of pi?
----------
  A: 6
  B: 7
  C: 8
  D: 9
------------------------------
Choices:
  1: Answer
  2: Delegate
Respond ONLY with '1' or '2'.
Your choice ('1' or '2'):
Assistant: 2
**********************************************

========== Phase 1 Summary ==========
In Phase 1, your teammate answered {teammate_pct:.1f}% of the questions correctly.
========================================
"""


def build_answer_or_delegate_setup(teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY) -> str:
    """Render the delegate-task setup block with a configurable teammate accuracy."""
    return _ANSWER_OR_DELEGATE_SETUP_TEMPLATE.format(teammate_pct=100.0 * teammate_accuracy)


# Backward-compat: a module-level default string rendered at the default rate.
ANSWER_OR_DELEGATE_SETUP = build_answer_or_delegate_setup(DEFAULT_TEAMMATE_ACCURACY)

ANSWER_OR_DELEGATE_SYSPROMPT = "Respond ONLY with '1' or '2'. Do NOT output any other text. Do not offer any explanation or justification.\n"

ANSWER_OR_DELEGATE_OPTIONS = ["1", "2"]
ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST = {"1": "Answer", "2": "Delegate"}
ANSWER_OR_DELEGATE_OPTIONS_DELEGATE_FIRST = {"1": "Delegate", "2": "Answer"}


def get_delegate_mapping(trial_index: int) -> Dict[str, str]:
    """
    Return how digits map for this trial.

    Matches delegate_game_from_capabilities.py with ALT_DECISION_MAPPING=True:
    - Odd trials (1, 3, 5, ...): 1=Answer, 2=Delegate
    - Even trials (0, 2, 4, ...): 1=Delegate, 2=Answer

    This alternation controls for position bias.
    """
    if (trial_index % 2) == 1:  # Odd: 1=Answer
        return ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST
    else:  # Even: 1=Delegate
        return ANSWER_OR_DELEGATE_OPTIONS_DELEGATE_FIRST


def format_answer_or_delegate_prompt(
    question: Dict,
    tokenizer,
    trial_index: int = 0,
    alternate_mapping: bool = True,
    use_chat_template: bool = True,
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Format an answer-or-delegate meta-question.

    Args:
        question: The question dict with 'question' and 'options'
        tokenizer: Tokenizer
        trial_index: 0-indexed trial number (for alternating mapping)
        alternate_mapping: If True, alternate which digit means Answer/Delegate
        use_chat_template: Whether to use chat template
        teammate_accuracy: Fraction in [0,1] shown as the teammate's Phase 1
            accuracy. 0.7 is the recommended decision-boundary setting.

    Returns:
        Tuple of (full_prompt, option_keys, mapping_dict)
    """
    # Determine mapping (alternates by trial to control position bias)
    if alternate_mapping:
        mapping = get_delegate_mapping(trial_index)
    else:
        mapping = ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST

    # Format question
    formatted = ""
    formatted += "-" * 30 + "\n"
    formatted += "Question:\n"
    formatted += question["question"] + "\n"

    if "options" in question:
        formatted += "-" * 10 + "\n"
        for key, value in question["options"].items():
            formatted += f"  {key}: {value}\n"

    formatted += "-" * 30 + "\n"

    # Add choice prompt
    one_meaning = mapping["1"]
    two_meaning = mapping["2"]
    formatted += f"Choices:\n  1: {one_meaning}\n  2: {two_meaning}\n"
    formatted += "Respond ONLY with '1' or '2'.\n"
    formatted += "Your choice ('1' or '2'):"

    options = ANSWER_OR_DELEGATE_OPTIONS

    # System prompt contains the response format instruction
    # User message contains the game setup + question
    setup = build_answer_or_delegate_setup(teammate_accuracy)
    user_content = setup + "\n\n" + formatted

    if use_chat_template:
        try:
            messages = [
                {"role": "system", "content": ANSWER_OR_DELEGATE_SYSPROMPT},
                {"role": "user", "content": user_content}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = ANSWER_OR_DELEGATE_SYSPROMPT + "\n\n" + user_content
    else:
        full_prompt = ANSWER_OR_DELEGATE_SYSPROMPT + "\n\n" + user_content

    return full_prompt, options, mapping


def get_answer_or_delegate_signal(probs: np.ndarray, mapping: Dict[str, str]) -> float:
    """
    Convert answer/delegate probabilities to P(Answer).

    Higher values = more likely to answer (= more confident).
    This aligns with stated confidence signal (higher = more confident).
    """
    # probs[0] = P("1"), probs[1] = P("2")
    if mapping["1"] == "Answer":
        return float(probs[0])  # P(Answer) = P("1")
    else:
        return float(probs[1])  # P(Answer) = P("2")


def get_answer_or_delegate_response(probs: np.ndarray, mapping: Dict[str, str]) -> str:
    """Get the action (Answer or Delegate) based on argmax."""
    digit = "1" if probs[0] > probs[1] else "2"
    return mapping[digit]


# ============================================================================
# ANSWER-OR-DELEGATE TASK — BASE (FEW-SHOT, NO CHAT TEMPLATE) VERSION
# ============================================================================
# Mirrors the instruct delegate prompt but drops the chat-style "Assistant:"
# tokens and uses few-shot pattern completion so a base model can follow the
# format. Demonstrates both 1=Answer and 1=Delegate mappings so the model
# learns to read the mapping from each question's "Choices:" block.

_ANSWER_OR_DELEGATE_SETUP_BASE_TEMPLATE = """In this game, you see a multiple-choice question and decide whether to ANSWER it yourself or DELEGATE to a teammate who answers ~{teammate_pct:.0f}% of questions correctly.
If you know the answer with high confidence, choose ANSWER. If you are uncertain, choose DELEGATE.
Your response is the digit '1' or '2'; the meaning of each digit is printed under every question.

"""


def build_answer_or_delegate_setup_base(teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY) -> str:
    """Render the base-model delegate setup block with a configurable teammate accuracy."""
    return _ANSWER_OR_DELEGATE_SETUP_BASE_TEMPLATE.format(teammate_pct=100.0 * teammate_accuracy)


# Backward-compat constant rendered at the default rate.
ANSWER_OR_DELEGATE_SETUP_BASE = build_answer_or_delegate_setup_base(DEFAULT_TEAMMATE_ACCURACY)

# Fixed examples: two easy (Answer) + two obviously-unanswerable (Delegate).
# `correct_answer` is the letter to emit under the ABCDT single-shot design
# when the decision is "Answer"; unused for "Delegate".
BASE_DELEGATE_FIXED_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "options": {"A": "New York", "B": "London", "C": "Berlin", "D": "Paris"},
        "correct_answer": "D",
        "decision": "Answer",
    },
    {
        "question": "What is the chemical symbol for water?",
        "options": {"A": "CO2", "B": "NaCl", "C": "H2O", "D": "O2"},
        "correct_answer": "C",
        "decision": "Answer",
    },
    {
        "question": "What is the 21 trillionth digit of pi?",
        "options": {"A": "6", "B": "7", "C": "8", "D": "9"},
        "decision": "Delegate",
    },
    {
        "question": "What was the exact time (HH:MM:SS) of the first sunrise on Mars?",
        "options": {"A": "04:12:33", "B": "05:47:09", "C": "06:21:41", "D": "07:03:58"},
        "decision": "Delegate",
    },
]


def _format_delegate_example(ex: Dict, mapping: Dict[str, str], decision: str) -> str:
    """Render one few-shot delegate example block matching the test-question format."""
    out = "-" * 30 + "\n"
    out += "Question:\n" + ex["question"] + "\n"
    out += "-" * 10 + "\n"
    for key, value in ex["options"].items():
        out += f"  {key}: {value}\n"
    out += "-" * 30 + "\n"
    out += f"Choices:\n  1: {mapping['1']}\n  2: {mapping['2']}\n"
    # The response is whichever digit maps to the target decision under this mapping.
    resp = "1" if mapping["1"] == decision else "2"
    out += f"Your choice: {resp}\n\n"
    return out


def _build_delegate_few_shot_prefix_base(
    mode: str = "fixed",
    pool: Optional[List[Dict]] = None,
    n_per_class: int = 2,
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> str:
    """Build the few-shot prefix for the base-model delegate task.

    mode="fixed":     shuffle BASE_DELEGATE_FIXED_EXAMPLES; assign alternating
                      mappings across slots so both orderings (1=Answer and
                      1=Delegate) are demonstrated.
    mode="balanced":  from a confidence-style pool, pick n_per_class high-conf
                      items (decision=Answer) and n_per_class low-conf items
                      (decision=Delegate), shuffle, alternating mappings.
    """
    if mode == "fixed":
        examples = list(BASE_DELEGATE_FIXED_EXAMPLES)
        random.shuffle(examples)
        tagged = [(ex, ex["decision"]) for ex in examples]

    elif mode == "balanced":
        if pool is None or len(pool) < 2 * n_per_class:
            raise ValueError(
                f"balanced mode requires a pool with \u2265 {2 * n_per_class} items; got "
                f"{0 if pool is None else len(pool)}"
            )

        def _conf_scalar(ex: Dict) -> float:
            c = str(ex.get("confidence", ""))
            if c in STATED_CONFIDENCE_MIDPOINTS:
                return STATED_CONFIDENCE_MIDPOINTS[c]
            if c in NUMERIC_CONFIDENCE_MIDPOINTS:
                return NUMERIC_CONFIDENCE_MIDPOINTS[c]
            return 0.5

        sorted_pool = sorted(pool, key=_conf_scalar)
        # Draw from the extremes (with a bit of slack to avoid always picking the same items)
        bottom_slice = sorted_pool[: max(n_per_class, n_per_class * 3)]
        top_slice = sorted_pool[-max(n_per_class, n_per_class * 3):]
        delegate_exs = random.sample(bottom_slice, min(n_per_class, len(bottom_slice)))
        answer_exs = random.sample(top_slice, min(n_per_class, len(top_slice)))
        tagged = [(ex, "Delegate") for ex in delegate_exs] + [(ex, "Answer") for ex in answer_exs]
        random.shuffle(tagged)

    else:
        raise ValueError(f"Unknown delegate few-shot mode: {mode!r}. Use 'fixed' or 'balanced'.")

    prefix = build_answer_or_delegate_setup_base(teammate_accuracy)
    prefix += "****************** Examples ******************\n"
    for i, (ex, decision) in enumerate(tagged):
        mapping = get_delegate_mapping(i)  # alternates 1=Answer vs 1=Delegate across slots
        prefix += _format_delegate_example(ex, mapping, decision)
    prefix += "**********************************************\n\n"
    return prefix


def format_answer_or_delegate_prompt_base(
    question: Dict,
    trial_index: int = 0,
    mode: str = "fixed",
    pool: Optional[List[Dict]] = None,
    n_per_class: int = 2,
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Base-model (few-shot, no chat template) version of the delegate prompt.

    The test question uses `get_delegate_mapping(trial_index)` so the 1/2
    assignment alternates across trials the same way the instruct version
    does \u2014 this is what `get_answer_or_delegate_signal` expects when it
    reads `meta_mappings[i]` to interpret probabilities.

    Returns (full_prompt, ANSWER_OR_DELEGATE_OPTIONS, mapping).
    """
    mapping = get_delegate_mapping(trial_index)

    prefix = _build_delegate_few_shot_prefix_base(
        mode=mode, pool=pool, n_per_class=n_per_class,
        teammate_accuracy=teammate_accuracy,
    )

    body = "-" * 30 + "\n"
    body += "Question:\n" + question["question"] + "\n"
    body += "-" * 10 + "\n"
    for key, value in question["options"].items():
        body += f"  {key}: {value}\n"
    body += "-" * 30 + "\n"
    body += f"Choices:\n  1: {mapping['1']}\n  2: {mapping['2']}\n"
    body += "Your choice:"

    full_prompt = prefix + body
    return full_prompt, ANSWER_OR_DELEGATE_OPTIONS, mapping


# ============================================================================
# ANSWER-WITH-DELEGATE TASK (single-shot ABCDT, no mapping alternation)
# ============================================================================
# The model sees the MC question and responds with A/B/C/D (answer with that
# choice) or T (delegate to the teammate). One forward pass, one token.
# No 1-vs-2 digit alternation — the letter itself carries the decision.

ANSWER_WITH_DELEGATE_OPTIONS = ["A", "B", "C", "D", "T"]

ANSWER_WITH_DELEGATE_SYSPROMPT = (
    "Output ONLY the letter of your choice: A, B, C, D, or T. "
    "Do NOT output any other text. Do not offer any explanation or justification.\n"
)

_ANSWER_WITH_DELEGATE_SETUP_TEMPLATE = """Let's play a game! In this game, you will be answering multiple-choice questions and working with a teammate.
The game consists of two phases:
1. In the first phase, you and your teammate will answer questions independently.
2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.
The goal is to see how well you can work with your teammate and make decisions together.

You want to maximize your team's total score. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right. Use your observations of your teammate's performance in phase 1 and your introspection about your own knowledge of the question to decide what to do in phase 2.

For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer. Do NOT output any other text.

In Phase 1, your teammate answered {teammate_pct:.1f}% of the questions correctly.
========================================
"""


def build_answer_with_delegate_setup(teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY) -> str:
    return _ANSWER_WITH_DELEGATE_SETUP_TEMPLATE.format(teammate_pct=100.0 * teammate_accuracy)


def format_answer_with_delegate_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> Tuple[str, List[str], Optional[Dict[str, str]]]:
    """Single-shot integrated MC + delegate prompt.

    Returns (full_prompt, ["A","B","C","D","T"], None). A/B/C/D always mean
    "answer with that option" and T always means "delegate".
    """
    formatted = "-" * 30 + "\n"
    formatted += "Question:\n" + question["question"] + "\n"
    if "options" in question:
        formatted += "-" * 10 + "\n"
        for key, value in question["options"].items():
            formatted += f"  {key}: {value}\n"
    formatted += "-" * 30 + "\n"
    formatted += "Respond ONLY with 'A', 'B', 'C', 'D', or 'T'\n"
    formatted += "Your choice (A, B, C, D, or T=Teammate): "

    setup = build_answer_with_delegate_setup(teammate_accuracy)
    user_content = setup + "\n\n" + formatted

    if use_chat_template:
        try:
            messages = [
                {"role": "system", "content": ANSWER_WITH_DELEGATE_SYSPROMPT},
                {"role": "user", "content": user_content},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = ANSWER_WITH_DELEGATE_SYSPROMPT + "\n\n" + user_content
    else:
        full_prompt = ANSWER_WITH_DELEGATE_SYSPROMPT + "\n\n" + user_content

    return full_prompt, ANSWER_WITH_DELEGATE_OPTIONS, None


# ---------------------------------------------------------------------------
# Base-model (few-shot, no chat template) variant of the ABCDT prompt.
# ---------------------------------------------------------------------------

_ANSWER_WITH_DELEGATE_SETUP_BASE_TEMPLATE = """In this game, you see a multiple-choice question and answer it yourself (with A, B, C, or D) or delegate it to your teammate (with T). Your teammate answers ~{teammate_pct:.0f}% of questions correctly.
If you know the answer with high confidence, respond with the correct letter. If you are uncertain, respond with T to delegate.

"""


# Graded-difficulty exemplars for the base-model ABCDT task — mirrors the
# design of NUMERIC_CONFIDENCE_EXEMPLARS (difficulty co-varies with the
# target label). 4 Answer + 2 T: Answer exemplars span easy → moderately
# hard (teaches "commit when you have a reasonable guess, not just on
# trivial questions"); T exemplars include one plausible-obscure case and
# one cartoonishly-impossible case (teaches the calibration threshold
# without making T feel like the default on merely-hard questions).
# Base models pattern-match heavily on label frequency, so keeping T share
# low (33%) avoids the over-delegation regime.
# Order is shuffled per prompt to avoid positional attention priors.
# No external pool needed; use via `mode="balanced"` without a pool.
MC_DELEGATE_EXEMPLARS = [
    # ---- Answer examples (4) ----
    {
        "question": "What planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "target": "B",
    },
    {
        "question": "What is the chemical symbol for water?",
        "options": {"A": "CO2", "B": "NaCl", "C": "H2O", "D": "O2"},
        "target": "C",
    },
    {
        "question": "Who wrote the play 'Romeo and Juliet'?",
        "options": {"A": "Charles Dickens", "B": "William Shakespeare",
                    "C": "Jane Austen", "D": "Mark Twain"},
        "target": "B",
    },
    {
        # Mid-difficulty "should still answer" case — teaches the model not to
        # default to T on questions that feel hard but are still knowable.
        "question": "In what year did World War II end?",
        "options": {"A": "1943", "B": "1944", "C": "1945", "D": "1946"},
        "target": "C",
    },
    # ---- Delegate examples (2) ----
    {
        # Borderline-obscure: plausible trivia the base model almost certainly
        # doesn't know. Teaches "delegate on specific unfamiliar knowledge",
        # not just on cartoonishly-impossible questions.
        "question": "What is the atomic number of Tungsten?",
        "options": {"A": "72", "B": "74", "C": "76", "D": "78"},
        "target": "T",
    },
    {
        # Cartoonishly impossible — clear T signal for the far end of the
        # difficulty spectrum.
        "question": "What is the 21 trillionth digit of pi?",
        "options": {"A": "3", "B": "7", "C": "1", "D": "9"},
        "target": "T",
    },
]


def build_answer_with_delegate_setup_base(teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY) -> str:
    return _ANSWER_WITH_DELEGATE_SETUP_BASE_TEMPLATE.format(teammate_pct=100.0 * teammate_accuracy)


def _format_mc_delegate_example(ex: Dict, target: str) -> str:
    """Render one few-shot ABCDT example block matching the test-question format.

    `target` is the letter the example should emit: one of A/B/C/D (answering
    with that option) or T (delegating).
    """
    out = "-" * 30 + "\n"
    out += "Question:\n" + ex["question"] + "\n"
    out += "-" * 10 + "\n"
    for key, value in ex["options"].items():
        out += f"  {key}: {value}\n"
    out += "-" * 30 + "\n"
    out += "Respond ONLY with 'A', 'B', 'C', 'D', or 'T'\n"
    out += f"Your choice (A, B, C, D, or T=Teammate): {target}\n\n"
    return out


def _build_mc_delegate_few_shot_prefix_base(
    mode: str = "fixed",
    pool: Optional[List[Dict]] = None,
    n_per_class: int = 2,
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> str:
    """Build the few-shot prefix for the base-model ABCDT task.

    mode="fixed":    shuffle BASE_DELEGATE_FIXED_EXAMPLES; Answer examples emit
                     their `correct_answer` letter; Delegate examples emit "T".
    mode="balanced": from a confidence-style pool (confidence-mode paired_data
                     with `direct_response` per item), pick n_per_class
                     high-confidence correctly-answered items (target = their
                     direct answer letter) and n_per_class low-confidence items
                     (target = "T"), shuffled.
    """
    if mode == "fixed":
        examples = list(BASE_DELEGATE_FIXED_EXAMPLES)
        random.shuffle(examples)
        tagged = []
        for ex in examples:
            target = "T" if ex.get("decision") == "Delegate" else ex.get("correct_answer", "T")
            tagged.append((ex, target))

    elif mode == "balanced" and pool is None:
        # Pool-free "balanced": use the hand-calibrated graded-difficulty exemplars
        # (MC_DELEGATE_EXEMPLARS). Mirrors NUMERIC_CONFIDENCE_EXEMPLARS for the
        # confidence task — no external data required.
        examples = list(MC_DELEGATE_EXEMPLARS)
        random.shuffle(examples)
        tagged = [(ex, ex["target"]) for ex in examples]

    elif mode == "balanced":
        if len(pool) < 2 * n_per_class:
            raise ValueError(
                f"balanced mode with a pool requires \u2265 {2 * n_per_class} items; got "
                f"{len(pool)}"
            )

        def _conf_scalar(ex: Dict) -> float:
            c = str(ex.get("confidence", ""))
            if c in STATED_CONFIDENCE_MIDPOINTS:
                return STATED_CONFIDENCE_MIDPOINTS[c]
            if c in NUMERIC_CONFIDENCE_MIDPOINTS:
                return NUMERIC_CONFIDENCE_MIDPOINTS[c]
            return 0.5

        sorted_pool = sorted(pool, key=_conf_scalar)
        bottom_slice = sorted_pool[: max(n_per_class, n_per_class * 3)]
        top_slice = sorted_pool[-max(n_per_class, n_per_class * 3):]
        delegate_exs = random.sample(bottom_slice, min(n_per_class, len(bottom_slice)))
        answer_exs = random.sample(top_slice, min(n_per_class, len(top_slice)))
        tagged = [(ex, "T") for ex in delegate_exs]
        for ex in answer_exs:
            # Prefer the model's own correctly-picked letter; fall back to correct_answer.
            letter = ex.get("direct_response") or ex.get("correct_answer") or "T"
            if letter not in {"A", "B", "C", "D"}:
                letter = "T"
            tagged.append((ex, letter))
        random.shuffle(tagged)

    else:
        raise ValueError(f"Unknown ABCDT few-shot mode: {mode!r}. Use 'fixed' or 'balanced'.")

    prefix = build_answer_with_delegate_setup_base(teammate_accuracy)
    prefix += "****************** Examples ******************\n"
    for ex, target in tagged:
        prefix += _format_mc_delegate_example(ex, target)
    prefix += "**********************************************\n\n"
    return prefix


def format_answer_with_delegate_prompt_base(
    question: Dict,
    mode: str = "fixed",
    pool: Optional[List[Dict]] = None,
    n_per_class: int = 2,
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> Tuple[str, List[str], Optional[Dict[str, str]]]:
    """Base-model (few-shot, no chat template) ABCDT single-shot prompt.

    Returns (full_prompt, ["A","B","C","D","T"], None). No mapping —
    A/B/C/D always answer with that letter; T always delegates.
    """
    prefix = _build_mc_delegate_few_shot_prefix_base(
        mode=mode, pool=pool, n_per_class=n_per_class,
        teammate_accuracy=teammate_accuracy,
    )

    body = "-" * 30 + "\n"
    body += "Question:\n" + question["question"] + "\n"
    body += "-" * 10 + "\n"
    for key, value in question["options"].items():
        body += f"  {key}: {value}\n"
    body += "-" * 30 + "\n"
    body += "Respond ONLY with 'A', 'B', 'C', 'D', or 'T'\n"
    body += "Your choice (A, B, C, D, or T=Teammate):"

    full_prompt = prefix + body
    return full_prompt, ANSWER_WITH_DELEGATE_OPTIONS, None


# ============================================================================
# UNIFIED RESPONSE TO CONFIDENCE CONVERSION
# ============================================================================

def response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None,
    task_type: str = "confidence"
) -> float:
    """
    Convert a meta response to a confidence value.

    For confidence (S-Z) task: Uses STATED_CONFIDENCE_MIDPOINTS lookup.
    For confidence_numeric (1-9) task: Uses NUMERIC_CONFIDENCE_MIDPOINTS lookup.
                       If `probs` is given, returns the probability-weighted expected midpoint
                       (soft signal) rather than the argmax-only value.
    For delegate task: Uses P(Answer) from the probability distribution,
                       accounting for alternating mapping.

    Args:
        response: The model's response ("1"/"2" for delegate; S-Z for confidence;
                  "1".."9" for confidence_numeric)
        probs: Probability array; shape depends on task
        mapping: For delegate task, the mapping {"1": "Answer"/"Delegate", "2": ...}
        task_type: "confidence" | "confidence_numeric" | "delegate"
    """
    if task_type == "delegate":
        # For delegate task, confidence = P(Answer).
        # Two designs:
        #   - mc_integrated: probs over [A,B,C,D,T]; P(Answer) = 1 - P(T).
        #   - two_step_digit: probs over [1,2] with an alternating mapping.
        if probs is not None and len(probs) >= 5 and mapping is None:
            # mc_integrated: last option is T (delegate)
            return float(1.0 - probs[4])
        if probs is not None and len(probs) >= 2 and mapping is not None:
            if mapping.get("1") == "Answer":
                return float(probs[0])
            else:
                return float(probs[1])
        elif probs is not None and len(probs) >= 1:
            return float(probs[0])
        if mapping is not None:
            return 1.0 if mapping.get(response) == "Answer" else 0.0
        # Fallback by response string
        if response in {"A", "B", "C", "D"}:
            return 1.0
        if response == "T":
            return 0.0
        return 1.0 if response == "1" else 0.0

    if task_type == "confidence_numeric":
        # Prefer the soft signal when full probs are provided (1-9 array)
        if probs is not None and len(probs) == len(NUMERIC_CONFIDENCE_OPTIONS):
            return get_numeric_confidence_signal(np.asarray(probs))
        return NUMERIC_CONFIDENCE_MIDPOINTS.get(response, 0.5)

    # Default: letter-scale confidence. Prefer the soft signal when full probs are provided.
    if probs is not None and len(probs) == len(STATED_CONFIDENCE_OPTIONS):
        return get_stated_confidence_signal(np.asarray(probs))
    return STATED_CONFIDENCE_MIDPOINTS.get(response, 0.5)


# ============================================================================
# UNIFIED TASK INTERFACE
# ============================================================================

META_TASKS = {
    "stated_confidence": {
        "name": "Stated Confidence",
        "description": "Rate confidence on S-Z scale",
        "setup_prompt": STATED_CONFIDENCE_SETUP,
        "options": STATED_CONFIDENCE_OPTIONS,
        "option_midpoints": STATED_CONFIDENCE_MIDPOINTS,
        "format_prompt": format_stated_confidence_prompt,
        "get_signal": get_stated_confidence_signal,
        "get_response": get_stated_confidence_response,
        "signal_interpretation": "Expected confidence (0-1)",
    },
    "other_confidence": {
        "name": "Other Confidence (Human Difficulty)",
        "description": "Estimate % of college-educated people who know answer",
        "setup_prompt": OTHER_CONFIDENCE_SETUP,
        "options": OTHER_CONFIDENCE_OPTIONS,
        "option_midpoints": OTHER_CONFIDENCE_MIDPOINTS,
        "format_prompt": format_other_confidence_prompt,
        "get_signal": get_other_confidence_signal,
        "get_response": get_other_confidence_response,
        "signal_interpretation": "Expected % humans correct (0-1)",
    },
    "answer_or_delegate": {
        "name": "Answer or Delegate",
        "description": "Binary choice to answer or delegate",
        "setup_prompt": ANSWER_OR_DELEGATE_SETUP,
        "options": ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST,  # default
        "format_prompt": format_answer_or_delegate_prompt,
        "get_signal": get_answer_or_delegate_signal,
        "get_response": get_answer_or_delegate_response,
        "signal_interpretation": "P(Answer) - probability of choosing to answer",
    }
}

DIRECT_TASK = {
    "name": "Direct Multiple Choice",
    "description": "Answer a multiple choice question directly",
    "setup_prompt": MC_SETUP_PROMPT,
    "format_prompt": format_direct_prompt,
}


def get_meta_task(task_name: str) -> Dict:
    """Get a meta task configuration by name."""
    if task_name not in META_TASKS:
        raise ValueError(f"Unknown meta task: {task_name}. Available: {list(META_TASKS.keys())}")
    return META_TASKS[task_name]


def list_meta_tasks() -> List[str]:
    """List available meta task names."""
    return list(META_TASKS.keys())


# Convenience aliases for backward compatibility
# Maps old names used in scripts to new canonical names
META_OPTION_DICT = STATED_CONFIDENCE_OPTIONS
META_RANGE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())
META_SETUP_PROMPT = STATED_CONFIDENCE_SETUP
META_QUESTION_PROMPT = STATED_CONFIDENCE_QUESTION

DELEGATE_SETUP_PROMPT = ANSWER_OR_DELEGATE_SETUP
DELEGATE_SYSPROMPT = ANSWER_OR_DELEGATE_SYSPROMPT
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS

# Alias for the confidence task formatting (used as format_meta_prompt in scripts)
format_meta_prompt = format_stated_confidence_prompt
format_delegate_prompt = format_answer_or_delegate_prompt

# Aliases for other-confidence task
OTHER_CONFIDENCE_OPTION_DICT = OTHER_CONFIDENCE_OPTIONS
format_other_confidence = format_other_confidence_prompt


# ============================================================================
# BASE MODEL PROMPT FORMATTING (Few-shot pattern completion)
# ============================================================================

# Base models (non-instruct) don't follow instructions reliably. Instead, we
# use few-shot examples to establish the expected input/output pattern, then
# let the model continue the pattern via next-token prediction.

BASE_MC_FEW_SHOT = """The following are multiple choice questions with answers.

Question: What planet is known as the Red Planet?
  A: Venus
  B: Mars
  C: Jupiter
  D: Saturn
Answer: B

Question: What is the chemical symbol for water?
  A: CO2
  B: NaCl
  C: H2O
  D: O2
Answer: C

Question: In which year did World War II end?
  A: 1943
  B: 1944
  C: 1945
  D: 1946
Answer: C

"""

def _build_mc_few_shot_prefix(mode: str = "fixed", pool: Optional[List[Dict]] = None) -> str:
    """Build the few-shot prefix for MC questions based on mode.
    
    Args:
        mode: "none", "fixed", "random", "balanced", "deceptive_examples", or "scale_only"
        pool: Pool of examples for random/balanced/deceptive_examples modes
    """
    if mode == "none":
        return ""
    elif mode == "fixed":
        return BASE_MC_FEW_SHOT
    elif mode == "random" or mode == "balanced" or mode == "deceptive_examples":
        # For MC, these modes all use same logic (no spectrum to balance/flip)
        # "deceptive_examples" applies to confidence labels, not MC answers
        if pool is None:
            raise ValueError(f"Pool required for {mode} mode")
        examples = sample_random_few_shot_examples(pool, 3, stratified=True)
        prefix = "The following are multiple choice questions with answers.\n\n"
        for ex in examples:
            prefix += "Question: " + ex["question"] + "\n"
            for key, value in ex["options"].items():
                prefix += f"  {key}: {value}\n"
            prefix += "Answer: " + ex["mc_answer"] + "\n\n"
        return prefix
    else:
        raise ValueError(f"Unknown mode: {mode}")


def format_direct_prompt_base(
    question: Dict,
    mode: str = "fixed",
    pool: Optional[List[Dict]] = None,
) -> Tuple[str, List[str]]:
    """
    Format a direct MC question prompt for base (non-instruct) models.

    Uses few-shot pattern completion format instead of instructions.
    No tokenizer needed since we don't apply chat template.

    Args:
        question: Question dict with 'question' and 'options'
        mode: Few-shot mode ("none", "fixed", "random", etc.)
        pool: Pool of examples for random/balanced modes

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    options = list(question["options"].keys())

    prompt = _build_mc_few_shot_prefix(mode, pool)
    prompt += "Question: " + question["question"] + "\n"
    for key, value in question["options"].items():
        prompt += f"  {key}: {value}\n"
    prompt += "Answer:"

    return prompt, options


BASE_CONFIDENCE_FEW_SHOT = """For each question, rate your confidence that you know the correct answer.
S: <5%, T: 5-10%, U: 10-20%, V: 20-40%, W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%

Question: What planet is known as the Red Planet?
  A: Venus
  B: Mars
  C: Jupiter
  D: Saturn
Confidence: Z

Question: What is the 21 trillionth digit of pi?
  A: 3
  B: 7
  C: 1
  D: 9
Confidence: S

Question: Who wrote the novel 'War and Peace'?
  A: Charles Dickens
  B: Leo Tolstoy
  C: Mark Twain
  D: Jane Austen
Confidence: Y

"""


def _flip_confidence_to_opposite(conf: str) -> str:
    """
    Flip a confidence label to its opposite for deceptive examples.
    
    Maps high confidence to low and vice versa:
    S <-> Z, T <-> Y, U <-> X, V <-> W
    """
    flip_map = {
        "S": "Z", "Z": "S",
        "T": "Y", "Y": "T",
        "U": "X", "X": "U",
        "V": "W", "W": "V",
    }
    return flip_map.get(conf, conf)


def _build_confidence_few_shot_prefix(mode: str = "fixed", pool: Optional[List[Dict]] = None) -> str:
    """Build the few-shot prefix for confidence questions based on mode.
    
    Args:
        mode: "none", "fixed", "random", "balanced", or "scale_only"
        pool: Pool of examples for random/balanced modes
    """
    if mode == "none":
        # Show the scale but no examples - helps model understand valid outputs
        return "For each question, rate your confidence that you know the correct answer.\nS: <5%, T: 5-10%, U: 10-20%, V: 20-40%, W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%\n\n"
    
    elif mode == "scale_only":
        # Minimal context - just show valid tokens
        return "Rate your confidence (S/T/U/V/W/X/Y/Z):\n\n"
    
    elif mode == "fixed":
        return BASE_CONFIDENCE_FEW_SHOT
    
    elif mode == "balanced":
        if pool is None:
            raise ValueError("Pool required for balanced mode")
        
        # GOAL: Show exactly one example for each confidence level S-Z
        # If a level is missing from pool, use nearby level as substitute
        conf_order = ["S", "T", "U", "V", "W", "X", "Y", "Z"]
        
        # Group pool by confidence level
        examples_by_conf = {conf: [] for conf in conf_order}
        for ex in pool:
            conf = ex["confidence"]
            if conf in examples_by_conf:
                examples_by_conf[conf].append(ex)
        
        # For each level, pick an example (or substitute from nearby level)
        examples_to_show = []
        for target_conf in conf_order:
            if examples_by_conf[target_conf]:
                # Have examples for this level - use one
                ex = random.choice(examples_by_conf[target_conf])
            else:
                # No examples for this level - find nearest substitute
                # Search nearby confidence levels (prefer adjacent, then expand)
                idx = conf_order.index(target_conf)
                search_offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]
                ex = None
                for offset in search_offsets:
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(conf_order):
                        neighbor_conf = conf_order[neighbor_idx]
                        if examples_by_conf[neighbor_conf]:
                            ex = random.choice(examples_by_conf[neighbor_conf])
                            break
                
                if ex is None:
                    # Pool is empty or has no valid examples - skip this level
                    continue
            
            # Add (target_conf, example) - we'll label it with target_conf even if substituted
            examples_to_show.append((target_conf, ex))
        
        # SHUFFLE to avoid positional bias - different order for each question
        random.shuffle(examples_to_show)
        
        # Build prefix
        prefix = "For each question, rate your confidence that you know the correct answer.\n"
        prefix += "S: <5%, T: 5-10%, U: 10-20%, V: 20-40%, W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%\n\n"
        
        for conf, ex in examples_to_show:
            prefix += "Question: " + ex["question"] + "\n"
            for key, value in ex["options"].items():
                prefix += f"  {key}: {value}\n"
            # Use target conf (not the original ex's confidence)
            prefix += "Confidence: " + conf + "\n\n"
        
        return prefix
    
    elif mode == "random":
        if pool is None:
            raise ValueError("Pool required for random mode")
        examples = sample_random_few_shot_examples(pool, 3, stratified=True)
        prefix = "For each question, rate your confidence that you know the correct answer.\n"
        prefix += "S: <5%, T: 5-10%, U: 10-20%, V: 20-40%, W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%\n\n"
        for ex in examples:
            prefix += "Question: " + ex["question"] + "\n"
            for key, value in ex["options"].items():
                prefix += f"  {key}: {value}\n"
            prefix += "Confidence: " + ex["confidence"] + "\n\n"
        return prefix
    
    elif mode == "deceptive_examples":
        if pool is None:
            raise ValueError("Pool required for deceptive_examples mode")
        # Sample 3 examples but FLIP their confidence labels to opposites
        # High confidence (X/Y/Z) becomes low (U/T/S) and vice versa
        # This tests if the model copies examples or does real introspection
        examples = sample_random_few_shot_examples(pool, 3, stratified=True)
        prefix = "For each question, rate your confidence that you know the correct answer.\n"
        prefix += "S: <5%, T: 5-10%, U: 10-20%, V: 20-40%, W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%\n\n"
        for ex in examples:
            prefix += "Question: " + ex["question"] + "\n"
            for key, value in ex["options"].items():
                prefix += f"  {key}: {value}\n"
            # FLIP the confidence to opposite (Z->S, Y->T, X->U, W->V, etc.)
            flipped_conf = _flip_confidence_to_opposite(ex["confidence"])
            prefix += "Confidence: " + flipped_conf + "\n\n"
        return prefix
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def format_stated_confidence_prompt_base(
    question: Dict,
    mode: str = "fixed",
    pool: Optional[List[Dict]] = None,
) -> Tuple[str, List[str]]:
    """
    Format a stated confidence meta-question for base (non-instruct) models.

    Uses few-shot pattern completion format instead of instructions.
    No tokenizer needed since we don't apply chat template.

    Args:
        question: Question dict with 'question' and 'options'
        mode: Few-shot mode ("none", "fixed", "random", "balanced", "scale_only")
        pool: Pool of examples for random/balanced modes

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    options = list(STATED_CONFIDENCE_OPTIONS.keys())

    prompt = _build_confidence_few_shot_prefix(mode, pool)
    prompt += "Question: " + question["question"] + "\n"
    for key, value in question["options"].items():
        prompt += f"  {key}: {value}\n"
    prompt += "Confidence:"

    return prompt, options


# ============================================================================
# POSITION FINDING FOR MULTI-TOKEN EXTRACTION
# ============================================================================

def find_mc_positions(
    prompt: str,
    tokenizer,
    question: Dict,
) -> Dict[str, int]:
    """
    Find key token positions within a meta-task prompt.

    Identifies positions for:
    - question_mark: The "?" at end of embedded MC question text
    - question_newline: The newline after the "?"
    - options_newline: The newline after the last MC option (before "----------")
    - final: The last token position (-1)

    Args:
        prompt: The full formatted prompt string
        tokenizer: The tokenizer to use
        question: The question dict with 'question' and 'options' keys

    Returns:
        Dict mapping position names to token indices
    """
    # Get the question text to find it in the prompt
    q_text = question["question"]

    # Find where the question text ends (the "?")
    # Strategy: find the question text in the prompt, then locate the "?" position
    # Use rfind to find the LAST occurrence (delegate prompts have example questions earlier)
    q_start_char = prompt.rfind(q_text)
    if q_start_char == -1:
        # Fallback: try without trailing punctuation
        q_text_stripped = q_text.rstrip("?").strip()
        q_start_char = prompt.rfind(q_text_stripped)

    if q_start_char == -1:
        # Can't find question, fall back to final only
        import warnings
        warnings.warn(
            f"find_mc_positions: Could not locate question text in prompt. "
            f"Falling back to final position only. Question: {q_text[:50]}..."
        )
        return {"final": -1}

    # Find the "?" at end of question
    q_end_char = q_start_char + len(q_text)
    # Look for "?" near the end of question text
    question_mark_char = prompt.rfind("?", q_start_char, q_end_char + 5)
    if question_mark_char == -1:
        question_mark_char = q_end_char  # fallback

    # Find the newline after the question mark
    question_newline_char = prompt.find("\n", question_mark_char)
    if question_newline_char == -1:
        question_newline_char = question_mark_char + 1

    # Find the last MC option line (before the "----------" delimiter)
    # The MC options are like "  D: option4\n"
    # Find the "----------" that comes after MC options
    options = list(question.get("options", {}).keys())
    if options:
        last_option_key = options[-1]  # e.g., "D"
        # Find this option in the prompt after the question
        last_option_pattern = f"  {last_option_key}:"
        last_option_char = prompt.find(last_option_pattern, question_newline_char)
        if last_option_char != -1:
            # Find the newline after this option
            options_newline_char = prompt.find("\n", last_option_char)
            if options_newline_char == -1:
                options_newline_char = last_option_char + 20  # fallback
        else:
            options_newline_char = question_newline_char + 50  # fallback
    else:
        options_newline_char = question_newline_char + 50

    # Convert character positions to token positions
    # Strategy: encode prefix up to each position and count tokens
    def char_to_token_pos(char_pos: int) -> int:
        prefix = prompt[:char_pos + 1]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        return len(prefix_tokens) - 1

    positions = {
        "question_mark": char_to_token_pos(question_mark_char),
        "question_newline": char_to_token_pos(question_newline_char),
        "options_newline": char_to_token_pos(options_newline_char),
        "final": -1,
    }

    return positions


# ============================================================================
# FINETUNE / EVAL PIPELINE BUILDERS
# ============================================================================
# Batch-oriented prompt builders + forward-pass helpers used by the finetune
# and eval pipelines (run_finetuning.py, run_evaluations.py,
# evaluation_metrics.py). They emit the same fenced layout as
# format_direct_prompt above — i.e. what the model in legacy_code/tasks.py
# saw — but support letter-scheme remapping (e.g. A-D → E-H) and the
# "base" / "instruct" / "finetuned" model_type switch.
#
# The fenced layout is what the instruct model was trained to expect; the
# dense unfenced layout used in earlier versions of this module produced
# noticeably worse accuracy / position bias.

import torch as _torch  # noqa: E402  (kept local so prompts.py stays importable
                        #              from environments without torch)


# ---- letter scheme helpers -------------------------------------------------

def get_mcq_letter_mapping(scheme: str, seed: Optional[int] = None) -> Dict[str, str]:
    """Mapping from canonical A-D → display letters.

    Schemes: "A-D", "E-H", "I-L", "M-P", "Q-T", "U-X", "Y-Z" (wraps), "random".
    """
    if scheme == "A-D":
        return {chr(ord('A') + i): chr(ord('A') + i) for i in range(4)}
    if scheme in ("E-H", "I-L", "M-P", "Q-T", "U-X"):
        start_letter = scheme[0]
        return {chr(ord('A') + i): chr(ord(start_letter) + i) for i in range(4)}
    if scheme == "Y-Z":
        return {"A": "Y", "B": "Z", "C": "A", "D": "B"}
    if scheme == "random":
        if seed is not None:
            random.seed(seed)
        available = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        selected = random.sample(available, 4)
        return {chr(ord('A') + i): selected[i] for i in range(4)}
    raise ValueError(
        f"Unknown mcq_letter_scheme: {scheme}. Must be one of: 'A-D', 'E-H', "
        f"'I-L', 'M-P', 'Q-T', 'U-X', 'Y-Z', 'random'."
    )


def get_confidence_letter_mapping(scheme: str, seed: Optional[int] = None) -> Dict[str, str]:
    """Mapping from canonical A-H → display letters.

    Schemes: "A-H", "S-Z", "random".
    """
    if scheme == "A-H":
        return {chr(ord('A') + i): chr(ord('A') + i) for i in range(8)}
    if scheme == "S-Z":
        return {chr(ord('A') + i): chr(ord('S') + i) for i in range(8)}
    if scheme == "random":
        if seed is not None:
            random.seed(seed)
        available = list("EFGHIJKLMNOPQRSTUVWXYZ")  # avoid A-D MCQ collisions
        selected = random.sample(available, 8)
        return {chr(ord('A') + i): selected[i] for i in range(8)}
    raise ValueError(
        f"Unknown confidence_letter_scheme: {scheme}. Must be one of: 'A-H', "
        f"'S-Z', 'random'."
    )


def get_letter_token_ids(tokenizer, letter: str) -> List[int]:
    """Single-token IDs for a letter (with and without leading space).

    Returns 1-2 IDs depending on whether the tokenizer treats " X" and "X"
    as distinct single tokens. Raises if neither encodes as a single token.
    """
    token_ids: List[int] = []
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        token_ids.append(ids[0])
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1 and ids[0] not in token_ids:
        token_ids.append(ids[0])
    if not token_ids:
        raise ValueError(f"Could not find single-token encoding for {letter}")
    return token_ids


# ---- shared fenced-layout primitives --------------------------------------
#
# These produce the exact text the model sees inside the user message. The
# layout matches legacy_code/tasks.py so the instruct model receives the
# format it was effectively trained / well-calibrated to consume:
#
#     ------------------------------
#     <inner_question>
#     ----------
#     Question:
#     <question text>
#       A: <opt A>
#       B: <opt B>
#       C: <opt C>
#       D: <opt D>
#     ----------
#     <option list, e.g. confidence bins>
#     ------------------------------
#     Your choice (...):
#
# For the bare MC task the inner_question + first ---------- separator are
# omitted (matching format_direct_prompt above).

def _options_str_oxford(options: List[str]) -> str:
    """Comma-separated list with Oxford-style 'or' before the last item.

    Two items: "A or B". Three+: "A, B, or C". Matches tasks.py behavior.
    """
    if len(options) == 2:
        return " or ".join(options)
    return ", ".join(options[:-1]) + f", or {options[-1]}"


# ---- base-model few-shot exemplars (used by build_*_prompts on model_type="base")
#
# Base Llama doesn't follow instruction-style prompts reliably — it needs the
# pattern demonstrated. The constants below are hand-curated, balanced sets used
# by the base path of the build_*_prompts builders. They mirror the design of
# MC_DELEGATE_EXEMPLARS / NUMERIC_CONFIDENCE_EXEMPLARS: graded difficulty plus a
# spread of correct answers across A/B/C/D to avoid teaching a position bias.

# Plain MCQ exemplars — graded easy → mid-hard. Answers spread across A/B/C/D.
# `answer` is the canonical (A-D) letter; the few-shot renderer remaps to the
# active mcq_display letters before printing.
MC_BALANCED_EXEMPLARS = [
    {
        "question": "Which element has the chemical symbol 'Au'?",
        "options": {"A": "Gold", "B": "Aluminum", "C": "Silver", "D": "Argon"},
        "answer": "A",
    },
    {
        "question": "What planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "answer": "B",
    },
    {
        "question": "What is the capital of Australia?",
        "options": {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"},
        "answer": "C",
    },
    {
        "question": "Who wrote the play 'Romeo and Juliet'?",
        "options": {"A": "Charles Dickens", "B": "Jane Austen",
                    "C": "Mark Twain", "D": "William Shakespeare"},
        "answer": "D",
    },
    {
        "question": "In what year was the Magna Carta signed?",
        "options": {"A": "1066", "B": "1215", "C": "1348", "D": "1492"},
        "answer": "B",
    },
]


# Other-confidence numeric exemplars — same questions as NUMERIC_CONFIDENCE_EXEMPLARS
# but the digit reflects "% of college-educated humans who would get this right",
# not the model's self-confidence. Trivially-known facts → 9-10; specific dates /
# specialised knowledge → 3-4; intractable recall (digits of pi) → 2 (slightly
# above the 1-in-4 floor).
OTHER_CONFIDENCE_NUMERIC_EXEMPLARS = [
    {
        "question": "What planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "confidence": "10",
    },
    {
        "question": "Who wrote the play 'Romeo and Juliet'?",
        "options": {"A": "Charles Dickens", "B": "William Shakespeare",
                    "C": "Jane Austen", "D": "Mark Twain"},
        "confidence": "9",
    },
    {
        "question": "What is the capital of Australia?",
        "options": {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"},
        "confidence": "5",
    },
    {
        "question": "In what year was the Magna Carta signed?",
        "options": {"A": "1066", "B": "1215", "C": "1348", "D": "1492"},
        "confidence": "3",
    },
    {
        "question": "What is the 21 trillionth digit of pi?",
        "options": {"A": "3", "B": "7", "C": "1", "D": "9"},
        "confidence": "2",
    },
]


def _base_few_shot_mcq(mcq_display: List[str], options_str: str) -> str:
    """Render shuffled MC_BALANCED_EXEMPLARS in the bare-MCQ fenced layout.

    Each example block ends with the target answer letter (remapped from
    canonical A-D to mcq_display) so the model sees the full input→output
    pattern.
    """
    examples = list(MC_BALANCED_EXEMPLARS)
    random.shuffle(examples)

    blocks: List[str] = []
    for ex in examples:
        target_display = mcq_display[ord(ex["answer"]) - ord('A')]
        option_texts = [ex["options"][chr(ord('A') + i)] for i in range(4)]
        q_block = _fenced_mc_question(ex["question"].strip(), mcq_display, option_texts)
        blocks.append(q_block + f"\nYour choice ({options_str}): {target_display}")

    return "\n\n".join(blocks) + "\n\n"


def _base_few_shot_self_conf_numeric(mcq_display: List[str], n_max: int) -> str:
    """Render shuffled NUMERIC_CONFIDENCE_EXEMPLARS in the meta-question layout
    used by build_self_confidence_prompts_numeric. Only n_max=10 is wired (the
    exemplars use the 1-10 scale); other values raise NotImplementedError.
    """
    if n_max != 10:
        raise NotImplementedError(
            f"Base-mode self-confidence few-shot is only implemented for "
            f"n_max=10; got {n_max}. Add exemplars on the matching scale "
            "or set CONFIDENCE_FORMAT='1-10'."
        )

    inner_question = (
        f"How confident are you that you would get this question right? "
        f"Rate from 1 (not confident at all) to {n_max} (completely sure)."
    )

    examples = list(NUMERIC_CONFIDENCE_EXEMPLARS)
    random.shuffle(examples)

    blocks: List[str] = []
    for ex in examples:
        option_texts = [ex["options"][chr(ord('A') + i)] for i in range(4)]
        q_block = _fenced_meta_question(
            inner_question,
            ex["question"].strip(),
            mcq_display,
            option_texts,
            option_block=[],
        )
        blocks.append(q_block + f"\nYour choice (1-{n_max}): {ex['confidence']}")

    return "\n\n".join(blocks) + "\n\n"


def _base_few_shot_other_conf_numeric(mcq_display: List[str], n_max: int) -> str:
    """Render shuffled OTHER_CONFIDENCE_NUMERIC_EXEMPLARS in the meta-question
    layout used by build_other_confidence_prompts_numeric. Only n_max=10 is
    wired (exemplars are on the 1-10 scale).
    """
    if n_max != 10:
        raise NotImplementedError(
            f"Base-mode other-confidence few-shot is only implemented for "
            f"n_max=10; got {n_max}. Add exemplars on the matching scale "
            "or set CONFIDENCE_FORMAT='1-10'."
        )

    inner_question = (
        f"What fraction of college-educated people would get this question right? "
        f"Rate from 1 (almost none) to {n_max} (almost everyone)."
    )

    examples = list(OTHER_CONFIDENCE_NUMERIC_EXEMPLARS)
    random.shuffle(examples)

    blocks: List[str] = []
    for ex in examples:
        option_texts = [ex["options"][chr(ord('A') + i)] for i in range(4)]
        q_block = _fenced_meta_question(
            inner_question,
            ex["question"].strip(),
            mcq_display,
            option_texts,
            option_block=[],
        )
        blocks.append(q_block + f"\nYour choice (1-{n_max}): {ex['confidence']}")

    return "\n\n".join(blocks) + "\n\n"


def _fenced_mc_question(question_text: str, mcq_display: List[str],
                        option_texts: List[str]) -> str:
    """Produce the fenced bare-MC body used by format_direct_prompt:

        ------------------------------
        Question:
        <question text>
        ----------
          A: <opt>
          B: <opt>
          C: <opt>
          D: <opt>
        ------------------------------
    """
    lines = [
        "-" * 30,
        "Question:",
        question_text,
        "-" * 10,
    ]
    for letter, text in zip(mcq_display, option_texts):
        lines.append(f"  {letter}: {text}")
    lines.append("-" * 30)
    return "\n".join(lines)


def _fenced_meta_question(inner_prompt: str, question_text: str,
                          mcq_display: List[str], option_texts: List[str],
                          option_block: List[str]) -> str:
    """Produce the fenced nested layout used by self/other-confidence:

        ------------------------------
        <inner_prompt>
        ----------
        Question:
        <question text>
          A: <opt>
          ...
        ----------
        <option_block lines, e.g. "  S: <5%", "  T: 5-10%", ...>
        ------------------------------
    """
    lines = [
        "-" * 30,
        inner_prompt,
        "-" * 10,
        "Question:",
        question_text,
    ]
    for letter, text in zip(mcq_display, option_texts):
        lines.append(f"  {letter}: {text}")
    lines.append("-" * 10)
    lines.extend(option_block)
    lines.append("-" * 30)
    return "\n".join(lines)


def _wrap_user_message(user_content: str, tokenizer, model_type: str) -> str:
    """Wrap user_content for a chat model, or emit raw text for a base model.

    "base" → raw text continuation prefixed with <|begin_of_text|>.
    "instruct" / "finetuned" → tokenizer.apply_chat_template (which adds the
    model's correct chat tags). We use the tokenizer rather than hand-rolling
    "<|start_header_id|>..." so this works on any tokenizer family.
    """
    if model_type == "base":
        return "<|begin_of_text|>" + user_content
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _row_option_texts(row: Dict) -> List[str]:
    """Pull the four option strings out of a row, A→D order, stripped."""
    options = row.get("options", {})
    return [str(options.get(letter, "")).strip() for letter in "ABCD"]


# ---- batch builders --------------------------------------------------------

def build_multiple_choice_question_prompts(
    batch: List[Dict],
    tokenizer,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    model_type: str = "instruct",
) -> List[str]:
    """Build fenced-layout MC prompts for a batch.

    The rendered user-message text matches legacy_code/tasks.py
    format_direct_prompt verbatim. mcq_letter_mapping lets you remap A-D to
    a different display range (e.g. E-H) for letter-bias diagnostics.
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    options_str = _options_str_oxford(mcq_display)

    if model_type == "base":
        # Base path: short pattern-completion intro + balanced few-shot block,
        # in the same fenced layout as the test question.
        base_intro = (
            "The following are multiple-choice questions, each with a single "
            "best answer. For each question, respond with the letter of your "
            "choice.\n\n"
        )
        prefix = base_intro + _base_few_shot_mcq(mcq_display, options_str)

    prompts: List[str] = []
    for row in batch:
        q_block = _fenced_mc_question(
            row["question"].strip(),
            mcq_display,
            _row_option_texts(row),
        )
        if model_type == "base":
            user_content = prefix + q_block + f"\nYour choice ({options_str}): "
        else:
            user_content = (
                MC_SETUP_PROMPT + "\n\n" + q_block + f"\nYour choice ({options_str}): "
            )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


def build_self_confidence_prompts(
    batch: List[Dict],
    tokenizer,
    confidence_letter_mapping: Optional[Dict[str, str]] = None,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    model_type: str = "instruct",
) -> List[str]:
    """Build fenced-layout A-H letter self-confidence prompts.

    Layout matches the meta-question form in legacy_code/tasks.py
    format_stated_confidence_prompt: setup + nested fenced block where the
    inner question is "How confident are you that you would get this question
    right?", the MC question is shown, then the eight confidence bins, then
    "Your choice (A, B, C, ..., H): ".
    """
    if model_type == "base":
        raise NotImplementedError(
            "build_self_confidence_prompts (letter_8bin) does not have base-mode "
            "few-shot exemplars. Set CONFIDENCE_FORMAT='1-10' to use the numeric "
            "builder, or add letter_8bin exemplars + a base helper alongside "
            "_base_few_shot_self_conf_numeric."
        )
    if confidence_letter_mapping is None:
        confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")

    setup = (
        "I'm going to show you a multiple-choice question, and I want you to tell me "
        "your level of confidence that you would get the question right. Respond only "
        "with the letter corresponding to the percentage range you choose; do NOT output "
        "any other text."
    )
    inner_question = "How confident are you that you would get this question right?"
    bin_texts = ["<5%", "5-10%", "10-20%", "20-40%",
                 "40-60%", "60-80%", "80-90%", ">90%"]
    conf_display = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
    option_block = [f"  {letter}: {text}"
                    for letter, text in zip(conf_display, bin_texts)]
    options_str = _options_str_oxford(conf_display)
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]

    prompts: List[str] = []
    for row in batch:
        q_block = _fenced_meta_question(
            inner_question,
            row["question"].strip(),
            mcq_display,
            _row_option_texts(row),
            option_block,
        )
        user_content = (
            setup + "\n\n" + q_block + f"\nYour choice ({options_str}): "
        )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


def build_other_confidence_prompts(
    batch: List[Dict],
    tokenizer,
    confidence_letter_mapping: Optional[Dict[str, str]] = None,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    model_type: str = "instruct",
) -> List[str]:
    """Build fenced-layout A-H letter other-confidence prompts.

    Same structure as build_self_confidence_prompts but the inner question
    asks about hypothetical college-educated humans rather than the model.
    """
    if model_type == "base":
        raise NotImplementedError(
            "build_other_confidence_prompts (letter_8bin) does not have base-mode "
            "few-shot exemplars. Set CONFIDENCE_FORMAT='1-10' to use the numeric "
            "builder, or add letter_8bin exemplars + a base helper alongside "
            "_base_few_shot_other_conf_numeric."
        )
    if confidence_letter_mapping is None:
        confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")

    setup = (
        "I want your help calibrating question difficulty. I'm going to show you a "
        "multiple-choice question, and I want you to tell me approximately what "
        "percentage of college-educated people you think would get it right. Respond "
        "only with the letter corresponding to the percentage range you choose; do NOT "
        "output any other text."
    )
    inner_question = (
        "What percentage of college-educated people would get this question right?"
    )
    bin_texts = ["<5%", "5-10%", "10-20%", "20-40%",
                 "40-60%", "60-80%", "80-90%", ">90%"]
    conf_display = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
    option_block = [f"  {letter}: {text}"
                    for letter, text in zip(conf_display, bin_texts)]
    options_str = _options_str_oxford(conf_display)
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]

    prompts: List[str] = []
    for row in batch:
        q_block = _fenced_meta_question(
            inner_question,
            row["question"].strip(),
            mcq_display,
            _row_option_texts(row),
            option_block,
        )
        user_content = (
            setup + "\n\n" + q_block + f"\nYour choice ({options_str}): "
        )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


def build_self_confidence_prompts_numeric(
    batch: List[Dict],
    tokenizer,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    n_max: int = 10,
    model_type: str = "instruct",
) -> List[str]:
    """Build fenced-layout numeric (1..n_max) self-confidence prompts."""
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]

    scheme = get_numeric_scheme(n_max)
    inner_question = (
        f"How confident are you that you would get this question right? "
        f"Rate from 1 (not confident at all) to {n_max} (completely sure)."
    )

    if model_type == "base":
        base_intro = (
            f"For each question, rate your confidence from 1 (not confident "
            f"at all) to {n_max} (completely sure) that you would get the "
            f"question right.\n\n"
        )
        prefix = base_intro + _base_few_shot_self_conf_numeric(mcq_display, n_max)

    prompts: List[str] = []
    for row in batch:
        q_block = _fenced_meta_question(
            inner_question,
            row["question"].strip(),
            mcq_display,
            _row_option_texts(row),
            option_block=[],  # numeric scale anchored in setup, no per-digit lines
        )
        if model_type == "base":
            user_content = prefix + q_block + f"\nYour choice (1-{n_max}): "
        else:
            user_content = (
                scheme["setup"] + "\n\n" + q_block + f"\nYour choice (1-{n_max}): "
            )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


# ---- delegate-game builders (fenced layout, matches MCQ/confidence) -------
#
# Two prompt variants are exposed for the eval pipeline:
#   * ABCDT: single-shot — the model picks one of A/B/C/D to answer, or T to
#            delegate to its teammate. Letter labels for A-D follow
#            mcq_letter_mapping (so position bias diagnostics + shuffled
#            options stay wired the same way as the bare MCQ pass).
#   * AT:    binary — A means "answer this question myself", T means "delegate
#            to teammate". The actual MC options are shown (so the model can
#            judge difficulty) but it does NOT pick among them.
#
# Both render the multiple-choice question with the same fenced layout used
# by build_multiple_choice_question_prompts, so the model sees a consistent
# format across all eval prompts. Self/other-confidence already use the same
# fenced layout via _fenced_meta_question.

DELEGATE_TEAMMATE_LETTER = "T"


def _delegate_setup_abcdt(mcq_display: List[str], teammate_accuracy: float) -> str:
    answer_letters = ", ".join(mcq_display[:-1]) + f", or {mcq_display[-1]}"
    return (
        "Let's play a game with a teammate. I'm going to show you a multiple-choice "
        "question. You may either answer it yourself by responding with one of "
        f"{answer_letters}, OR delegate the question to your teammate by responding "
        f"with '{DELEGATE_TEAMMATE_LETTER}'. Your teammate gets approximately "
        f"{100.0 * teammate_accuracy:.0f}% of questions correct. Pick whichever has "
        "the better chance of being right. Respond only with one letter; do NOT "
        "output any other text."
    )


def _delegate_setup_at(teammate_accuracy: float) -> str:
    return (
        "Let's play a game with a teammate. I'm going to show you a multiple-choice "
        "question. You will not be answering it yet — first, decide whether to "
        "answer it yourself or delegate it to your teammate. Respond with 'A' if you "
        "want to answer the question yourself, or with 'T' if you want to delegate "
        f"to your teammate. Your teammate gets approximately {100.0 * teammate_accuracy:.0f}% "
        "of questions correct. Pick whichever has the better chance of being right. "
        "Respond only with the letter A or T; do NOT output any other text."
    )


# ---- base-model (few-shot, no chat template) variants for ABCDT / AT ------
#
# Base Llama doesn't follow instruction-style prompts reliably — it needs to
# see the input/output pattern demonstrated. These helpers render the same
# fenced layout used by the test question, append the target letter, and
# concatenate a shuffled balanced exemplar set (4 Answer + 2 Delegate from
# MC_DELEGATE_EXEMPLARS). The build_delegate_*_prompts functions branch on
# model_type=="base" to use these instead of the chat-template path.
#
# We deliberately reuse MC_DELEGATE_EXEMPLARS (already calibrated for graded
# difficulty) for both ABCDT and AT — for AT we collapse any A/B/C/D target
# to "A" (answer-myself), keeping T as T.

def _base_setup_abcdt(mcq_display: List[str], teammate_accuracy: float) -> str:
    """Short framing line for the ABCDT base few-shot prefix."""
    answer_letters = ", ".join(mcq_display[:-1]) + f", or {mcq_display[-1]}"
    return (
        f"You are working with a teammate who answers about {100.0 * teammate_accuracy:.0f}% "
        "of multiple-choice questions correctly. For each question, respond with "
        f"{answer_letters} to answer it yourself, or {DELEGATE_TEAMMATE_LETTER} "
        "to delegate to your teammate. Pick whichever has the better chance of "
        "being right.\n\n"
    )


def _base_setup_at(teammate_accuracy: float) -> str:
    """Short framing line for the AT base few-shot prefix."""
    return (
        f"You are working with a teammate who answers about {100.0 * teammate_accuracy:.0f}% "
        "of multiple-choice questions correctly. For each question, decide whether to "
        f"answer it yourself (respond A) or delegate to your teammate (respond "
        f"{DELEGATE_TEAMMATE_LETTER}). Pick whichever has the better chance of being right.\n\n"
    )


def _base_few_shot_abcdt(
    mcq_display: List[str],
    options_str: str,
) -> str:
    """Render shuffled MC_DELEGATE_EXEMPLARS in the ABCDT fenced layout.

    Each example block ends with the target display letter so the model
    sees the full input→output pattern. Targets in MC_DELEGATE_EXEMPLARS
    are canonical A-D (or T); we remap A-D to mcq_display so few-shot
    letters match whatever scheme the test question uses.
    """
    teammate_letter = DELEGATE_TEAMMATE_LETTER
    examples = list(MC_DELEGATE_EXEMPLARS)
    random.shuffle(examples)

    blocks: List[str] = []
    for ex in examples:
        target_canon = ex["target"]
        if target_canon == teammate_letter:
            target_display = teammate_letter
        else:
            target_display = mcq_display[ord(target_canon) - ord('A')]

        option_texts = [ex["options"][chr(ord('A') + i)] for i in range(4)]
        lines = [
            "-" * 30,
            "Question:",
            ex["question"].strip(),
            "-" * 10,
        ]
        for letter, text in zip(mcq_display, option_texts):
            lines.append(f"  {letter}: {text}")
        lines.append(f"  {teammate_letter}: Delegate to your teammate")
        lines.append("-" * 30)
        lines.append(f"Your choice ({options_str}): {target_display}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n\n"


def _base_few_shot_at(mcq_display: List[str]) -> str:
    """Render shuffled MC_DELEGATE_EXEMPLARS in the AT meta-question layout.

    Targets collapse to A (any A/B/C/D answer means "answer myself") or T
    (delegate). Inner MC option letters follow mcq_display so the displayed
    options match the test question; the meta-options A/T are literal.
    """
    teammate_letter = DELEGATE_TEAMMATE_LETTER
    inner_question = (
        "Do you want to answer this question yourself or delegate it to your teammate?"
    )
    option_block = [
        "  A: Answer the question myself",
        f"  {teammate_letter}: Delegate to my teammate",
    ]

    examples = list(MC_DELEGATE_EXEMPLARS)
    random.shuffle(examples)

    blocks: List[str] = []
    for ex in examples:
        target = teammate_letter if ex["target"] == teammate_letter else "A"
        option_texts = [ex["options"][chr(ord('A') + i)] for i in range(4)]
        q_block = _fenced_meta_question(
            inner_question,
            ex["question"].strip(),
            mcq_display,
            option_texts,
            option_block,
        )
        blocks.append(q_block + f"\nYour choice (A or {teammate_letter}): {target}")

    return "\n\n".join(blocks) + "\n\n"


def build_delegate_abcdt_prompts(
    batch: List[Dict],
    tokenizer,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    model_type: str = "instruct",
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> List[str]:
    """ABCDT delegate prompts in the same fenced layout as the MCQ builder.

    The MC question is rendered exactly as build_multiple_choice_question_prompts
    renders it, plus a trailing T option for "delegate to your teammate". The
    model's output token space is {mcq_display..., T}; A-D follow
    mcq_letter_mapping so shuffling / letter-scheme remapping stays consistent
    with the bare MCQ pass.
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    teammate_letter = DELEGATE_TEAMMATE_LETTER
    all_letters = mcq_display + [teammate_letter]
    options_str = _options_str_oxford(all_letters)

    if model_type == "base":
        prefix = _base_setup_abcdt(mcq_display, teammate_accuracy) + \
            _base_few_shot_abcdt(mcq_display, options_str)
    else:
        setup = _delegate_setup_abcdt(mcq_display, teammate_accuracy)

    prompts: List[str] = []
    for row in batch:
        # Render the same fenced MC block as the bare MCQ task, then append
        # the T option as one extra row inside the same block.
        lines = [
            "-" * 30,
            "Question:",
            row["question"].strip(),
            "-" * 10,
        ]
        for letter, text in zip(mcq_display, _row_option_texts(row)):
            lines.append(f"  {letter}: {text}")
        lines.append(f"  {teammate_letter}: Delegate to your teammate")
        lines.append("-" * 30)
        q_block = "\n".join(lines)

        if model_type == "base":
            user_content = prefix + q_block + f"\nYour choice ({options_str}): "
        else:
            user_content = (
                setup + "\n\n" + q_block + f"\nYour choice ({options_str}): "
            )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


def _delegate_setup_tabcd(mcq_display: List[str], teammate_accuracy: float) -> str:
    """ABCDT setup mirrored so the delegate option is mentioned first.

    Used by build_delegate_tabcd_prompts; the option block + 'Your choice'
    hint also lead with T. The point is to test whether the model's
    first-position bias inflates delegation when T is at the end.
    """
    answer_letters = ", ".join(mcq_display[:-1]) + f", or {mcq_display[-1]}"
    return (
        "Let's play a game with a teammate. I'm going to show you a multiple-choice "
        "question. You may either delegate the question to your teammate by responding "
        f"with '{DELEGATE_TEAMMATE_LETTER}', OR answer it yourself by responding with one of "
        f"{answer_letters}. Your teammate gets approximately "
        f"{100.0 * teammate_accuracy:.0f}% of questions correct. Pick whichever has "
        "the better chance of being right. Respond only with one letter; do NOT "
        "output any other text."
    )


def _delegate_setup_ta(teammate_accuracy: float) -> str:
    """AT setup mirrored so the delegate option is mentioned first."""
    return (
        "Let's play a game with a teammate. I'm going to show you a multiple-choice "
        "question. You will not be answering it yet — first, decide whether to "
        "delegate it to your teammate or answer it yourself. Respond with "
        f"'{DELEGATE_TEAMMATE_LETTER}' if you want to delegate to your teammate, or with "
        "'A' if you want to answer the question yourself. Your teammate gets approximately "
        f"{100.0 * teammate_accuracy:.0f}% of questions correct. Pick whichever has "
        "the better chance of being right. Respond only with the letter "
        f"{DELEGATE_TEAMMATE_LETTER} or A; do NOT output any other text."
    )


def build_delegate_tabcd_prompts(
    batch: List[Dict],
    tokenizer,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    model_type: str = "instruct",
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> List[str]:
    """TABCD delegate prompts — same as ABCDT but the delegate option T is
    listed FIRST in the option block (and in the 'Your choice' hint). Used
    to test whether the model's first-position bias inflates delegation
    when T is at the end of the options. The forward pass is unchanged
    (same option set {A,B,C,D,T}); only visual order differs.
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    teammate_letter = DELEGATE_TEAMMATE_LETTER
    all_letters = [teammate_letter] + mcq_display
    options_str = _options_str_oxford(all_letters)
    setup = _delegate_setup_tabcd(mcq_display, teammate_accuracy)

    prompts: List[str] = []
    for row in batch:
        lines = [
            "-" * 30,
            "Question:",
            row["question"].strip(),
            "-" * 10,
            f"  {teammate_letter}: Delegate to your teammate",
        ]
        for letter, text in zip(mcq_display, _row_option_texts(row)):
            lines.append(f"  {letter}: {text}")
        lines.append("-" * 30)
        q_block = "\n".join(lines)

        user_content = (
            setup + "\n\n" + q_block + f"\nYour choice ({options_str}): "
        )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


def build_delegate_ta_prompts(
    batch: List[Dict],
    tokenizer,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    model_type: str = "instruct",
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> List[str]:
    """TA delegate prompts — same as AT but T (delegate) is listed FIRST in
    the meta-option block, the inner question, and the 'Your choice' hint.
    Forward pass over {A,T} is identical to AT.
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    setup = _delegate_setup_ta(teammate_accuracy)
    inner_question = (
        "Do you want to delegate this question to your teammate or answer it yourself?"
    )
    option_block = [
        f"  {DELEGATE_TEAMMATE_LETTER}: Delegate to my teammate",
        "  A: Answer the question myself",
    ]

    prompts: List[str] = []
    for row in batch:
        q_block = _fenced_meta_question(
            inner_question,
            row["question"].strip(),
            mcq_display,
            _row_option_texts(row),
            option_block,
        )
        user_content = (
            setup + "\n\n" + q_block + f"\nYour choice ({DELEGATE_TEAMMATE_LETTER} or A): "
        )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


def build_delegate_at_prompts(
    batch: List[Dict],
    tokenizer,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    model_type: str = "instruct",
    teammate_accuracy: float = DEFAULT_TEAMMATE_ACCURACY,
) -> List[str]:
    """AT delegate prompts (binary Answer / Teammate) in fenced layout.

    Same nested fenced block as self/other-confidence, but the meta-options
    are A=Answer-myself / T=Delegate. The MC options are still shown inside
    the inner question so the model can judge difficulty. Letter A here is
    the meta-decision token, not an MC option letter — when MCQ display
    uses A-D this is a deliberate overload (same letter, different token
    space) and the setup text disambiguates.
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    inner_question = (
        "Do you want to answer this question yourself or delegate it to your teammate?"
    )
    option_block = [
        "  A: Answer the question myself",
        f"  {DELEGATE_TEAMMATE_LETTER}: Delegate to my teammate",
    ]

    if model_type == "base":
        prefix = _base_setup_at(teammate_accuracy) + _base_few_shot_at(mcq_display)
    else:
        setup = _delegate_setup_at(teammate_accuracy)

    prompts: List[str] = []
    for row in batch:
        q_block = _fenced_meta_question(
            inner_question,
            row["question"].strip(),
            mcq_display,
            _row_option_texts(row),
            option_block,
        )
        if model_type == "base":
            user_content = prefix + q_block + "\nYour choice (A or T): "
        else:
            user_content = (
                setup + "\n\n" + q_block + "\nYour choice (A or T): "
            )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


def build_other_confidence_prompts_numeric(
    batch: List[Dict],
    tokenizer,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
    n_max: int = 10,
    model_type: str = "instruct",
) -> List[str]:
    """Build fenced-layout numeric (1..n_max) other-confidence prompts."""
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]

    setup = (
        "I want your help calibrating question difficulty. I'm going to show you "
        "a multiple-choice question, and I want you to tell me approximately how "
        f"likely a college-educated person is to get it right, on a scale of 1 to "
        f"{n_max} where 1 means \"almost no one would get it right\" and {n_max} "
        f"means \"almost everyone would get it right\". Respond only with a number "
        f"from 1 to {n_max}; do NOT output any other text."
    )
    inner_question = (
        f"What fraction of college-educated people would get this question right? "
        f"Rate from 1 (almost none) to {n_max} (almost everyone)."
    )

    if model_type == "base":
        base_intro = (
            f"For each question, estimate from 1 (almost no one would get it "
            f"right) to {n_max} (almost everyone would get it right) how a "
            f"college-educated person would do.\n\n"
        )
        prefix = base_intro + _base_few_shot_other_conf_numeric(mcq_display, n_max)

    prompts: List[str] = []
    for row in batch:
        q_block = _fenced_meta_question(
            inner_question,
            row["question"].strip(),
            mcq_display,
            _row_option_texts(row),
            option_block=[],
        )
        if model_type == "base":
            user_content = prefix + q_block + f"\nYour choice (1-{n_max}): "
        else:
            user_content = (
                setup + "\n\n" + q_block + f"\nYour choice (1-{n_max}): "
            )
        prompts.append(_wrap_user_message(user_content, tokenizer, model_type))
    return prompts


# ---- forward-pass helpers --------------------------------------------------

def run_mcq_forward_pass(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    temperature: float = 0.0,  # unused, kept for API symmetry
    requires_grad: bool = False,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
):
    """Forward pass over MCQ prompts; returns A-D-ordered logits and probs.

    Aggregates over both " X" and "X" single-token IDs for each display letter
    via logsumexp, then maps display letters back to canonical A-D order.
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    reverse_mapping = {v: k for k, v in mcq_letter_mapping.items()}

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with _torch.no_grad():
            out = model(**enc, use_cache=False)
    final_logits = out.logits[:, -1, :]

    display_letters = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    letter_token_ids = {
        d: get_letter_token_ids(tokenizer, d) for d in display_letters
    }

    logits_dict = {}
    for d in display_letters:
        sub = final_logits[:, letter_token_ids[d]]
        logits_dict[reverse_mapping[d]] = _torch.logsumexp(sub, dim=-1)

    logits4 = _torch.stack(
        [logits_dict[chr(ord('A') + i)] for i in range(4)], dim=-1
    )
    probs4 = _torch.softmax(logits4, dim=-1)
    entropy = -(probs4 * _torch.log(probs4 + 1e-12)).sum(dim=-1)

    idx = logits4.argmax(dim=-1).tolist()
    return {
        "pred_letters": [display_letters[i] for i in idx],
        "pred_positions": idx,
        "logits4": logits4,
        "probs4": probs4,
        "entropy": entropy,
    }


def run_confidence_forward_pass(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    temperature: float = 0.0,
    requires_grad: bool = False,
    confidence_letter_mapping: Optional[Dict[str, str]] = None,
):
    """Forward pass over A-H confidence prompts; returns A-H-ordered logits, probs."""
    if confidence_letter_mapping is None:
        confidence_letter_mapping = get_confidence_letter_mapping("A-H")
    reverse_mapping = {v: k for k, v in confidence_letter_mapping.items()}

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with _torch.no_grad():
            out = model(**enc, use_cache=False)
    final_logits = out.logits[:, -1, :]

    display_letters = [confidence_letter_mapping[chr(ord('A') + i)] for i in range(8)]
    bin_token_ids = {
        d: get_letter_token_ids(tokenizer, d) for d in display_letters
    }

    logits_dict = {}
    for d in display_letters:
        sub = final_logits[:, bin_token_ids[d]]
        logits_dict[reverse_mapping[d]] = _torch.logsumexp(sub, dim=-1)

    logits8 = _torch.stack(
        [logits_dict[chr(ord('A') + i)] for i in range(8)], dim=-1
    )
    probs8 = _torch.softmax(logits8, dim=-1)

    mids = _torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95],
                         dtype=_torch.float32, device=device)
    expected_conf = (probs8 * mids).sum(dim=-1)

    idx = logits8.argmax(dim=-1).tolist()
    return {
        "logits8": logits8,
        "probs8": probs8,
        "expected_conf": expected_conf,
        "pred_bins": [display_letters[i] for i in idx],
        "pred_bin_indices": idx,
    }


# Cache: id(tokenizer) → {n_max: list[list[int]]}
_NUMERIC_TOKEN_ID_CACHE: dict = {}


def _get_numeric_token_ids(tokenizer, n_max: int) -> List[List[int]]:
    """Cached single-token IDs for digits 1..n_max (with and without space)."""
    key = id(tokenizer)
    bucket = _NUMERIC_TOKEN_ID_CACHE.setdefault(key, {})
    if n_max in bucket:
        return bucket[n_max]
    ids_per_digit = [get_letter_token_ids(tokenizer, str(d))
                     for d in range(1, n_max + 1)]
    bucket[n_max] = ids_per_digit
    return ids_per_digit


def run_confidence_forward_pass_numeric(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    temperature: float = 0.0,
    requires_grad: bool = False,
    n_max: int = 10,
):
    """Forward pass over numeric (1..n_max) confidence prompts."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with _torch.no_grad():
            out = model(**enc, use_cache=False)
    final_logits = out.logits[:, -1, :]

    digit_token_ids = _get_numeric_token_ids(tokenizer, n_max)
    logits_list = []
    for ids in digit_token_ids:
        if not ids:
            raise ValueError(
                f"No single-token encoding for one of the digits 1..{n_max}. "
                "Check that your tokenizer encodes each digit as a single token."
            )
        logits_list.append(_torch.logsumexp(final_logits[:, ids], dim=-1))

    logits = _torch.stack(logits_list, dim=-1)
    probs = _torch.softmax(logits, dim=-1)

    scheme = get_numeric_scheme(n_max)
    mids = _torch.tensor(
        [scheme["midpoints"][str(d)] for d in range(1, n_max + 1)],
        dtype=_torch.float32, device=device,
    )
    expected_conf = (probs * mids).sum(dim=-1)

    idx = logits.argmax(dim=-1).tolist()
    return {
        "logits": logits,
        "probs": probs,
        "expected_conf": expected_conf,
        "pred_digits": [str(i + 1) for i in idx],
        "pred_digit_indices": idx,
        "n_max": n_max,
    }


def run_delegate_abcdt_forward_pass(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    requires_grad: bool = False,
    mcq_letter_mapping: Optional[Dict[str, str]] = None,
):
    """Forward pass over ABCDT delegate prompts.

    Logits/probs are returned in canonical [A, B, C, D, T] order (A-D follow
    mcq_letter_mapping for tokenization, then mapped back to canonical
    positions). Index 4 is always T (delegate).
    """
    if mcq_letter_mapping is None:
        mcq_letter_mapping = get_mcq_letter_mapping("A-D")
    reverse_mapping = {v: k for k, v in mcq_letter_mapping.items()}
    mcq_display = [mcq_letter_mapping[chr(ord('A') + i)] for i in range(4)]
    teammate_letter = DELEGATE_TEAMMATE_LETTER

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with _torch.no_grad():
            out = model(**enc, use_cache=False)
    final_logits = out.logits[:, -1, :]

    letter_token_ids = {
        d: get_letter_token_ids(tokenizer, d) for d in mcq_display + [teammate_letter]
    }

    # Canonical-order logits: A, B, C, D, T (A-D mapped back from display).
    logits_canon: Dict[str, _torch.Tensor] = {}
    for d in mcq_display:
        sub = final_logits[:, letter_token_ids[d]]
        logits_canon[reverse_mapping[d]] = _torch.logsumexp(sub, dim=-1)
    logits_canon[teammate_letter] = _torch.logsumexp(
        final_logits[:, letter_token_ids[teammate_letter]], dim=-1
    )

    logits5 = _torch.stack(
        [logits_canon[chr(ord('A') + i)] for i in range(4)] + [logits_canon[teammate_letter]],
        dim=-1,
    )
    probs5 = _torch.softmax(logits5, dim=-1)

    idx = logits5.argmax(dim=-1).tolist()
    canonical_letters = ["A", "B", "C", "D", teammate_letter]
    display_letters = mcq_display + [teammate_letter]

    return {
        "logits5": logits5,
        "probs5": probs5,
        # Canonical (positional) — A means "answer with the option in slot 0",
        # regardless of which display letter was actually shown.
        "pred_canonical": [canonical_letters[i] for i in idx],
        "pred_positions": idx,  # 0..4; 4 == delegate
        # Display letter — what the model actually emits for this prompt.
        "pred_display": [display_letters[i] for i in idx],
        # P(answer = T) and P(answer with one of the four MC options).
        "p_delegate": probs5[:, 4],
        "p_answer": 1.0 - probs5[:, 4],
    }


def run_delegate_at_forward_pass(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    requires_grad: bool = False,
):
    """Forward pass over AT (binary Answer / Teammate) delegate prompts.

    Returns logits/probs in [A, T] order. Index 0 = answer, 1 = delegate.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if requires_grad:
        out = model(**enc, use_cache=False)
    else:
        with _torch.no_grad():
            out = model(**enc, use_cache=False)
    final_logits = out.logits[:, -1, :]

    a_ids = get_letter_token_ids(tokenizer, "A")
    t_ids = get_letter_token_ids(tokenizer, DELEGATE_TEAMMATE_LETTER)

    logits2 = _torch.stack(
        [
            _torch.logsumexp(final_logits[:, a_ids], dim=-1),
            _torch.logsumexp(final_logits[:, t_ids], dim=-1),
        ],
        dim=-1,
    )
    probs2 = _torch.softmax(logits2, dim=-1)

    idx = logits2.argmax(dim=-1).tolist()
    pred_letters = ["A" if i == 0 else DELEGATE_TEAMMATE_LETTER for i in idx]
    return {
        "logits2": logits2,
        "probs2": probs2,
        "pred_letters": pred_letters,
        "pred_positions": idx,
        "p_answer": probs2[:, 0],
        "p_delegate": probs2[:, 1],
    }
