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
# NUMERIC CONFIDENCE TASK (1-10 scale)
# ============================================================================
# In the Llama 3 tokenizer, "10" is a single token, so we can do 1-10 cleanly.
# The scale is framed as "1 = not confident at all, 10 = completely sure" with
# each digit representing a 10%-wide confidence band. Midpoints span [5%, 95%].
# A runtime tokenizer check in run_introspection_experiment.py aborts if "10"
# turns out to be multi-token on whatever tokenizer the user loads.

NUMERIC_CONFIDENCE_SETUP = (
    "I'm going to show you a question, and I want you to tell me your level "
    "of confidence that you know the correct answer to this question, on a "
    "scale of 1 to 10 where 1 means \"not confident at all\" and 10 means "
    "\"completely sure\". Respond only with a number from 1 to 10; do NOT "
    "output any other text."
)

NUMERIC_CONFIDENCE_QUESTION = (
    "How confident are you that you know the correct answer (1-10)?"
)

# 10 equal-width bins over [0, 1], midpoints at {5%, 15%, ..., 95%}.
# "1" = bin [0%, 10%) → midpoint 5%; "10" = bin [90%, 100%] → midpoint 95%.
NUMERIC_CONFIDENCE_OPTIONS = {str(i): f"~{(i*10-5)}% confident" for i in range(1, 11)}
NUMERIC_CONFIDENCE_MIDPOINTS = {str(i): (i * 10 - 5) / 100.0 for i in range(1, 11)}


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
