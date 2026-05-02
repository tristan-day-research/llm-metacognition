"""
Question loading and management for consistent question sets across experiments.

Ensures the same questions are used across:
- Direct entropy extraction
- Meta-judgment tasks
- Probe training/testing

Uses deterministic seeding to guarantee reproducibility.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Import centralized prompt formatting from tasks.py
from prompts import format_direct_prompt as _format_direct_prompt_impl


def load_questions(
    dataset_name: str,
    num_questions: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = True
) -> List[Dict]:
    """
    Load and optionally shuffle questions from a dataset.

    Args:
        dataset_name: Name of dataset (e.g., "SimpleMC", "GPQA")
        num_questions: Max number of questions to return (None = all)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle questions

    Returns:
        List of question dicts with 'question', 'options', 'correct_answer', etc.
    """
    from core.datasets import load_and_format_dataset

    questions = load_and_format_dataset(dataset_name, num_questions_needed=num_questions)

    if questions is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(questions)

    if num_questions is not None:
        questions = questions[:num_questions]

    return questions


def get_question_hash(questions: List[Dict]) -> str:
    """
    Compute a hash of the question IDs to verify consistency.

    Use this to ensure the same questions are used across scripts.
    """
    ids = [q.get("id", q.get("question", "")[:50]) for q in questions]
    id_str = "|".join(sorted(ids))
    return hashlib.md5(id_str.encode()).hexdigest()[:12]


def save_question_set(
    questions: List[Dict],
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save a question set to disk for reuse.

    Args:
        questions: List of question dicts
        output_path: Path to save JSON file
        metadata: Optional dict with dataset_name, seed, etc.
    """
    data = {
        "questions": questions,
        "hash": get_question_hash(questions),
        "count": len(questions),
    }
    if metadata:
        data["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(questions)} questions to {output_path} (hash: {data['hash']})")


def load_question_set(input_path: str) -> List[Dict]:
    """
    Load a previously saved question set.

    Args:
        input_path: Path to JSON file

    Returns:
        List of question dicts
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    questions = data["questions"]
    expected_hash = data.get("hash")

    if expected_hash:
        actual_hash = get_question_hash(questions)
        if actual_hash != expected_hash:
            print(f"Warning: Question hash mismatch! Expected {expected_hash}, got {actual_hash}")

    print(f"Loaded {len(questions)} questions from {input_path}")
    return questions


def verify_question_consistency(
    questions: List[Dict],
    expected_hash: str,
    raise_on_mismatch: bool = True
) -> bool:
    """
    Verify that a question set matches an expected hash.

    Args:
        questions: List of question dicts
        expected_hash: Expected hash string
        raise_on_mismatch: If True, raise ValueError on mismatch

    Returns:
        True if hashes match
    """
    actual_hash = get_question_hash(questions)

    if actual_hash != expected_hash:
        msg = f"Question hash mismatch! Expected {expected_hash}, got {actual_hash}"
        if raise_on_mismatch:
            raise ValueError(msg)
        print(f"Warning: {msg}")
        return False

    return True


def format_direct_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    setup_prompt: Optional[str] = None
) -> tuple:
    """
    Format a direct MC question prompt.

    This is a thin wrapper around tasks.format_direct_prompt for backward compatibility.

    Args:
        question: Question dict with 'question' and 'options'
        tokenizer: Tokenizer for chat template
        use_chat_template: Whether to use chat template
        setup_prompt: Optional custom setup prompt

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    return _format_direct_prompt_impl(question, tokenizer, use_chat_template, setup_prompt)


def split_questions(
    questions: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple:
    """
    Split questions into train/test sets.

    Returns indices rather than copies to allow alignment with activations.

    Args:
        questions: List of question dicts
        train_ratio: Fraction for training
        seed: Random seed

    Returns:
        Tuple of (train_indices, test_indices)
    """
    n = len(questions)
    indices = np.arange(n)

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    split_idx = int(n * train_ratio)
    train_idx = np.sort(indices[:split_idx])
    test_idx = np.sort(indices[split_idx:])

    return train_idx, test_idx
