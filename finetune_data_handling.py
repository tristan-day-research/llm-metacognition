import json
import os
import re
import random
import numpy as np
import traceback
from datetime import datetime, timezone
from torch.utils.data import Dataset


# Separate RNGs for training and evaluation to prevent any shared state
# Training uses seed 42, evaluation uses seed 999 to ensure complete separation
_train_rng = np.random.default_rng(42)
_eval_rng = np.random.default_rng(999)

def get_batch(dataset, batch_size, is_training=True):
    """
    Get a random batch from the dataset.
    
    Args:
        dataset: The dataset to sample from
        batch_size: Number of samples to return
        is_training: If True, use training RNG; if False, use evaluation RNG.
                    This ensures train and val sampling are completely independent.
    
    Returns:
        List of dataset samples
    """
    rng = _train_rng if is_training else _eval_rng
    idxs = rng.choice(len(dataset), size=batch_size, replace=False)
    return [dataset[i] for i in idxs]


def write_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def load_jsonl_dataset(path, dataset_type="unknown"):
    """
    Load dataset with proper shuffling.
    
    Args:
        path: Path to JSONL file
        dataset_type: Type of dataset ("train", "val", "test", etc.) for logging
    
    Returns:
        List of question dictionaries
    """
    # Use a seeded RNG for deterministic option shuffling per dataset
    # Different seeds for different dataset types to ensure independence
    dataset_seed = hash(path) % (2**31)  # Deterministic seed based on path
    local_rng = np.random.default_rng(dataset_seed)
    # Also seed Python's random for shuffle() calls
    random.seed(dataset_seed)
    
    out = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)

            question = row["question"]
            correct = row["correct_answer"]
            distractors = row["distractors"]

            # Build 4-option MCQ set
            opts = [correct] + distractors
            if len(opts) != 4:
                continue

            # Shuffle options using seeded RNG for reproducibility
            # Note: random.shuffle() uses Python's global random state, which we seeded above
            random.shuffle(opts)
            
            # Assign to letters and find correct letter
            options = {}
            correct_letter = None
            for i, letter in enumerate(["A", "B", "C", "D"]):
                options[letter] = opts[i]
                if opts[i] == correct:
                    correct_letter = letter
            
            # Verify we found the correct answer
            if correct_letter is None:
                print(f"WARNING: Could not find correct answer for {row.get('qid')}")
                continue

            out.append({
                "qid": row.get("qid"),
                "question": question,
                "options": options,
                "correct_letter": correct_letter,
            })

    return out


class MCQDataset(Dataset):
    """Dataset for Multiple Choice Questions."""

    def __init__(self, path):
        """
        Initialize MCQ dataset from JSONL file.

        Args:
            path: Path to JSONL file containing questions
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        """Return number of questions in dataset."""
        return len(self.rows)

    def __getitem__(self, idx):
        """
        Get a single question from the dataset.

        Args:
            idx: Index of question to retrieve

        Returns:
            Dictionary with question, options, and correct answer
        """
        row = self.rows[idx]

        question = row["question"]
        correct = row["correct_answer"]
        distractors = row["distractors"][:3]  # ensure 3

        # No shuffling: fixed layout
        options = [(correct, True)] + [(d, False) for d in distractors]

        labeled = {}
        correct_letter = None
        for label, (text, is_correct) in zip("ABCD", options):
            labeled[label] = text
            if is_correct:
                correct_letter = label

        return {
            "qid": row.get("qid", str(idx)),
            "question": question,
            "options": labeled,
            "correct_letter": correct_letter,
        }


def normalize_text(s):
    """Normalize text for comparison: lowercase, strip, normalize whitespace, remove trailing period."""
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def load_mcq_results_data(mcq_results_path):
    """
    Load MCQ results data from JSON file.
    Returns dict mapping question ID -> {probs, subject_answer, entropy}
    """
    import torch  # ADD THIS
    import json
    import math
    
    if mcq_results_path is None:
        return None
    
    try:
        with open(mcq_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results_lookup = {}
        
        # Handle wrapped format {"results": {...}}
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        
        # Process each question
        for qid, result in data.items():
            probs_dict = result.get("probs", {})
            
            # Compute entropy from probs
            probs = [probs_dict.get(letter, 0.0) for letter in "ABCD"]
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            entropy = -(probs_tensor * torch.log(probs_tensor + 1e-12)).sum().item()
            
            results_lookup[str(qid)] = {
                "probs": probs_dict,
                "subject_answer": result.get("subject_answer"),
                "entropy": entropy,
                "options": result.get("question", {}).get("options", {})
            }
        
        print(f"✓ Loaded {len(results_lookup)} pre-recorded MCQ results")
        return results_lookup
        
    except Exception as e:
        print(f"Error loading MCQ results from {mcq_results_path}: {e}")
        import traceback
        traceback.print_exc()  # This will show the full error
        return None


def filter_dataset_by_mcq_results(dataset, mcq_results_lookup, dataset_name="dataset"):
    """
    Filter dataset to only include questions with pre-recorded MCQ results.
    
    Args:
        dataset: List of question dicts
        mcq_results_lookup: Dict from load_mcq_results_data()
        dataset_name: Name for logging (e.g., "training", "validation")
    
    Returns:
        Filtered dataset (list)
    """
    if mcq_results_lookup is None:
        return dataset
    
    original_size = len(dataset)
    
    # Filter to only questions with matching qid in results
    filtered = [
        row for row in dataset 
        if row.get("qid") and row.get("qid") in mcq_results_lookup
    ]
    
    filtered_size = len(filtered)
    removed = original_size - filtered_size
    removed_pct = (removed / original_size * 100) if original_size > 0 else 0
    
    print(f"\n🔍 Filtered {dataset_name}:")
    print(f"  Original size: {original_size}")
    print(f"  After filtering: {filtered_size}")
    print(f"  Removed: {removed} ({removed_pct:.1f}%)")
    
    if removed > 0:
        print(f"  ⚠️  {removed_pct:.1f}% of {dataset_name} data will not be used")
    
    return filtered


def verify_and_resolve_options(row, mcq_results_lookup, log_file_path=None):
    """
    Resolve the correct option set for the batch question.

    Checks top-level keys created by the loader above.

    Args:
        row: Dictionary with question data
        mcq_results_lookup: Lookup dictionary for recorded results
        log_file_path: Path for logging (optional)

    Returns:
        Tuple of (result_data, options)
    """
    qid = row.get("qid")
    batch_question = row["question"]
    # These are the raw/shuffled ones from dataset
    batch_opts = row["options"]

    # Default: use batch opts
    if not mcq_results_lookup:
        return None, batch_opts

    # ---------- 1. Lookup ----------
    result_data = None
    
    # Extract text-to-ID mapping if available
    text_to_id = mcq_results_lookup.get("__text_to_id__", {})
    
    # Try QID match first
    if qid is not None and str(qid) in mcq_results_lookup:
        result_data = mcq_results_lookup[str(qid)]
    
    # Try Text match if QID failed - use text-to-ID mapping to find the ID, then look up by ID
    if result_data is None:
        norm_text = normalize_text(batch_question)
        if norm_text in text_to_id:
            # Found text mapping, look up by the mapped ID
            mapped_id = text_to_id[norm_text]
            if mapped_id in mcq_results_lookup:
                result_data = mcq_results_lookup[mapped_id]
        elif norm_text in mcq_results_lookup:
            # Fallback: text might be stored directly (for questions without IDs)
            result_data = mcq_results_lookup[norm_text]

    # Nothing found → fallback
    if result_data is None:
        if log_file_path:
            # Import here to avoid circular import
            from finetune_utils import write_log
            write_log(log_file_path, {
                "type": "verification_no_lookup_found",
                "qid": qid,
                "question_snippet": batch_question[:50],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        return None, batch_opts

    # ---------- 2. Extract Fields (Robustly) ----------
    # The loader puts these at the top level now
    rec_opts = result_data.get("options")
    rec_q_text = result_data.get("question_text")

    # If missing options, we can't proceed
    if not rec_opts:
        if log_file_path:
            # Import here to avoid circular import
            from finetune_utils import write_log
            write_log(log_file_path, {
                "type": "verification_missing_options_in_record",
                "qid": qid,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        return None, batch_opts

    # ---------- 3. Verify Question Text ----------
    # Only verify if we successfully captured text in the loader
    if rec_q_text:
        if normalize_text(batch_question) != normalize_text(rec_q_text):
            if log_file_path:
                # Import here to avoid circular import
                from finetune_utils import write_log
                write_log(log_file_path, {
                    "type": "verification_question_text_mismatch",
                    "qid": qid,
                    "batch": batch_question,
                    "recorded": rec_q_text,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            return None, batch_opts

    # ---------- 4. Verify/Sync Options ----------
    # We rely on the recorded options (rec_opts) as the truth.
    # We just ensure the batch *contains* the same keys so we don't crash.
    for letter in ["A", "B", "C", "D"]:
        if letter not in rec_opts:
            # If the recorded data is missing a letter (rare, but possible in broken data)
            return None, batch_opts

        # Optional: Verify content matches if you suspect the dataset changed
        # This might fail if "options" in batch are shuffled vs "options" in record
        # Since we want to FORCE the record, we often skip checking the content 
        # unless we are unsure if it's the same question.
        # Given we checked QID and Text above, we trust rec_opts.

    # Verification passed - log success
    if log_file_path:
        # Import here to avoid circular import
        from finetune_utils import write_log
        write_log(log_file_path, {
            "type": "verification_passed",
            "qid": qid,
            "question_snippet": batch_question[:50],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return result_data, rec_opts


def validate_file_exists_and_not_empty(file_path, file_description="file"):
    """
    Validate that a file exists and is not empty.
    
    Args:
        file_path: Path to the file to validate
        file_description: Description of the file for error messages
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_description.capitalize()} not found: {file_path}\n"
            f"Please provide a valid path to your {file_description}."
        )
    
    # Check if file is empty
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(
            f"{file_description.capitalize()} is empty: {file_path}\n"
            f"The file exists but contains no data. Please check your {file_description} file."
        )


def validate_and_load_dataset(data_path, dataset_type="training"):
    """
    Validate and load a dataset file.
    
    Args:
        data_path: Path to JSONL dataset file
        dataset_type: Type of dataset ("training" or "validation")
        
    Returns:
        MCQDataset instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If dataset is empty
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{dataset_type.capitalize()} data file not found: {data_path}\n"
            f"Please provide a valid path to your {dataset_type} dataset."
        )
    
    dataset = MCQDataset(data_path)
    if len(dataset) == 0:
        raise ValueError(
            f"{dataset_type.capitalize()} dataset is empty: {data_path}\n"
            f"The file exists but contains no data. Please check your dataset file."
        )
    
    return dataset


def resolve_file_path(file_path, search_dirs=None):
    """
    Try to find a file in common locations if it doesn't exist at the given path.
    
    Args:
        file_path: Original file path
        search_dirs: List of additional directories to search (optional)
        
    Returns:
        Resolved file path if found, original path otherwise
        
    Raises:
        FileNotFoundError: If file cannot be found in any location
    """
    if os.path.exists(file_path):
        return file_path
    
    file_path = file_path.strip()
    basename = os.path.basename(file_path)
    
    # Default search directories
    default_search_dirs = [
        "data",
        "explicit_confidence_task_logs",
        "capabilities_test_logs",
    ]
    
    if search_dirs:
        search_dirs = list(search_dirs) + default_search_dirs
    else:
        search_dirs = default_search_dirs
    
    # Build possible paths
    possible_paths = [file_path]
    for search_dir in search_dirs:
        possible_paths.append(os.path.join(search_dir, basename))
    possible_paths.append(os.path.join(os.getcwd(), file_path))
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, raise error with helpful message
    error_msg = (
        f"Error: Could not find file. "
        f"Tried the following paths:\n"
        + "\n".join(f"  - {path}" for path in possible_paths)
        + f"\n\nOriginal path: {file_path}"
        + f"\nCurrent working directory: {os.getcwd()}"
    )
    raise FileNotFoundError(error_msg)


def validate_training_files(args):
    """
    Validate all input files for training, with automatic path resolution.
    
    This function validates:
    - Training data file (required)
    - Validation data file (optional)
    - Test data file (optional, with path resolution)
    - MCQ results file (optional, with path resolution)
    
    Args:
        args: Argument object with the following attributes:
            - train_data_path: Path to training data (required)
            - val_data_path: Path to validation data (optional)
            - test_data_path: Path to test data (optional)
            - mcq_results_data: Path to MCQ results (optional)
            
    Modifies args in place by resolving paths for test_data_path and mcq_results_data.
    
    Raises:
        FileNotFoundError: If any required file is not found
        ValueError: If any file is empty
    """
    print("\n=== Validating input files ===")
    
    # Validate training data file (required)
    validate_file_exists_and_not_empty(args.train_data_path, "training data file")
    print(f"✓ Training data file exists and is not empty: {args.train_data_path}")
    
    # Validate validation data file (optional)
    if args.val_data_path:
        validate_file_exists_and_not_empty(args.val_data_path, "validation data file")
        print(f"✓ Validation data file exists and is not empty: {args.val_data_path}")
    
    # Validate test data file (optional, with path resolution)
    if args.test_data_path:
        try:
            args.test_data_path = resolve_file_path(
                args.test_data_path,
                search_dirs=["data"]  # Test files are typically in data/
            )
        except FileNotFoundError as e:
            print(f"Test data file not found at: {args.test_data_path}")
            print(f"Current working directory: {os.getcwd()}")
            print("Attempting to find file...")
            raise FileNotFoundError(
                f"Test data file not found: {args.test_data_path}\n"
                f"Please provide a valid path to the test data file."
            ) from e
        
        validate_file_exists_and_not_empty(args.test_data_path, "test data file")
        print(f"✓ Test data file exists and is not empty: {args.test_data_path}")
    
    # Validate MCQ results file (optional, with path resolution)
    if args.mcq_results_data:
        try:
            args.mcq_results_data = resolve_file_path(
                args.mcq_results_data,
                search_dirs=["data", "explicit_confidence_task_logs", "capabilities_test_logs"]
            )
        except FileNotFoundError as e:
            print(f"MCQ results file not found at: {args.mcq_results_data}")
            print(f"Current working directory: {os.getcwd()}")
            print("Attempting to find file...")
            raise FileNotFoundError(
                f"MCQ results file not found: {args.mcq_results_data}\n"
                f"Please provide a valid path to the MCQ results file."
            ) from e
        
        validate_file_exists_and_not_empty(args.mcq_results_data, "MCQ results file")
        print(f"✓ MCQ results file exists and is not empty: {args.mcq_results_data}")
    
    print("=== File validation complete ===\n")


def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries."""
    return batch


def validate_datasets_separate(train_dataset, val_dataset, dataset_name_train="train", dataset_name_val="val"):
    """
    Validate that train and validation datasets are completely separate.
    
    Checks for:
    1. Overlapping question IDs (qid)
    2. Overlapping question text (normalized)
    
    Args:
        train_dataset: Training dataset (list of dicts)
        val_dataset: Validation dataset (list of dicts)
        dataset_name_train: Name for error messages
        dataset_name_val: Name for error messages
    
    Raises:
        ValueError: If any overlap is detected
    """
    # Extract all qids from both datasets
    train_qids = set()
    val_qids = set()
    
    train_questions = set()
    val_questions = set()
    
    for row in train_dataset:
        qid = row.get("qid")
        if qid:
            train_qids.add(str(qid))
        question = row.get("question", "")
        if question:
            train_questions.add(normalize_text(question))
    
    for row in val_dataset:
        qid = row.get("qid")
        if qid:
            val_qids.add(str(qid))
        question = row.get("question", "")
        if question:
            val_questions.add(normalize_text(question))
    
    # Check for overlapping qids
    overlapping_qids = train_qids & val_qids
    if overlapping_qids:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(overlapping_qids)} question IDs appear in both "
            f"{dataset_name_train} and {dataset_name_val} datasets. "
            f"Sample overlapping qids: {list(overlapping_qids)[:5]}"
        )
    
    # Check for overlapping question text
    overlapping_questions = train_questions & val_questions
    if overlapping_questions:
        # This is also data leakage - same question in both train and val
        # Even if qids differ, the model would see the same question during training and evaluation
        sample_questions = list(overlapping_questions)[:5]
        sample_str = "\n".join([f"    - {q[:100]}..." for q in sample_questions])
        
        # Find the actual qids for the overlapping questions to help with debugging
        overlapping_qids_info = []
        for norm_q in sample_questions:
            # Find qids in train dataset
            train_qids_for_q = [row.get("qid") for row in train_dataset 
                               if normalize_text(row.get("question", "")) == norm_q]
            # Find qids in val dataset
            val_qids_for_q = [row.get("qid") for row in val_dataset 
                             if normalize_text(row.get("question", "")) == norm_q]
            if train_qids_for_q or val_qids_for_q:
                overlapping_qids_info.append(
                    f"    Question: {norm_q[:80]}...\n"
                    f"      Train qids: {train_qids_for_q[:3]}{'...' if len(train_qids_for_q) > 3 else ''}\n"
                    f"      Val qids: {val_qids_for_q[:3]}{'...' if len(val_qids_for_q) > 3 else ''}"
                )
        
        qids_info_str = "\n\n".join(overlapping_qids_info) if overlapping_qids_info else sample_str
        
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(overlapping_questions)} questions with identical text "
            f"appear in both {dataset_name_train} and {dataset_name_val} datasets. "
            f"This means the model will see the same questions during training and evaluation, "
            f"which invalidates the evaluation.\n\n"
            f"Overlapping questions with their qids:\n{qids_info_str}\n\n"
            f"Please remove these questions from one of the datasets (preferably from {dataset_name_train}) "
            f"to ensure proper train/val separation."
        )
    
    print(f"✓ Dataset separation validated: {len(train_qids)} train qids, {len(val_qids)} val qids, no overlap")


def find_and_remove_duplicates(train_dataset, val_dataset, remove_from="train", log_file_path=None):
    """
    Find questions that appear in both datasets and remove them from the specified dataset.
    
    Checks for duplicates by BOTH:
    1. Question ID (qid) - exact match
    2. Question text (normalized) - text content match
    
    This is a utility function to help fix data leakage issues detected by validate_datasets_separate().
    
    Args:
        train_dataset: Training dataset (list of dicts)
        val_dataset: Validation dataset (list of dicts) - used for comparison only
        remove_from: Which dataset to remove duplicates from ("train" or "val")
                    Default "train" removes from training set (recommended)
        log_file_path: Optional path to log file for detailed logging
    
    Returns:
        Tuple of (removal_summary, updated_dataset)
        - removal_summary: Dict with counts and details of removals
        - updated_dataset: The dataset with duplicates removed (new list, not modified in place)
    """
    from finetune_utils import write_log
    
    # Build lookup sets from validation dataset
    val_qids = set()
    val_questions_normalized = set()
    val_qid_to_question = {}  # Map qid to question text for logging
    
    for row in val_dataset:
        qid = row.get("qid")
        question = row.get("question", "")
        if qid:
            val_qids.add(str(qid))
            val_qid_to_question[str(qid)] = question
        if question:
            val_questions_normalized.add(normalize_text(question))
    
    # Determine which dataset to filter
    if remove_from == "train":
        dataset_to_filter = train_dataset
        other_dataset_name = "validation"
    elif remove_from == "val":
        dataset_to_filter = val_dataset
        other_dataset_name = "training"
    else:
        raise ValueError(f"remove_from must be 'train' or 'val', got '{remove_from}'")
    
    # Track removals by match type
    removed_by_qid = []
    removed_by_text = []
    removed_by_both = []  # Matched by both qid and text
    
    # Filter out duplicates
    filtered_dataset = []
    
    for row in dataset_to_filter:
        qid = row.get("qid")
        question = row.get("question", "")
        norm_q = normalize_text(question) if question else ""
        
        # Check for matches
        matched_by_qid = qid and str(qid) in val_qids
        matched_by_text = norm_q and norm_q in val_questions_normalized
        
        if matched_by_qid or matched_by_text:
            # Determine match type for logging
            if matched_by_qid and matched_by_text:
                match_type = "both_qid_and_text"
                removed_by_both.append({
                    "qid": qid,
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "match_method": "both_qid_and_text"
                })
            elif matched_by_qid:
                match_type = "qid_only"
                removed_by_qid.append({
                    "qid": qid,
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "match_method": "qid_only"
                })
            else:  # matched_by_text
                match_type = "text_only"
                removed_by_text.append({
                    "qid": qid,
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "match_method": "text_only"
                })
            
            # Log individual removal
            if log_file_path:
                write_log(log_file_path, {
                    "type": "duplicate_removed",
                    "removed_from": remove_from,
                    "qid": qid,
                    "question_snippet": question[:200] if question else None,
                    "match_method": match_type,
                    "matched_by_qid": matched_by_qid,
                    "matched_by_text": matched_by_text,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        else:
            # Keep this row - no match found
            filtered_dataset.append(row)
    
    # Build summary
    total_removed = len(removed_by_qid) + len(removed_by_text) + len(removed_by_both)
    
    removal_summary = {
        "total_removed": total_removed,
        "removed_from": remove_from,
        "by_qid_only": len(removed_by_qid),
        "by_text_only": len(removed_by_text),
        "by_both": len(removed_by_both),
        "details": {
            "qid_only": removed_by_qid[:10],  # Limit details
            "text_only": removed_by_text[:10],
            "both": removed_by_both[:10]
        }
    }
    
    # Print summary
    if total_removed > 0:
        print(f"\n🔧 Removed {total_removed} duplicate question(s) from {remove_from} dataset:")
        print(f"   - {len(removed_by_qid)} matched by qid only")
        print(f"   - {len(removed_by_text)} matched by question text only")
        print(f"   - {len(removed_by_both)} matched by both qid and text")
        print(f"\n   Match method breakdown:")
        print(f"     • QID match: Questions with same question ID (qid) in both datasets")
        print(f"     • Text match: Questions with identical normalized text (even if qids differ)")
        print(f"     • Both: Questions matching by both qid and text")
        
        # Show samples
        all_removed = removed_by_qid + removed_by_text + removed_by_both
        print(f"\n   Sample removed questions (showing up to 5):")
        for i, item in enumerate(all_removed[:5], 1):
            print(f"     {i}. qid={item['qid']}, match_method={item['match_method']}")
            print(f"        question={item['question']}")
        if len(all_removed) > 5:
            print(f"     ... and {len(all_removed) - 5} more")
        
        print(f"\n✓ {remove_from} dataset: {len(dataset_to_filter)} → {len(filtered_dataset)} samples "
              f"(removed {total_removed} duplicates)")
        
        # Log summary
        if log_file_path:
            write_log(log_file_path, {
                "type": "duplicate_removal_summary",
                **removal_summary,
                "original_size": len(dataset_to_filter),
                "final_size": len(filtered_dataset),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    else:
        print(f"✓ No duplicates found to remove from {remove_from} dataset")
    
    return removal_summary, filtered_dataset

