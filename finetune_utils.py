import json
import os
import re
import random
import traceback
from datetime import datetime, timezone
from torch.utils.data import Dataset
import torch


def write_log(log_file_path, entry_dict):
    """
    Simple one-line logging function.

    Logs dict as JSON to file if path provided.

    Args:
        log_file_path: Path to log file (None to skip logging)
        entry_dict: Dictionary to log as JSON
    """
    if log_file_path:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')


def _get_log_file_path(log_dir, model_name, suffix):
    """Helper function to create log file paths."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    model_name_safe = model_name.replace("/", "-").replace("_", "-")
    return os.path.join(
        log_dir, f"{model_name_safe}_{timestamp}_{suffix}.jsonl"
    )


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


def build_multiple_choice_question_prompts(batch):
    """
    Builds direct MCQ prompts matching explicit_confidence_task.py exactly.
    Includes dashed formatting and specific indentation.
    """
    setup_prompt = (
        "I'm going to ask you a series of multiple-choice questions. For each one, "
        "select the answer you think is best. Respond only with the letter of your choice; "
        "do NOT output any other text."
    )
    
    prompts = []
    for row in batch:
        question = row["question"]
        options = row["options"]
        
        # Structure matches BaseGameClass._present_question logic seen in logs
        # Header: Question + Text
        # Separator: 10 dashes
        # Body: Indented Options
        q_text = (
            "------------------------------\n"
            "Question:\n"
            f"{question}\n"
            "----------\n"
        )
        
        # Options are indented by 2 spaces
        for key in ["A", "B", "C", "D"]:
            if key in options:
                q_text += f"  {key}: {options[key]}\n"
        
        q_text += "------------------------------"
        
        # Full prompt assembly
        llm_prompt = q_text + "\nYour choice (A, B, C, or D): "
        full_prompt = setup_prompt + "\n\n" + llm_prompt
        prompts.append(full_prompt)
        
    return prompts


def build_self_confidence_prompts(batch):
    """
    Builds self-confidence prompts matching explicit_confidence_task.py exactly.
    """
    setup_prompt = (
        "I'm going to show you a multiple-choice question, and I want you to tell me "
        "your level of confidence that you would get the question right. Respond only "
        "with the letter corresponding to the percentage range you choose; do NOT output "
        "any other text."
    )
    
    # Matching explicit_confidence_task.py (lines 88-90)
    nested_option_dict = {
        "A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", 
        "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"
    }
    nested_question_prompt = "How confident are you that you would get this question right?"
    
    prompts = []
    for row in batch:
        question = row["question"]
        options = row["options"]
        
        # Structure matches BaseGameClass._present_nested_question logic
        # Header: Nested Prompt
        # Separator: 10 dashes
        # Body: Question Text + Indented Options
        # Separator: 10 dashes
        # Nested Body: Indented Nested Options
        q_text = (
            "------------------------------\n"
            f"{nested_question_prompt}\n"
            "----------\n"
            f"{question}\n"
        )
        
        # Original Question Options
        for key in ["A", "B", "C", "D"]:
            if key in options:
                q_text += f"  {key}: {options[key]}\n"
        
        q_text += "----------\n"
        
        # Confidence Options
        for key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            q_text += f"  {key}: {nested_option_dict[key]}\n"
            
        q_text += "------------------------------"
        
        # Full prompt assembly
        options_str = "A, B, C, D, E, F, G, or H"
        llm_prompt = q_text + f"\nYour choice ({options_str}): "
        full_prompt = setup_prompt + "\n\n" + llm_prompt
        prompts.append(full_prompt)
        
    return prompts


def build_other_confidence_prompts(batch):
    """
    Builds other-confidence (college-educated) prompts matching explicit_confidence_task.py exactly.
    """
    setup_prompt = (
        "I want your help calibrating question difficulty. I'm going to show you a "
        "multiple-choice question, and I want you to tell me approximately what "
        "percentage of college-educated people you think would get it right. Respond "
        "only with the letter corresponding to the percentage range you choose; do NOT "
        "output any other text."
    )
    
    # Matching explicit_confidence_task.py (lines 83-85)
    nested_option_dict = {
        "A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", 
        "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"
    }
    nested_question_prompt = "What percentage of college-educated people would get this question right?"
    
    prompts = []
    for row in batch:
        question = row["question"]
        options = row["options"]
        
        # Structure matches BaseGameClass._present_nested_question logic
        q_text = (
            "------------------------------\n"
            f"{nested_question_prompt}\n"
            "----------\n"
            f"{question}\n"
        )
        
        # Original Question Options
        for key in ["A", "B", "C", "D"]:
            if key in options:
                q_text += f"  {key}: {options[key]}\n"
        
        q_text += "----------\n"
        
        # Confidence Options
        for key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            q_text += f"  {key}: {nested_option_dict[key]}\n"
            
        q_text += "------------------------------"
        
        # Full prompt assembly
        options_str = "A, B, C, D, E, F, G, or H"
        llm_prompt = q_text + f"\nYour choice ({options_str}): "
        full_prompt = setup_prompt + "\n\n" + llm_prompt
        prompts.append(full_prompt)
        
    return prompts

# ============================================================
# Dataset: no logprobs needed, only the raw MCQ fields
# ============================================================


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


def get_single_token_id(tokenizer, letter: str) -> int:
    """
    Find a single-token representation for a letter.
    Try ' A' first (common for LLaMA), then 'A'.
    """
    # Try with leading space (common SPM pattern)
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # Fallback: bare letter
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    raise ValueError(f"Could not find a single-token encoding for {letter}: got {ids}")


def load_mcq_results_data(mcq_results_path, log_file_path=None):
    """
    Load MCQ results data from JSON or JSONL file.
    Returns a dictionary mapping question ID or question text to result data.
    """
    if mcq_results_path is None:
        print("mcq_results_path is None")
        return None
    
    results_lookup = {}
    
    try:
        # Try different encodings in case of BOM or encoding issues
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        file_handle = None
        
        for encoding in encodings:
            try:
                file_handle = open(mcq_results_path, 'r', encoding=encoding)
                # Read first few bytes to check format
                first_bytes = file_handle.read(1024)
                file_handle.seek(0)
                
                # Strip BOM if present
                if first_bytes.startswith('\ufeff'):
                    first_bytes = first_bytes[1:]
                    file_handle.seek(1)
                
                first_char = first_bytes.strip()[0] if first_bytes.strip() else ''
                break
            except Exception as e:
                if file_handle:
                    file_handle.close()
                continue
        
        if file_handle is None:
            raise IOError(f"Could not open file with any encoding: {encodings}")
        
        try:
            if first_char == '{':
                # Likely JSON format - try to parse as single JSON object
                print("Attempting to parse as JSON format...")
                try:
                    data = json.load(file_handle)
                    if isinstance(data, dict) and "results" in data:
                        # JSON format with results dictionary
                        print(f"Found 'results' dictionary with {len(data['results'])} entries")
                        for qid, result in data["results"].items():
                            question_data = result.get("question", {})
                            question_id = question_data.get("id", qid)
                            question_text = question_data.get("question", "")
                            
                            # Store by ID and by question text for lookup
                            # Extract options from question_data if available
                            options = question_data.get("options", {})
                            
                            if question_id:
                                results_lookup[question_id] = {
                                    "subject_answer": result.get("subject_answer"),
                                    "probs": result.get("probs", {}),
                                    "options": options
                                }
                            if question_text:
                                results_lookup[question_text] = {
                                    "subject_answer": result.get("subject_answer"),
                                    "probs": result.get("probs", {}),
                                    "options": options
                                }
                    else:
                        # Single result object, not wrapped in "results"
                        question_data = data.get("question", {})
                        question_id = question_data.get("id")
                        question_text = question_data.get("question", "")
                        
                        if question_id:
                            results_lookup[question_id] = {
                                "subject_answer": data.get("subject_answer"),
                                "probs": data.get("probs", {})
                            }
                        if question_text:
                            results_lookup[question_text] = {
                                "subject_answer": data.get("subject_answer"),
                                "probs": data.get("probs", {})
                            }
                except json.JSONDecodeError as json_err:
                    print(f"JSON parsing error: {json_err}")
                    error_pos = json_err.pos if hasattr(json_err, 'pos') else None
                    print(f"Error at position {error_pos}")
                    
                    # Try to parse the results dictionary manually by extracting entries
                    print("Attempting to extract results entries manually...")
                    file_handle.seek(0)
                    content = file_handle.read()
                    
                    # Try to find and parse the "results" dictionary
                    # Look for the pattern: "results": {
                    results_start = content.find('"results": {')
                    if results_start != -1:
                        # Find the opening brace of results
                        brace_start = content.find(
                            '{', results_start + len('"results": '))
                        if brace_start != -1:
                            # Try to extract individual entries
                            # Pattern: "qid": { ... }
                            # Match pattern: "key": { ... } where key is a question ID
                            # This regex matches a quoted key followed by a JSON object
                            pattern = r'"([^"]+)":\s*\{'
                            matches = list(re.finditer(pattern, content[brace_start:]))
                            
                            print(f"Found {len(matches)} potential result entries")
                            
                            # Try to extract each entry
                            for i, match in enumerate(matches):
                                entry_start = brace_start + match.end() - 1  # -1 to include the {
                                # Find the matching closing brace
                                brace_count = 0
                                entry_end = entry_start
                                for j, char in enumerate(content[entry_start:], start=entry_start):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            entry_end = j + 1
                                            break
                                
                                if entry_end > entry_start:
                                    try:
                                        entry_json = content[entry_start:entry_end]
                                        result = json.loads(entry_json)
                                        
                                        # Ensure result is a dictionary
                                        if not isinstance(result, dict):
                                            continue
                                        
                                        question_data = result.get("question", {})
                                        
                                        # Ensure question_data is a dictionary
                                        if not isinstance(question_data, dict):
                                            # If question_data is not a dict, try to use the match group as question_id
                                            question_id = match.group(1)
                                            question_text = ""
                                        else:
                                            question_id = question_data.get("id", match.group(1))
                                            question_text = question_data.get("question", "")
                                        
                                        if question_id:
                                            results_lookup[question_id] = {
                                                "subject_answer": result.get("subject_answer"),
                                                "probs": result.get("probs", {})
                                            }
                                        if question_text:
                                            results_lookup[question_text] = {
                                                "subject_answer": result.get("subject_answer"),
                                                "probs": result.get("probs", {})
                                            }
                                    except (json.JSONDecodeError, AttributeError, TypeError) as e:
                                        # Skip this entry if it can't be parsed or has wrong structure
                                        continue
                                
                                # Progress update
                                if (i + 1) % 1000 == 0:
                                    print(f"Processed {i + 1} entries, found {len(results_lookup)} valid results so far...")
                            
                            if len(results_lookup) > 0:
                                print(f"Successfully extracted {len(results_lookup)} results from malformed JSON")
                            else:
                                print("Could not extract any results from malformed JSON")
                                # Fall back to JSONL attempt
                                print("Attempting to parse as JSONL format...")
                                file_handle.seek(0)
                                line_num = 0
                                for line in file_handle:
                                    line_num += 1
                                    if line.strip():
                                        try:
                                            result = json.loads(line)
                                            question_data = result.get("question", {})
                                            question_id = question_data.get("id")
                                            question_text = question_data.get("question", "")
                                            
                                            if question_id:
                                                results_lookup[question_id] = {
                                                    "subject_answer": result.get("subject_answer"),
                                                    "probs": result.get("probs", {})
                                                }
                                            if question_text:
                                                results_lookup[question_text] = {
                                                    "subject_answer": result.get("subject_answer"),
                                                    "probs": result.get("probs", {})
                                                }
                                        except json.JSONDecodeError as line_err:
                                            if line_num <= 5:  # Only print first few errors
                                                print(f"Warning: Failed to parse line {line_num}: {line_err}")
                        else:
                            print("Could not find results dictionary in file")
                    else:
                        print("Could not find 'results' key in file")
                        # Fall back to JSONL attempt
                        print("Attempting to parse as JSONL format...")
                        file_handle.seek(0)
                        line_num = 0
                        for line in file_handle:
                            line_num += 1
                            if line.strip():
                                try:
                                    result = json.loads(line)
                                    question_data = result.get("question", {})
                                    question_id = question_data.get("id")
                                    question_text = question_data.get("question", "")
                                    
                                    if question_id:
                                        results_lookup[question_id] = {
                                            "subject_answer": result.get("subject_answer"),
                                            "probs": result.get("probs", {})
                                        }
                                    if question_text:
                                        results_lookup[question_text] = {
                                            "subject_answer": result.get("subject_answer"),
                                            "probs": result.get("probs", {})
                                        }
                                except json.JSONDecodeError as line_err:
                                    if line_num <= 5:  # Only print first few errors
                                        print(f"Warning: Failed to parse line {line_num}: {line_err}")
            else:
                # Likely JSONL format - one JSON object per line
                print("Attempting to parse as JSONL format...")
                line_num = 0
                for line in file_handle:
                    line_num += 1
                    if line.strip():
                        try:
                            result = json.loads(line)
                            question_data = result.get("question", {})
                            question_id = question_data.get("id")
                            question_text = question_data.get("question", "")
                            
                            # Extract options from question_data if available
                            options = question_data.get("options", {})
                            
                            if question_id:
                                results_lookup[question_id] = {
                                    "subject_answer": result.get("subject_answer"),
                                    "probs": result.get("probs", {}),
                                    "options": options
                                }
                            if question_text:
                                results_lookup[question_text] = {
                                    "subject_answer": result.get("subject_answer"),
                                    "probs": result.get("probs", {}),
                                    "options": options
                                }
                        except json.JSONDecodeError as line_err:
                            if line_num <= 5:  # Only print first few errors
                                print(f"Warning: Failed to parse line {line_num}: {line_err}")
        finally:
            file_handle.close()
            
    except Exception as e:
        print(f"Warning: Failed to load MCQ results data from "
              f"{mcq_results_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None
    
    print(f"Loaded {len(results_lookup)} MCQ results entries")
    if log_file_path:
        write_log(log_file_path, {"message": f"Loaded {len(results_lookup)} MCQ results entries"})

    return results_lookup


def compute_ABCD_entropy(probs):
    """
    Compute entropy from probability distribution for A, B, C, D options.

    Args:
        probs: tensor, list, or dict - probabilities for A, B, C, D
               If dict, should have keys "A", "B", "C", "D"
               If list/tensor, should be in order [A, B, C, D]

    Returns:
        scalar entropy value
    """
    import torch
    
    # Handle dictionary format (keys: "A", "B", "C", "D")
    if isinstance(probs, dict):
        probs = [
            probs.get("A", 0.0),
            probs.get("B", 0.0),
            probs.get("C", 0.0),
            probs.get("D", 0.0)
        ]

    # Convert to tensor if needed
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)

    # Ensure probabilities sum to 1
    probs = probs / (probs.sum() + 1e-12)

    # Compute entropy (natural logs)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return entropy


def shuffle_options_and_update_correct_letter(row):
    """
    Shuffle the options (A, B, C, D) and update the correct_letter accordingly.
    
    This ensures the correct answer isn't always in position A, preventing
    position bias in the model.
    
    Args:
        row: Dictionary with "options" (dict with keys A, B, C, D) and "correct_letter"
        
    Returns:
        row: Modified row with shuffled options and updated correct_letter
    """
    if "options" not in row or "correct_letter" not in row:
        return row
    
    options = row["options"]
    correct_letter = row["correct_letter"]
    
    # Get the correct answer text
    correct_answer_text = options.get(correct_letter, "")
    
    # Create list of (letter, text) pairs
    option_pairs = [(letter, options[letter]) for letter in "ABCD" if letter in options]
    
    # Shuffle the pairs
    random.shuffle(option_pairs)
    
    # Rebuild options dict with new letter assignments
    new_options = {}
    new_correct_letter = None
    for new_letter, (old_letter, text) in zip("ABCD", option_pairs):
        new_options[new_letter] = text
        if old_letter == correct_letter:
            new_correct_letter = new_letter
    
    # Update row
    row["options"] = new_options
    row["correct_letter"] = new_correct_letter
    
    return row


def normalize_text(s):
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


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
    
    # Try QID match
    if qid is not None and str(qid) in mcq_results_lookup:
        result_data = mcq_results_lookup[str(qid)]
    
    # Try Text match if QID failed
    if result_data is None:
        norm_text = normalize_text(batch_question)
        if norm_text in mcq_results_lookup:
            result_data = mcq_results_lookup[norm_text]

    # Nothing found → fallback
    if result_data is None:
        if log_file_path:
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
        write_log(log_file_path, {
            "type": "verification_passed",
            "qid": qid,
            "question_snippet": batch_question[:50],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return result_data, rec_opts


def verify_model_answer_match(pred_probs, result_data, qid=None,
                               log_file_path=None):
    """
    Check whether the model's predicted answer matches pre-recorded answer.

    Args:
        pred_probs: tensor of shape [4] - probs for A,B,C,D in order
        result_data: dict containing "subject_answer"
        qid: question ID
        log_file_path: path for write_log()
    """
    if result_data is None:
        return

    rec_ans = result_data.get("subject_answer")
    if rec_ans is None:
        return

    # model's predicted answer letter
    pred_idx = pred_probs.argmax().item()
    pred_letter = "ABCD"[pred_idx]

    # match?
    matched = (pred_letter == rec_ans)

    if log_file_path:
        write_log(log_file_path, {
            "type": ("model_answer_match" if matched
                     else "model_answer_mismatch"),
            "qid": qid,
            "predicted_answer": pred_letter,
            "recorded_answer": rec_ans,
            "predicted_probs": pred_probs.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


# ============================================================
# Weights & Biases and HuggingFace Hub utilities
# ============================================================


def init_wandb(project, run_name=None, config=None, tags=None, notes=None,
               script_path=None):
    """
    Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        run_name: W&B run name (auto-generated if None)
        config: Dictionary of configuration parameters
        tags: List of tags for the run
        notes: Notes/description for the run
        script_path: Path to training script to save for reproducibility
    """
    try:
        import wandb
        wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags if tags else None,
            notes=notes if notes else None,
        )
        if script_path:
            wandb.save(
                script_path,
                base_path=os.path.dirname(os.path.abspath(script_path))
            )
        return wandb
    except ImportError:
        print("Warning: wandb not installed, skipping W&B logging")
        return None


def log_wandb_metrics(metrics, step=None):
    """Log metrics to Weights & Biases."""
    try:
        import wandb
        wandb.log(metrics, step=step)
    except (ImportError, AttributeError):
        pass  # Silently fail if wandb not available


def log_wandb_config(updates, allow_val_change=False):
    """Update W&B config with new values.
    
    Args:
        updates: Dictionary of config values to update
        allow_val_change: If True, allows changing existing config values
    """
    try:
        import wandb
        wandb.config.update(updates, allow_val_change=allow_val_change)
    except (ImportError, AttributeError):
        pass


def log_device_info(device):
    """Log device information to W&B."""
    try:
        import wandb
        import torch
        log_wandb_config({"actual_device": device})
        if device == "cuda" and torch.cuda.is_available():
            log_wandb_config({
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_memory_gb": (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
            })
    except (ImportError, AttributeError):
        pass


def save_hf_checkpoint(model, tokenizer, checkpoint_repo, step,
                       private=False):
    """
    Save checkpoint to HuggingFace Hub.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        checkpoint_repo: Repository name for checkpoint
        step: Training step number
        private: Whether to make the repo private

    Returns:
        True if successful, False otherwise
    """
    try:
        checkpoint_name = f"{checkpoint_repo}-step-{step}"
        model.push_to_hub(
            checkpoint_name,
            private=private,
            commit_message=f"Checkpoint at step {step}"
        )
        tokenizer.push_to_hub(
            checkpoint_name,
            private=private,
            commit_message=f"Tokenizer checkpoint at step {step}"
        )
        log_wandb_metrics({"checkpoint/hf_repo": checkpoint_name}, step=step)
        return True
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to HuggingFace Hub: {e}")
        return False


def save_model_final(model, tokenizer, output_dir, hf_repo=None,
                      hf_private=False, save_wandb_artifact=False):
    """
    Save final model locally and optionally to HuggingFace Hub.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Local directory to save model
        hf_repo: HuggingFace Hub repository name (optional)
        hf_private: Whether to make HF repo private
        save_wandb_artifact: Whether to save as W&B artifact

    Returns:
        True if successful, False otherwise
    """
    # Save locally
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Save as W&B artifact
    if save_wandb_artifact:
        try:
            import wandb
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.name}",
                type="model",
                description=(
                    f"Fine-tuned ECT model with LoRA. "
                    f"Config: {wandb.config}"
                )
            )
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact)
            print(f"Model saved as wandb artifact: {artifact.name}")
        except (ImportError, AttributeError):
            print("Warning: Could not save W&B artifact")

    # Push to HuggingFace Hub
    if hf_repo:
        try:
            model.push_to_hub(hf_repo, private=hf_private)
            tokenizer.push_to_hub(hf_repo, private=hf_private)
            log_wandb_config({"hf_repo": hf_repo})
            print(f"Model and tokenizer pushed to HuggingFace Hub: {hf_repo}")
            return True
        except Exception as e:
            print(f"Warning: Failed to push to HuggingFace Hub: {e}")
            return False

    return True


def finish_wandb():
    """Finish W&B run."""
    try:
        import wandb
        wandb.finish()
    except (ImportError, AttributeError):
        pass


def check_and_clear_gpu_memory(device, min_free_gb=5.0):
    """
    Check GPU memory status and clear cache if needed.
    
    Args:
        device: Device string ("cuda" or "cpu")
        min_free_gb: Minimum free memory in GB to warn about
        
    Returns:
        dict with memory info if CUDA, None otherwise
    """
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                return None
                
            # Clear any cached memory from previous runs
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            free_memory = total_memory - reserved
            
            total_memory_gb = total_memory / (1024**3)
            allocated_gb = allocated / (1024**3)
            reserved_gb = reserved / (1024**3)
            free_memory_gb = free_memory / (1024**3)
            
            print(f"GPU Memory Status (after cache clear):")
            print(f"  Total: {total_memory_gb:.2f} GB")
            print(f"  Allocated: {allocated_gb:.2f} GB")
            print(f"  Reserved: {reserved_gb:.2f} GB")
            print(f"  Free: {free_memory_gb:.2f} GB")
            
            # Warn if memory is low
            if free_memory_gb < min_free_gb:
                print(f"\n⚠️  WARNING: Low GPU memory ({free_memory_gb:.2f} GB free)")
                print("   Llama-3-8B needs ~16GB to load. This may cause out-of-memory errors.")
                print("\n   Solutions:")
                print("   1. Restart Python/Python process to clear reserved memory")
                print("   2. Run: python -c 'import torch; torch.cuda.empty_cache()' in another terminal")
                print("   3. Restart the vast.ai instance to fully clear GPU memory")
                print("   4. Check for zombie processes: ps aux | grep python")
            
            return {
                "total_gb": total_memory_gb,
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
                "free_gb": free_memory_gb
            }
        except Exception:
            return None
    return None


def load_model_with_error_handling(model_name, device):
    """
    Load model with memory-efficient settings and proper error handling.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Loaded model
        
    Raises:
        torch.cuda.OutOfMemoryError: If GPU out of memory with helpful message
    """
    import torch
    from transformers import AutoModelForCausalLM
    
    try:
        # Use low_cpu_mem_usage to reduce peak memory during loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )
        
        # Move to device after loading
        if device == "cuda":
            torch.cuda.empty_cache()
            model = model.to(device)
        else:
            model = model.to(device)
            
        return model
    except torch.cuda.OutOfMemoryError as e:
        print("\n❌ CUDA Out of Memory Error!")
        print("The GPU does not have enough free memory to load the model.")
        print("\nCurrent GPU memory status:")
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - reserved
            print(f"  Total: {total:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {free:.2f} GB")
        print("\n🔍 Check what's using GPU memory:")
        print("   nvidia-smi")
        print("\n💡 Solutions (try in order):")
        print("1. Kill other processes using the GPU:")
        print("   nvidia-smi  # Find PIDs")
        print("   kill <PID>  # Kill processes")
        print("\n2. Try setting expandable_segments to reduce fragmentation:")
        print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("   # Then re-run your script")
        print("\n3. Restart the vast.ai instance to fully clear GPU memory")
        print("\n4. Use a GPU with more memory or reduce model size")
        raise


def setup_tokenizer(model_name):
    """
    Load and configure tokenizer for causal LM.
    
    Args:
        model_name: HuggingFace model name or path
        
    Returns:
        Configured tokenizer
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token if it doesn't exist (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Standard for causal LMs
    return tokenizer


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


def log_prompts_and_responses(step, prompt_log_file_path, answer_logits4, conf_logits8,
                               batch, answer_prompts, confidence_prompts, soft_targets):
    """
    Log prompts and responses for first 2 steps.
    
    Args:
        step: Current training step
        prompt_log_file_path: Path to log file for prompts
        answer_logits4: Logits for MCQ answers [B, 4]
        conf_logits8: Logits for confidence bins [B, 8]
        batch: Batch of question data
        answer_prompts: List of MCQ prompts
        confidence_prompts: List of confidence prompts
        soft_targets: Soft target distributions [B, 8]
    """
    # if step is not None and step < 3 and prompt_log_file_path:
    #     # Compute probabilities and predictions for logging
    #     with torch.no_grad():
    #         # First forward pass: MCQ answer probabilities
    #         mcq_probs = torch.softmax(answer_logits4, dim=-1)  # [B, 4]
    #         mcq_predicted = mcq_probs.argmax(dim=-1)  # [B]
            
    #         # Second forward pass: Confidence bin probabilities
    #         conf_probs = torch.softmax(conf_logits8, dim=-1)  # [B, 8]
    #         conf_predicted = conf_probs.argmax(dim=-1)  # [B]
        
    #     for i in range(len(batch)):
    #         # Convert predictions to letters
    #         mcq_pred_letter = "ABCD"[mcq_predicted[i].item()]
    #         conf_pred_letter = "ABCDEFGH"[conf_predicted[i].item()]
            
    #         log_entry = {
    #             "type": "prompt_and_response_pair",
    #             "step": step,
    #             "batch_index": i,
    #             "qid": batch[i].get("qid"),
    #             "mcq_prompt": answer_prompts[i],
    #             "mcq_response": {
    #                 "logits": answer_logits4[i].cpu().tolist(),
    #                 "probabilities": mcq_probs[i].cpu().tolist(),
    #                 "predicted_answer": mcq_pred_letter,
    #                 "probabilities_dict": {
    #                     "A": float(mcq_probs[i][0].item()),
    #                     "B": float(mcq_probs[i][1].item()),
    #                     "C": float(mcq_probs[i][2].item()),
    #                     "D": float(mcq_probs[i][3].item()),
    #                 }
    #             },
    #             "confidence_prompt": confidence_prompts[i],
    #             "confidence_response": {
    #                 "logits": conf_logits8[i].cpu().tolist(),
    #                 "probabilities": conf_probs[i].cpu().tolist(),
    #                 "predicted_bin": conf_pred_letter,
    #                 "probabilities_dict": {
    #                     "A": float(conf_probs[i][0].item()),
    #                     "B": float(conf_probs[i][1].item()),
    #                     "C": float(conf_probs[i][2].item()),
    #                     "D": float(conf_probs[i][3].item()),
    #                     "E": float(conf_probs[i][4].item()),
    #                     "F": float(conf_probs[i][5].item()),
    #                     "G": float(conf_probs[i][6].item()),
    #                     "H": float(conf_probs[i][7].item()),
    #                 }
    #             },
    #             "soft_targets": soft_targets[i].cpu().tolist(),
    #             "timestamp": datetime.now(timezone.utc).isoformat()
    #         }
    #         write_log(prompt_log_file_path, log_entry)
            
    #         # Also print to console
    #         print(f"\n{'='*80}")
    #         print(f"STEP {step} | BATCH INDEX {i} | QID: {batch[i].get('qid')}")
    #         print(f"{'='*80}")
    #         print(f"\nMCQ PROMPT (First forward pass):")
    #         print(f"{'-'*80}")
    #         print(answer_prompts[i])
    #         print(f"\nMCQ RESPONSE:")
    #         print(f"{'-'*80}")
    #         print(f"  Predicted Answer: {mcq_pred_letter}")
    #         print(f"  Probabilities: A={mcq_probs[i][0]:.4f}, B={mcq_probs[i][1]:.4f}, "
    #               f"C={mcq_probs[i][2]:.4f}, D={mcq_probs[i][3]:.4f}")
    #         print(f"  Logits: {answer_logits4[i].cpu().tolist()}")
    #         print(f"\nCONFIDENCE PROMPT (Second forward pass - separate context):")
    #         print(f"{'-'*80}")
    #         print(confidence_prompts[i])
    #         print(f"\nCONFIDENCE RESPONSE:")
    #         print(f"{'-'*80}")
    #         print(f"  Predicted Bin: {conf_pred_letter}")
    #         conf_bin_labels = ["A: <5%", "B: 5-10%", "C: 10-20%", "D: 20-40%",
    #                           "E: 40-60%", "F: 60-80%", "G: 80-90%", "H: >90%"]
    #         print(f"  Probabilities:")
    #         for j, label in enumerate(conf_bin_labels):
    #             print(f"    {label}: {conf_probs[i][j]:.4f}")
    #         print(f"  Logits: {conf_logits8[i].cpu().tolist()}")
    #         print(f"\nSOFT TARGETS (Training target):")
    #         print(f"{'-'*80}")
    #         print(f"  {soft_targets[i].cpu().tolist()}")
    #         print(f"{'='*80}\n")

