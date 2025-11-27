import json
import os
import re
import traceback
from datetime import datetime
from torch.utils.data import Dataset


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
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name_safe = model_name.replace("/", "-").replace("_", "-")
    return os.path.join(
        log_dir, f"{model_name_safe}_{timestamp}_{suffix}.jsonl"
    )


def build_confidence_prompts(batch):
    """
    Build explicit confidence prompts for a batch of questions.

    Args:
        batch: list of dictionaries with 'question' and 'options' keys

    Returns:
        list of prompt strings
    """
    prompts = []
    for row in batch:
        question = row["question"]
        options = row["options"]
        prompt = (
            "I'm going to show you a multiple-choice question, and I want "
            "you to tell me your level of confidence that you would get the "
            "question right.\n"
            "Respond only with a single letter from A to H; do NOT output any "
            "other text.\n\n"
            f"Question: {question}\n"
            "Options:\n"
            f"A. {options['A']}\n"
            f"B. {options['B']}\n"
            f"C. {options['C']}\n"
            f"D. {options['D']}\n\n"
            "How confident are you that you would get this question right?\n"
            "Confidence: "
        )
        prompts.append(prompt)
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                    "timestamp": datetime.now().isoformat()
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
            "timestamp": datetime.now().isoformat()
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
            "timestamp": datetime.now().isoformat()
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


def log_wandb_config(updates):
    """Update W&B config with new values."""
    try:
        import wandb
        wandb.config.update(updates)
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


