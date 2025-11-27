import argparse
import json
import math
import os
import re
import torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import wandb


def write_log(log_file_path, entry_dict):
    """Simple one-line logging function. Logs dict as JSON to file if path provided."""
    if log_file_path:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')


def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries."""
    return batch


# ============================================================
# Dataset: no logprobs needed, only the raw MCQ fields
# ============================================================


class MCQDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
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


# ============================================================
# Entropy → scalar confidence → soft labels
# ============================================================

def compute_soft_labels(logits4, sigma=10.0):
    """
    Convert the model's 4-way answer logits into a soft 8-bin confidence
    distribution using the percentage-based Gaussian kernel from your colleague.

    logits4: tensor of shape [4]
    sigma: Gaussian width in percentage space (default: 10)
    """
    # 1. Softmax over the 4 MCQ options
    ps = torch.softmax(logits4, dim=0)

    # 2. Entropy (natural logs)
    H = -(ps * torch.log(ps + 1e-12)).sum()

    # 3. Convert entropy to "confidence percentage"
    #    confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - H / math.log(4)) * 100.0

    # 4. Bin midpoints + widths (exact values from your colleague)
    bin_edges = torch.tensor([0, 5, 10, 20, 40, 60, 80, 90, 100],
                             dtype=torch.float32,
                             device=logits4.device)
    bin_midpoints = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95],
                                 dtype=torch.float32,
                                 device=logits4.device)
    bin_widths = bin_edges[1:] - bin_edges[:-1]   # shape [8]

    # 5. Gaussian kernel in percentage space
    distances = (bin_midpoints - confidence_percent)**2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    return weights / weights.sum()


def compute_entropy(probs):
    """
    Compute entropy from probability distribution.
    
    probs: tensor, list, or dict - probabilities for A, B, C, D
           If dict, should have keys "A", "B", "C", "D"
           If list/tensor, should be in order [A, B, C, D]
    Returns: scalar entropy value
    """
    # Handle dictionary format (keys: "A", "B", "C", "D")
    if isinstance(probs, dict):
        probs = [probs.get("A", 0.0), probs.get("B", 0.0), probs.get("C", 0.0), probs.get("D", 0.0)]
    
    # Convert to tensor if needed
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)
    
    # Ensure probabilities sum to 1
    probs = probs / (probs.sum() + 1e-12)
    
    # Compute entropy (natural logs)
    H = -(probs * torch.log(probs + 1e-12)).sum()
    return H


def convert_entropy_to_soft_labels(entropy, sigma=10.0):
    """
    Convert entropy value to soft 8-bin confidence distribution.
    
    entropy: scalar entropy value
    sigma: Gaussian width in percentage space (default: 10)
    Returns: tensor of shape [8] with soft label distribution
    """
    # Convert entropy to "confidence percentage"
    # confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0
    
    # Bin midpoints + widths (same as compute_soft_labels)
    bin_edges = torch.tensor([0, 5, 10, 20, 40, 60, 80, 90, 100],
                             dtype=torch.float32)
    bin_midpoints = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95],
                                 dtype=torch.float32)
    bin_widths = bin_edges[1:] - bin_edges[:-1]   # shape [8]
    
    # Gaussian kernel in percentage space
    distances = (bin_midpoints - confidence_percent)**2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths
    
    return weights / weights.sum()


def build_confidence_prompts(batch):
    """
    Build explicit confidence prompts for a batch of questions.
    
    batch: list of dictionaries with 'question' and 'options' keys
    Returns: list of prompt strings
    """
    prompts = []
    for row in batch:
        q = row["question"]
        opts = row["options"]
        p = (
            "I'm going to show you a multiple-choice question, and I want you to tell me "
            "your level of confidence that you would get the question right.\n"
            "Respond only with a single letter from A to H; do NOT output any other text.\n\n"
            f"Question: {q}\n"
            "Options:\n"
            f"A. {opts['A']}\n"
            f"B. {opts['B']}\n"
            f"C. {opts['C']}\n"
            f"D. {opts['D']}\n\n"
            "How confident are you that you would get this question right?\n"
            "Confidence: "
        )
        prompts.append(p)
    return prompts


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
                        brace_start = content.find('{', results_start + len('"results": '))
                        if brace_start != -1:
                            # Try to extract individual entries
                            # Pattern: "qid": { ... }
                            import re
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
        import traceback
        print(f"Warning: Failed to load MCQ results data from {mcq_results_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None
    
    print(f"Loaded {len(results_lookup)} MCQ results entries") or (log_file_path and write_log(log_file_path, {"message": f"Loaded {len(results_lookup)} MCQ results entries"}))

    
    return results_lookup

# def load_mcq_results_data(mcq_results_path):
#     """
#     Load MCQ results data. 
#     FIX: Now explicitly stores 'question_text' to allow string comparison later.
#     """
#     if mcq_results_path is None:
#         return None
    
#     results_lookup = {}
    
#     try:
#         with open(mcq_results_path, 'r', encoding='utf-8') as f:
#             content = f.read().strip()
            
#             # --- Parsing Strategy ---
#             data_entries = []
            
#             # 1. Try JSONL (Line-delimited)
#             if content.startswith('{') and not content.startswith('{"results":'):
#                  for line in content.split('\n'):
#                     if line.strip():
#                         try:
#                             data_entries.append(json.loads(line))
#                         except: pass
            
#             # 2. Try Standard JSON with "results" key
#             else:
#                 try:
#                     full_data = json.loads(content)
#                     if "results" in full_data:
#                         # Convert dict items to list of values
#                         data_entries = list(full_data["results"].values())
#                     elif "question" in full_data: 
#                         # Single object
#                         data_entries = [full_data]
#                 except: pass

#             # --- Populating Lookup ---
#             for res in data_entries:
#                 # Handle both nested "question" dict and flat structures
#                 q_block = res.get("question", {})
                
#                 # Get ID
#                 qid = q_block.get("id") or res.get("id")
#                 if not qid: continue # Skip if no ID
                
#                 # Get Options
#                 options = q_block.get("options") or res.get("options")
                
#                 # Get Text
#                 q_text = q_block.get("question") or res.get("question")
                
#                 # Store in lookup
#                 entry = {
#                     "subject_answer": res.get("subject_answer"),
#                     "probs": res.get("probs", {}),
#                     "options": options,       # Top-level for easy access
#                     "question_text": q_text,  # CRITICAL for verification
#                 }
                
#                 results_lookup[str(qid)] = entry
                
#                 # Also index by text if available
#                 if q_text:
#                     results_lookup[normalize_text(q_text)] = entry

#     except Exception as e:
#         print(f"Warning: Failed to load MCQ results: {e}")
#         return None
    
#     print(f"Loaded {len(results_lookup)} MCQ results entries")
#     return results_lookup



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
    FIX: Checks top-level keys created by the loader above.
    """
    qid = row.get("qid")
    batch_question = row["question"]
    batch_opts = row["options"] # These are the raw/shuffled ones from dataset

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

def verify_model_answer_match(pred_probs, result_data, qid=None, log_file_path=None):
    """
    Check whether the model's predicted answer (A/B/C/D) matches the 
    pre-recorded subject_answer from the MCQ results.

    Args:
        pred_probs: tensor of shape [4] — probs for A,B,C,D in order
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
            "type": "model_answer_match" if matched else "model_answer_mismatch",
            "qid": qid,
            "predicted_answer": pred_letter,
            "recorded_answer": rec_ans,
            "predicted_probs": pred_probs.tolist(),
            "timestamp": datetime.now().isoformat()
        })



# ------------------------------------------------------------------
# Training Step
# ------------------------------------------------------------------


def train_step(
    model,
    tokenizer,
    batch,
    sigma=10.0,
    device="cuda",
    mcq_results_lookup=None,
    log_file_path=None,
    args=None
):
    """
    Single training step for Explicit Confidence Task.
    Supports two modes:
        - args.use_recorded_responses == True:
            Use recorded MCQ results (frozen teacher) as target.
        - args.use_recorded_responses == False:
            Use live logits from current model (dynamic teacher).
    """

    B = len(batch)

    mc_setup_prompt = (
        "I'm going to ask you a series of multiple-choice questions. For each one, "
        "select the answer you think is best. Respond only with the letter of your choice; "
        "do NOT output any other text."
    )

    answer_prompts = []
    resolved_results = []  # store recorded MCQ results for training targets

    # ------------------------------------------------------------------
    # 1. Build MCQ prompts + verify question/choices
    # ------------------------------------------------------------------
    for row in batch:
        result_data, opts = verify_and_resolve_options(
            row,
            mcq_results_lookup,
            log_file_path
        )
        row["options"] = opts
        resolved_results.append(result_data)  # save for later target lookup

        p = (
            mc_setup_prompt + "\n\n"
            "Question:\n"
            f"{row['question']}\n"
            f"A: {opts['A']}\n"
            f"B: {opts['B']}\n"
            f"C: {opts['C']}\n"
            f"D: {opts['D']}\n"
            "Your choice (A, B, C, or D): "
        )
        answer_prompts.append(p)

    # ------------------------------------------------------------------
    # 2. First forward pass (NO GRAD)
    # ------------------------------------------------------------------
    enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model(**enc)

        # last token logits
        final_logits = out.logits[:, -1, :]  # [B, vocab]

        # extract A/B/C/D logits
        abcd_ids = torch.tensor(
            [get_single_token_id(tokenizer, c) for c in "ABCD"],
            device=device,
            dtype=torch.long
        )

        answer_logits4 = final_logits[:, abcd_ids]  # [B, 4]
        answer_probs = torch.softmax(answer_logits4, dim=-1)

    # ------------------------------------------------------------------
    # 3. Compute soft targets (frozen teacher or dynamic)
    # ------------------------------------------------------------------
    soft_targets_list = []

    for i, row in enumerate(batch):

        if args.use_recorded_responses:
            # ------------------------------
            # FROZEN TEACHER MODE (default)
            # ------------------------------
            qid = str(row.get("qid"))
            rd = resolved_results[i]

            if rd is None:
                # Fallback to dynamic teacher mode when recorded data is missing
                error_msg = f"No recorded MCQ results for qid: {qid}, using dynamic teacher fallback"
                print(error_msg)
                if log_file_path:
                    write_log(log_file_path, {"error": error_msg, "qid": qid})
                # Use current model's logits as fallback
                soft_target = compute_soft_labels(answer_logits4[i], sigma=sigma).to(device)
            else:
                # The pre-recorded file may store probabilities in different formats:
                # - `predicted_probs` or `answer_probs` (list or dict)
                # - `probs` (dict with keys "A", "B", "C", "D")
                teacher_probs = rd.get("predicted_probs") or rd.get("answer_probs") or rd.get("probs")
                if teacher_probs is None:
                    # Fallback to dynamic teacher mode when probability data is missing
                    error_msg = f"Recorded MCQ results missing probability data for qid {qid}, using dynamic teacher fallback"
                    print(error_msg)
                    if log_file_path:
                        write_log(log_file_path, {"error": error_msg, "qid": qid})
                    # Use current model's logits as fallback
                    soft_target = compute_soft_labels(answer_logits4[i], sigma=sigma).to(device)
                else:
                    entropy = compute_entropy(teacher_probs)
                    soft_target = convert_entropy_to_soft_labels(entropy).to(device)

        else:
            # ------------------------------
            # DYNAMIC TEACHER MODE (old)
            # ------------------------------
            soft_target = compute_soft_labels(answer_logits4[i], sigma=sigma).to(device)

        soft_targets_list.append(soft_target)

    soft_targets = torch.stack(soft_targets_list)  # [B, 8]

    # ------------------------------------------------------------------
    # 4. Build explicit-confidence prompt and run second forward pass
    # ------------------------------------------------------------------
    confidence_prompts = build_confidence_prompts(batch)
    enc2 = tokenizer(confidence_prompts, return_tensors="pt", padding=True).to(device)
    out2 = model(**enc2)

    # Extract logits for confidence bins A–H
    final_logits2 = out2.logits[:, -1, :]
    bins_ids = torch.tensor(
        [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
        device=device,
        dtype=torch.long
    )
    conf_logits8 = final_logits2[:, bins_ids]  # [B,8]

    # ------------------------------------------------------------------
    # 5. Compute loss
    # ------------------------------------------------------------------
    log_probs = torch.log_softmax(conf_logits8, dim=-1)  # [B,8]
    loss = (-soft_targets * log_probs).sum(dim=1).mean()

    return loss



# def train_step(model, tokenizer, batch, sigma=10.0, device="cuda", mcq_results_lookup=None, log_file_path=None):

#     B = len(batch)

#     mc_setup_prompt = (
#         "I'm going to ask you a series of multiple-choice questions. For each one, "
#         "select the answer you think is best. Respond only with the letter of your choice; "
#         "do NOT output any other text."
#     )

#     answer_prompts = []
#     resolved_results = []   # <-- store (result_data, opts) for use later

#     # ------------------------------------------------------------------
#     # 1. Build prompts and verify question/option match
#     # ------------------------------------------------------------------
#     for row in batch:
#         result_data, opts = verify_and_resolve_options(
#             row,
#             mcq_results_lookup,
#             log_file_path
#         )
#         resolved_results.append(result_data)   # <-- save it

#         p = (
#             mc_setup_prompt + "\n\n"
#             "Question:\n"
#             f"{row['question']}\n"
#             f"A: {opts['A']}\n"
#             f"B: {opts['B']}\n"
#             f"C: {opts['C']}\n"
#             f"D: {opts['D']}\n"
#             "Your choice (A, B, C, or D): "
#         )
#         answer_prompts.append(p)

#     # ------------------------------------------------------------------
#     # 2. First forward pass (NO GRAD)
#     # ------------------------------------------------------------------
#     enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)

#     with torch.no_grad():
#         out = model(**enc)

#         final_logits = out.logits[:, -1, :]     # [B, vocab]

#         abcd_ids = torch.tensor(
#             [get_single_token_id(tokenizer, c) for c in "ABCD"],
#             device=device,
#             dtype=torch.long
#         )

#         answer_logits4 = final_logits[:, abcd_ids]     # [B,4]
#         answer_probs = torch.softmax(answer_logits4, dim=-1)

#         soft_targets = torch.stack(
#             [compute_soft_labels(ans, sigma=sigma) for ans in answer_logits4]
#         ).to(device)  # [B,8]

#     # ------------------------------------------------------------------
#     # 3. NEW: verify model’s new answers vs pre-recorded answers
#     # ------------------------------------------------------------------
#     for i, row in enumerate(batch):
#         pred_probs_i = answer_probs[i]
#         result_data_i = resolved_results[i]    # <-- correct source

#         verify_model_answer_match(
#             pred_probs_i,
#             result_data=result_data_i,
#             qid=row.get("qid"),
#             log_file_path=log_file_path
#         )




#     # ------------------------------------------------------------------
#     # 2. Second pass: explicit confidence, independent prompt
#     # ------------------------------------------------------------------

#     conf_prompts = []
#     for row in batch:
#         q = row["question"]
#         opts = row["options"]
#         p = (
#             "I'm going to show you a multiple-choice question, and I want you to tell me "
#             "your level of confidence that you would get the question right.\n"
#             "Respond only with a single letter from A to H; do NOT output any other text.\n\n"
#             f"Question: {q}\n"
#             "Options:\n"
#             f"A. {opts['A']}\n"
#             f"B. {opts['B']}\n"
#             f"C. {opts['C']}\n"
#             f"D. {opts['D']}\n\n"
#             "How confident are you that you would get this question right?\n"
#             "Confidence: "
#         )
#         conf_prompts.append(p)

#     enc2 = tokenizer(conf_prompts, return_tensors="pt", padding=True).to(device)
#     out2 = model(**enc2)
#     final_logits2 = out2.logits[:, -1, :]  # [B, vocab]

#     # Extract logits for confidence tokens A–H (again robustly)
#     conf_ids = torch.tensor(
#         [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
#         device=device,
#         dtype=torch.long
#     )  # [8]

#     conf_logits = final_logits2[:, conf_ids]  # [B, 8]

#     # Soft-label cross entropy: only second pass gets gradients
#     log_probs = torch.log_softmax(conf_logits, dim=-1)  # [B, 8]
#     loss = -(soft_targets * log_probs).sum(dim=1).mean()

#     return loss

# ============================================================
# Main training
# ============================================================

def train(args):
    # Set up parameters log file immediately at the start
    log_dir = "fine_tune_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name_safe = args.model_name.replace("/", "-").replace("_", "-")
    params_log_file_path = os.path.join(log_dir, f"{model_name_safe}_{timestamp}_parameters.jsonl")
    
    # Log all parameters (including defaults) before any other operations
    all_params = vars(args)
    write_log(params_log_file_path, {
        "type": "script_parameters",
        "timestamp": datetime.now().isoformat(),
        "parameters": all_params
    })
    print(f"All parameters logged to: {params_log_file_path}")
    
    # Debug: Print received arguments
    print(f"DEBUG: mcq_results_data argument value: {repr(args.mcq_results_data)}")
    print(f"DEBUG: mcq_results_data type: {type(args.mcq_results_data)}")
    print(f"DEBUG: mcq_results_data truthiness: {bool(args.mcq_results_data)}")
    if args.mcq_results_data:
        print(f"DEBUG: mcq_results_data length: {len(args.mcq_results_data)}")
    
    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),  # Log all arguments as config
        tags=args.wandb_tags if args.wandb_tags else None,
        notes=args.wandb_notes if args.wandb_notes else None,
    )
    
    # Save the training script to wandb for reproducibility
    script_path = __file__
    wandb.save(script_path, base_path=os.path.dirname(os.path.abspath(script_path)))
    
    # Handle device selection with CPU fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device
    
    # Log device info
    wandb.config.update({"actual_device": device})
    if device == "cuda":
        wandb.config.update({
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        })
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Set pad_token if it doesn't exist (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Standard for causal LMs
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Ensure model config also has pad_token_id set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA configuration
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )
    model = get_peft_model(model, lora).to(device)

    ds = MCQDataset(args.data_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Load MCQ results data if provided
    mcq_results_lookup = None
    # Strip whitespace in case there's any
    if args.mcq_results_data:
        args.mcq_results_data = args.mcq_results_data.strip()
    
    # Set up logging file for question matches/mismatches (create early so we can use it in load_mcq_results_data)
    log_file_path = None
    if args.mcq_results_data:
        log_dir = "fine_tune_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_name_safe = args.model_name.replace("/", "-").replace("_", "-")
        log_file_path = os.path.join(log_dir, f"{model_name_safe}_{timestamp}_question_comparison.jsonl")
    
    if args.mcq_results_data:
        print(f"Loading pre-recorded Multiple Choice Results data from: {args.mcq_results_data}")
        # Check if file exists
        if not os.path.exists(args.mcq_results_data):
            print(f"Warning: MCQ results file not found at: {args.mcq_results_data}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Attempting to find file...")
            # Try to find the file in common locations
            basename = os.path.basename(args.mcq_results_data)
            possible_paths = [
                args.mcq_results_data,
                os.path.join("data", basename),
                os.path.join("explicit_confidence_task_logs", basename),
                os.path.join("capabilities_test_logs", basename),
                os.path.join(os.getcwd(), args.mcq_results_data),
            ]
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found file at: {path}")
                    args.mcq_results_data = path
                    found = True
                    break
            if not found:
                print(f"Error: Could not find MCQ results file. Skipping verification.")
                mcq_results_lookup = None
            else:
                mcq_results_lookup = load_mcq_results_data(args.mcq_results_data, log_file_path)
        else:
            mcq_results_lookup = load_mcq_results_data(args.mcq_results_data, log_file_path)
    else:
        print("No pre-recorded Multiple Choice Results data has been loaded")
    
    # Log file path already created above
    if log_file_path:
        print(f"Question comparison log will be written to: {log_file_path}")
    
    # Log dataset info
    wandb.config.update({
        "dataset_size": len(ds),
        "num_batches": len(dl)
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Log optimizer config
    wandb.config.update({
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate
    })

    for step, batch in enumerate(dl):
        if step >= args.max_steps:
            break
            
        loss = train_step(model, tokenizer, batch, device=device, sigma=args.sigma, mcq_results_lookup=mcq_results_lookup, log_file_path=log_file_path, args=args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to wandb
        log_dict = {
            "train/loss": loss.item(),
            "train/step": step
        }
        wandb.log(log_dict, step=step)

        if step % args.log_interval == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        # Save checkpoint to HuggingFace Hub
        if args.save_hf_checkpoints and args.hf_checkpoint_repo:
            if (step + 1) % args.checkpoint_steps == 0:
                checkpoint_repo = f"{args.hf_checkpoint_repo}-step-{step+1}"
                try:
                    model.push_to_hub(
                        checkpoint_repo,
                        private=args.hf_checkpoint_private,
                        commit_message=f"Checkpoint at step {step+1}"
                    )
                    tokenizer.push_to_hub(
                        checkpoint_repo,
                        private=args.hf_checkpoint_private,
                        commit_message=f"Tokenizer checkpoint at step {step+1}"
                    )
                    print(f"Checkpoint saved to HuggingFace Hub: {checkpoint_repo}")
                    wandb.log({"checkpoint/hf_repo": checkpoint_repo}, step=step)
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint to HuggingFace Hub: {e}")

    # Save model locally
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Save model as wandb artifact for reproducibility
    if args.save_wandb_artifact:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.name}",
            type="model",
            description=f"Fine-tuned ECT model with LoRA. Config: {wandb.config}"
        )
        artifact.add_dir(args.output_dir)
        wandb.log_artifact(artifact)
        print(f"Model saved as wandb artifact: {artifact.name}")
    
    # Optionally push to HuggingFace Hub
    if args.save_hf:
        if args.hf_repo is None:
            raise ValueError("--hf_repo must be provided when --save_hf is set")
        model.push_to_hub(args.hf_repo, private=args.hf_checkpoint_private)
        tokenizer.push_to_hub(args.hf_repo, private=args.hf_checkpoint_private)
        wandb.config.update({"hf_repo": args.hf_repo})
        print(f"Model and tokenizer pushed to HuggingFace Hub: {args.hf_repo}")
    
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train dynamic metacognition model (Explicit Confidence Task)"
    )

    # -----------------------
    # Model
    # -----------------------
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HF model name or path")
    parser.add_argument("--device", type=str,
                        default="cuda",
                        choices=["cuda", "cpu"],
                        help="Compute device")

    # -----------------------
    # Data
    # -----------------------
    parser.add_argument("--data_path", type=str,
                        default="questions.jsonl",
                        help="Path to JSONL MCQ dataset")

    parser.add_argument("--batch_size", type=int,
                        default=4,
                        help="Training batch size")

    parser.add_argument("--mcq_results_data", type=str,
                        default=None,
                        help="Path to JSON/JSONL file with previous MCQ results for verification")

    # -----------------------
    # LoRA
    # -----------------------
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Modules to apply LoRA to")

    # -----------------------
    # Training
    # -----------------------
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000000000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Sigma parameter for soft label distribution")
    parser.add_argument("--use_recorded_responses", action="store_true", default=True,
                        help="Use recorded MCQ responses (frozen teacher) as training targets instead of recomputing logits. (Default: True)")
    parser.add_argument("--no_use_recorded_responses", dest="use_recorded_responses", action="store_false",
                        help="Disable using recorded responses, use dynamic teacher (current model logits) instead.")


    # -----------------------
    # Output
    # -----------------------
    parser.add_argument("--output_dir", type=str,
                        default="dynamic_ect_lora",
                        help="Directory to save final model")

    parser.add_argument("--save_hf", action="store_true",
                        help="If set, push LoRA model to HuggingFace Hub")

    parser.add_argument("--hf_repo", type=str,
                        default=None,
                        help="HF repo name if pushing to Hub")

    parser.add_argument("--save_hf_checkpoints", action="store_true",
                        help="If set, save checkpoints to HuggingFace Hub during training")

    parser.add_argument("--hf_checkpoint_repo", type=str,
                        default=None,
                        help="Base HF repo name for checkpoints (e.g., 'username/model-name')")

    parser.add_argument("--checkpoint_steps", type=int,
                        default=500,
                        help="Save checkpoint every N steps")

    parser.add_argument("--hf_checkpoint_private", action="store_true",
                        help="If set, make checkpoint repos private")

    # -----------------------
    # Weights & Biases
    # -----------------------
    parser.add_argument("--wandb_project", type=str,
                        default="llm-metacognition-ect",
                        help="W&B project name")
    
    parser.add_argument("--wandb_run_name", type=str,
                        default=None,
                        help="W&B run name (auto-generated if not provided)")
    
    parser.add_argument("--wandb_tags", type=str, nargs="+",
                        default=None,
                        help="Tags for W&B run")
    
    parser.add_argument("--wandb_notes", type=str,
                        default=None,
                        help="Notes/description for W&B run")
    
    parser.add_argument("--save_wandb_artifact", action="store_true",
                        help="Save model as W&B artifact for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
