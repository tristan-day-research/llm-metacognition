
# --- repo path bootstrap (so root-level imports like `prompts`,
# `finetune_config` resolve when run from anywhere) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import os
import random
from datetime import datetime, timezone
from torch.utils.data import Dataset
import torch
from argparse import Namespace


from data_handling import (
    load_mcq_results_data,
    collate_fn as data_collate_fn,
    normalize_text
)

def load_tokenizer(args):
    """Load and configure tokenizer for causal LM."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.truncation_side = "right"

    return tokenizer


def prepare_model_and_tokenizer(model, tokenizer):
    """Sync model config with tokenizer."""
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Sync model config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    
    # Ensure padding settings (in case they were changed)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    return model, tokenizer


def load_model_with_lora(args, tokenizer):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    # LoRA
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lcfg)

    return model


def parse_letter_from_model_text(model_text, valid_letters):
    """Extract letter from generated text"""
    if model_text is None:
        return None

    cleaned = model_text.upper().strip()
    if len(cleaned) == 0:
        return None

    # Check first character
    if cleaned[0] in valid_letters:
        return cleaned[0]

    # Check last character  
    if cleaned[-1] in valid_letters:
        return cleaned[-1]

    return None


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


# ============================================================
# Weights & Biases and HuggingFace Hub utilities
# ============================================================

def generate_checkpoint_tag(run_name, run_id, step, timestamp=None):
    """
    Generate a checkpoint tag that maps 1:1 to WandB run.
    
    Format: {run_name}-runid-{run_id}-step-{step}-{timestamp}
    
    Args:
        run_name: WandB run name (human-readable)
        run_id: WandB run ID (unique stable ID)
        step: Training step number (int or string like "final")
        timestamp: Optional timestamp string (YYYYMMDD-HHMMSS format). 
                   If None, generates current timestamp.
    
    Returns:
        Checkpoint tag string
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    
    # Sanitize run_name for use in repo names (replace spaces/special chars with hyphens)
    safe_run_name = run_name.replace(" ", "-").replace("/", "-").replace("_", "-")
    # Remove any consecutive hyphens
    while "--" in safe_run_name:
        safe_run_name = safe_run_name.replace("--", "-")
    # Remove leading/trailing hyphens
    safe_run_name = safe_run_name.strip("-")
    
    # Convert step to string for formatting
    step_str = str(step)
    
    checkpoint_tag = f"{safe_run_name}-runid-{run_id}-step-{step_str}-{timestamp}"
    return checkpoint_tag


def _sanitize_for_hf(name: str) -> str:
    """Convert a human-readable string to an HF-repo-safe slug.

    HF allows alphanumerics, '-', '_', '.'. We collapse any other run of
    characters (spaces, commas, etc.) to a single '-' and strip leading /
    trailing '-' or '.' (which HF disallows at the boundary).
    """
    import re
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    s = s.strip("-.")
    return s or "run"


def build_structured_hf_repo_path(hf_checkpoint_repo, run_name, run_id, step, timestamp):
    """
    Build a structured HF repo path that mirrors the WandB run name so the
    same identifier can be found in both places.

    Format: {username}/{sanitized_wandb_run_name}_step_{step}

    The wandb run name already includes a UTC timestamp prefix (added in
    run_finetuning.py), so the HF repo path inherits that prefix too —
    searching either platform with the timestamp or the descriptive part
    finds the matching artifact.

    Args:
        hf_checkpoint_repo: 'username/anything' — only the username is used.
        run_name: WandB run name (timestamp-prefixed, human-readable).
        run_id: WandB run ID — used only as a fallback if run_name is empty.
        step: training step, int or string like 'final'.
        timestamp: fallback timestamp string if run_name is empty.

    Returns:
        Full HF repo path string, max 96 chars, no spaces.
    """
    if hf_checkpoint_repo and "/" in hf_checkpoint_repo:
        username = hf_checkpoint_repo.split("/")[0]
    else:
        username = "checkpoints"

    step_str = str(step).replace("/", "-").replace(" ", "-")

    base = run_name if run_name else f"{timestamp}_{run_id or 'run'}"
    base_slug = _sanitize_for_hf(base)

    suffix = f"_step_{step_str}"
    max_total = 96
    available_for_base = max_total - len(f"{username}/") - len(suffix)
    if available_for_base < 1:
        # Username + suffix already saturate the budget; degrade gracefully
        # by dropping the slug entirely.
        return f"{username}/{suffix.lstrip('_')}"[:max_total]
    if len(base_slug) > available_for_base:
        base_slug = base_slug[:available_for_base].rstrip("-.")

    return f"{username}/{base_slug}{suffix}"


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


def save_model_final(model, tokenizer, output_dir, hf_repo=None,
                      hf_private=False, save_wandb_artifact=False,
                      wandb_run_name=None, wandb_run_id=None, step=None,
                      final_timestamp=None, keep_local=True):
    """
    Save final model locally and optionally to HuggingFace Hub.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Local directory to save model
        hf_repo: HuggingFace Hub repository name (optional, base name)
        hf_private: Whether to make HF repo private
        save_wandb_artifact: Whether to save as W&B artifact
        wandb_run_name: WandB run name (for checkpoint tagging)
        wandb_run_id: WandB run ID (for checkpoint tagging)
        step: Final training step number (for checkpoint tagging)
        final_timestamp: Optional timestamp for final checkpoint tag (YYYYMMDD-HHMMSS format)

    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save locally
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Verify files were created
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")
        adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
        tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
        
        files_exist = (
            os.path.exists(adapter_config_path) or 
            os.path.exists(adapter_model_path) or
            os.path.exists(os.path.join(output_dir, "adapter_model.bin"))
        ) and os.path.exists(tokenizer_config_path)
        
        if files_exist:
            print(f"✓ Model and tokenizer saved to {output_dir}")
        else:
            print(f"⚠️  Warning: Model save may have failed. Check {output_dir}")
            return False
    except Exception as e:
        print(f"❌ Error saving model to {output_dir}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save as W&B artifact
    if save_wandb_artifact:
        try:
            import wandb
            # Use passed wandb_run_name if available, otherwise try wandb.run.name
            artifact_name = f"model-{wandb_run_name}" if wandb_run_name else f"model-{wandb.run.name if wandb.run else 'unknown'}"
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=(
                    f"Fine-tuned ECT model with LoRA. "
                    f"Step: {step if step is not None else 'unknown'}"
                )
            )
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact)
            print(f"✓ Model saved as wandb artifact: {artifact.name}")
        except (ImportError, AttributeError) as e:
            print(f"⚠️  Warning: Could not save W&B artifact (wandb not available): {e}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save W&B artifact: {e}")
            import traceback
            traceback.print_exc()

    # Push to HuggingFace Hub
    if hf_repo:
        # Generate checkpoint tag if WandB info is available
        final_hf_repo = hf_repo
        if wandb_run_name and wandb_run_id and step is not None:
            checkpoint_tag = generate_checkpoint_tag(
                wandb_run_name, wandb_run_id, step, final_timestamp
            )
            # Build structured repo path that includes run_name and run_id for traceability
            final_hf_repo = build_structured_hf_repo_path(
                hf_repo, wandb_run_name, wandb_run_id, step, final_timestamp
            )
            
            # Log checkpoint identifier to WandB
            # Note: Log without step parameter to avoid step ordering issues
            try:
                import wandb
                wandb.log({"hf_checkpoint_name": checkpoint_tag})
                log_wandb_config({"hf_repo": final_hf_repo})
                # Store in run summary
                wandb.run.summary["final_checkpoint_tag"] = checkpoint_tag
                wandb.run.summary["final_hf_repo"] = final_hf_repo
            except (ImportError, AttributeError):
                pass  # Silently fail if wandb not available
            
            print(f"Final checkpoint tag: {checkpoint_tag}")
        elif wandb_run_name and wandb_run_id:
            # If step is not provided, use "final" as step identifier
            checkpoint_tag = generate_checkpoint_tag(
                wandb_run_name, wandb_run_id, "final", final_timestamp
            )
            # Build structured repo path that includes run_name and run_id for traceability
            final_hf_repo = build_structured_hf_repo_path(
                hf_repo, wandb_run_name, wandb_run_id, "final", final_timestamp
            )
            
            # Log checkpoint identifier to WandB
            try:
                import wandb
                wandb.log({"hf_checkpoint_name": checkpoint_tag})
                log_wandb_config({"hf_repo": final_hf_repo})
                # Store in run summary
                wandb.run.summary["final_checkpoint_tag"] = checkpoint_tag
                wandb.run.summary["final_hf_repo"] = final_hf_repo
            except (ImportError, AttributeError):
                pass
            
            print(f"Final checkpoint tag: {checkpoint_tag}")
        else:
            log_wandb_config({"hf_repo": final_hf_repo})
            print("⚠️  Warning: WandB run info not available, using base repo name without checkpoint tag")
        
        try:
            model.push_to_hub(final_hf_repo, private=hf_private)
            tokenizer.push_to_hub(final_hf_repo, private=hf_private)
            print(f"✓ Model and tokenizer pushed to HuggingFace Hub: {final_hf_repo}")
            push_ok = True
        except Exception as e:
            print(f"⚠️  Warning: Failed to push to HuggingFace Hub: {e}")
            import traceback
            traceback.print_exc()
            push_ok = False

        # Clean up local copy if not requested to keep it. Done after the
        # HF push so the local dir is available for the upload.
        if not keep_local:
            _cleanup_local_dir(output_dir, "final model")
        return push_ok

    # No HF push happened — clean up here if requested.
    if not keep_local:
        _cleanup_local_dir(output_dir, "final model")
    return True


def _cleanup_local_dir(path, label):
    """Remove a local checkpoint dir after it's been uploaded elsewhere."""
    import shutil
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"✓ Removed local {label} dir (KEEP_LOCAL_CHECKPOINTS=False): {path}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to remove local {label} dir {path}: {e}")


def save_training_parameters(args, checkpoint_base_dir):
    """
    Save all training parameters/arguments to a JSON file in the checkpoint directory.
    
    Args:
        args: argparse.Namespace or similar object with training parameters
        checkpoint_base_dir: Directory where parameters should be saved
    """
    params_file = os.path.join(checkpoint_base_dir, "training_parameters.json")
    
    # Convert args to dictionary, handling non-serializable values
    params_dict = {}
    if isinstance(args, Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        args_dict = args.__dict__ if hasattr(args, '__dict__') else {}
    
    # Convert to JSON-serializable format
    for key, value in args_dict.items():
        # Skip private attributes and non-serializable objects
        if key.startswith('_'):
            continue
        try:
            # Try to serialize the value
            json.dumps(value)
            params_dict[key] = value
        except (TypeError, ValueError):
            # Convert non-serializable values to strings
            params_dict[key] = str(value)
    
    # Add timestamp
    params_dict['_saved_at'] = datetime.now(timezone.utc).isoformat()
    
    try:
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Training parameters saved to: {os.path.abspath(params_file)}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save training parameters: {e}")


def save_checkpoint(model, tokenizer, checkpoint_base_dir, step,
                    save_hf_checkpoints=False, hf_checkpoint_repo=None,
                    hf_checkpoint_private=False, wandb_run_name=None,
                    wandb_run_id=None, checkpoint_timestamp=None,
                    keep_local=True):
    """
    Save a training checkpoint locally and optionally to HuggingFace Hub.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        checkpoint_base_dir: Base directory for checkpoints (e.g., 'local_checkpoints/2025-01-01-12-00-00_checkpoints')
        step: Training step number
        save_hf_checkpoints: Whether to push to HuggingFace Hub
        hf_checkpoint_repo: Base HF repo name for checkpoints (e.g., 'username/model-name')
        hf_checkpoint_private: Whether to make HF checkpoint repos private
        wandb_run_name: WandB run name (for checkpoint tagging)
        wandb_run_id: WandB run ID (for checkpoint tagging)
        checkpoint_timestamp: Optional timestamp for checkpoint tag (YYYYMMDD-HHMMSS format)
    
    Returns:
        True if successful, False otherwise
    """
    ckpt_dir = os.path.join(checkpoint_base_dir, f"ckpt_step_{step}")
    ckpt_dir_abs = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving checkpoint at step {step}")
    print(f"Local path: {ckpt_dir_abs}")
    print(f"{'='*60}")
    
    try:
        # Save locally
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"✓ Checkpoint saved locally to: {ckpt_dir_abs}")
        
        # Optionally push to HuggingFace Hub
        if save_hf_checkpoints:
            # Generate checkpoint tag if WandB info is available
            if wandb_run_name and wandb_run_id:
                checkpoint_tag = generate_checkpoint_tag(
                    wandb_run_name, wandb_run_id, step, checkpoint_timestamp
                )
                
                # Build structured repo path that includes run_name and run_id for traceability
                if hf_checkpoint_repo:
                    hf_ckpt_repo = build_structured_hf_repo_path(
                        hf_checkpoint_repo, wandb_run_name, wandb_run_id, step, checkpoint_timestamp
                    )
                else:
                    # If no base repo provided, use a default structure
                    timestamp_clean = checkpoint_timestamp.replace("-", "_") if checkpoint_timestamp else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    hf_ckpt_repo = f"checkpoints/ect_{timestamp_clean}_{wandb_run_id}_step_{step}"
                    print("⚠️  Warning: --hf_checkpoint_repo not specified, using default path structure")
                
                # Log checkpoint identifier to WandB
                # Note: Log without step parameter to avoid step ordering issues
                # Checkpoint names are metadata, not step-aligned metrics
                try:
                    import wandb
                    wandb.log({"hf_checkpoint_name": checkpoint_tag})
                    # Store checkpoint info in summary with step-specific key
                    wandb.run.summary[f"checkpoint_step_{step}"] = checkpoint_tag
                    wandb.run.summary[f"checkpoint_hf_repo_step_{step}"] = hf_ckpt_repo
                except (ImportError, AttributeError):
                    pass  # Silently fail if wandb not available
                
                print(f"Pushing checkpoint to HuggingFace Hub: {hf_ckpt_repo}")
                print(f"Checkpoint tag: {checkpoint_tag}")
                try:
                    model.push_to_hub(hf_ckpt_repo, private=hf_checkpoint_private)
                    tokenizer.push_to_hub(hf_ckpt_repo, private=hf_checkpoint_private)
                    print(f"✓ Checkpoint pushed to HuggingFace Hub: {hf_ckpt_repo}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to push checkpoint to HF Hub: {e}")
                    import traceback
                    traceback.print_exc()
            elif hf_checkpoint_repo:
                # Fallback to old format if WandB info not available
                hf_ckpt_repo = f"{hf_checkpoint_repo}-step-{step}"
                print(f"Pushing checkpoint to HuggingFace Hub: {hf_ckpt_repo}")
                print("⚠️  Warning: WandB run info not available, using fallback naming")
                try:
                    model.push_to_hub(hf_ckpt_repo, private=hf_checkpoint_private)
                    tokenizer.push_to_hub(hf_ckpt_repo, private=hf_checkpoint_private)
                    print(f"✓ Checkpoint pushed to HuggingFace Hub: {hf_ckpt_repo}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to push checkpoint to HF Hub: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️  Warning: --save_hf_checkpoints is set but --hf_checkpoint_repo is not specified and WandB info is not available. Skipping HF Hub upload.")

        # Clean up local copy if not requested to keep it. Done last so the
        # HF push above can read from the dir.
        if not keep_local:
            _cleanup_local_dir(ckpt_dir, f"step-{step} checkpoint")

        return True

    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def finish_wandb():
    """Finish W&B run."""
    try:
        import wandb
        wandb.finish()
    except (ImportError, AttributeError):
        pass


# ============================================================
# Data Leakage Prevention - Defensive Checks
# ============================================================

def validate_train_batch(batch, train_dataset_qids=None, train_dataset_questions=None, function_name="train_step"):
    """
    Defensive check: Verify batch contains only questions from train_dataset.
    Check both qid and question text to catch any leakage.
    
    Args:
        batch: Batch from train_dataset (list of question dicts)
        train_dataset_qids: Set of qids from train_dataset for validation (optional)
        train_dataset_questions: Set of normalized question texts from train_dataset (optional)
        function_name: Name of the calling function for error messages
    
    Raises:
        ValueError: If any question in batch is not in train_dataset
    """
    if train_dataset_qids is None and train_dataset_questions is None:
        return  # No validation sets provided, skip check
    
    for row in batch:
        qid = row.get("qid")
        question = row.get("question", "")
        
        # Check qid if available
        if train_dataset_qids is not None and qid:
            if str(qid) not in train_dataset_qids:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED: {function_name}() received a question with qid={qid} "
                    f"that is not in train_dataset. This should never happen!"
                )
        
        # Check question text if available
        if train_dataset_questions is not None and question:
            norm_q = normalize_text(question)
            if norm_q not in train_dataset_questions:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED: {function_name}() received a question with text='{question[:50]}...' "
                    f"that is not in train_dataset. This should never happen!"
                )


def validate_eval_dataset(val_dataset, train_dataset_qids=None, train_dataset_questions=None, function_name="run_evaluation"):
    """
    Defensive check: Verify val_dataset doesn't contain any train_dataset questions.
    Check both qid and question text to catch any leakage.
    
    Args:
        val_dataset: Validation or test dataset (NEVER train_dataset)
        train_dataset_qids: Set of qids from train_dataset for validation (optional)
        train_dataset_questions: Set of normalized question texts from train_dataset (optional)
        function_name: Name of the calling function for error messages
    
    Raises:
        ValueError: If any question in val_dataset is also in train_dataset
    """
    if train_dataset_qids is None and train_dataset_questions is None:
        return  # No validation sets provided, skip check
    
    for row in val_dataset:
        qid = row.get("qid")
        question = row.get("question", "")
        
        # Check qid if available
        if train_dataset_qids is not None and qid:
            if str(qid) in train_dataset_qids:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED: {function_name}() received a question with qid={qid} "
                    f"that is in train_dataset. This should never happen!"
                )
        
        # Check question text if available
        if train_dataset_questions is not None and question:
            norm_q = normalize_text(question)
            if norm_q in train_dataset_questions:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED: {function_name}() received a question with text='{question[:50]}...' "
                    f"that is in train_dataset. This should never happen!"
                )
