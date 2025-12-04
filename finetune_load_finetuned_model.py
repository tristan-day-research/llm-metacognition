"""
finetuned_model_loader.py

Reusable module for loading Tristan's LoRA-finetuned Llama models.

This module provides clean, importable functions:

    from finetuned_model_loader import load_finetuned_model
    model, tokenizer = load_finetuned_model(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        lora_repo="Tristan-Day/llama_3.1_finetuned",
        merge=True
    )

It removes dataset loading or CLI behavior. Pure utility.
"""

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

###########################################################
# Load base + LoRA adapter
###########################################################
def load_finetuned_model(base_model: str, lora_repo: str, merge: bool = False):
    """
    Load a base Llama model and apply LoRA adapter.
    
    Supports both local paths and HuggingFace repositories:
    - Local path: e.g., "local_checkpoints/2025-12-03-22-26-59_checkpoints/ckpt_step_1280"
    - HF repo: e.g., "Tristan-Day/llama_3.1_finetuned"
    - Auto-detects by checking if path exists locally first

    Args:
        base_model: str — e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
        lora_repo:  str — Local path or HuggingFace repository containing LoRA checkpoints
        merge:      bool — if True, merges LoRA weights into base model

    Returns:
        model: torch.nn.Module — ready for inference
        tokenizer: transformers.AutoTokenizer
    """
    # Check if lora_repo is a local path
    lora_path = Path(lora_repo)
    is_local = lora_path.exists() and lora_path.is_dir()
    
    # If not found as direct path, try searching in common checkpoint locations
    if not is_local:
        # Try in local_checkpoints directory
        possible_paths = [
            lora_path,  # Original path
            Path("local_checkpoints") / lora_repo,
            Path("local_checkpoints") / f"{lora_repo}_checkpoints",
        ]
        
        # Also try if lora_repo looks like a checkpoint name, search for it
        if "step" in lora_repo.lower() or "ckpt" in lora_repo.lower():
            # Search in local_checkpoints for matching directories
            local_checkpoints_dir = Path("local_checkpoints")
            if local_checkpoints_dir.exists():
                for checkpoint_dir in local_checkpoints_dir.iterdir():
                    if checkpoint_dir.is_dir():
                        # Look for ckpt_step_* directories
                        for ckpt_dir in checkpoint_dir.iterdir():
                            if ckpt_dir.is_dir() and lora_repo in str(ckpt_dir):
                                lora_path = ckpt_dir
                                is_local = True
                                break
                        if is_local:
                            break
                        # Also check if the checkpoint name matches the directory name
                        if lora_repo in checkpoint_dir.name:
                            # Look for step directories inside
                            for ckpt_dir in checkpoint_dir.iterdir():
                                if ckpt_dir.is_dir() and "ckpt_step" in ckpt_dir.name:
                                    lora_path = ckpt_dir
                                    is_local = True
                                    break
                            if is_local:
                                break
        
        # Check the found paths
        for path in possible_paths:
            if path.exists() and path.is_dir():
                lora_path = path
                is_local = True
                break
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA weights from local path or HF repo
    if is_local:
        print(f"📁 Loading LoRA adapter from local path: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))
    else:
        print(f"🌐 Loading LoRA adapter from HuggingFace: {lora_repo}")
        try:
            model = PeftModel.from_pretrained(model, lora_repo)
        except Exception as e:
            print(f"\n❌ Error loading from HuggingFace: {e}")
            print(f"\n💡 Tip: If this is a local checkpoint, provide the full path, e.g.:")
            print(f"   local_checkpoints/2025-12-03-22-26-59_checkpoints/ckpt_step_1280")
            print(f"\n   Or search for it with:")
            print(f"   find local_checkpoints -name '*{lora_repo.split('/')[-1]}*' -type d")
            raise

    # Optional merge
    if merge:
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer

###########################################################
# Load base model only (no LoRA)
###########################################################
def load_base_model(base_model: str):
    """
    Load a base model without any LoRA adapter.
    
    Args:
        base_model: str — e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    Returns:
        model: torch.nn.Module — ready for inference
        tokenizer: transformers.AutoTokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    return model, tokenizer

###########################################################
# Helper: list available checkpoints in repo
###########################################################
from huggingface_hub import list_repo_files

def list_lora_checkpoints(lora_repo: str):
    """
    List files in the LoRA HF repository. Useful to discover checkpoints
    like step_200/, step_400/, etc.
    """
    return list_repo_files(lora_repo)

###########################################################
# Helper: load specific checkpoint folder
###########################################################

def load_finetuned_checkpoint(base_model: str, lora_repo: str, checkpoint_folder: str, merge: bool = False):
    """
    Load a specific checkpoint inside a HF LoRA repo.

    Example:
        model, tokenizer = load_finetuned_checkpoint(
            base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            lora_repo="Tristan-Day/llama_3.1_finetuned",
            checkpoint_folder="step_2000",
            merge=True,
        )
    """
    full_path = f"{lora_repo}/{checkpoint_folder}"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, full_path)

    if merge:
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer
