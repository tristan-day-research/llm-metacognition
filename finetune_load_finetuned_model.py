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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

###########################################################
# Load base + LoRA adapter
###########################################################
def load_finetuned_model(base_model: str, lora_repo: str, merge: bool = False):
    """
    Load a base Llama model and apply LoRA adapter.

    Args:
        base_model: str — e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
        lora_repo:  str — HuggingFace repository containing LoRA checkpoints
        merge:      bool — if True, merges LoRA weights into base model

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

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_repo)

    # Optional merge
    if merge:
        model = model.merge_and_unload()

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
