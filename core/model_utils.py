"""
Model loading and naming utilities for introspection experiments.

Provides consistent model loading, run naming, and detection of model properties
(base vs instruct, chat template availability, etc.)
"""

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from typing import Optional, Tuple

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Device detection
# Note: MPS disabled due to segfault issues with Llama 3.1 models
# DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_base_model(model_name: str) -> bool:
    """Check if model is a base model (not instruction-tuned)."""
    model_lower = model_name.lower()
    instruct_indicators = ['instruct', 'chat', '-it', 'rlhf', 'sft', 'dpo']
    return not any(ind in model_lower for ind in instruct_indicators)


def has_chat_template(tokenizer) -> bool:
    """Check if tokenizer has a chat template."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True
        )
        return True
    except Exception:
        return False


def get_model_short_name(model_name: str) -> str:
    """
    Extract a short, filesystem-safe name from a model path.

    Examples:
        "meta-llama/Llama-3.1-8B-Instruct" -> "Llama-3.1-8B-Instruct"
        "/path/to/adapter" -> "adapter"
    """
    # Handle HuggingFace paths
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def get_run_name(
    base_model: str,
    dataset: str,
    task: str = "probe",
    adapter: Optional[str] = None,
    num_questions: Optional[int] = None,
    seed: Optional[int] = None
) -> str:
    """
    Generate a consistent run name for output files.

    Format: {model_short}[_adapter]_{dataset}_{task}[_n{num}][_s{seed}]

    Args:
        base_model: Base model name/path
        dataset: Dataset name (e.g., "SimpleMC", "GPQA")
        task: Task type (e.g., "probe", "steer", "delegate")
        adapter: Optional adapter path
        num_questions: Optional number of questions
        seed: Optional random seed

    Returns:
        Filesystem-safe run name string
    """
    model_short = get_model_short_name(base_model)

    parts = [model_short]

    if adapter:
        adapter_short = get_model_short_name(adapter)
        parts.append(f"adapter-{adapter_short}")

    parts.append(dataset)
    parts.append(task)

    if num_questions:
        parts.append(f"n{num_questions}")

    if seed is not None:
        parts.append(f"s{seed}")

    return "_".join(parts)


def _resolve_adapter_path(adapter_path: str) -> str:
    """Resolve an adapter spec to a usable path or HF repo id.

    Tries (in order):
      1. The string as-is, if it's an existing local directory.
      2. ``local_checkpoints/<spec>`` and ``local_checkpoints/<spec>_checkpoints``.
      3. If the spec mentions ``step`` or ``ckpt``, scans subdirectories of
         ``local_checkpoints/`` for a matching ``ckpt_step_*`` folder.

    If none of the local paths exist, returns the original string unchanged
    so HuggingFace ``PeftModel.from_pretrained`` can treat it as a repo id.
    """
    p = Path(adapter_path)
    if p.exists() and p.is_dir():
        return str(p)

    # Common local layout: ./local_checkpoints/<run>/ckpt_step_<N>/
    candidates = [
        Path("local_checkpoints") / adapter_path,
        Path("local_checkpoints") / f"{adapter_path}_checkpoints",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return str(cand)

    if "step" in adapter_path.lower() or "ckpt" in adapter_path.lower():
        root = Path("local_checkpoints")
        if root.exists():
            for run_dir in root.iterdir():
                if not run_dir.is_dir():
                    continue
                for ckpt_dir in run_dir.iterdir():
                    if ckpt_dir.is_dir() and adapter_path in str(ckpt_dir):
                        return str(ckpt_dir)
                if adapter_path in run_dir.name:
                    for ckpt_dir in run_dir.iterdir():
                        if ckpt_dir.is_dir() and "ckpt_step" in ckpt_dir.name:
                            return str(ckpt_dir)
    return adapter_path


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    merge: bool = False,
) -> Tuple:
    """
    Load a model and tokenizer, optionally with a PEFT adapter.

    Args:
        base_model_name: HuggingFace model name or path
        adapter_path: Optional path to PEFT adapter. May be a local directory,
            an HF repo id, or a checkpoint name to be resolved against
            ``local_checkpoints/`` (e.g. ``"ckpt_step_1280"``).
        device_map: Device mapping strategy
        torch_dtype: Data type (auto-detected if None)
        load_in_4bit: Load model in 4-bit quantization (recommended for 70B+ models)
        load_in_8bit: Load model in 8-bit quantization
        merge: If True and an adapter was loaded, merge LoRA weights into
            the base model (`PeftModel.merge_and_unload`). Useful when you
            want a plain `AutoModelForCausalLM` for downstream code that
            doesn't know about PEFT.

    Returns:
        Tuple of (model, tokenizer, num_layers)
    """
    if torch_dtype is None:
        # Force float32 on CPU to avoid segfaults
        torch_dtype = torch.float32 if DEVICE == "cpu" else torch.float16 if DEVICE == "cuda" else torch.float32

    print(f"Loading model: {base_model_name}")

    # On CPU (macOS), use minimal kwargs to avoid segfaults from accelerate
    if DEVICE == "cpu":
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "token": HF_TOKEN,
            "low_cpu_mem_usage": False,  # Disable accelerate-based loading
        }
    else:
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "token": HF_TOKEN
        }

    # Build quantization config if requested
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,  # Nested quantization for memory savings
                    bnb_4bit_quant_type="nf4"  # NormalFloat4 for better quality
                )
                print("  Using 4-bit quantization (NF4)")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload if needed
                )
                model_kwargs["device_map"] = {"": 0}
                print("  Using 8-bit quantization (with CPU offload if needed)")
        except ImportError:
            print("  Warning: bitsandbytes not installed, falling back to fp16")
            print("  Install with: pip install bitsandbytes")
            quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-pad for proper batched generation

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs
    )

    if adapter_path:
        try:
            from peft import PeftModel
            resolved = _resolve_adapter_path(adapter_path)
            if resolved != adapter_path:
                print(f"Loading adapter (resolved local): {resolved}")
            else:
                print(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, resolved)
            if merge:
                model = model.merge_and_unload()
        except Exception as e:
            raise RuntimeError(f"Error loading adapter {adapter_path}: {e}")

    # Get number of layers
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
        num_layers = len(base.model.layers)
    else:
        num_layers = len(model.model.layers)

    model.eval()
    print(f"Model has {num_layers} layers, device: {DEVICE}")

    return model, tokenizer, num_layers


def should_use_chat_template(model_name: str, tokenizer) -> bool:
    """Determine whether to use chat template based on model type and tokenizer."""
    return has_chat_template(tokenizer) and not is_base_model(model_name)
