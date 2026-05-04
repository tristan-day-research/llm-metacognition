"""
Build a stratified dataset for entropy prediction experiments.
OPTIMIZED VERSION: Uses batched inference and vectorized entropy calculation.

This script:
1. Samples diverse text from multiple HuggingFace datasets
2. Runs a pilot to compute actual output entropy for each prompt (BATCHED)
3. Stratifies by entropy and samples evenly across deciles
4. Saves the final dataset
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import torch
import numpy as np
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Dict, Tuple

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_run_name,
    get_model_short_name,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME
PILOT_SIZE = 10000
FINAL_SIZE = 5000
MIN_PROMPT_LENGTH = 20
MAX_PROMPT_LENGTH = 500
CHECKPOINT_INTERVAL = 500
SEED = 42

# --- OPTIMIZATION CONFIG ---
# Batch size depends on VRAM. 16-32 is usually safe for 8B models on 24GB VRAM.
# Decrease if you hit OOM errors.
BATCH_SIZE = 16
# ---------------------------

# Quantization (for large models like 70B)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_nexttoken")
    return str(OUTPUTS_DIR / f"{model_short}_nexttoken")


def load_diverse_texts(num_samples: int) -> List[str]:
    """Load diverse text samples from multiple sources."""
    print("Loading diverse text sources...")
    all_texts = []
    samples_per_source = num_samples // 4

    # Helper to safely load streams
    def load_stream(dataset_name, subset, split, data_dir=None, text_col="text"):
        print(f"  Loading {dataset_name}...")
        try:
            kwargs = {"split": split, "streaming": True}
            if subset: kwargs["name"] = subset
            if data_dir: kwargs["data_dir"] = data_dir
            
            ds = load_dataset(dataset_name, **kwargs)
            collected = []
            for i, item in enumerate(ds):
                if i >= samples_per_source:
                    break
                text = item[text_col]
                if len(text) > 100:
                    collected.append(text)
            print(f"    Loaded {len(collected)} samples")
            return collected
        except Exception as e:
            print(f"    Warning: Could not load {dataset_name}: {e}")
            return []

    # Wikipedia
    all_texts.extend(load_stream("wikimedia/wikipedia", "20231101.en", "train"))
    # Code
    all_texts.extend(load_stream("bigcode/the-stack-smol", None, "train", data_dir="data/python", text_col="content"))
    # FineWeb
    all_texts.extend(load_stream("HuggingFaceFW/fineweb", "sample-10BT", "train"))
    # C4
    all_texts.extend(load_stream("allenai/c4", "en", "train"))

    print(f"Total texts loaded: {len(all_texts)}")
    return all_texts


def create_prompts(texts: List[str], tokenizer, num_prompts: int) -> List[Dict]:
    """
    Create prompts and PRE-CALCULATE input_ids to avoid re-tokenization.
    """
    print(f"Creating {num_prompts} prompts...")
    prompts = []

    # Optimization: Use a progress bar for creation
    pbar = tqdm(total=num_prompts)
    
    while len(prompts) < num_prompts:
        text = random.choice(texts)
        
        # Tokenize without special tokens initially to control length exactly
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) < MIN_PROMPT_LENGTH + 1:
            continue

        max_len = min(len(tokens) - 1, MAX_PROMPT_LENGTH)
        prompt_length = random.randint(MIN_PROMPT_LENGTH, max_len)
        
        # Extract prompt tokens
        prompt_tokens = tokens[:prompt_length]
        
        # We store input_ids immediately. 
        # Note: Llama 3 usually doesn't need a BOS token if using Instruct, 
        # but if your tokenizer expects one, add_special_tokens=True handled that usually.
        # Here we manually kept it raw, so let's decode to get clean text, 
        # and re-encode efficiently or just use tokens if tokenizer is reversible.
        # To be safe and consistent with original logic: decode -> store text.
        prompt_text = tokenizer.decode(prompt_tokens)
        
        # Store dict
        prompts.append({
            "text": prompt_text,
            "prompt_length": prompt_length,
            # We will generate input_ids for the batcher on the fly or pre-compute here.
            # Pre-computing is safer for consistency.
            "input_ids": tokenizer.encode(prompt_text, add_special_tokens=True)
        })
        pbar.update(1)

    pbar.close()
    print(f"Created {len(prompts)} valid prompts")
    return prompts


def load_checkpoint(checkpoint_path: Path) -> List[Dict]:
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path) as f:
            return json.load(f)
    return []


def save_checkpoint(results: List[Dict], checkpoint_path: Path):
    with open(checkpoint_path, "w") as f:
        json.dump(results, f)
    # Print less frequently to avoid clutter
    # print(f"  Checkpoint saved: {len(results)} prompts")


def compute_entropy_batched(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of next-token distribution for a batch.
    logits: [batch_size, vocab_size]
    Returns: [batch_size]
    """
    # Use float32 for stability in entropy calculation even if model is fp16/bf16
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    # Entropy = - sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def run_pilot_inference_batched(
    prompts: List[Dict],
    model,
    tokenizer,
    checkpoint_path: Path
) -> List[Dict]:
    """
    Run inference in batches.
    """
    # Load existing results
    results = load_checkpoint(checkpoint_path)
    processed_count = len(results)
    
    if processed_count >= len(prompts):
        print("All prompts already processed.")
        return results

    if processed_count > 0:
        print(f"Resuming from prompt {processed_count}/{len(prompts)}")

    # Filter out already processed prompts
    # (Assuming prompts list order is deterministic/preserved)
    prompts_to_process = prompts[processed_count:]

    print(f"Running batched inference on {len(prompts_to_process)} remaining prompts...")
    model.eval()

    # Ensure pad token is set (Llama 3 sometimes misses this)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Batch generator
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    pbar = tqdm(total=len(prompts_to_process))
    
    # Process in batches
    for batch_prompts in batch(prompts_to_process, BATCH_SIZE):
        
        # Prepare batch tensors
        # We need to pad manually or use tokenizer.pad. 
        # Since we have pre-computed input_ids, let's just pad them.
        batch_input_ids = [p["input_ids"] for p in batch_prompts]
        
        # Pad to longest in this batch
        max_len_batch = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in batch_input_ids:
            # Left padding is often preferred for generation, but for pure classification/
            # next-token prediction, right padding is fine as long as we extract the 
            # correct last token index.
            # Using Right Padding here for simplicity in indexing.
            pad_len = max_len_batch - len(ids)
            padded_ids = ids + [tokenizer.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        input_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=DEVICE)
        mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=DEVICE)

        with torch.inference_mode(): # Faster than no_grad
            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
            logits = outputs.logits

            # We need the logits of the last *real* token for each sequence
            # Index is: sum(mask) - 1
            last_token_indices = mask_tensor.sum(dim=1) - 1
            
            # Select relevant logits: [batch_size, vocab_size]
            # torch.arange creates indices [0, 1, ... batch_size-1]
            batch_logits = logits[torch.arange(logits.shape[0], device=DEVICE), last_token_indices, :]
            
            # Compute entropy vectorized
            entropies = compute_entropy_batched(batch_logits)

        # Append results
        for i, prompt_data in enumerate(batch_prompts):
            # Create a clean copy without the 'input_ids' to save space in JSON
            result_entry = {
                "text": prompt_data["text"],
                "prompt_length": prompt_data["prompt_length"],
                "entropy": entropies[i].item()
            }
            results.append(result_entry)

        processed_count += len(batch_prompts)
        pbar.update(len(batch_prompts))

        # Checkpoint based on total processed
        if processed_count % CHECKPOINT_INTERVAL < BATCH_SIZE:
             save_checkpoint(results, checkpoint_path)

    pbar.close()
    save_checkpoint(results, checkpoint_path)
    return results


def stratify_and_sample(
    prompts_with_entropy: List[Dict],
    num_samples: int,
    num_bins: int = 10
) -> List[Dict]:
    """Stratify prompts by entropy into bins and sample evenly."""
    print(f"Stratifying by entropy into {num_bins} bins...")

    # Sort by entropy
    sorted_prompts = sorted(prompts_with_entropy, key=lambda x: x["entropy"])

    # Get entropy range
    entropies = [p["entropy"] for p in sorted_prompts]
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    print(f"  Entropy range: [{min_entropy:.3f}, {max_entropy:.3f}]")

    # Create bins using percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(entropies, percentiles)

    # Assign each prompt to a bin
    bins = [[] for _ in range(num_bins)]
    for prompt in sorted_prompts:
        entropy = prompt["entropy"]
        bin_idx = np.searchsorted(bin_edges[1:], entropy)
        bin_idx = min(bin_idx, num_bins - 1)
        bins[bin_idx].append(prompt)

    # Sample evenly
    samples_per_bin = num_samples // num_bins
    stratified_sample = []

    for i, bin_prompts in enumerate(bins):
        if len(bin_prompts) >= samples_per_bin:
            sampled = random.sample(bin_prompts, samples_per_bin)
        else:
            print(f"    Warning: Bin {i} has only {len(bin_prompts)} prompts, using all")
            sampled = bin_prompts
        stratified_sample.extend(sampled)

    print(f"Final dataset size: {len(stratified_sample)}")
    return stratified_sample


def main():
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")

    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    samples_raw_path = Path(f"{output_prefix}_samples_raw.json")
    checkpoint_path = Path(f"{output_prefix}_checkpoint.json")
    final_output = Path(f"{output_prefix}_entropy_dataset.json")

    # Load model
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    print(f"Model has {num_layers} layers")

    # Step 1: Load diverse texts
    texts = load_diverse_texts(PILOT_SIZE)

    # Step 2: Create prompts
    # NOTE: prompts now contains "input_ids" for efficiency
    prompts = create_prompts(texts, tokenizer, PILOT_SIZE)

    # Step 3: Run BATCHED pilot inference
    prompts_with_entropy = run_pilot_inference_batched(
        prompts, model, tokenizer, checkpoint_path
    )

    # Save raw samples
    with open(samples_raw_path, "w") as f:
        json.dump(prompts_with_entropy, f, indent=2)
    print(f"Saved raw samples to {samples_raw_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Step 4: Stratify and sample
    final_dataset = stratify_and_sample(prompts_with_entropy, FINAL_SIZE)

    # Save final dataset
    output_data = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "pilot_size": PILOT_SIZE,
            "final_size": FINAL_SIZE,
            "min_prompt_length": MIN_PROMPT_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "seed": SEED,
            "batch_size": BATCH_SIZE
        },
        "data": final_dataset
    }

    with open(final_output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved final dataset to {final_output}")

    entropies = [p["entropy"] for p in final_dataset]
    print("\nFinal dataset statistics:")
    print(f"  Size: {len(final_dataset)}")
    print(f"  Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
    print(f"  Entropy mean: {np.mean(entropies):.3f}")
    print(f"  Entropy std: {np.std(entropies):.3f}")


if __name__ == "__main__":
    main()