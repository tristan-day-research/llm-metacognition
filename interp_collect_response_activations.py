"""
collect_activations.py

Comprehensive activation collection for introspection experiments.
Collects activations, outputs, and metadata for MCQ, self-confidence, and other-confidence tasks.

Usage:
    # Test run with 10 questions on base model (all question types, auto-named output dir)
    # NOTE: By default, shuffle_answers=True to match training/evaluation behavior
    python interp_collect_activations.py --model_type base --num_questions 10 --dataset_path data/PopMC_0_difficulty_filtered_test.jsonl
    
    # Run only MCQ questions
    python interp_collect_activations.py --model_type base --num_questions 10 --which_questions mcq --dataset_path data/PopMC_0_difficulty_filtered_test.jsonl
    
    # Run MCQ and self-confidence only
    python interp_collect_activations.py --model_type base --num_questions 10 --which_questions mcq,self --dataset_path data/PopMC_0_difficulty_filtered_test.jsonl
    
    # Full run with 1000 questions on finetuned model
    python interp_collect_activations.py --model_type finetuned --num_questions 1000 --dataset_path data/PopMC_0_difficulty_filtered_train.jsonl
    
    # Specific checkpoint with custom output dir
    python interp_collect_activations.py --model_type finetuned --lora_checkpoint ckpt_step_1280 --num_questions 100 --output_dir my_custom_dir --dataset_path data/PopMC_0_difficulty_filtered_test.jsonl
    
    # Without answer shuffling (if needed for specific experiments)
    python interp_collect_activations.py --model_type base --num_questions 10 --no_shuffle_answers --dataset_path data/PopMC_0_difficulty_filtered_test.jsonl
"""

import argparse
import json
import os
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Import your helper functions
from finetune_load_finetuned_model import load_base_model, load_finetuned_model
from finetune_utils import prepare_model_and_tokenizer
from finetune_prompting import (
    build_multiple_choice_question_prompts,
    build_self_confidence_prompts,
    build_other_confidence_prompts,
    get_letter_token_ids
)


###########################################################
# Activation Hook Manager
###########################################################

class ActivationCollector:
    """Collects activations from all transformer layers."""
    
    def __init__(self, model, num_layers=32):
        self.model = model
        self.num_layers = num_layers
        self.activations = {}  # {layer_idx: tensor}
        self.hooks = []
        
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # Output is tuple, we want the hidden states
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Store on CPU to save GPU memory, convert to float16
            self.activations[layer_idx] = hidden_states.detach().cpu().half()
        return hook
    
    def register_hooks(self):
        """Register hooks on all transformer layers."""
        # Handle both base models and PeftModel (LoRA)
        # Base model: model.model.layers[i]
        # PeftModel: Need to get the actual base model, then model.model.layers[i]
        from peft import PeftModel
        
        # Try to find the layers attribute
        layers = None
        
        if isinstance(self.model, PeftModel):
            # LoRA/PEFT model - need to get the actual base model
            # PeftModel.base_model is a LoraModel wrapper, need to go deeper
            # Use get_base_model() to get the unwrapped base model
            try:
                base_model = self.model.get_base_model()
            except AttributeError:
                # Fallback: try accessing base_model.base_model
                base_model = self.model.base_model
                if hasattr(base_model, 'base_model'):
                    base_model = base_model.base_model
            
            # Now base_model should be LlamaForCausalLM
            if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
                layers = base_model.model.layers
            elif hasattr(base_model, 'layers'):
                layers = base_model.layers
            else:
                raise AttributeError(
                    f"Could not find layers in PeftModel. "
                    f"base_model type: {type(base_model)}, "
                    f"base_model attributes: {[a for a in dir(base_model) if not a.startswith('_')][:15]}"
                )
        else:
            # Base model - direct access
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layers = self.model.model.layers
            elif hasattr(self.model, 'layers'):
                layers = self.model.layers
            else:
                raise AttributeError(
                    f"Could not find layers in model. "
                    f"Model type: {type(self.model)}, "
                    f"Model attributes: {[a for a in dir(self.model) if not a.startswith('_')][:15]}"
                )
        
        for i in range(self.num_layers):
            layer = layers[i]
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)
    
    def clear_activations(self):
        """Clear stored activations to free memory."""
        self.activations = {}
        torch.cuda.empty_cache()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations_dict(self):
        """Return copy of current activations."""
        return {k: v.clone() for k, v in self.activations.items()}


###########################################################
# Feature Extraction
###########################################################

def extract_surface_features(question_text, tokenizer):
    """Extract surface-level difficulty features from question."""
    
    # Tokenize to get token-level info
    tokens = tokenizer.encode(question_text, add_special_tokens=False)
    
    # Basic length features
    char_length = len(question_text)
    token_length = len(tokens)
    
    # Question type detection
    question_lower = question_text.lower()
    question_type = "other"
    for qtype in ["what", "where", "when", "who", "how", "why", "which"]:
        if qtype in question_lower[:20]:  # Check beginning
            question_type = qtype
            break
    
    # Negation detection
    has_negation = any(neg in question_lower for neg in ["not", "n't", "never", "no ", "none"])
    
    # Complexity indicators
    has_ambiguity = any(word in question_lower for word in ["possibly", "might", "could", "may"])
    
    # Token rarity (approximation using token IDs - higher IDs tend to be rarer)
    token_rarity_mean = np.mean(tokens)
    token_rarity_max = max(tokens)
    
    return {
        "char_length": char_length,
        "token_length": token_length,
        "question_type": question_type,
        "has_negation": has_negation,
        "has_ambiguity": has_ambiguity,
        "token_rarity_mean": float(token_rarity_mean),
        "token_rarity_max": int(token_rarity_max),
    }


###########################################################
# MCQ Pass with Activation Collection
###########################################################

def run_mcq_pass_with_activations(
    model,
    tokenizer,
    batch,
    collector,
    device="cuda",
    model_name="",
    checkpoint_id="",
):
    """Run MCQ forward pass and collect activations."""
    
    # Build prompts
    prompts = build_multiple_choice_question_prompts(batch, tokenizer)
    
    # Tokenize
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    # Clear previous activations
    collector.clear_activations()
    
    # Forward pass (hooks will collect activations)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    
    final_logits = out.logits[:, -1, :]  # [B, vocab]
    
    # Get letter token IDs and aggregate
    letter_token_ids = {
        letter: get_letter_token_ids(tokenizer, letter) 
        for letter in "ABCD"
    }
    
    # Validate that all letters have token IDs (handle case where get_letter_token_ids returns empty list)
    for letter in "ABCD":
        if not letter_token_ids[letter]:
            raise ValueError(
                f"Could not find single-token encoding for letter '{letter}'. "
                f"This is required for MCQ token aggregation."
            )
    
    logits4_list = []
    for letter in "ABCD":
        token_ids = letter_token_ids[letter]
        letter_logits = final_logits[:, token_ids]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)
        logits4_list.append(aggregated_logit)
    
    logits4 = torch.stack(logits4_list, dim=-1)  # [B, 4]
    probs4 = F.softmax(logits4, dim=-1)
    entropy = -(probs4 * torch.log(probs4 + 1e-12)).sum(dim=-1)
    
    # Predicted answers
    pred_indices = logits4.argmax(dim=-1).cpu().tolist()
    pred_letters = ["ABCD"[i] for i in pred_indices]
    
    # Get activations (already on CPU in float16)
    activations = collector.get_activations_dict()
    
    # Build results for each question in batch
    results = []
    for i, row in enumerate(batch):
        correct_letter = row["answer"]
        is_correct = pred_letters[i] == correct_letter
        
        # Calculate logit margin (correct vs best other)
        correct_idx = "ABCD".index(correct_letter)
        correct_logit = logits4[i, correct_idx].item()
        other_logits = [logits4[i, j].item() for j in range(4) if j != correct_idx]
        logit_margin = correct_logit - max(other_logits)
        
        # Find answer token position (last token before padding)
        attention_mask = enc.attention_mask[i]
        answer_token_idx = attention_mask.sum().item() - 1
        
        # Extract activations ONLY for response token (not full sequence)
        # For MCQ, response is always 1 token (A/B/C/D)
        question_activations = {
            layer_idx: acts[i, answer_token_idx].numpy()  # [4096] in float16
            for layer_idx, acts in activations.items()
        }
        
        # Response metadata
        response_length = 1  # MCQ always generates 1 token
        full_answer = pred_letters[i]  # The actual letter generated
        
        result = {
            # Question metadata
            "question_id": row.get("id", f"q_{i}"),
            "question_text": row["question"],
            "options": row["options"],
            "correct_answer_letter": correct_letter,
            
            # Prompt & execution
            "model_name": model_name,
            "checkpoint_id": checkpoint_id,
            "prompt_type": "mcq",
            "prompt_text": prompts[i],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 1,
            
            # Model output
            "output_text": pred_letters[i],
            "parsed_answer": pred_letters[i],
            "is_correct": is_correct,
            "response_length": response_length,  # Number of tokens in response
            "full_answer": full_answer,  # Full text of answer
            
            # Token-level distributions
            "logits": logits4[i].cpu().tolist(),  # [4] for A/B/C/D
            "probs": probs4[i].cpu().tolist(),
            "entropy": entropy[i].item(),
            "logit_margin": logit_margin,
            "answer_token_index": answer_token_idx,
            
            # Surface features
            "surface_features": extract_surface_features(row["question"], tokenizer),
            
            # Activations (stored separately due to size)
            "activations": question_activations,
        }
        
        results.append(result)
    
    return results


###########################################################
# Confidence Pass with Activation Collection
###########################################################

def run_confidence_pass_with_activations(
    model,
    tokenizer,
    batch,
    collector,
    confidence_type="self",  # "self" or "other"
    device="cuda",
    model_name="",
    checkpoint_id="",
):
    """Run confidence forward pass and collect activations."""
    
    # Build prompts
    if confidence_type == "self":
        prompts = build_self_confidence_prompts(batch, tokenizer)
    else:
        prompts = build_other_confidence_prompts(batch, tokenizer)
    
    # Tokenize
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    # Clear previous activations
    collector.clear_activations()
    
    # Forward pass
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    
    final_logits = out.logits[:, -1, :]  # [B, vocab]
    
    # Get bin token IDs and aggregate
    bin_token_ids = {
        letter: get_letter_token_ids(tokenizer, letter) 
        for letter in "ABCDEFGH"
    }
    
    # Validate that all letters have token IDs (handle case where get_letter_token_ids returns empty list)
    for letter in "ABCDEFGH":
        if not bin_token_ids[letter]:
            raise ValueError(
                f"Could not find single-token encoding for letter '{letter}'. "
                f"This is required for confidence bin token aggregation."
            )
    
    logits8_list = []
    for letter in "ABCDEFGH":
        token_ids = bin_token_ids[letter]
        letter_logits = final_logits[:, token_ids]
        aggregated_logit = torch.logsumexp(letter_logits, dim=-1)
        logits8_list.append(aggregated_logit)
    
    logits8 = torch.stack(logits8_list, dim=-1)  # [B, 8]
    probs8 = F.softmax(logits8, dim=-1)
    
    # Expected confidence (midpoints of bins)
    mids = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95], dtype=torch.float32, device=device)
    expected_conf = (probs8 * mids).sum(dim=-1)
    
    # Entropy
    entropy = -(probs8 * torch.log(probs8 + 1e-12)).sum(dim=-1)
    
    # Predicted bins
    pred_indices = logits8.argmax(dim=-1).cpu().tolist()
    pred_bins = ["ABCDEFGH"[i] for i in pred_indices]
    
    # Get activations
    activations = collector.get_activations_dict()
    
    # Build results
    results = []
    for i, row in enumerate(batch):
        # Find confidence token position (last token before padding)
        attention_mask = enc.attention_mask[i]
        conf_token_idx = attention_mask.sum().item() - 1
        
        # Extract activations ONLY for response token (not full sequence)
        # For confidence, response is always 1 token (A-H)
        question_activations = {
            layer_idx: acts[i, conf_token_idx].numpy()  # [4096] in float16
            for layer_idx, acts in activations.items()
        }
        
        # Response metadata
        response_length = 1  # Confidence always generates 1 token (bin letter)
        full_answer = pred_bins[i]  # The actual bin letter generated
        
        result = {
            # Question metadata
            "question_id": row.get("id", f"q_{i}"),
            "question_text": row["question"],
            "options": row["options"],
            
            # Prompt & execution
            "model_name": model_name,
            "checkpoint_id": checkpoint_id,
            "prompt_type": f"{confidence_type}_confidence",
            "prompt_text": prompts[i],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 1,
            
            # Model output
            "output_text": pred_bins[i],
            "parsed_answer": pred_bins[i],
            f"{confidence_type}_confidence": expected_conf[i].item(),
            "response_length": response_length,  # Number of tokens in response
            "full_answer": full_answer,  # Full text of answer
            
            # Token-level distributions
            "logits": logits8[i].cpu().tolist(),  # [8] for A-H
            "probs": probs8[i].cpu().tolist(),
            "entropy": entropy[i].item(),
            "confidence_token_index": conf_token_idx,
            
            # Activations
            "activations": question_activations,
        }
        
        results.append(result)
    
    return results


###########################################################
# Dataset Loading Utilities
###########################################################

def load_popmc_from_jsonl(jsonl_path, num_questions=None, shuffle_answers=True, seed=42):
    """
    Load PopMC dataset from JSONL file and convert to expected format.
    
    Args:
        jsonl_path: Path to JSONL file (e.g., "data/PopMC_0_difficulty_filtered_test.jsonl")
        num_questions: Number of questions to load (None = all)
        shuffle_answers: Whether to shuffle answer options
        seed: Random seed for shuffling
    
    Returns:
        List of dicts with format: {"id", "question", "options": {"A": ..., "B": ..., ...}, "answer": "A"}
    """
    rng = random.Random(seed)
    
    with open(jsonl_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
    
    formatted = []
    for item in raw_data:
        question_text = item.get('question', '').strip()
        qid = item.get('qid', '')
        correct_answer_text = item.get('correct_answer', '').strip()
        distractors = item.get('distractors', [])
        
        # Validation
        if not question_text or not correct_answer_text:
            continue
        if not distractors or len(distractors) < 3:
            continue
        if any(len(d.strip()) == 0 for d in distractors):
            continue
        
        # Create options list and shuffle
        options_list = [correct_answer_text] + distractors[:3]
        if shuffle_answers:
            rng.shuffle(options_list)
        
        # Build options dict and find correct letter
        options_dict = {}
        correct_letter = None
        for i, option_text in enumerate(options_list):
            letter = "ABCD"[i]
            options_dict[letter] = option_text
            if option_text == correct_answer_text:
                correct_letter = letter
        
        if correct_letter is None:
            continue  # Shouldn't happen, but safety check
        
        formatted.append({
            "id": qid if qid else f"popmc_{len(formatted)}",
            "question": question_text,
            "options": options_dict,
            "answer": correct_letter,  # Expected by the code
        })
        
        if num_questions and len(formatted) >= num_questions:
            break
    
    return formatted


###########################################################
# Main Collection Pipeline
###########################################################

def collect_all_data(
    model_type="base",
    lora_checkpoint=None,
    num_questions=10,
    batch_size=8,
    output_dir=None,
    device="cuda",
    dataset_path=None,
    base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    which_questions=None,
    shuffle_answers=True,  # Default True to match training/evaluation behavior
):
    """Main pipeline to collect all data."""
    
    # Parse which_questions
    if which_questions is None:
        which_questions = ["mcq", "self", "other"]
    elif isinstance(which_questions, str):
        # Handle comma-separated string
        which_questions = [q.strip().lower() for q in which_questions.split(",")]
    
    # Validate which_questions
    valid_questions = {"mcq", "self", "other"}
    which_questions = [q for q in which_questions if q in valid_questions]
    if not which_questions:
        raise ValueError(f"which_questions must contain at least one of: {valid_questions}")
    
    # Load model first to get model_name for output_dir
    print("Loading model...")
    
    if model_type == "base":
        model, tokenizer = load_base_model(base_model_name)
        model_name = "llama-3.1-8b-base"
        checkpoint_id = "base"
    else:  # finetuned
        if lora_checkpoint is None:
            lora_checkpoint = "ckpt_step_1280"  # Default
        model, tokenizer = load_finetuned_model(
            base_model=base_model_name,
            lora_repo=lora_checkpoint,
            merge=False
        )
        model_name = "llama-3.1-8b-finetuned"
        checkpoint_id = lora_checkpoint
    
    # CRITICAL: Prepare model and tokenizer to match training/evaluation setup
    # This ensures padding_side="left" and proper tokenizer configuration
    model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)
    
    model.eval()
    print(f"✓ Loaded {model_name} (checkpoint: {checkpoint_id})")
    print(f"✓ Prepared model and tokenizer (padding_side={tokenizer.padding_side}, truncation_side={tokenizer.truncation_side})")
    
    # Auto-generate output_dir if not provided
    if output_dir is None:
        # Format: {which_questions}+{base or finetuned}_{model_name}_{num_questions}_{datetime}
        which_str = "+".join(sorted(which_questions))  # Sort for consistency
        model_type_str = "base" if model_type == "base" else "finetuned"
        # Extract model name from base_model_name (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct" -> "Meta-Llama-3.1-8B-Instruct")
        # Or use the derived model_name if it's simpler
        if "/" in base_model_name:
            base_name = base_model_name.split("/")[-1]  # Get last part after /
        else:
            base_name = base_model_name
        # Sanitize for filesystem (remove special chars, convert to lowercase)
        safe_model_name = base_name.replace("-", "_").replace(".", "_").lower()
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"{which_str}+{model_type_str}_{safe_model_name}_{num_questions}_{datetime_str}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ACTIVATION COLLECTION PIPELINE")
    print(f"{'='*60}")
    print(f"Model type: {model_type}")
    print(f"Model: {model_name} (checkpoint: {checkpoint_id})")
    print(f"Number of questions: {num_questions}")
    print(f"Batch size: {batch_size}")
    print(f"Question types: {', '.join(which_questions)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load dataset
    if dataset_path is None:
        raise ValueError("--dataset_path is required. Specify path to JSONL file (e.g., data/PopMC_0_difficulty_filtered_test.jsonl)")
    
    print(f"\nLoading dataset from {dataset_path}...")
    # Use hash-based seed like finetune_data_handling.py for consistency
    # This ensures same shuffling pattern as training/evaluation
    dataset_seed = hash(dataset_path) % (2**31) if shuffle_answers else 42
    dataset = load_popmc_from_jsonl(dataset_path, num_questions=num_questions, shuffle_answers=shuffle_answers, seed=dataset_seed)
    if not dataset:
        raise ValueError(f"No valid questions found in {dataset_path}")
    print(f"✓ Loaded {len(dataset)} questions")
    print(f"  Shuffle answers: {shuffle_answers} (seed: {dataset_seed})")
    
    # Create comprehensive metadata log with all parameters
    metadata_log = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "script": "interp_collect_activations.py",
        },
        "model_config": {
            "model_type": model_type,
            "base_model_name": base_model_name,
            "final_model_name": model_name,
            "lora_checkpoint": lora_checkpoint,
            "final_checkpoint_id": checkpoint_id,
            "device": device,
        },
        "dataset_config": {
            "dataset_path": dataset_path,
            "num_questions_requested": num_questions,
            "num_questions_loaded": len(dataset),
        },
        "processing_config": {
            "batch_size": batch_size,
            "output_dir": str(output_path),
            "shuffle_answers": shuffle_answers,
        },
        "collection_info": {
            "passes": which_questions,
            "num_layers": 32,
            "activation_dtype": "float16",
            "activation_shape_per_layer": "[num_questions, seq_length, 4096]",
        }
    }
    
    # Save metadata log to root of output_dir
    metadata_log_path = output_path / "metadata_log.json"
    with open(metadata_log_path, "w") as f:
        json.dump(metadata_log, f, indent=2)
    print(f"✓ Saved metadata log to {metadata_log_path}")
    
    # Save run config (for backward compatibility)
    run_config = {
        "model_type": model_type,
        "base_model_name": base_model_name,
        "lora_checkpoint": lora_checkpoint,
        "num_questions": num_questions,
        "batch_size": batch_size,
        "dataset_path": dataset_path,
        "which_questions": which_questions,
        "timestamp": datetime.now().isoformat(),
        "device": device,
    }
    with open(output_path / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    
    # Setup activation collector
    collector = ActivationCollector(model, num_layers=32)
    collector.register_hooks()
    print("✓ Registered activation hooks on 32 layers")
    
    # Debug: Print first prompt to verify format matches training/evaluation
    if dataset:
        test_batch = dataset[0:1]
        test_prompts = build_multiple_choice_question_prompts(test_batch, tokenizer)
        print(f"\n{'='*60}")
        print("DEBUG: First MCQ Prompt (for comparison with training/evaluation)")
        print(f"{'='*60}")
        print(test_prompts[0])
        print(f"{'='*60}\n")
    
    # Process in batches
    all_mcq_results = []
    all_self_conf_results = []
    all_other_conf_results = []
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    # Run MCQ pass if requested
    if "mcq" in which_questions:
        print(f"\n{'='*60}")
        print("PASS: MCQ Questions")
        print(f"{'='*60}")
        
        for batch_idx in tqdm(range(num_batches), desc="MCQ batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            
            results = run_mcq_pass_with_activations(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                collector=collector,
                device=device,
                model_name=model_name,
                checkpoint_id=checkpoint_id,
            )
            all_mcq_results.extend(results)
            
            # Save intermediate results every 10 batches
            # NOTE: This saves ALL accumulated results (cumulative), not just the current batch
            if (batch_idx + 1) % 10 == 0:
                save_results_batch(all_mcq_results, output_path / "mcq", batch_idx)
                print(f"  [Batch {batch_idx + 1}/{num_batches}] Saved intermediate checkpoint: {len(all_mcq_results)} questions (cumulative)")
        
        # Final batch save if not already saved
        if num_batches % 10 != 0:
            save_results_batch(all_mcq_results, output_path / "mcq", num_batches - 1)
        
        print(f"✓ Completed {len(all_mcq_results)} MCQ questions (expected {len(dataset)})")
        if len(all_mcq_results) != len(dataset):
            print(f"  ⚠️  WARNING: Question count mismatch! Expected {len(dataset)}, got {len(all_mcq_results)}")
        
        # Save MCQ results immediately to free memory
        print(f"\nSaving MCQ results to free memory...")
        save_final_results(all_mcq_results, output_path / "mcq")
        # Clear from memory
        del all_mcq_results
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✓ MCQ results saved and memory freed")
    
    # Run self-confidence pass if requested
    if "self" in which_questions:
        print(f"\n{'='*60}")
        print("PASS: Self-Confidence Questions")
        print(f"{'='*60}")
        
        for batch_idx in tqdm(range(num_batches), desc="Self-conf batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            
            results = run_confidence_pass_with_activations(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                collector=collector,
                confidence_type="self",
                device=device,
                model_name=model_name,
                checkpoint_id=checkpoint_id,
            )
            all_self_conf_results.extend(results)
            
            if (batch_idx + 1) % 10 == 0:
                save_results_batch(all_self_conf_results, output_path / "self_conf", batch_idx)
                print(f"  [Batch {batch_idx + 1}/{num_batches}] Saved intermediate checkpoint: {len(all_self_conf_results)} questions (cumulative)")
        
        # Final batch save if not already saved
        if num_batches % 10 != 0:
            save_results_batch(all_self_conf_results, output_path / "self_conf", num_batches - 1)
        
        print(f"✓ Completed {len(all_self_conf_results)} self-confidence questions (expected {len(dataset)})")
        if len(all_self_conf_results) != len(dataset):
            print(f"  ⚠️  WARNING: Question count mismatch! Expected {len(dataset)}, got {len(all_self_conf_results)}")
        
        # Save self-confidence results immediately to free memory
        print(f"\nSaving self-confidence results to free memory...")
        save_final_results(all_self_conf_results, output_path / "self_conf")
        # Clear from memory
        del all_self_conf_results
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✓ Self-confidence results saved and memory freed")
    
    # Run other-confidence pass if requested
    if "other" in which_questions:
        print(f"\n{'='*60}")
        print("PASS: Other-Confidence Questions")
        print(f"{'='*60}")
        
        for batch_idx in tqdm(range(num_batches), desc="Other-conf batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            
            results = run_confidence_pass_with_activations(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                collector=collector,
                confidence_type="other",
                device=device,
                model_name=model_name,
                checkpoint_id=checkpoint_id,
            )
            all_other_conf_results.extend(results)
            
            if (batch_idx + 1) % 10 == 0:
                save_results_batch(all_other_conf_results, output_path / "other_conf", batch_idx)
                print(f"  [Batch {batch_idx + 1}/{num_batches}] Saved intermediate checkpoint: {len(all_other_conf_results)} questions (cumulative)")
        
        # Final batch save if not already saved
        if num_batches % 10 != 0:
            save_results_batch(all_other_conf_results, output_path / "other_conf", num_batches - 1)
        
        print(f"✓ Completed {len(all_other_conf_results)} other-confidence questions (expected {len(dataset)})")
        if len(all_other_conf_results) != len(dataset):
            print(f"  ⚠️  WARNING: Question count mismatch! Expected {len(dataset)}, got {len(all_other_conf_results)}")
    
    # Remove hooks
    collector.remove_hooks()
    
    # Final save for other-confidence (if not already saved)
    print(f"\n{'='*60}")
    print("FINAL SAVE")
    print(f"{'='*60}")
    
    # Other-confidence results are saved here if they exist
    if "other" in which_questions and all_other_conf_results:
        print(f"✓ Completed {len(all_other_conf_results)} other-confidence questions (expected {len(dataset)})")
        if len(all_other_conf_results) != len(dataset):
            print(f"  ⚠️  WARNING: Question count mismatch! Expected {len(dataset)}, got {len(all_other_conf_results)}")
        
        print(f"\nSaving other-confidence results...")
        save_final_results(all_other_conf_results, output_path / "other_conf")
        del all_other_conf_results
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✓ Other-confidence results saved")
    
    print(f"\n✓ All data saved to {output_path}")
    print(f"\nData summary:")
    
    # Check what was actually saved by reading the files
    if "mcq" in which_questions:
        mcq_meta = output_path / "mcq" / "metadata.json"
        if mcq_meta.exists():
            with open(mcq_meta) as f:
                mcq_data = json.load(f)
            print(f"  - MCQ results: {len(mcq_data)} questions")
    if "self" in which_questions:
        self_meta = output_path / "self_conf" / "metadata.json"
        if self_meta.exists():
            with open(self_meta) as f:
                self_data = json.load(f)
            print(f"  - Self-conf results: {len(self_data)} questions")
    if "other" in which_questions:
        other_meta = output_path / "other_conf" / "metadata.json"
        if other_meta.exists():
            with open(other_meta) as f:
                other_data = json.load(f)
            print(f"  - Other-conf results: {len(other_data)} questions")
    
    print(f"  - Activations: response tokens only (1 token per question)")
    print(f"  - Format: compressed .npz [num_layers, num_questions, 4096]")
    print(f"  - Storage: ~100-500x smaller than full sequences")
    
    # Return None since we've cleared from memory
    return None, None, None


def save_results_batch(results, output_path, batch_idx):
    """Save intermediate batch results."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata (without activations) as JSON
    metadata = []
    for r in results:
        meta = {k: v for k, v in r.items() if k != "activations"}
        metadata.append(meta)
    
    with open(output_path / f"metadata_batch_{batch_idx}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_final_results(results, output_path):
    """Save final results with activations compressed in npz format."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("  ⚠️  Warning: No results to save")
        return
    
    num_questions = len(results)
    print(f"  Processing {num_questions} questions...")
    
    # Save all metadata (without activations)
    metadata = []
    for r in results:
        meta = {k: v for k, v in r.items() if k != "activations"}
        metadata.append(meta)
    
    # Save metadata to temporary file first, then rename (atomic write)
    metadata_tmp = output_path / "metadata.json.tmp"
    metadata_final = output_path / "metadata.json"
    with open(metadata_tmp, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(metadata_tmp, metadata_final)
    
    print(f"  ✓ Saved metadata: {output_path / 'metadata.json'} ({num_questions} questions)")
    
    # Check if we have activations
    if "activations" not in results[0]:
        print("  ⚠️  Warning: No activations found to save")
        return
    
    # Collect activations by layer
    # New format: each activation is [4096] (single token), not [seq_len, 4096]
    num_layers = len(results[0]["activations"])
    
    print(f"  Collecting activations from {num_layers} layers...")
    
    # Build activation array: [num_layers, num_questions, 4096]
    all_activations = np.zeros((num_layers, num_questions, 4096), dtype=np.float16)
    
    missing_activations = 0
    for q_idx, r in enumerate(results):
        if "activations" in r and r["activations"]:
            for layer_idx, acts in r["activations"].items():
                # acts is now [4096] not [seq_len, 4096]
                if isinstance(acts, np.ndarray):
                    all_activations[layer_idx, q_idx, :] = acts
                else:
                    # Handle if it's already a numpy array or needs conversion
                    all_activations[layer_idx, q_idx, :] = np.array(acts, dtype=np.float16)
        else:
            missing_activations += 1
    
    if missing_activations > 0:
        print(f"  ⚠️  Warning: {missing_activations} questions missing activations")
    
    # Save as compressed npz (one file per condition)
    npz_final = output_path / "activations.npz"
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save directly
    print(f"  Saving activations...")
    np.savez_compressed(
        str(npz_final),
        activations=all_activations,
        layer_indices=np.arange(num_layers),
        num_questions=num_questions,
        hidden_dim=4096
    )
    
    if not npz_final.exists():
        raise RuntimeError(f"Failed to save activations: {npz_final}")
    
    file_size_mb = npz_final.stat().st_size / (1024**2)
    print(f"  ✓ Saved compressed activations: {output_path / 'activations.npz'}")
    print(f"    Shape: [num_layers={num_layers}, num_questions={num_questions}, hidden_dim=4096]")
    print(f"    Size: {file_size_mb:.1f} MB (compressed)")
    uncompressed_size = (num_questions * num_layers * 4096 * 2) / (1024**2)
    if file_size_mb > 0:
        print(f"    Compression ratio: ~{uncompressed_size / file_size_mb:.1f}x")


###########################################################
# CLI
###########################################################

def main():
    parser = argparse.ArgumentParser(description="Collect activations for introspection experiments")
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "finetuned"],
        default="base",
        help="Which model to use"
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="LoRA checkpoint name (for finetuned model)"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of questions to process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not specified)"
    )
    parser.add_argument(
        "--which_questions",
        type=str,
        default="mcq,self,other",
        help="Comma-separated list of question types to run: mcq, self, other (e.g., 'mcq,self' or 'mcq')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to JSONL dataset file (e.g., data/PopMC_0_difficulty_filtered_test.jsonl)"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--shuffle_answers",
        action="store_true",
        default=None,  # Will be set to True if not specified
        help="Shuffle answer options (default: True to match training/evaluation behavior)"
    )
    parser.add_argument(
        "--no_shuffle_answers",
        dest="shuffle_answers",
        action="store_false",
        help="Disable shuffling of answer options"
    )
    
    args = parser.parse_args()
    
    # Set default to True if neither flag was specified
    if args.shuffle_answers is None:
        args.shuffle_answers = True
    
    # Run collection
    collect_all_data(
        model_type=args.model_type,
        lora_checkpoint=args.lora_checkpoint,
        num_questions=args.num_questions,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        dataset_path=args.dataset_path,
        base_model_name=args.base_model_name,
        which_questions=args.which_questions,
        shuffle_answers=args.shuffle_answers,
    )


if __name__ == "__main__":
    main()
