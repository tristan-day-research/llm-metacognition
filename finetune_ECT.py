import argparse
import math
import numpy as np
import os
import torch
import wandb
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

# Imports from helper files
from finetune_evaluation_metrics import (
      run_evaluation,
)
from finetune_utils import (
    save_model_final,
    save_checkpoint,
    save_training_parameters,
    load_tokenizer,
    load_model_with_lora,
    convert_entropy_to_soft_labels,
    prepare_model_and_tokenizer
)
from finetune_prompting import (
    build_self_confidence_prompts,
    build_multiple_choice_question_prompts,
    run_mcq_forward_pass,
    run_confidence_forward_pass
)
from finetune_data_handling import (
    load_mcq_results_data,
    get_batch,
    load_jsonl_dataset,
    filter_dataset_by_mcq_results,
    collate_fn
)



# ============================================================
# Entropy → scalar confidence → soft labels
# ============================================================

def compute_soft_labels(logits4, sigma=10.0):
    """
    Convert 4-way answer logits into soft 8-bin confidence distribution.

    Uses percentage-based Gaussian kernel to create soft labels.

    Args:
        logits4: tensor of shape [4] with logits for A, B, C, D
        sigma: Gaussian width in percentage space (default: 10)

    Returns:
        tensor of shape [8] with soft label distribution
    """
    # 1. Softmax over the 4 MCQ options
    probs = torch.softmax(logits4, dim=0)

    # 2. Entropy (natural logs)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()

    # 3. Convert entropy to "confidence percentage"
    #    confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

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


# ------------------------------------------------------------------
# Training Step
# ------------------------------------------------------------------

def train_step(model, tokenizer, batch, device, sigma, args, mcq_results_lookup=None):
    """
    Train step with support for frozen teacher (pre-recorded) or dynamic teacher.
    
    Args:
        mcq_results_lookup: Dict from load_mcq_results_data() or None
    
    Returns:
        loss tensor, or None if batch should be skipped
    """
    model.train()

    # ----------------------------------------------
    # 1. Get entropy (frozen vs dynamic teacher)
    # ----------------------------------------------

    # ----------------------------------------------
    # If using frozen teacher
    # ----------------------------------------------

    if args.use_recorded_responses:
        if mcq_results_lookup is None:
            raise ValueError(
                "--use_recorded_responses is True but no mcq_results_data provided!"
            )
        
        # Look up pre-recorded entropy for each question in batch
        entropies = []
        valid_indices = []  # Track which samples in batch are valid
        
        for i, row in enumerate(batch):
            qid = row.get("qid")
            if qid and qid in mcq_results_lookup:
                entropies.append(mcq_results_lookup[qid]["entropy"])
                valid_indices.append(i)
            else:
                # Skip this question
                print(f"⏭️  Skipping question without pre-recorded data: qid={qid}")
        
        # If no valid samples in batch, skip this training step
        if len(entropies) == 0:
            print(f"⚠️  Entire batch skipped - no pre-recorded data available")
            return None
        
        # Filter batch to only valid samples
        if len(valid_indices) < len(batch):
            batch = [batch[i] for i in valid_indices]
            print(f"  Batch reduced from {len(valid_indices)} to {len(batch)} samples")
        
        entropy = torch.tensor(entropies, dtype=torch.float32, device=device)

    # ----------------------------------------------
    # If using dynamic teacher: compute entropy from current model
    # ----------------------------------------------

    else:
        mcq_prompts = build_multiple_choice_question_prompts(batch, tokenizer)
        
        mcq_out = run_mcq_forward_pass(
            model=model,
            tokenizer=tokenizer,
            prompts=mcq_prompts,
            device=device,
            temperature=0.0,
            requires_grad=True,  # KEEP GRADIENTS for dynamic teacher
        )
        
        entropy = mcq_out["entropy"]  # [B]

    # Convert to soft labels
    soft = convert_entropy_to_soft_labels(entropy)  # [B,8]

    # ----------------------------------------------
    # 2. Confidence forward pass
    # ----------------------------------------------

    conf_prompts = build_self_confidence_prompts(batch, tokenizer)

    conf_out = run_confidence_forward_pass(
        model=model,
        tokenizer=tokenizer,
        prompts=conf_prompts,
        device=device,
        temperature=0.0,
        requires_grad=True,  # Need gradients for training
    )

    logits8 = conf_out["logits8"]  # [B, 8]

    # ----------------------------------------------
    # 3. Compute loss
    # ----------------------------------------------
    log_p = torch.log_softmax(logits8, dim=-1)
    loss = -(soft * log_p).sum(dim=-1).mean()

    # ----------------------------------------------
    # 4. Backprop
    # ----------------------------------------------
    loss.backward()

    return loss.detach()

    
# ============================================================
# Main training
# ============================================================

def train(args):
    """
    Trainer for Expected Confidence Task (ECT).
    Uses:
        - run_mcq_forward_pass()
        - run_confidence_forward_pass()
        - train_step()
        - val_step()
        - run_evaluation()
    """

    # ============================================================
    # Setup / Load model and data
    # ============================================================
    device = args.device
    tokenizer = load_tokenizer(args)
    model = load_model_with_lora(args, tokenizer).to(device)

    # Canonicalize model/tokenizer setup (fix pad_token warnings)
    model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)

    # Dataset loading ------------------------------------------------
    train_dataset = load_jsonl_dataset(args.train_data_path)
    val_dataset   = load_jsonl_dataset(args.val_data_path)

    print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
    # print(f"  Validation will run every {args.val_interval} steps")

    # Check if option A is somehow special
    # print("\nDEBUG: Checking first 50 questions:")
    # for i in range(50):
    #     q = val_dataset[i]
    #     print(f"Q{i}: correct={q.get('correct_letter')}, A={q['options']['A'][:40]}")
    
    # # Load pre-recorded MCQ results if using frozen teacher
    # mcq_results_lookup = None
    # if args.use_recorded_responses:
    #     if args.mcq_results_data is None:
    #         raise ValueError(
    #             "--use_recorded_responses requires --mcq_results_data to be specified"
    #         )
    #     mcq_results_lookup = load_mcq_results_data(args.mcq_results_data)
    #     if mcq_results_lookup is None:
    #         raise ValueError(f"Failed to load MCQ results from {args.mcq_results_data}")

    # Load pre-recorded MCQ results if using frozen teacher OR if provided for evaluation
    mcq_results_lookup = None
    if args.mcq_results_data is not None:
        mcq_results_lookup = load_mcq_results_data(args.mcq_results_data)
        if mcq_results_lookup is None:
            if args.use_recorded_responses:
                raise ValueError(f"Failed to load MCQ results from {args.mcq_results_data}")
            else:
                print(f"⚠️  Warning: Could not load MCQ results from {args.mcq_results_data}, continuing without pre-recorded entropy logging")
        
        # Filter datasets to only include questions with pre-recorded results (if using frozen teacher)
        if args.use_recorded_responses:
            train_dataset = filter_dataset_by_mcq_results(
                train_dataset, mcq_results_lookup, dataset_name="training"
            )
            val_dataset = filter_dataset_by_mcq_results(
                val_dataset, mcq_results_lookup, dataset_name="validation"
            )
    elif args.use_recorded_responses:
        raise ValueError(
            "--use_recorded_responses requires --mcq_results_data to be specified"
        )

    print(f"\n✓ Training dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
    

    if args.use_recorded_responses:
        print(f"✓ Using FROZEN TEACHER (pre-recorded responses)")
    else:
        print(f"✓ Using DYNAMIC TEACHER (current model)")

    # Optimizer ------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0
    )

    # Logging --------------------------------------------------------
    if args.save_wandb_artifact:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # Output / checkpoints
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Setup checkpoint directory with datetime
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_base_dir = os.path.join("local_checkpoints", f"{timestamp}_checkpoints")
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    print(f"✓ Checkpoints will be saved to: {os.path.abspath(checkpoint_base_dir)}")
    
    # Save training parameters to checkpoint directory
    save_training_parameters(args, checkpoint_base_dir)

    # Setup evaluation log file path
    log_dir = "finetune_logs"
    os.makedirs(log_dir, exist_ok=True)
    model_name_safe = args.model_name.replace("/", "-").replace("_", "-")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{model_name_safe}_evaluation_metrics.jsonl")

    # ============================================================
    # Baseline evaluation BEFORE training
    # ============================================================
    print("\n" + "="*60)
    print("Running baseline validation (before training)...")
    print("="*60)

    # DEBUG printing
    # test_batch = val_dataset[0:1]
    # test_prompts = build_multiple_choice_question_prompts(test_batch, tokenizer)
    # print("\n" + "="*80)
    # print("DEBUG: Sample MCQ Prompt")
    # print("="*80)
    # print(test_prompts[0])
    # print("="*80 + "\n")

    baseline_metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        step_name="baseline",
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=0,
        mcq_results_lookup=mcq_results_lookup,
    )

    print(f"\nBaseline Accuracy: {baseline_metrics['mcq_accuracy']:.4f}")
    print(f"Baseline Avg Entropy: {baseline_metrics['avg_entropy']:.4f}")
    print(f"Baseline Avg Confidence: {baseline_metrics['avg_confidence']:.4f}")
    print(f"- samples: {baseline_metrics['n_samples']}\n")

    # ============================================================
    # TRAINING LOOP
    # ============================================================



    step = 0
    losses = []

    while step < args.max_steps:
        batch = get_batch(train_dataset, args.batch_size)

        # -----------------------------
        # Train step 
        # -----------------------------
        optimizer.zero_grad()
        loss = train_step(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            sigma=args.sigma,
            device=device,
            args=args,
            mcq_results_lookup=mcq_results_lookup,
        )
        
        # Skip optimizer step if batch was skipped
        if loss is None:
            continue  # Don't increment step, try next batch
        
        optimizer.step()

        losses.append(loss.item())

        # W&B logging
        if args.save_wandb_artifact:
            wandb.log({"train/loss": loss.item(), "step": step})

        # -----------------------------
        # Periodic validation (Validation Step)
        # -----------------------------
        if (step % args.val_interval) == 0 and step > 0:
            print("\n" + "="*60)
            print(f"Validation at step {step}")
            print("="*60)

            val_metrics = run_evaluation(
                model=model,
                tokenizer=tokenizer,
                val_dataset=val_dataset,
                device=device,
                args=args,
                step_name="validation",
                num_samples=args.val_num_samples,
                log_file_path=log_file_path,
                step=step,
                mcq_results_lookup=mcq_results_lookup,
            )

            print(f"Val Accuracy: {val_metrics['mcq_accuracy']:.4f}")
            print(f"Val Loss:     {val_metrics['avg_loss']:.4f}")
            print(f"Val Entropy:  {val_metrics['avg_entropy']:.4f}")
            print(f"Val Conf:     {val_metrics['avg_confidence']:.4f}")
            print(f"Samples:      {val_metrics['n_samples']}")

        # -----------------------------
        # Periodic checkpointing
        # -----------------------------
        if (step % args.checkpoint_steps) == 0 and step > 0:
            # Include timestamp in repo name so each training run gets its own repo
            hf_checkpoint_repo_with_timestamp = None
            if args.save_hf_checkpoints and args.hf_checkpoint_repo:
                hf_checkpoint_repo_with_timestamp = f"{args.hf_checkpoint_repo}-{timestamp}"
            
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_base_dir=checkpoint_base_dir,
                step=step,
                save_hf_checkpoints=args.save_hf_checkpoints,
                hf_checkpoint_repo=hf_checkpoint_repo_with_timestamp,
                hf_checkpoint_private=args.hf_checkpoint_private
            )

        step += 1

    # ============================================================
    # Final metrics
    # ============================================================
    print("\n" + "="*60)
    print("Final evaluation:")
    print("="*60)

    final_metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        step_name="final",
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=step,
        mcq_results_lookup=mcq_results_lookup,
    )

    print(f"\nFinal Accuracy:  {final_metrics['mcq_accuracy']:.4f}")
    print(f"Final Loss:      {final_metrics['avg_loss']:.4f}")
    print(f"Final Confidence:{final_metrics['avg_confidence']:.4f}")

    if args.save_wandb_artifact:
        wandb.finish()

    print("Saving model.")
    success = save_model_final(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        hf_repo=args.hf_repo if args.save_hf else None,
        hf_private=args.hf_checkpoint_private,
        save_wandb_artifact=args.save_wandb_artifact
    )
    if success:
        print("✓ Model saved successfully!")
    else:
        print("❌ Model save failed! Check error messages above.")
        raise RuntimeError("Failed to save model")
    


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train dynamic metacognition model "
                    "(Explicit Confidence Task)"
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
    parser.add_argument("--train_data_path", type=str,
                        required=True,
                        help="Path to JSONL training dataset")
    
    parser.add_argument("--val_data_path", type=str,
                        default=None,
                        help="Path to JSONL validation dataset (optional)")
    
    parser.add_argument("--test_data_path", type=str,
                        default=None,
                        help="Path to JSONL test dataset (optional, will be evaluated after training)")

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
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Run validation every N training steps")
    parser.add_argument("--limit_val_batches", type=int, default=None,
                        help="Limit validation to N batches (None = use all validation data)")
    parser.add_argument("--val_num_samples", type=int, default=500,
                        help="Number of random questions to sample from validation dataset for validation steps (default: 500)")
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Sigma parameter for soft label distribution")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for sampling predictions (0.0 = deterministic/argmax, >0 = sampling)")
    parser.add_argument(
        "--no_shuffle_options", dest="shuffle_options", action="store_false",
        help="Disable shuffling of multiple choice answer options (shuffling is enabled by default)"
    )
    parser.add_argument(
        "--use_recorded_responses", action="store_true", default=None,
        help=("Use recorded MCQ responses (frozen teacher) as training "
              "targets instead of recomputing logits.")
    )
    parser.add_argument(
        "--no_use_recorded_responses", dest="use_recorded_responses",
        action="store_false",
        help=("Disable using recorded responses, use dynamic teacher "
              "(current model logits) instead.")
    )


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

    args = parser.parse_args()
    
    # Set default for shuffle_options (True if --no_shuffle_options was not provided)
    if not hasattr(args, 'shuffle_options'):
        args.shuffle_options = True
    
    # Validate that exactly one of --use_recorded_responses or --no_use_recorded_responses is set
    if args.use_recorded_responses is None:
        parser.error(
            "Exactly one of --use_recorded_responses or --no_use_recorded_responses must be specified. "
            "You must explicitly choose whether to use recorded responses (frozen teacher) or live responses (dynamic teacher)."
        )
    
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
