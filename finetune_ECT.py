import argparse
import math
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from finetune_utils import (
    write_log,
    MCQDataset,
    get_single_token_id,
    load_mcq_results_data,
    verify_and_resolve_options,
    init_wandb,
    log_wandb_metrics,
    log_wandb_config,
    log_device_info,
    save_hf_checkpoint,
    save_model_final,
    finish_wandb,
    build_confidence_prompts,
    _get_log_file_path,
)


def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries."""
    return batch


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


def compute_entropy(probs):
    """
    Compute entropy from probability distribution.

    Args:
        probs: tensor, list, or dict - probabilities for A, B, C, D
               If dict, should have keys "A", "B", "C", "D"
               If list/tensor, should be in order [A, B, C, D]

    Returns:
        scalar entropy value
    """
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


def convert_entropy_to_soft_labels(entropy, sigma=10.0):
    """
    Convert entropy value to soft 8-bin confidence distribution.

    Args:
        entropy: scalar entropy value
        sigma: Gaussian width in percentage space (default: 10)

    Returns:
        tensor of shape [8] with soft label distribution
    """
    # Convert entropy to "confidence percentage"
    # confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

    # Bin midpoints + widths (same as compute_soft_labels)
    bin_edges = torch.tensor(
        [0, 5, 10, 20, 40, 60, 80, 90, 100],
        dtype=torch.float32
    )
    bin_midpoints = torch.tensor(
        [2.5, 7.5, 15, 30, 50, 70, 85, 95],
        dtype=torch.float32
    )
    bin_widths = bin_edges[1:] - bin_edges[:-1]  # shape [8]

    # Gaussian kernel in percentage space
    distances = (bin_midpoints - confidence_percent) ** 2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    return weights / weights.sum()





# ------------------------------------------------------------------
# Training Step
# ------------------------------------------------------------------

def train_step(model, tokenizer, batch, sigma=10.0, device="cuda",
               mcq_results_lookup=None, log_file_path=None, args=None):
    """
    Single training step for Explicit Confidence Task.

    Supports two modes:
        - args.use_recorded_responses == True:
            Use recorded MCQ results (frozen teacher) as target.
        - args.use_recorded_responses == False:
            Use live logits from current model (dynamic teacher).

    Args:
        model: Language model
        tokenizer: Tokenizer
        batch: Batch of training examples
        sigma: Gaussian width for soft labels
        device: Device to run on
        mcq_results_lookup: Lookup dict for recorded MCQ results
        log_file_path: Path for logging
        args: Training arguments

    Returns:
        loss tensor
    """

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
            row, mcq_results_lookup, log_file_path
        )
        row["options"] = opts
        resolved_results.append(result_data)  # save for later target lookup

        prompt = (
            mc_setup_prompt + "\n\n"
            "Question:\n"
            f"{row['question']}\n"
            f"A: {opts['A']}\n"
            f"B: {opts['B']}\n"
            f"C: {opts['C']}\n"
            f"D: {opts['D']}\n"
            "Your choice (A, B, C, or D): "
        )
        answer_prompts.append(prompt)

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
                # Fallback to dynamic teacher mode when recorded data missing
                error_msg = (
                    f"No recorded MCQ results for qid: {qid}, "
                    "using dynamic teacher fallback"
                )
                print(error_msg)
                if log_file_path:
                    write_log(log_file_path, {"error": error_msg, "qid": qid})
                # Use current model's logits as fallback
                soft_target = compute_soft_labels(
                    answer_logits4[i], sigma=sigma
                ).to(device)
            else:
                # Pre-recorded file may store probabilities in different formats:
                # - `predicted_probs` or `answer_probs` (list or dict)
                # - `probs` (dict with keys "A", "B", "C", "D")
                teacher_probs = (
                    rd.get("predicted_probs") or
                    rd.get("answer_probs") or
                    rd.get("probs")
                )
                if teacher_probs is None:
                    # Fallback when probability data is missing
                    error_msg = (
                        f"Recorded MCQ results missing probability data for "
                        f"qid {qid}, using dynamic teacher fallback"
                    )
                    print(error_msg)
                    if log_file_path:
                        write_log(log_file_path,
                                  {"error": error_msg, "qid": qid})
                    # Use current model's logits as fallback
                    soft_target = compute_soft_labels(
                        answer_logits4[i], sigma=sigma
                    ).to(device)
                else:
                    entropy = compute_entropy(teacher_probs)
                    soft_target = convert_entropy_to_soft_labels(
                        entropy
                    ).to(device)

        else:
            # ------------------------------
            # DYNAMIC TEACHER MODE (old)
            # ------------------------------
            soft_target = compute_soft_labels(
                answer_logits4[i], sigma=sigma
            ).to(device)

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


# ============================================================
# Main training
# ============================================================


def train(args):
    """Main training function."""
    # Set up parameters log file immediately at the start
    log_dir = "fine_tune_logs"
    params_log_file_path = _get_log_file_path(
        log_dir, args.model_name, "parameters"
    )

    # Log all parameters (including defaults) before any other operations
    all_params = vars(args)
    write_log(params_log_file_path, {
        "type": "script_parameters",
        "timestamp": datetime.now().isoformat(),
        "parameters": all_params
    })
    print(f"All parameters logged to: {params_log_file_path}")

    # Initialize Weights & Biases
    init_wandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        script_path=__file__
    )

    # Handle device selection with CPU fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # Log device info
    log_device_info(device)
    
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

    dataset = MCQDataset(args.data_path)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    
    # Load MCQ results data if provided
    mcq_results_lookup = None
    # Strip whitespace in case there's any
    if args.mcq_results_data:
        args.mcq_results_data = args.mcq_results_data.strip()

    # Set up logging file for question matches/mismatches
    log_file_path = None
    if args.mcq_results_data:
        log_file_path = _get_log_file_path(
            log_dir, args.model_name, "question_comparison"
        )

    if args.mcq_results_data:
        print(f"Loading pre-recorded Multiple Choice Results data from: "
              f"{args.mcq_results_data}")
        # Check if file exists
        if not os.path.exists(args.mcq_results_data):
            print(f"Warning: MCQ results file not found at: "
                  f"{args.mcq_results_data}")
            print(f"Current working directory: {os.getcwd()}")
            print("Attempting to find file...")
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
                print("Error: Could not find MCQ results file. "
                      "Skipping verification.")
                mcq_results_lookup = None
            else:
                mcq_results_lookup = load_mcq_results_data(
                    args.mcq_results_data, log_file_path
                )
        else:
            mcq_results_lookup = load_mcq_results_data(
                args.mcq_results_data, log_file_path
            )
    else:
        print("No pre-recorded Multiple Choice Results data has been loaded")

    # Log file path already created above
    if log_file_path:
        print(f"Question comparison log will be written to: {log_file_path}")

    # Log dataset info
    log_wandb_config({
        "dataset_size": len(dataset),
        "num_batches": len(dataloader)
    })

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )

    # Log optimizer config
    log_wandb_config({
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate
    })

    for step, batch in enumerate(dataloader):
        if step >= args.max_steps:
            break

        loss = train_step(
            model, tokenizer, batch, device=device, sigma=args.sigma,
            mcq_results_lookup=mcq_results_lookup,
            log_file_path=log_file_path, args=args
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to wandb
        log_wandb_metrics({
            "train/loss": loss.item(),
            "train/step": step
        }, step=step)

        if step % args.log_interval == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        # Save checkpoint to HuggingFace Hub
        if args.save_hf_checkpoints and args.hf_checkpoint_repo:
            if (step + 1) % args.checkpoint_steps == 0:
                save_hf_checkpoint(
                    model, tokenizer, args.hf_checkpoint_repo, step + 1,
                    private=args.hf_checkpoint_private
                )
                print(f"Checkpoint saved to HuggingFace Hub: "
                      f"{args.hf_checkpoint_repo}-step-{step+1}")

    # Save model locally and optionally to HuggingFace Hub
    if args.save_hf and args.hf_repo is None:
        raise ValueError("--hf_repo must be provided when --save_hf is set")

    save_model_final(
        model, tokenizer, args.output_dir,
        hf_repo=args.hf_repo if args.save_hf else None,
        hf_private=args.hf_checkpoint_private,
        save_wandb_artifact=args.save_wandb_artifact
    )

    finish_wandb()

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
    parser.add_argument(
        "--use_recorded_responses", action="store_true", default=True,
        help=("Use recorded MCQ responses (frozen teacher) as training "
              "targets instead of recomputing logits. (Default: True)")
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
