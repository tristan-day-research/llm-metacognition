import argparse
import json
import math
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import wandb


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

        # 4 options here as unlabeled list
        options = [(correct, True)] + [(d, False) for d in distractors]
        random.shuffle(options)

        # Assign shuffled answers to A/B/C/D
        labeled = {}
        correct_letter = None
        for label, (text, is_correct) in zip("ABCD", options):
            labeled[label] = text
            if is_correct:
                correct_letter = label

        return {
            "qid": row["qid"],
            "question": question,
            "options": labeled,
            "correct_letter": correct_letter
        }

# ============================================================
# Entropy → scalar confidence → soft labels
# ============================================================

def compute_soft_labels(logits4, sigma=0.15):
    ps = torch.softmax(logits4, dim=0)
    H = -(ps * torch.log(ps + 1e-12)).sum()
    c = 1 - (H / math.log(4))
    # Ensure centers is on the same device as logits4
    centers = torch.linspace(1/16, 15/16, 8, device=logits4.device)
    soft = torch.exp(-(centers - c)**2 / (2*sigma*sigma))
    return soft / soft.sum()
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


def train_step(model, tokenizer, batch, sigma=0.15, device="cuda"):
    B = len(batch)

    # ------------------------------------------------------------------
    # 1. First pass: answer MCQ (teacher signal) — NO GRAD, no leakage
    # ------------------------------------------------------------------
    answer_prompts = []
    for row in batch:
        q = row["question"]
        opts = row["options"]
        p = (
            f"Question: {q}\n"
            "Options:\n"
            f"A. {opts['A']}\n"
            f"B. {opts['B']}\n"
            f"C. {opts['C']}\n"
            f"D. {opts['D']}\n\n"
            "Answer with one letter: A, B, C, or D.\n"
            "Answer: "
        )
        answer_prompts.append(p)

    enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model(**enc)
        # logits for *next* token after the prompt
        final_logits = out.logits[:, -1, :]  # [B, vocab]

        # Extract logits for A/B/C/D tokens (robustly)
        abcd_ids = torch.tensor(
            [get_single_token_id(tokenizer, c) for c in "ABCD"],
            device=device,
            dtype=torch.long
        )  # [4]

        answer_logits4 = final_logits[:, abcd_ids]  # [B, 4]

        # Soft labels from current logits (teacher entropy)
        soft_targets = torch.stack(
            [compute_soft_labels(ans, sigma=sigma) for ans in answer_logits4]
        ).to(device)  # [B, 8]

    # ------------------------------------------------------------------
    # 2. Second pass: explicit confidence, independent prompt
    # ------------------------------------------------------------------
    
    conf_prompts = []
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
        conf_prompts.append(p)

    enc2 = tokenizer(conf_prompts, return_tensors="pt", padding=True).to(device)
    out2 = model(**enc2)
    final_logits2 = out2.logits[:, -1, :]  # [B, vocab]

    # Extract logits for confidence tokens A–H (again robustly)
    conf_ids = torch.tensor(
        [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
        device=device,
        dtype=torch.long
    )  # [8]

    conf_logits = final_logits2[:, conf_ids]  # [B, 8]

    # Soft-label cross entropy: only second pass gets gradients
    log_probs = torch.log_softmax(conf_logits, dim=-1)  # [B, 8]
    loss = -(soft_targets * log_probs).sum(dim=1).mean()

    return loss

# ============================================================
# Main training
# ============================================================

def train(args):
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
            
        loss = train_step(model, tokenizer, batch, device=device, sigma=args.sigma)

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
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Sigma parameter for soft label distribution")

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
