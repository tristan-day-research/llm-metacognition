"""
run_evaluations.py

Utility script/module to:
  ✓ Load (optionally LoRA-adapted) model via core.model_utils.load_model_and_tokenizer()
  ✓ Optionally evaluate base (un-finetuned) model first for comparison
  ✓ Run evaluate_model() from evaluation_metrics.py
  ✓ Evaluate any dataset in PopMC / SimpleMC / TriviaMC format

WORKFLOW REPLICATION:
  This script follows the EXACT same workflow as run_finetuning.py for perfect replication:
  - Uses load_jsonl_dataset() for data loading (same function, same shuffling)
  - Uses prepare_model_and_tokenizer() for model setup (same function)
  - Uses evaluate_model() which calls run_evaluation() (same function)
  - Uses build_multiple_choice_question_prompts() for prompts (same function)
  - Uses run_mcq_forward_pass() for MCQ inference (same function, same params)
  - Uses build_self_confidence_prompts() and build_other_confidence_prompts() (same functions)
  - Uses run_confidence_forward_pass() for confidence inference (same function)
  
  This ensures identical behavior to training-time evaluation in run_finetuning.py.

Usage example:
    from run_evaluations import run_evaluations

    results = run_evaluations(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        lora_repo="",
        dataset_path="data/SimpleMC.jsonl",
        merge=True,
        evaluate_base_first=True
    )

"""

# --- repo path bootstrap (so root-level imports like `finetune_prompting`,
# `finetune_config` resolve when run from anywhere) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import os
import torch
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.model_utils import load_model_and_tokenizer
from evaluation_metrics import evaluate_model
from data_handling import load_jsonl_dataset
from utils import prepare_model_and_tokenizer


def _print_letter_diagnostics(log_file_path: str, log_prefix: str = "") -> None:
    """Position-bias diagnostics computed from the per-sample JSONL log.

    Reads the rows just written by evaluate_model() and reports two checks
    that the standard summary doesn't expose:
      - chi-square test for uniformity of predicted positions (detects
        A/B/C/D answer bias independent of correctness)
      - per-position conditional accuracy P(correct | true_position=p),
        which catches the "always predicts B" failure mode

    Position-based (not display-letter-based) so it's robust to the
    randomized letter mapping used in training.
    """
    sample_type = f"{log_prefix}eval_sample" if log_prefix else "eval_sample"
    rows = []
    with open(log_file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == sample_type:
                rows.append(r)
    if not rows:
        return

    pred_pos = [r["model_answer_position"] for r in rows
                if r.get("model_answer_position") is not None]
    n = len(pred_pos)
    if n == 0:
        return

    print(f"\n--- POSITION-BIAS DIAGNOSTICS ({log_prefix or 'eval'}) ---")
    pred_dist = Counter(pred_pos)
    expected = n / 4.0
    chi2 = sum((pred_dist.get(p, 0) - expected) ** 2 / expected for p in range(4))
    print(f"  Chi-square (predicted-position uniformity): {chi2:.2f}  "
          f"(crit @ p=0.05: 7.81 → {'uniform' if chi2 < 7.81 else 'biased'})")
    print(f"  Predicted-position counts: " +
          ", ".join(f"pos{p}={pred_dist.get(p, 0)}" for p in range(4)))

    print("  Accuracy conditioned on correct position:")
    for pos in range(4):
        total = sum(1 for r in rows if r.get("correct_answer_position") == pos)
        if total == 0:
            continue
        n_correct = sum(1 for r in rows
                        if r.get("correct_answer_position") == pos
                        and r.get("is_correct"))
        print(f"    correct=pos{pos}: {n_correct}/{total} ({n_correct/total:.2%})")

###############################################################################
# Main evaluation wrapper
###############################################################################
def run_evaluations(
    base_model: str,
    lora_repo: str,
    dataset_path: str,
    sigma: float,
    merge: bool = False,
    max_samples: int = None,
    evaluate_base_first: bool = False,
    use_wandb: bool = False,
    wandb_project: str = None,
    compute_confidence: bool = True,
    compute_other_confidence: bool = True,
    loss_type: str = "gaussian_soft_bin_ce",
    log_dir: str = "finetuned_evals",
    confidence_letter_scheme: str = "A-H",
    confidence_letter_random_seed: int = None,
):
    """
    Load finetuned model + tokenizer and run full evaluation pipeline
    on a dataset using your existing evaluate_model() utilities.

    Args:
        base_model:  Name of base Llama model.
        lora_repo:   HF repo containing LoRA weights.
        dataset_path: JSONL dataset of {'question','choices','answer'} items.
        sigma: Gaussian width parameter for soft label conversion (REQUIRED - no default to prevent silent errors).
        merge:       If True, merges LoRA weights into base model.
        max_samples: Optional limit on evaluation samples.
        evaluate_base_first: If True, evaluate base model first, then finetuned model.
        use_wandb: If True, log results to Weights & Biases.
        wandb_project: WandB project name (only used if use_wandb=True).
        compute_confidence: If True, compute self-confidence predictions.
        compute_other_confidence: If True, compute other-confidence predictions.
        loss_type: Loss type for evaluation ('gaussian_soft_bin_ce' or 'scalar_confidence_mse').
        log_dir: Directory to save evaluation logs (default: "finetuned_evals").
        confidence_letter_scheme: Letter scheme for confidence bins ('A-H', 'S-Z', or 'random').
        confidence_letter_random_seed: Random seed for 'random' scheme (optional).

    Returns:
        results: dict with evaluation metrics. If evaluate_base_first=True,
                 returns dict with 'base' and 'finetuned' keys containing results.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Setup logging directory and file paths
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    dataset_name = dataset_path.stem
    lora_name = lora_repo.split("/")[-1]  # Get just the repo name, not full path
    base_model_safe = base_model.replace("/", "-").replace("_", "-")
    
    # Load dataset using the proper loader (handles PopMC format conversion)
    print(f"\n📂 Loading dataset from: {dataset_path}")
    data = load_jsonl_dataset(str(dataset_path), dataset_type="evaluation")
    print(f"✓ Loaded {len(data)} samples from dataset")

    if max_samples is not None:
        data = data[:max_samples]
        print(f"✓ Limited to {max_samples} samples for evaluation")

    results = {}

    # Setup single log file (used for both models if evaluate_base_first=True)
    if evaluate_base_first:
        # Use a single log file for both evaluations
        combined_log_file = os.path.join(
            log_dir,
            f"{timestamp}_{base_model_safe}_{lora_name}_{dataset_name}_n{len(data)}_comparison.jsonl"
        )
        print(f"📝 Logging both models to: {combined_log_file}")
    else:
        # Single model evaluation - use standard naming
        combined_log_file = os.path.join(
            log_dir,
            f"{timestamp}_{base_model_safe}_{lora_name}_{dataset_name}_n{len(data)}.jsonl"
        )
        print(f"📝 Logging finetuned model results to: {combined_log_file}")

    # Optionally evaluate base model first
    if evaluate_base_first:
        print("\n" + "="*60)
        print(" EVALUATING BASE (UN-FINETUNED) MODEL")
        print("="*60 + "\n")
        
        base_model_obj, base_tokenizer, _ = load_model_and_tokenizer(base_model)
        # Ensure padding token is set
        base_model_obj, base_tokenizer = prepare_model_and_tokenizer(base_model_obj, base_tokenizer)
        
        base_results = evaluate_model(
            model=base_model_obj,
            tokenizer=base_tokenizer,
            dataset=data,
            sigma=sigma,
            compute_confidence=compute_confidence,
            compute_other_confidence=compute_other_confidence,
            loss_type=loss_type,
            num_samples=max_samples,
            log_file_path=combined_log_file,  # Same file for both
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=f"{dataset_name}_base",
            log_prefix="instruct_",
            confidence_letter_scheme=confidence_letter_scheme,
            confidence_letter_random_seed=confidence_letter_random_seed,
        )
        
        results['base'] = base_results
        _print_letter_diagnostics(combined_log_file, log_prefix="instruct_")

        # Clean up base model to free memory
        del base_model_obj
        del base_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n" + "="*60)
        print(" BASE MODEL EVALUATION COMPLETE")
        print("="*60 + "\n")

    # Evaluate finetuned model
    print("\n" + "="*60)
    print(" EVALUATING FINETUNED MODEL")
    print("="*60 + "\n")
    
    print("\n==============================")
    print(" Loading finetuned model ")
    print("==============================\n")

    model, tokenizer, _ = load_model_and_tokenizer(
        base_model_name=base_model,
        adapter_path=lora_repo,
        merge=merge,
    )
    # Ensure padding token is set
    model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)

    finetuned_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=data,
        sigma=sigma,
        compute_confidence=compute_confidence,
        compute_other_confidence=compute_other_confidence,
        loss_type=loss_type,
        num_samples=max_samples,
        log_file_path=combined_log_file,  # Same file for both
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=f"{dataset_name}_finetuned",
        log_prefix="finetuned_",
        confidence_letter_scheme=confidence_letter_scheme,
        confidence_letter_random_seed=confidence_letter_random_seed,
    )
    
    _print_letter_diagnostics(combined_log_file, log_prefix="finetuned_")

    if evaluate_base_first:
        results['finetuned'] = finetuned_results
    else:
        results = finetuned_results

    return results

###############################################################################
# Optional CLI entry point
###############################################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation on finetuned Llama model")
    parser.add_argument("--base_model", required=True, help="Base model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--lora_repo", required=True, help="HuggingFace repo containing LoRA weights")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights into base model")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--evaluate_base_first", action="store_true", 
                       help="Evaluate base (un-finetuned) model first for comparison")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Log results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="WandB project name (default: 'llm-evaluation')")
    parser.add_argument("--no_confidence", action="store_true",
                       help="Skip confidence computation (faster evaluation)")
    parser.add_argument("--no_other_confidence", action="store_true",
                       help="Skip other-confidence computation")
    parser.add_argument("--loss_type", type=str, default="gaussian_soft_bin_ce",
                       choices=["gaussian_soft_bin_ce", "scalar_confidence_mse"],
                       help="Loss type for evaluation (default: gaussian_soft_bin_ce)")
    parser.add_argument("--sigma", type=float, required=True,
                       help="Gaussian width parameter for soft label conversion (REQUIRED - no default to prevent silent errors)")
    parser.add_argument("--log_dir", type=str, default="finetuned_evals",
                       help="Directory to save evaluation logs (default: finetuned_evals)")
    parser.add_argument("--confidence_letter_scheme", type=str, default="A-H",
                       choices=["A-H", "S-Z", "random"],
                       help="Letter scheme for confidence bins (default: A-H)")
    parser.add_argument("--confidence_letter_random_seed", type=int, default=None,
                       help="Random seed for 'random' confidence_letter_scheme (optional)")

    args = parser.parse_args()

    out = run_evaluations(
        base_model=args.base_model,
        lora_repo=args.lora_repo,
        dataset_path=args.dataset,
        sigma=args.sigma,
        merge=args.merge,
        max_samples=args.max_samples,
        evaluate_base_first=args.evaluate_base_first,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        compute_confidence=not args.no_confidence,
        compute_other_confidence=not args.no_other_confidence,
        loss_type=args.loss_type,
        log_dir=args.log_dir,
        confidence_letter_scheme=args.confidence_letter_scheme,
        confidence_letter_random_seed=args.confidence_letter_random_seed,
    )

    print("\n" + "="*60)
    print(" FINAL EVALUATION RESULTS")
    print("="*60)
    print(json.dumps(out, indent=2))
    
    # Print comparison if base was evaluated
    if args.evaluate_base_first and isinstance(out, dict) and 'base' in out and 'finetuned' in out:
        print("\n" + "="*60)
        print(" COMPARISON: BASE vs FINETUNED")
        print("="*60)
        base_acc = out['base'].get('mcq_accuracy', 0)
        finetuned_acc = out['finetuned'].get('mcq_accuracy', 0)
        improvement = finetuned_acc - base_acc
        print(f"Base Model Accuracy:      {base_acc:.4f}")
        print(f"Finetuned Model Accuracy: {finetuned_acc:.4f}")
        print(f"Improvement:              {improvement:+.4f} ({improvement*100:+.2f}%)")
