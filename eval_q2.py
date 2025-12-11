import json
import argparse
from pathlib import Path
from tqdm import tqdm

# === your utilities ===
from finetune_utils import run_generation, load_jsonl  # adapt if names differ
from finetune_load_finetuned_model import load_finetuned_model
from finetune_data_handling import load_tokenizer_and_base  # guessing name: adjust if needed


# ------------------------------------------------------------
# Simple abstention classifier
# ------------------------------------------------------------
ABSTENTION_MARKERS = [
    "i don't know",
    "unknown",
    "cannot answer",
    "can't answer",
    "no king of france",
    "not enough information",
    "n/a",
    "no answer",
    "undefined",
    "not answerable",
    "unanswerable",
    "does not exist",
]

def classify_response(response: str):
    """
    Returns:
        "abstain"     → if model expresses uncertainty
        "hallucinate" → if model gives a specific answer
    """

    resp = response.strip().lower()

    # Abstention / refusal detection
    for marker in ABSTENTION_MARKERS:
        if marker in resp:
            return "abstain"

    # If it looks like a concrete answer (names, numbers, places)
    # Very lightweight heuristic; you can refine.
    if any(token in resp for token in [" is ", " was ", " are ", " were ", " born ", " in "]):
        return "hallucinate"

    # Otherwise default to hallucination (conservative)
    return "hallucinate"


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------
def evaluate_model_on_q2(model, tokenizer, q2_data, max_new_tokens=64):
    outputs = []
    hallucinations = 0
    abstentions = 0

    for item in tqdm(q2_data, desc="Evaluating"):
        q = item["question"]
        prompt = f"Q: {q}\nA:"

        # Run through your generation stack
        generated = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        category = classify_response(generated)

        if category == "hallucinate":
            hallucinations += 1
        else:
            abstentions += 1

        outputs.append({
            "id": item.get("id", None),
            "question": q,
            "answer": generated,  # Full model response (exact text)
            "classification": category,  # "abstain" or "hallucinate" based on classify_response()
        })

    total = len(q2_data)
    return {
        "total": total,
        "hallucinations": hallucinations,
        "abstentions": abstentions,
        "hallucination_rate": hallucinations / total,
        "abstention_rate": abstentions / total,
        "outputs": outputs,
    }


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main(args):
    # -------- Load Q2 dataset --------
    q2_data = load_jsonl(args.q2_path)  # You may need to adapt this name
    total_questions = len(q2_data)
    print(f"Loaded {total_questions} Q2 questions.")
    
    # Limit number of questions if specified
    if args.num_questions is not None:
        q2_data = q2_data[:args.num_questions]
        print(f"Using first {len(q2_data)} questions (limited by --num_questions).")

    # -------- Load models based on selection --------
    base_model = None
    base_tokenizer = None
    ft_model = None
    ft_tokenizer = None
    
    if args.model in ["base", "both"]:
        print("\nLoading BASE model...")
        base_model, base_tokenizer = load_tokenizer_and_base(args.base_model)
    
    if args.model in ["finetuned", "both"]:
        print("\nLoading FINETUNED model...")
        ft_model, ft_tokenizer = load_finetuned_model(
            model_name=args.base_model,
            checkpoint_path=args.finetuned_checkpoint
        )

    # -------- Evaluate models --------
    base_results = None
    ft_results = None
    
    if args.model in ["base", "both"]:
        print("\nRunning Q2 on BASE model...")
        base_results = evaluate_model_on_q2(base_model, base_tokenizer, q2_data)
    
    if args.model in ["finetuned", "both"]:
        print("\nRunning Q2 on FINETUNED model...")
        ft_results = evaluate_model_on_q2(ft_model, ft_tokenizer, q2_data)

    # -------- Save outputs --------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if base_results is not None:
        with open(out_dir / "base_q2_results.json", "w") as f:
            json.dump(base_results, f, indent=2)

    if ft_results is not None:
        with open(out_dir / "finetuned_q2_results.json", "w") as f:
            json.dump(ft_results, f, indent=2)

    print("\n=== SUMMARY ===")
    if base_results is not None:
        print("BASE:")
        print(f"  Abstention:    {base_results['abstention_rate']:.3f}")
        print(f"  Hallucination: {base_results['hallucination_rate']:.3f}")

    if ft_results is not None:
        print("\nFINETUNED:")
        print(f"  Abstention:    {ft_results['abstention_rate']:.3f}")
        print(f"  Hallucination: {ft_results['hallucination_rate']:.3f}")

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--q2_path", type=str, required=True,
                        help="Path to Q2 dataset (.jsonl)")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model identifier (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--finetuned_checkpoint", type=str, default=None,
                        help="HF checkpoint path for the finetuned model (required if --model is 'finetuned' or 'both')")
    parser.add_argument("--output_dir", type=str, default="finetune_evals",
                        help="Where to store results")
    parser.add_argument("--num_questions", type=int, default=None,
                        help="Number of questions to evaluate (default: all questions)")
    parser.add_argument("--model", type=str, default="both", choices=["base", "finetuned", "both"],
                        help="Which model(s) to evaluate: 'base', 'finetuned', or 'both' (default: both)")

    args = parser.parse_args()
    
    # Validate finetuned_checkpoint is provided if evaluating finetuned model
    if args.model in ["finetuned", "both"] and not args.finetuned_checkpoint:
        parser.error("--finetuned_checkpoint is required when --model is 'finetuned' or 'both'")
    
    main(args)
