"""
Analyze correlation between MC answer positions (A/B/C/D) and uncertainty metrics.

This script checks whether answer letter positions correlate with model uncertainty,
which could explain why the MC probe's logit lens projection shows B/C-like tokens.

Usage:
    python analyze_mc_answer_bias.py
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from scipy import stats

from core import get_model_short_name

# Configuration - set these to match your experiment
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path if using one

OUTPUTS_DIR = Path("outputs")


def analyze_dataset(dataset_path: Path) -> Dict:
    """Analyze a single MC dataset for answer position biases."""
    with open(dataset_path) as f:
        data = json.load(f)

    items = data["data"]

    # Extract data
    correct_letters = []
    predicted_letters = []
    metrics = {name: [] for name in ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]}

    for item in items:
        correct_letters.append(item["correct_answer"])
        predicted_letters.append(item["predicted_answer"])
        for name in metrics:
            if name in item:
                metrics[name].append(item[name])

    # Convert to arrays
    correct_letters = np.array(correct_letters)
    predicted_letters = np.array(predicted_letters)
    metrics = {name: np.array(vals) for name, vals in metrics.items() if vals}

    # Map letters to positions (A=0, B=1, C=2, D=3)
    letter_to_pos = {"A": 0, "B": 1, "C": 2, "D": 3}
    correct_pos = np.array([letter_to_pos.get(l, -1) for l in correct_letters])
    predicted_pos = np.array([letter_to_pos.get(l, -1) for l in predicted_letters])

    # Filter out invalid positions
    valid_mask = (correct_pos >= 0) & (predicted_pos >= 0)
    correct_pos = correct_pos[valid_mask]
    predicted_pos = predicted_pos[valid_mask]
    metrics = {name: vals[valid_mask] for name, vals in metrics.items()}

    results = {
        "dataset": data["config"]["dataset"],
        "n_questions": len(correct_pos),
        "correct_letter_counts": {},
        "predicted_letter_counts": {},
        "correlations": {}
    }

    # Count letter frequencies
    for letter in ["A", "B", "C", "D"]:
        pos = letter_to_pos[letter]
        results["correct_letter_counts"][letter] = int(np.sum(correct_pos == pos))
        results["predicted_letter_counts"][letter] = int(np.sum(predicted_pos == pos))

    # Compute correlations between letter position and metrics
    for metric_name, metric_vals in metrics.items():
        # Correlation with correct answer position
        if len(np.unique(correct_pos)) > 1:
            r_correct, p_correct = stats.spearmanr(correct_pos, metric_vals)
        else:
            r_correct, p_correct = np.nan, np.nan

        # Correlation with predicted answer position
        if len(np.unique(predicted_pos)) > 1:
            r_predicted, p_predicted = stats.spearmanr(predicted_pos, metric_vals)
        else:
            r_predicted, p_predicted = np.nan, np.nan

        # Mean metric by correct answer letter
        mean_by_correct = {}
        for letter in ["A", "B", "C", "D"]:
            pos = letter_to_pos[letter]
            mask = correct_pos == pos
            if np.sum(mask) > 0:
                mean_by_correct[letter] = float(np.mean(metric_vals[mask]))

        # Mean metric by predicted answer letter
        mean_by_predicted = {}
        for letter in ["A", "B", "C", "D"]:
            pos = letter_to_pos[letter]
            mask = predicted_pos == pos
            if np.sum(mask) > 0:
                mean_by_predicted[letter] = float(np.mean(metric_vals[mask]))

        results["correlations"][metric_name] = {
            "correct_position_r": float(r_correct) if not np.isnan(r_correct) else None,
            "correct_position_p": float(p_correct) if not np.isnan(p_correct) else None,
            "predicted_position_r": float(r_predicted) if not np.isnan(r_predicted) else None,
            "predicted_position_p": float(p_predicted) if not np.isnan(p_predicted) else None,
            "mean_by_correct_letter": mean_by_correct,
            "mean_by_predicted_letter": mean_by_predicted,
        }

    return results


def plot_results(all_results: List[Dict], output_path: Path):
    """Plot answer letter distributions and metric correlations."""
    n_datasets = len(all_results)

    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 8))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    letters = ["A", "B", "C", "D"]
    x = np.arange(len(letters))
    width = 0.35

    for i, results in enumerate(all_results):
        dataset_name = results["dataset"]

        # Top row: letter frequency distribution
        ax = axes[0, i]
        correct_counts = [results["correct_letter_counts"].get(l, 0) for l in letters]
        predicted_counts = [results["predicted_letter_counts"].get(l, 0) for l in letters]

        ax.bar(x - width/2, correct_counts, width, label="Correct", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, predicted_counts, width, label="Predicted", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(letters)
        ax.set_xlabel("Answer Letter")
        ax.set_ylabel("Count")
        ax.set_title(f"{dataset_name}\nAnswer Distribution")
        ax.legend()

        # Bottom row: mean entropy by letter (entropy is the canonical uncertainty metric)
        ax = axes[1, i]
        if "entropy" in results["correlations"]:
            metric_data = results["correlations"]["entropy"]
            correct_means = [metric_data["mean_by_correct_letter"].get(l, 0) for l in letters]
            predicted_means = [metric_data["mean_by_predicted_letter"].get(l, 0) for l in letters]

            ax.bar(x - width/2, correct_means, width, label="By Correct", color="steelblue", alpha=0.8)
            ax.bar(x + width/2, predicted_means, width, label="By Predicted", color="coral", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(letters)
            ax.set_xlabel("Answer Letter")
            ax.set_ylabel("Mean entropy")

            r_correct = metric_data["correct_position_r"]
            r_predicted = metric_data["predicted_position_r"]
            r_str = f"r(correct)={r_correct:.3f}" if r_correct else ""
            if r_predicted:
                r_str += f", r(pred)={r_predicted:.3f}"
            ax.set_title(f"entropy by Letter\n{r_str}")
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def print_results(results: Dict):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"Dataset: {results['dataset']} (n={results['n_questions']})")
    print(f"{'='*60}")

    print("\nAnswer Letter Distribution:")
    print(f"  Correct:   ", end="")
    for letter in ["A", "B", "C", "D"]:
        count = results["correct_letter_counts"].get(letter, 0)
        pct = 100 * count / results["n_questions"]
        print(f"{letter}={count} ({pct:.1f}%)  ", end="")
    print()

    print(f"  Predicted: ", end="")
    for letter in ["A", "B", "C", "D"]:
        count = results["predicted_letter_counts"].get(letter, 0)
        pct = 100 * count / results["n_questions"]
        print(f"{letter}={count} ({pct:.1f}%)  ", end="")
    print()

    print("\nCorrelations with Answer Position (Spearman):")
    for metric_name, corr_data in results["correlations"].items():
        r_correct = corr_data["correct_position_r"]
        p_correct = corr_data["correct_position_p"]
        r_predicted = corr_data["predicted_position_r"]
        p_predicted = corr_data["predicted_position_p"]

        r_correct_str = f"{r_correct:+.3f}" if r_correct is not None else "N/A"
        p_correct_str = f"(p={p_correct:.3f})" if p_correct is not None else ""
        r_predicted_str = f"{r_predicted:+.3f}" if r_predicted is not None else "N/A"
        p_predicted_str = f"(p={p_predicted:.3f})" if p_predicted is not None else ""

        print(f"  {metric_name:12s}: correct_pos r={r_correct_str} {p_correct_str:12s}  predicted_pos r={r_predicted_str} {p_predicted_str}")

    print(f"\nMean entropy by Answer Letter:")
    if "entropy" in results["correlations"]:
        metric_data = results["correlations"]["entropy"]
        print("  By correct answer:   ", end="")
        for letter in ["A", "B", "C", "D"]:
            val = metric_data["mean_by_correct_letter"].get(letter)
            print(f"{letter}={val:.3f}  " if val else f"{letter}=N/A  ", end="")
        print()
        print("  By predicted answer: ", end="")
        for letter in ["A", "B", "C", "D"]:
            val = metric_data["mean_by_predicted_letter"].get(letter)
            print(f"{letter}={val:.3f}  " if val else f"{letter}=N/A  ", end="")
        print()


def find_mc_datasets() -> List[Path]:
    """Find all MC dataset files for the configured model/adapter."""
    model_short = get_model_short_name(BASE_MODEL_NAME)

    if MODEL_NAME != BASE_MODEL_NAME:
        # Adapter model - look for adapter-specific files
        adapter_short = get_model_short_name(MODEL_NAME)
        pattern = f"{model_short}_adapter-{adapter_short}_*_mc_dataset.json"
    else:
        # Base model - exclude adapter files
        pattern = f"{model_short}_*_mc_dataset.json"

    all_matches = list(OUTPUTS_DIR.glob(pattern))

    # If base model, filter out adapter files
    if MODEL_NAME == BASE_MODEL_NAME:
        all_matches = [p for p in all_matches if "_adapter-" not in p.name]

    return sorted(all_matches)


def main():
    # Find all applicable dataset files
    dataset_paths = find_mc_datasets()

    if not dataset_paths:
        model_short = get_model_short_name(BASE_MODEL_NAME)
        print(f"No MC dataset files found for {model_short} in {OUTPUTS_DIR}")
        return

    print(f"Found {len(dataset_paths)} MC dataset(s):")
    for p in dataset_paths:
        print(f"  {p.name}")

    all_results = []
    for path in dataset_paths:
        results = analyze_dataset(path)
        all_results.append(results)
        print_results(results)

    if all_results:
        model_short = get_model_short_name(BASE_MODEL_NAME)
        if MODEL_NAME != BASE_MODEL_NAME:
            adapter_short = get_model_short_name(MODEL_NAME)
            output_prefix = f"{model_short}_adapter-{adapter_short}"
        else:
            output_prefix = model_short

        output_path = OUTPUTS_DIR / f"{output_prefix}_mc_answer_bias.png"
        plot_results(all_results, output_path)

        # Save JSON results
        json_output = output_path.with_suffix(".json")
        with open(json_output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved JSON results to {json_output}")


if __name__ == "__main__":
    main()
