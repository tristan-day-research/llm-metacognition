"""
Explicit Confidence Task (ECT) - Test model introspection capabilities.

For each question:
1. Ask the model the MC question directly, record answer + probabilities
2. In a separate prompt (independent context), ask the model its confidence
   on the S-Z scale

Analyzes how well the model's stated confidence correlates with actual accuracy.

Metrics computed:
- ECE (Expected Calibration Error): lower = better calibrated
- AUROC: how well confidence discriminates correct vs incorrect (0.5=random, 1.0=perfect)
- Brier Score: mean squared error between confidence and correctness (lower=better)
- Point-biserial correlation: between stated confidence and correctness
- Spearman correlation: between stated confidence and MC top probability

Configuration is set at the top of the script - no CLI args needed.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats as scipy_stats

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    should_use_chat_template,
    DEVICE,
)
from core.questions import load_questions
from prompts import (
    format_direct_prompt,
    format_stated_confidence_prompt,
    format_direct_prompt_base,
    format_stated_confidence_prompt_base,
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    get_stated_confidence_signal,
    get_stated_confidence_response,
)

# =============================================================================
# CONFIGURATION — edit values in experiment_config.ConfidenceTestsConfig
# =============================================================================
from experiment_config import ConfidenceTestsConfig as _C

MODEL_TYPE = _C.MODEL_TYPE
FEW_SHOT_MODE = _C.FEW_SHOT_MODE
RANDOM_FEW_SHOT_SOURCE = _C.RANDOM_FEW_SHOT_SOURCE
MODEL_CONFIGS = {k: dict(v) for k, v in _C.MODEL_CONFIGS.items()}
DATASET = _C.DATASET
NUM_QUESTIONS = _C.NUM_QUESTIONS
SEED = _C.SEED
BATCH_SIZE = _C.BATCH_SIZE
LOAD_IN_4BIT = _C.LOAD_IN_4BIT
LOAD_IN_8BIT = _C.LOAD_IN_8BIT
OUTPUT_DIR = _C.OUTPUT_DIR


# =============================================================================
# INFERENCE
# =============================================================================

def get_option_probs(model, tokenizer, prompts, option_token_ids, add_special_tokens=False):
    """
    Run batched forward pass and extract option probabilities at the last token.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of formatted prompt strings
        option_token_ids: List of token IDs for the answer options
        add_special_tokens: Whether to prepend BOS token.
            True for base model (raw prompts need BOS).
            False for instruct model (chat template already includes BOS).

    Returns:
        all_probs: List[np.ndarray] - softmax over option tokens (normalized)
        all_logits: List[np.ndarray] - raw logits for option tokens
        all_option_mass: List[float] - fraction of total probability on option tokens
            (diagnostic: high = model understands the task format)
    """
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    all_probs = []
    all_logits = []
    all_option_mass = []

    for i in range(input_ids.shape[0]):
        logits_i = outputs.logits[i, -1, :]

        # Total probability mass on option tokens (full-vocabulary softmax)
        full_probs = torch.softmax(logits_i, dim=-1)
        option_mass = full_probs[option_token_ids].sum().item()
        all_option_mass.append(option_mass)

        # Probabilities normalized over option tokens only
        option_logits = logits_i[option_token_ids]
        probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
        logits_np = option_logits.cpu().numpy()
        all_probs.append(probs)
        all_logits.append(logits_np)

    return all_probs, all_logits, all_option_mass


# =============================================================================
# ANALYSIS METRICS
# =============================================================================

def compute_ece(confidences, correctness, n_bins=10):
    """
    Expected Calibration Error using equal-width bins.
    Lower = better calibrated. Range [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_bin = mask.sum()
        if n_bin > 0:
            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += n_bin * abs(bin_acc - bin_conf)
    return ece / total


def compute_auroc(confidences, correctness):
    """
    Area Under ROC Curve for confidence discriminating correct vs incorrect.
    0.5 = random, 1.0 = perfect discrimination.
    """
    n_pos = int(correctness.sum())
    n_neg = len(correctness) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sorted_idx = np.argsort(-confidences)
    sorted_correct = correctness[sorted_idx]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp, fp = 0, 0
    for c in sorted_correct:
        if c:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Use trapezoid (trapz was deprecated in numpy 2.0)
    try:
        return float(np.trapezoid(tpr_list, fpr_list))
    except AttributeError:
        # Fallback for older numpy versions
        return float(np.trapz(tpr_list, fpr_list))


def compute_brier_score(confidences, correctness):
    """
    Brier score = mean((confidence - correctness)^2).
    Lower = better. Range [0, 1].
    """
    return float(np.mean((confidences - correctness) ** 2))


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(results, output_path, model_label):
    """
    Create comprehensive calibration analysis plots and compute summary stats.

    6-panel figure:
    1. Calibration curve (stated confidence vs actual accuracy)
    2. Accuracy by confidence level (bar chart)
    3. Confidence distribution by correctness
    4. Stated confidence vs MC top probability (scatter)
    5. Calibration gap (over/under-confidence per bin)
    6. Summary statistics text

    Returns:
        dict of summary statistics
    """
    confidences = np.array([r["stated_confidence_value"] for r in results])
    correctness = np.array([r["is_correct"] for r in results], dtype=float)
    mc_top_probs = np.array([max(r["mc_probs"]) for r in results])
    confidence_responses = [r["stated_confidence_response"] for r in results]
    mc_option_mass = np.array([r["mc_option_mass"] for r in results])
    conf_option_mass = np.array([r["confidence_option_mass"] for r in results])

    conf_options = list(STATED_CONFIDENCE_OPTIONS.keys())
    conf_midpoints = [STATED_CONFIDENCE_MIDPOINTS[k] for k in conf_options]
    conf_labels_short = [f"{k}\n({STATED_CONFIDENCE_OPTIONS[k]})" for k in conf_options]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Panel 1: Calibration curve ---
    ax = axes[0, 0]
    bin_accs, bin_confs, bin_counts, bin_opts = [], [], [], []
    for opt, mid in zip(conf_options, conf_midpoints):
        mask = np.array([r == opt for r in confidence_responses])
        if mask.sum() > 0:
            bin_accs.append(float(correctness[mask].mean()))
            bin_confs.append(mid)
            bin_counts.append(int(mask.sum()))
            bin_opts.append(opt)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    if bin_confs:
        ax.plot(bin_confs, bin_accs, "o-", color="steelblue", markersize=8, zorder=5)
        for conf, acc, cnt in zip(bin_confs, bin_accs, bin_counts):
            ax.annotate(f"n={cnt}", (conf, acc), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7)
    ax.set_xlabel("Stated Confidence (midpoint)")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title("Calibration Curve")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Accuracy by confidence level ---
    ax = axes[0, 1]
    accs_by_opt, counts_by_opt = [], []
    for opt in conf_options:
        mask = np.array([r == opt for r in confidence_responses])
        if mask.sum() > 0:
            accs_by_opt.append(float(correctness[mask].mean()))
        else:
            accs_by_opt.append(0.0)
        counts_by_opt.append(int(mask.sum()))

    bars = ax.bar(range(len(conf_options)), accs_by_opt, color="steelblue",
                  edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(conf_options)))
    ax.set_xticklabels(conf_labels_short, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Confidence Level")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, cnt in zip(bars, counts_by_opt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={cnt}", ha="center", va="bottom", fontsize=7)

    # --- Panel 3: Confidence distribution by correctness ---
    ax = axes[0, 2]
    correct_counts, incorrect_counts = [], []
    for opt in conf_options:
        c_mask = np.array(
            [r == opt and bool(c) for r, c in zip(confidence_responses, correctness)]
        )
        i_mask = np.array(
            [r == opt and not bool(c) for r, c in zip(confidence_responses, correctness)]
        )
        correct_counts.append(int(c_mask.sum()))
        incorrect_counts.append(int(i_mask.sum()))

    x = np.arange(len(conf_options))
    w = 0.35
    ax.bar(x - w / 2, correct_counts, w, label="Correct", color="green",
           alpha=0.7, edgecolor="black")
    ax.bar(x + w / 2, incorrect_counts, w, label="Incorrect", color="red",
           alpha=0.7, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(conf_options, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution by Correctness")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: MC top probability vs stated confidence ---
    ax = axes[1, 0]
    c_mask = correctness.astype(bool)
    ax.scatter(confidences[c_mask], mc_top_probs[c_mask],
               alpha=0.3, s=15, c="green", label="Correct")
    ax.scatter(confidences[~c_mask], mc_top_probs[~c_mask],
               alpha=0.3, s=15, c="red", label="Incorrect")

    try:
        rho_mc, p_mc = scipy_stats.spearmanr(confidences, mc_top_probs)
    except Exception:
        rho_mc, p_mc = float("nan"), float("nan")

    ax.set_xlabel("Stated Confidence")
    ax.set_ylabel("MC Top Probability (normalized)")
    rho_str = f"{rho_mc:.3f}" if not np.isnan(rho_mc) else "N/A"
    ax.set_title(f"Stated Conf vs MC Certainty (\u03c1={rho_str})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Calibration gap ---
    ax = axes[1, 1]
    if bin_confs:
        gaps = [c - a for c, a in zip(bin_confs, bin_accs)]
        colors_gap = ["red" if g > 0 else "blue" for g in gaps]
        ax.bar(range(len(gaps)), gaps, color=colors_gap, edgecolor="black", alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(gaps)))
        ax.set_xticklabels(bin_opts, fontsize=8)
    ax.set_ylabel("Confidence \u2212 Accuracy")
    ax.set_title("Calibration Gap (red=overconfident)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 6: Summary statistics ---
    ax = axes[1, 2]
    ax.axis("off")

    overall_acc = float(correctness.mean())
    mean_conf = float(confidences.mean())
    ece = compute_ece(confidences, correctness)
    auroc = compute_auroc(confidences, correctness)
    brier = compute_brier_score(confidences, correctness)

    try:
        rpb, p_rpb = scipy_stats.pointbiserialr(correctness, confidences)
        rpb, p_rpb = float(rpb), float(p_rpb)
    except Exception:
        rpb, p_rpb = float("nan"), float("nan")

    if np.isnan(rho_mc):
        rho_mc, p_mc = float("nan"), float("nan")

    summary = {
        "overall_accuracy": overall_acc,
        "mean_stated_confidence": mean_conf,
        "ece": ece,
        "auroc": auroc,
        "brier_score": brier,
        "point_biserial_r": rpb,
        "point_biserial_p": p_rpb,
        "spearman_conf_mc_rho": float(rho_mc),
        "spearman_conf_mc_p": float(p_mc),
        "mean_mc_option_mass": float(mc_option_mass.mean()),
        "mean_confidence_option_mass": float(conf_option_mass.mean()),
        "n_questions": len(results),
    }

    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if not np.isnan(v) else "N/A"

    lines = [
        f"Overall Accuracy:  {overall_acc:.1%}",
        f"Mean Stated Conf:  {mean_conf:.1%}",
        "",
        "Calibration:",
        f"  ECE:           {_fmt(ece)}",
        f"  Brier Score:   {_fmt(brier)}",
        f"  AUROC:         {_fmt(auroc)}",
        "",
        "Correlations:",
        f"  Conf\u2194Correct:  r={_fmt(rpb, '.3f')} (p={_fmt(p_rpb, '.1e')})",
        f"  Conf\u2194MC prob:  \u03c1={_fmt(float(rho_mc), '.3f')} (p={_fmt(float(p_mc), '.1e')})",
        "",
        "Task Compliance (prob mass on options):",
        f"  MC (A-D):      {mc_option_mass.mean():.1%}",
        f"  Conf (S-Z):    {conf_option_mass.mean():.1%}",
        "",
        f"N = {len(results)}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Summary Statistics")

    plt.suptitle(f"Explicit Confidence Task \u2014 {model_label}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Resolve configuration ---
    config = MODEL_CONFIGS[MODEL_TYPE]
    model_name = config["model"]
    adapter_path = config["adapter"]
    is_base = MODEL_TYPE == "base"

    model_short = get_model_short_name(model_name)
    if adapter_path:
        adapter_short = get_model_short_name(adapter_path)
        run_label = f"{model_short}_adapter-{adapter_short}"
    else:
        run_label = model_short
    
    # Add few-shot mode suffix for base model
    if is_base:
        run_label += f"_fs-{FEW_SHOT_MODE}"
    
    base_name = f"{run_label}_{DATASET}_ect"

    print("=" * 70)
    print("EXPLICIT CONFIDENCE TASK (ECT)")
    print("=" * 70)
    print(f"Model type:  {MODEL_TYPE}")
    print(f"Model:       {model_name}")
    if adapter_path:
        print(f"Adapter:     {adapter_path}")
    print(f"Dataset:     {DATASET}")
    print(f"Questions:   {NUM_QUESTIONS}")
    print(f"Batch size:  {BATCH_SIZE}")
    if is_base:
        print(f"Few-shot:    {FEW_SHOT_MODE}")
        if FEW_SHOT_MODE == "random":
            print(f"  Source:    {RANDOM_FEW_SHOT_SOURCE}")
    print(f"Output:      {base_name}")
    print()

    # --- Configure few-shot mode for base model ---
    few_shot_pool = None
    if is_base and FEW_SHOT_MODE in ["random", "balanced", "deceptive_examples"]:
        print(f"Loading few-shot examples from: {RANDOM_FEW_SHOT_SOURCE}")
        with open(RANDOM_FEW_SHOT_SOURCE, "r") as f:
            source_data = json.load(f)
        # Build pool of (question, options, mc_answer, confidence) tuples
        few_shot_pool = []
        for item in source_data["data"]:
            few_shot_pool.append({
                "question": item["question"],
                "options": item["options"],
                "mc_answer": item["predicted_answer"],
                "confidence": item["stated_confidence_response"],
            })
        print(f"  Loaded {len(few_shot_pool)} examples for sampling")
        print()

    # --- Load model ---
    print("Loading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        model_name,
        adapter_path=adapter_path,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat = should_use_chat_template(model_name, tokenizer)
    # Base model prompts are raw text and need BOS prepended during tokenization.
    # Instruct/finetuned prompts go through chat template which already includes BOS.
    add_bos = is_base

    print(f"  Layers: {num_layers}")
    print(f"  Chat template: {use_chat}")
    print(f"  Base model mode: {is_base}")

    # --- Load questions ---
    print(f"\nLoading {DATASET}...")
    questions = load_questions(DATASET, num_questions=NUM_QUESTIONS, seed=SEED)
    print(f"  Loaded {len(questions)} questions")

    # --- Token IDs ---
    # MC answer options: A, B, C, D
    mc_keys = list(questions[0]["options"].keys())
    mc_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in mc_keys]
    print(f"  MC tokens: {dict(zip(mc_keys, mc_token_ids))}")

    # Confidence options: S, T, U, V, W, X, Y, Z
    conf_keys = list(STATED_CONFIDENCE_OPTIONS.keys())
    conf_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in conf_keys]
    print(f"  Confidence tokens: {dict(zip(conf_keys, conf_token_ids))}")

    # Sanity check: verify single-token encoding round-trips
    for key, tid in zip(mc_keys + conf_keys, mc_token_ids + conf_token_ids):
        decoded = tokenizer.decode([tid])
        if decoded.strip() != key:
            print(f"  WARNING: Token {tid} decodes to '{decoded}', expected '{key}'")

    # --- Run both tasks ---
    print(f"\nRunning ECT ({len(questions)} questions, batch_size={BATCH_SIZE})...")
    results = []
    correct_count = 0
    first_batch = True

    for batch_start in tqdm(range(0, len(questions), BATCH_SIZE), desc="Processing"):
        batch_qs = questions[batch_start : batch_start + BATCH_SIZE]

        # --- Direct MC prompts ---
        mc_prompts = []
        for q in batch_qs:
            if is_base:
                prompt, _ = format_direct_prompt_base(q, FEW_SHOT_MODE, few_shot_pool)
            else:
                prompt, _ = format_direct_prompt(q, tokenizer, use_chat)
            mc_prompts.append(prompt)

        mc_probs, mc_logits, mc_mass = get_option_probs(
            model, tokenizer, mc_prompts, mc_token_ids, add_special_tokens=add_bos
        )

        # --- Confidence prompts (independent context, model unaware of MC answer) ---
        conf_prompts = []
        for q in batch_qs:
            if is_base:
                prompt, _ = format_stated_confidence_prompt_base(q, FEW_SHOT_MODE, few_shot_pool)
            else:
                prompt, _ = format_stated_confidence_prompt(q, tokenizer, use_chat)
            conf_prompts.append(prompt)

        conf_probs, conf_logits, conf_mass = get_option_probs(
            model, tokenizer, conf_prompts, conf_token_ids, add_special_tokens=add_bos
        )

        # --- Debug: Print first example ---
        if first_batch:
            first_batch = False
            print("\n" + "=" * 70)
            print("FIRST EXAMPLE (for verification)")
            print("=" * 70)
            print("\n--- MC PROMPT ---")
            print(mc_prompts[0])
            print("\n--- MC RESPONSE ---")
            mc_pred = mc_keys[np.argmax(mc_probs[0])]
            print(f"Predicted: {mc_pred}")
            print(f"Correct:   {batch_qs[0]['correct_answer']}")
            print(f"Probs:     {dict(zip(mc_keys, [f'{p:.4f}' for p in mc_probs[0]]))}")
            print(f"Option mass (full vocab): {mc_mass[0]:.4f}")
            
            print("\n--- CONFIDENCE PROMPT ---")
            print(conf_prompts[0])
            print("\n--- CONFIDENCE RESPONSE ---")
            conf_pred = conf_keys[np.argmax(conf_probs[0])]
            print(f"Predicted: {conf_pred} ({STATED_CONFIDENCE_OPTIONS[conf_pred]})")
            print(f"Value:     {get_stated_confidence_signal(np.array(conf_probs[0])):.3f}")
            print(f"Probs:     {dict(zip(conf_keys, [f'{p:.4f}' for p in conf_probs[0]]))}")
            print(f"Option mass (full vocab): {conf_mass[0]:.4f}")
            print("=" * 70 + "\n")

        # --- Record per-question results ---
        for i, q in enumerate(batch_qs):
            predicted = mc_keys[np.argmax(mc_probs[i])]
            is_correct = predicted == q["correct_answer"]
            if is_correct:
                correct_count += 1

            conf_response = get_stated_confidence_response(np.array(conf_probs[i]))
            conf_value = get_stated_confidence_signal(np.array(conf_probs[i]))

            results.append({
                "question_id": q.get("id", ""),
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "mc_probs": mc_probs[i].tolist(),
                "mc_logits": mc_logits[i].tolist(),
                "mc_option_mass": mc_mass[i],
                "stated_confidence_response": conf_response,
                "stated_confidence_value": conf_value,
                "confidence_probs": conf_probs[i].tolist(),
                "confidence_logits": conf_logits[i].tolist(),
                "confidence_option_mass": conf_mass[i],
            })

    # --- Print summary ---
    accuracy = correct_count / len(questions)
    print(f"\n  MC Accuracy: {accuracy:.1%} ({correct_count}/{len(questions)})")

    conf_responses = [r["stated_confidence_response"] for r in results]
    print("  Confidence distribution:")
    for opt in conf_keys:
        count = sum(1 for r in conf_responses if r == opt)
        pct = count / len(results) * 100
        print(f"    {opt} ({STATED_CONFIDENCE_OPTIONS[opt]:>6s}): {count:4d} ({pct:.1f}%)")

    mean_mc_mass = np.mean([r["mc_option_mass"] for r in results])
    mean_conf_mass = np.mean([r["confidence_option_mass"] for r in results])
    print(f"  MC option mass (A-D):      {mean_mc_mass:.1%}")
    print(f"  Conf option mass (S-Z):    {mean_conf_mass:.1%}")

    # --- Plot and analyze ---
    print("\nGenerating analysis plots...")
    plot_path = OUTPUT_DIR / f"{base_name}_results.png"
    summary_stats = plot_results(results, plot_path, run_label)

    # --- Save results JSON ---
    output_json = {
        "config": {
            "model_type": MODEL_TYPE,
            "model": model_name,
            "adapter": adapter_path,
            "dataset": DATASET,
            "num_questions": len(questions),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
        },
        "summary": summary_stats,
        "confidence_distribution": {
            opt: sum(1 for r in conf_responses if r == opt)
            for opt in conf_keys
        },
        "data": results,
    }
    
    # Add few-shot config for base model
    if is_base:
        output_json["config"]["few_shot_mode"] = FEW_SHOT_MODE
        if FEW_SHOT_MODE == "random":
            output_json["config"]["few_shot_source"] = RANDOM_FEW_SHOT_SOURCE

    json_path = OUTPUT_DIR / f"{base_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Saved results: {json_path}")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Accuracy:          {summary_stats['overall_accuracy']:.1%}")
    print(f"  Mean Confidence:   {summary_stats['mean_stated_confidence']:.1%}")
    print(f"  ECE:               {summary_stats['ece']:.4f}")
    auroc = summary_stats['auroc']
    auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
    print(f"  AUROC:             {auroc_str}")
    print(f"  Brier Score:       {summary_stats['brier_score']:.4f}")
    rpb = summary_stats['point_biserial_r']
    rpb_str = f"r={rpb:.3f}" if not np.isnan(rpb) else "r=N/A"
    print(f"  Conf<->Correct:    {rpb_str}")
    rho = summary_stats['spearman_conf_mc_rho']
    rho_str = f"rho={rho:.3f}" if not np.isnan(rho) else "rho=N/A"
    print(f"  Conf<->MC prob:    {rho_str}")
    print(f"\nOutput files:")
    print(f"  {json_path}")
    print(f"  {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
