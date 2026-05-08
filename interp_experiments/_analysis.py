"""
Behavioural + calibration analysis, reporting, and plots for the introspection
collection pipeline. Read-only over numpy/list inputs (no model state, no GPU).

Entrypoints used by `run_collect_activations.run_single_experiment`:
  - `compute_calibration_masks` / `split_results_by_calibration` — calibrated-vs-
    miscalibrated subsetting of probe predictions.
  - `analyze_behavioral_introspection` — correlation between stated confidence
    and direct entropy, plus delegate-task summary stats.
  - `analyze_other_confidence_control` — Steiger's-Z comparison of self vs
    other confidence correlations against the direct uncertainty signal.
  - `print_results`, `plot_results`, `plot_calibration_split`,
    `plot_other_confidence_comparison` — terminal + PNG reporting.
  - `save_example_prompts_and_responses_txt`, `save_quick_summary_png` —
    eyeball-friendly diagnostics written before the slow probe analysis.

All functions read static config from `IntrospectionExperimentConfig` and
per-run state from `_io.ctx`.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from experiment_config import IntrospectionExperimentConfig as _C
from prompts import (
    ANSWER_OR_DELEGATE_OPTIONS,
    ANSWER_WITH_DELEGATE_OPTIONS,
    _scale_options,
    format_delegate_prompt,
    format_direct_prompt,
    format_direct_prompt_base,
    format_meta_prompt,
    format_meta_prompt_base,
    format_other_confidence_prompt,
    get_meta_signal,
    meta_task_type,
    response_to_confidence,
)

from _io import _run_subfolder_name, ctx as _ctx, get_model_display_label


# =============================================================================
# Calibration split (test-time)
# =============================================================================

def compute_calibration_masks(
    stated_confidence: np.ndarray,
    direct_metric: np.ndarray,
    metric_is_uncertainty: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median-split calibrated-vs-miscalibrated masks.

    Calibrated = the model's stated confidence agrees with the direct
    uncertainty signal (high confidence + low uncertainty, or vice versa).
    Miscalibrated = they disagree.
    """
    conf_median = np.median(stated_confidence)
    metric_median = np.median(direct_metric)

    if metric_is_uncertainty:
        calibrated = (
            ((stated_confidence > conf_median) & (direct_metric < metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric > metric_median))
        )
        miscalibrated = (
            ((stated_confidence > conf_median) & (direct_metric > metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric < metric_median))
        )
    else:
        calibrated = (
            ((stated_confidence > conf_median) & (direct_metric > metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric < metric_median))
        )
        miscalibrated = (
            ((stated_confidence > conf_median) & (direct_metric < metric_median)) |
            ((stated_confidence < conf_median) & (direct_metric > metric_median))
        )

    return calibrated, miscalibrated


def split_results_by_calibration(
    results: Dict,
    y_test: np.ndarray,
    calibrated_mask: np.ndarray,
    miscalibrated_mask: np.ndarray,
) -> Dict:
    """Recompute R² separately on the calibrated- and miscalibrated-test rows."""
    split_results = {}
    for layer_idx, layer_data in results.items():
        d2d_preds = np.array(layer_data["direct_to_direct"]["predictions"])
        d2m_preds = np.array(layer_data["direct_to_meta_fixed"]["predictions"])
        shuffled_r2 = layer_data["shuffled_baseline"]["r2"]

        if calibrated_mask.sum() > 1:
            d2d_r2_cal = r2_score(y_test[calibrated_mask], d2d_preds[calibrated_mask])
            d2m_r2_cal = r2_score(y_test[calibrated_mask], d2m_preds[calibrated_mask])
        else:
            d2d_r2_cal = float("nan")
            d2m_r2_cal = float("nan")

        if miscalibrated_mask.sum() > 1:
            d2d_r2_mis = r2_score(y_test[miscalibrated_mask], d2d_preds[miscalibrated_mask])
            d2m_r2_mis = r2_score(y_test[miscalibrated_mask], d2m_preds[miscalibrated_mask])
        else:
            d2d_r2_mis = float("nan")
            d2m_r2_mis = float("nan")

        split_results[layer_idx] = {
            "calibrated": {"d2d_r2": d2d_r2_cal, "d2m_r2": d2m_r2_cal},
            "miscalibrated": {"d2d_r2": d2d_r2_mis, "d2m_r2": d2m_r2_mis},
            "shuffled_r2": shuffled_r2,
        }
    return split_results


def plot_calibration_split(
    split_results: Dict,
    n_calibrated: int,
    n_miscalibrated: int,
    output_path: str,
):
    """Side-by-side D2D / D2M R² curves for calibrated vs miscalibrated trials."""
    layers = sorted(split_results.keys())
    shuffled_r2 = [split_results[l]["shuffled_r2"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    cal_d2d = [split_results[l]["calibrated"]["d2d_r2"] for l in layers]
    cal_d2m = [split_results[l]["calibrated"]["d2m_r2"] for l in layers]
    ax1.plot(layers, cal_d2d, "o-", label="Direct→Direct", linewidth=2, color="C0")
    ax1.plot(layers, cal_d2m, "s-", label="Direct→Meta", linewidth=2, color="C1")
    ax1.plot(layers, shuffled_r2, "x--", label="Shuffled baseline", linewidth=1, alpha=0.5, color="gray")
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("R² Score")
    ax1.set_title(f"Calibrated Trials (n={n_calibrated})")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    mis_d2d = [split_results[l]["miscalibrated"]["d2d_r2"] for l in layers]
    mis_d2m = [split_results[l]["miscalibrated"]["d2m_r2"] for l in layers]
    ax2.plot(layers, mis_d2d, "o-", label="Direct→Direct", linewidth=2, color="C0")
    ax2.plot(layers, mis_d2m, "s-", label="Direct→Meta", linewidth=2, color="C1")
    ax2.plot(layers, shuffled_r2, "x--", label="Shuffled baseline", linewidth=1, alpha=0.5, color="gray")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("R² Score")
    ax2.set_title(f"Miscalibrated Trials (n={n_miscalibrated})")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Calibration split — {get_model_display_label()} / {_ctx.dataset_name}",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Calibration split plot saved to {output_path}")


def plot_other_confidence_comparison(
    main_results: Dict,
    mc_results: Dict,
    other_results: Dict,
    output_path: str,
):
    """Two-panel comparison of D→M (Self) vs D→M (Other) for entropy + MC probes."""
    layers = sorted(main_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    d2d = [main_results[l]["direct_to_direct"]["test_r2"] for l in layers]
    d2m_self = [main_results[l]["direct_to_meta_fixed"]["r2"] for l in layers]
    d2m_other = [other_results[l]["d2m_other_entropy_r2"] for l in layers]
    shuffled = [main_results[l]["shuffled_baseline"]["r2"] for l in layers]
    ax1.plot(layers, d2d, "o-", label="D→D", linewidth=2, color="C0")
    ax1.plot(layers, d2m_self, "s-", label="D→M (Self)", linewidth=2, color="C1")
    ax1.plot(layers, d2m_other, "^-", label="D→M (Other)", linewidth=2, color="C2")
    ax1.plot(layers, shuffled, "x--", label="Shuffled", linewidth=1, alpha=0.5, color="gray")
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index"); ax1.set_ylabel("R² Score")
    ax1.set_title("Entropy Probe Transfer")
    ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    mc_d2d = [mc_results[l]["d2d_accuracy"] for l in layers]
    mc_d2m_self = [mc_results[l]["d2m_centered_accuracy"] for l in layers]
    mc_d2m_other = [other_results[l]["d2m_other_mc_accuracy"] for l in layers]
    mc_shuffled = [mc_results[l]["shuffled_accuracy"] for l in layers]
    ax2.plot(layers, mc_d2d, "o-", label="D→D", linewidth=2, color="C0")
    ax2.plot(layers, mc_d2m_self, "s-", label="D→M (Self)", linewidth=2, color="C1")
    ax2.plot(layers, mc_d2m_other, "^-", label="D→M (Other)", linewidth=2, color="C2")
    ax2.plot(layers, mc_shuffled, "x--", label="Shuffled", linewidth=1, alpha=0.5, color="gray")
    ax2.axhline(y=0.25, color="black", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index"); ax2.set_ylabel("Accuracy")
    ax2.set_title("MC Answer Probe Transfer")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Self vs Other confidence transfer — {get_model_display_label()} / {_ctx.dataset_name}",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Other-confidence transfer comparison saved to {output_path}")


# =============================================================================
# Behavioral analysis (correlations + delegate stats)
# =============================================================================

def analyze_behavioral_introspection(
    meta_responses: List[str],
    direct_entropies: np.ndarray,
    test_idx: np.ndarray,
    meta_probs: List[List[float]] = None,
    meta_mappings: List[Dict[str, str]] = None,
    direct_probs: List[List[float]] = None,
    questions: List[Dict] = None,
) -> Dict:
    """Correlation between stated confidence and direct-task entropy + delegate stats.

    Negative correlation suggests introspection (high confidence ↔ low entropy).
    """
    if _ctx.meta_task == "delegate":
        stated_confidence = np.array([
            response_to_confidence(r, np.array(p) if p else None, m,
                                   meta_task_type(meta_task=_ctx.meta_task, scale=_C.CONFIDENCE_SCALE))
            for r, p, m in zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses),
                meta_mappings or [None] * len(meta_responses),
            )
        ])
    else:
        stated_confidence = np.array([
            get_meta_signal(np.array(p), scale=_C.CONFIDENCE_SCALE) if p else 0.5
            for p in meta_probs
        ])

    test_confidence = stated_confidence[test_idx]
    test_entropy = direct_entropies[test_idx]

    # Subsample-to-m intervals (Fisher-z transformed correlations)
    n_subsamples = 200
    n = len(direct_entropies)
    m = len(test_idx)

    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def fisher_z_inv(z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    full_r = np.corrcoef(stated_confidence, direct_entropies)[0, 1]
    full_z = fisher_z(full_r)
    test_r = np.corrcoef(test_confidence, test_entropy)[0, 1]
    n_test = len(test_confidence)

    full_subsample_deviations = []
    for k in range(n_subsamples):
        rng = np.random.default_rng(k)
        idx = rng.choice(n, size=m, replace=False)
        sub_r = np.corrcoef(stated_confidence[idx], direct_entropies[idx])[0, 1]
        if not np.isnan(sub_r):
            sub_z = fisher_z(sub_r)
            full_subsample_deviations.append(sub_z - full_z)

    dev_lower = np.percentile(full_subsample_deviations, 2.5)
    dev_upper = np.percentile(full_subsample_deviations, 97.5)
    full_ci_lower = fisher_z_inv(full_z + dev_lower)
    full_ci_upper = fisher_z_inv(full_z + dev_upper)
    full_ci_std = np.std([fisher_z_inv(full_z + d) for d in full_subsample_deviations])

    test_subsample_correlations = []
    for k in range(n_subsamples):
        rng = np.random.default_rng(k)
        idx = rng.choice(n, size=m, replace=False)
        sub_r = np.corrcoef(stated_confidence[idx], direct_entropies[idx])[0, 1]
        if not np.isnan(sub_r):
            test_subsample_correlations.append(sub_r)

    test_ci_mean = np.mean(test_subsample_correlations)
    test_ci_std = np.std(test_subsample_correlations)
    test_ci_lower = np.percentile(test_subsample_correlations, 2.5)
    test_ci_upper = np.percentile(test_subsample_correlations, 97.5)

    if len(full_subsample_deviations) > 0:
        subsample_correlations = [fisher_z_inv(full_z + d) for d in full_subsample_deviations]
        n_below_zero = sum(1 for r in subsample_correlations if r < 0)
        n_above_zero = sum(1 for r in subsample_correlations if r > 0)
        n_total = len(subsample_correlations)
        full_pvalue = max(2 * min(n_below_zero, n_above_zero) / n_total, 1 / n_subsamples)
    else:
        full_pvalue = np.nan

    if abs(test_r) < 1 and n_test > 2:
        t_stat = test_r * np.sqrt((n_test - 2) / (1 - test_r**2))
        test_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_test - 2))
    else:
        test_pvalue = np.nan

    # Partial correlation (placeholder for future controls)
    import pandas as pd
    from core.logres_helpers import partial_correlation_on_decision

    control_series_list = []
    full_partial_result = partial_correlation_on_decision(
        dv_series=pd.Series(direct_entropies, name="entropy"),
        iv_series=pd.Series(stated_confidence, name="confidence"),
        control_series_list=control_series_list,
    )

    result = {
        "full_correlation": float(full_r),
        "full_correlation_pvalue": float(full_pvalue),
        "test_correlation": float(test_r),
        "test_correlation_pvalue": float(test_pvalue),
        "full_correlation_ci95": [float(full_ci_lower), float(full_ci_upper)],
        "full_correlation_ci_std": float(full_ci_std),
        "test_correlation_ci95": [float(test_ci_lower), float(test_ci_upper)],
        "test_correlation_ci_mean": float(test_ci_mean),
        "test_correlation_ci_std": float(test_ci_std),
        "n_subsamples": n_subsamples,
        "partial_correlation": float(full_partial_result["correlation"]),
        "partial_correlation_ci95": [
            float(full_partial_result["ci_lower"]),
            float(full_partial_result["ci_upper"]),
        ],
        "partial_correlation_pvalue": float(full_partial_result["p_value"]),
        "partial_correlation_controls": [s.name for s in control_series_list],
        "n_samples_full": n,
        "n_samples_test": n_test,
        "test_confidence_mean": float(test_confidence.mean()),
        "test_confidence_std": float(test_confidence.std()),
        "test_entropy_mean": float(test_entropy.mean()),
        "test_entropy_std": float(test_entropy.std()),
    }

    if _ctx.meta_task == "delegate":
        delegated = []
        self_answers = []
        if _C.DELEGATE_PROMPT_DESIGN == "mc_integrated":
            for i, response in enumerate(meta_responses):
                is_delegated = (str(response) == "T")
                delegated.append(is_delegated)
                if not is_delegated:
                    self_answers.append(i)
        elif meta_mappings is not None:
            for i, (response, mapping) in enumerate(zip(meta_responses, meta_mappings)):
                if mapping is not None:
                    decision = mapping.get(response, "Unknown")
                    is_delegated = (decision == "Delegate")
                    delegated.append(is_delegated)
                    if not is_delegated:
                        self_answers.append(i)

        delegation_rate = sum(delegated) / len(delegated) if delegated else 0.0
        result["delegation_rate"] = float(delegation_rate)
        result["num_delegated"] = sum(delegated)
        result["num_self_answered"] = len(self_answers)
        result["response_distribution"] = {
            r: sum(1 for x in meta_responses if str(x) == r)
            for r in set(str(x) for x in meta_responses)
        }

        if questions is not None and self_answers:
            self_correct = 0
            for idx in self_answers:
                if idx >= len(questions):
                    continue
                q = questions[idx]
                correct = q.get("correct_answer")
                if correct is None:
                    continue
                if _C.DELEGATE_PROMPT_DESIGN == "mc_integrated":
                    if idx < len(meta_responses) and str(meta_responses[idx]) == correct:
                        self_correct += 1
                elif direct_probs is not None and idx < len(direct_probs):
                    probs = direct_probs[idx]
                    if probs and "options" in q:
                        options = list(q["options"].keys())
                        if options[np.argmax(probs)] == correct:
                            self_correct += 1

            result["self_answer_accuracy"] = float(self_correct / len(self_answers))
            result["self_correct"] = self_correct
            result["teammate_accuracy"] = float(_C.TEAMMATE_ACCURACY)

            team_score = self_correct + sum(delegated) * _C.TEAMMATE_ACCURACY
            result["team_score"] = float(team_score)
            result["team_score_normalized"] = float(team_score / len(delegated)) if delegated else 0.0

            if direct_probs is not None:
                overall_correct = 0
                overall_graded = 0
                for idx in range(min(len(direct_probs), len(questions))):
                    probs = direct_probs[idx]
                    q = questions[idx]
                    if probs and "correct_answer" in q and "options" in q:
                        options = list(q["options"].keys())
                        if options[np.argmax(probs)] == q["correct_answer"]:
                            overall_correct += 1
                        overall_graded += 1
                if overall_graded:
                    result["overall_accuracy"] = float(overall_correct / overall_graded)

    result["stated_confidence"] = stated_confidence.tolist()
    return result


def analyze_other_confidence_control(
    other_signals: np.ndarray,
    self_confidence: np.ndarray,
    direct_metric: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    """Steiger's-Z comparison: is self-confidence MORE correlated with the direct
    uncertainty signal than other-confidence (control) is?
    """
    n = len(direct_metric)

    self_r = np.corrcoef(self_confidence, direct_metric)[0, 1]
    self_pvalue = _two_tail_t_pvalue(self_r, n)

    other_r = np.corrcoef(other_signals, direct_metric)[0, 1]
    other_pvalue = _two_tail_t_pvalue(other_r, n)

    self_other_r = np.corrcoef(self_confidence, other_signals)[0, 1]
    self_other_pvalue = _two_tail_t_pvalue(self_other_r, n)

    # Hotelling-Williams test for dependent correlation difference
    r12, r13, r23 = self_r, other_r, self_other_r
    if abs(r12) < 1 and abs(r13) < 1 and abs(r23) < 1:
        r_avg = (r12 + r13) / 2
        det = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
        if det > 0:
            t_stat = (r12 - r13) * np.sqrt((n - 3) * (1 + r23) / (2 * det))
            diff_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 3))
        else:
            diff_pvalue = np.nan
    else:
        diff_pvalue = np.nan

    n_bootstrap = 1000
    diff_samples = []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(b)
        idx = rng.choice(n, size=n, replace=True)
        self_r_b = np.corrcoef(self_confidence[idx], direct_metric[idx])[0, 1]
        other_r_b = np.corrcoef(other_signals[idx], direct_metric[idx])[0, 1]
        if not np.isnan(self_r_b) and not np.isnan(other_r_b):
            diff_samples.append(self_r_b - other_r_b)

    if diff_samples:
        diff_ci_lower = np.percentile(diff_samples, 2.5)
        diff_ci_upper = np.percentile(diff_samples, 97.5)
    else:
        diff_ci_lower = diff_ci_upper = np.nan

    return {
        "self_vs_metric_r": float(self_r),
        "self_vs_metric_pvalue": float(self_pvalue),
        "other_vs_metric_r": float(other_r),
        "other_vs_metric_pvalue": float(other_pvalue),
        "self_vs_other_r": float(self_other_r),
        "self_vs_other_pvalue": float(self_other_pvalue),
        "correlation_difference": float(self_r - other_r),
        "correlation_difference_ci95": [float(diff_ci_lower), float(diff_ci_upper)],
        "correlation_difference_pvalue": float(diff_pvalue) if not np.isnan(diff_pvalue) else None,
        "self_confidence_mean": float(self_confidence.mean()),
        "self_confidence_std": float(self_confidence.std()),
        "other_confidence_mean": float(other_signals.mean()),
        "other_confidence_std": float(other_signals.std()),
        "n_samples": n,
    }


def _two_tail_t_pvalue(r: float, n: int) -> float:
    if abs(r) < 1 and n > 2:
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        return 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    return np.nan


# =============================================================================
# Reporting
# =============================================================================

def print_results(results: Dict, behavioral: Dict,
                  other_confidence_analysis: Optional[Dict] = None,
                  *, metric: str) -> None:
    """Terminal-friendly summary of a single (model, dataset, meta_task) run."""
    print("\n" + "=" * 100)
    print("INTROSPECTION EXPERIMENT RESULTS")
    print("=" * 100)

    print("\n--- Behavioral Analysis ---")
    n_full = behavioral.get("n_samples_full", "?")
    n_test = behavioral.get("n_samples_test", "?")
    n_subsamples = behavioral.get("n_subsamples", "?")
    print("Correlation (stated confidence vs direct entropy):")

    full_p = behavioral.get("full_correlation_pvalue")
    p_str = f", p = {full_p:.2e}" if full_p is not None and not np.isnan(full_p) else ""
    full_ci = behavioral.get("full_correlation_ci95", [None, None])
    full_ci_std = behavioral.get("full_correlation_ci_std")
    if full_ci[0] is not None:
        print(f"  Full  (n={n_full}):  r = {behavioral['full_correlation']:.4f} ± {full_ci_std:.4f}  "
              f"[95% CI: {full_ci[0]:.4f}, {full_ci[1]:.4f}]{p_str}")
    else:
        print(f"  Full  (n={n_full}):  r = {behavioral['full_correlation']:.4f}{p_str}")

    test_p = behavioral.get("test_correlation_pvalue")
    p_str = f", p = {test_p:.2e}" if test_p is not None and not np.isnan(test_p) else ""
    test_ci = behavioral.get("test_correlation_ci95", [None, None])
    test_ci_std = behavioral.get("test_correlation_ci_std")
    if test_ci[0] is not None:
        print(f"  Test  (n={n_test}):  r = {behavioral['test_correlation']:.4f} ± {test_ci_std:.4f}  "
              f"[95% CI: {test_ci[0]:.4f}, {test_ci[1]:.4f}]{p_str}")
    else:
        print(f"  Test  (n={n_test}):  r = {behavioral['test_correlation']:.4f}{p_str}")

    print(f"  (CIs from {n_subsamples} subsamples to test size, centered on point estimate)")

    partial_r = behavioral.get("partial_correlation")
    partial_ci = behavioral.get("partial_correlation_ci95", [None, None])
    partial_p = behavioral.get("partial_correlation_pvalue")
    controls = behavioral.get("partial_correlation_controls", [])
    if partial_r is not None:
        p_str = f", p = {partial_p:.2e}" if partial_p is not None and not np.isnan(partial_p) else ""
        ctrl_str = f" (controlling for {', '.join(controls)})" if controls else ""
        print(f"  Partial{ctrl_str}: r = {partial_r:.4f}  "
              f"[95% CI: {partial_ci[0]:.4f}, {partial_ci[1]:.4f}]{p_str}")

    print("  (Negative correlation suggests introspection; positive suggests miscalibration)")

    if other_confidence_analysis is not None:
        print("\n--- Other-Confidence Control (Human Difficulty Estimation) ---")
        self_r = other_confidence_analysis["self_vs_metric_r"]
        other_r = other_confidence_analysis["other_vs_metric_r"]
        diff = other_confidence_analysis["correlation_difference"]
        diff_ci = other_confidence_analysis["correlation_difference_ci95"]
        diff_p = other_confidence_analysis.get("correlation_difference_pvalue")
        self_p = other_confidence_analysis["self_vs_metric_pvalue"]
        other_p = other_confidence_analysis["other_vs_metric_pvalue"]

        self_p_str = f", p = {self_p:.2e}" if self_p is not None and not np.isnan(self_p) else ""
        other_p_str = f", p = {other_p:.2e}" if other_p is not None and not np.isnan(other_p) else ""

        print(f"  Self-confidence vs {metric}:    r = {self_r:.4f}{self_p_str}")
        print(f"  Other-confidence vs {metric}:   r = {other_r:.4f}{other_p_str}")
        print(f"  Self vs Other confidence:       r = {other_confidence_analysis['self_vs_other_r']:.4f}")
        print(f"")
        print(f"  Difference (self - other):      Δr = {diff:.4f}  "
              f"[95% CI: {diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")

        if diff_p is not None:
            print(f"  Steiger's test p-value:         p = {diff_p:.4e}")
            if diff_p < 0.05 and diff < 0:
                print(f"  → Self-confidence significantly MORE correlated with {metric} than other-confidence")
                print(f"    This suggests the model is introspecting on its own uncertainty,")
                print(f"    not just assessing question difficulty.")
            elif diff_p < 0.05 and diff > 0:
                print(f"  → Self-confidence significantly LESS correlated with {metric} than other-confidence")
                print(f"    This is unexpected - the model may be using question difficulty as a proxy.")
            else:
                print("  → No significant difference between self and other confidence correlations")
        else:
            print("  → Could not compute significance test")

    if _ctx.meta_task == "delegate" and "delegation_rate" in behavioral:
        print("\n--- Delegate Task Summary ---")
        print(f"  Delegation rate:      {behavioral['delegation_rate']:.1%} "
              f"({behavioral['num_delegated']} delegated, {behavioral['num_self_answered']} self-answered)")
        if "self_answer_accuracy" in behavioral:
            print(f"  Self-answer accuracy: {behavioral['self_answer_accuracy']:.1%} "
                  f"({behavioral['self_correct']}/{behavioral['num_self_answered']} correct)")
            print(f"  Teammate accuracy:    {behavioral['teammate_accuracy']:.1%} (configured)")
            print(f"  Team score:           {behavioral['team_score']:.1f} / "
                  f"{behavioral['num_delegated'] + behavioral['num_self_answered']} "
                  f"({behavioral['team_score_normalized']:.1%})")
            always_delegate = float(behavioral["teammate_accuracy"])
            if "overall_accuracy" in behavioral:
                always_answer = float(behavioral["overall_accuracy"])
                print(f"  Baselines:            always-answer = {always_answer:.1%}, "
                      f"always-delegate = {always_delegate:.1%}")
            else:
                print(f"  Baselines:            always-delegate = {always_delegate:.1%}  "
                      f"(always-answer baseline: see notebook)")

        dist = behavioral.get("response_distribution") or {}
        if dist:
            total = sum(dist.values()) or 1
            items = sorted(dist.items(), key=lambda kv: -kv[1])
            pretty = ", ".join(f"'{k}'={v}" for k, v in items)
            top_key, top_cnt = items[0]
            skew = top_cnt / total
            print(f"  Meta-response distribution: {pretty}  (top '{top_key}' = {skew:.1%})")
            if skew > 0.9:
                print(
                    "  ⚠  WARNING: model almost always picks one option — the observed "
                    "delegation rate is a prompt-format artifact, not a metacognitive signal. "
                    "Raising TEAMMATE_ACCURACY or changing the prompt framing may help."
                )

    print("\n--- Probe Analysis by Layer ---")
    print(f"{'Layer':<8} {'Direct→Direct':<15} {'D→M (fixed)':<15} {'D→M (orig)':<15} {'Meta→Meta':<15} {'Shuffled':<12}")
    print(f"{'':8} {'R²':<15} {'R²':<15} {'R²':<15} {'R²':<15} {'R²':<12}")
    print("-" * 110)

    for layer_idx in sorted(results.keys()):
        res = results[layer_idx]
        d2d = res["direct_to_direct"]["test_r2"]
        d2m_fixed = res["direct_to_meta_fixed"]["r2"]
        d2m_orig = res["direct_to_meta"]["r2"]
        m2m = res["meta_to_meta"]["test_r2"]
        shuf = res["shuffled_baseline"]["r2"]
        print(f"{layer_idx:<8} {d2d:<15.4f} {d2m_fixed:<15.4f} {d2m_orig:<15.4f} {m2m:<15.4f} {shuf:<12.4f}")

    print("=" * 110)

    layers = sorted(results.keys())
    best_d2d_layer = max(layers, key=lambda l: results[l]["direct_to_direct"]["test_r2"])
    best_d2d = results[best_d2d_layer]["direct_to_direct"]["test_r2"]
    best_d2m_fixed_layer = max(layers, key=lambda l: results[l]["direct_to_meta_fixed"]["r2"])
    best_d2m_fixed = results[best_d2m_fixed_layer]["direct_to_meta_fixed"]["r2"]
    best_m2m_layer = max(layers, key=lambda l: results[l]["meta_to_meta"]["test_r2"])
    best_m2m = results[best_m2m_layer]["meta_to_meta"]["test_r2"]

    print(f"\nBest Direct→Direct:      Layer {best_d2d_layer} (R² = {best_d2d:.4f})")
    print(f"Best Direct→Meta (fixed): Layer {best_d2m_fixed_layer} (R² = {best_d2m_fixed:.4f})")
    print(f"Best Meta→Meta:          Layer {best_m2m_layer} (R² = {best_m2m:.4f})")

    if best_d2d > 0:
        transfer_ratio = best_d2m_fixed / best_d2d
        print(f"\nTransfer ratio (best D→M fixed / best D→D): {transfer_ratio:.2%}")
        if transfer_ratio > 0.5:
            print("  → Strong evidence for introspection!")
        elif transfer_ratio > 0.25:
            print("  → Moderate evidence for introspection")
        elif transfer_ratio > 0:
            print("  → Weak evidence for introspection")
        else:
            print("  → No evidence for introspection (negative transfer)")


def plot_results(
    results: Dict,
    behavioral: Dict,
    direct_entropies: np.ndarray,
    test_idx: np.ndarray,
    output_path: str = "introspection_results.png",
    mc_answer_results: Optional[Dict] = None,
):
    """4-panel introspection summary figure."""
    layers = sorted(results.keys())

    ent_d2d_r2 = [results[l]["direct_to_direct"]["test_r2"] for l in layers]
    ent_d2m_separate_r2 = [results[l]["direct_to_meta_fixed"]["r2"] for l in layers]
    ent_d2m_centered_r2 = [results[l]["direct_to_meta_centered"]["r2"] for l in layers]
    ent_shuffled_r2 = [results[l]["shuffled_baseline"]["r2"] for l in layers]
    ent_pearson = [results[l]["direct_to_meta"]["pearson"] for l in layers]
    ent_d2d_pearson = [results[l]["direct_to_direct"]["test_pearson"] for l in layers]

    has_mc = mc_answer_results is not None
    if has_mc:
        mc_d2d_acc = [mc_answer_results[l]["d2d_accuracy"] for l in layers]
        first_layer = mc_answer_results[layers[0]]
        if "d2m_centered_accuracy" in first_layer:
            mc_d2m_sep_acc = [mc_answer_results[l]["d2m_separate_accuracy"] for l in layers]
            mc_d2m_cen_acc = [mc_answer_results[l]["d2m_centered_accuracy"] for l in layers]
        else:
            mc_d2m_sep_acc = [mc_answer_results[l]["d2m_accuracy"] for l in layers]
            mc_d2m_cen_acc = mc_d2m_sep_acc
        mc_chance = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle(
        f"Introspection Analysis: {get_model_display_label()} on {_ctx.dataset_name} ({_ctx.meta_task})",
        fontsize=16,
    )

    def normalize(data):
        arr = np.array(data)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    def plot_ent_baselines(ax, metric_d2d, metric_shuffled=None):
        ax.plot(layers, metric_d2d, "o-", label="Entropy D→D", color="tab:blue", linewidth=2)
        if metric_shuffled is not None:
            ax.plot(layers, metric_shuffled, ":", label="Entropy Shuffled", color="gray", alpha=0.6)

    def plot_mc_lines(ax, d2d, d2m, label_suffix=""):
        if has_mc:
            ax.plot(layers, d2d, "d-", label="MC Ans D→D", color="tab:green", linewidth=2)
            ax.plot(layers, d2m, "d-", label=f"MC Ans D→M {label_suffix}", color="tab:red", linewidth=2)
            ax.axhline(y=mc_chance, color="tab:green", linestyle=":", alpha=0.4, label="Chance")

    # Panel 1
    ax1 = axes[0, 0]
    plot_ent_baselines(ax1, ent_d2d_r2, ent_shuffled_r2)
    ax1.plot(layers, ent_d2m_separate_r2, "s-", label="Entropy D→M Sep", color="tab:orange", linewidth=2)
    if has_mc:
        plot_mc_lines(ax1, mc_d2d_acc, mc_d2m_sep_acc, "Sep ")
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index"); ax1.set_ylabel("$R^2$ / Accuracy")
    ax1.set_title("Method 1: Separate Scaler (Upper Bound)\n(Absolute Performance)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    bottom_y = min(min(ent_shuffled_r2), min(ent_d2m_separate_r2), -0.1)
    ax1.set_ylim(bottom=max(-2.0, bottom_y - 0.1), top=1.05)

    # Panel 2
    ax2 = axes[0, 1]
    plot_ent_baselines(ax2, ent_d2d_r2, ent_shuffled_r2)
    ax2.plot(layers, ent_d2m_centered_r2, "s-", label="Entropy D→M Cen", color="tab:orange", linewidth=2)
    if has_mc:
        plot_mc_lines(ax2, mc_d2d_acc, mc_d2m_cen_acc, "Cen ")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Layer Index"); ax2.set_ylabel("$R^2$ / Accuracy")
    ax2.set_title("Method 2: Centered Scaler (Rigorous)\n(Geometry Check)")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=max(-2.0, bottom_y - 0.1), top=1.05)

    # Panel 3
    ax3 = axes[1, 0]
    ax3.plot(layers, ent_d2d_pearson, "o-", label="Entropy D→D Pearson", color="tab:blue", linewidth=2)
    ax3.plot(layers, ent_pearson, "s-", label="Entropy D→M Pearson", color="tab:orange", linewidth=2)
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.set_xlabel("Layer Index"); ax3.set_ylabel("Correlation (r)")
    ax3.set_title("Method 3: Pearson Correlation\n(Shift Invariant Signal Check)")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.05)

    # Panel 4
    ax4 = axes[1, 1]
    ax4.plot(layers, normalize(ent_d2d_r2), "o-", label="Entropy D→D", color="tab:blue", alpha=0.4)
    ax4.plot(layers, normalize(ent_d2m_separate_r2), "s-", label="Entropy D→M",
             color="tab:orange", linewidth=2.5)
    if has_mc:
        ax4.plot(layers, normalize(mc_d2d_acc), "d-", label="MC Ans D→D",
                 color="tab:green", alpha=0.4)
        ax4.plot(layers, normalize(mc_d2m_sep_acc), "d-", label="MC Ans D→M",
                 color="tab:red", linewidth=2.5)
    ax4.set_xlabel("Layer Index"); ax4.set_ylabel("Normalized Score (0-1)")
    ax4.set_title("Signal Emergence (Min-Max Scaled)\nCheck: Do Red/Orange lines rise together?")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


# =============================================================================
# Eyeball-friendly diagnostic dumps (run before the slow analysis)
# =============================================================================

def save_example_prompts_and_responses_txt(
    data: dict,
    questions: list,
    tokenizer,
    is_base: bool,
    use_chat_template: bool,
    few_shot_mode: str,
    output_path: str,
    n_examples: int = 10,
    delegate_pool: Optional[list] = None,
    other_data: Optional[dict] = None,
) -> None:
    """Save a human-readable .txt with full config snapshot + per-example prompts + responses.

    For each of the first `n_examples` questions: question, options, correct
    answer; the EXACT direct prompt as sent to the model (chat tags + all)
    with per-option probabilities, argmax, correctness, uncertainty metrics;
    the EXACT meta prompt with the same fidelity; and (when `other_data` is
    provided) the EXACT other-confidence prompt.

    "Model response" = the next-token distribution at the answer position
    — the runner runs a single forward pass and reads that distribution; no
    autoregressive continuation is generated.
    """
    buf = io.StringIO()

    n = min(n_examples, len(questions))
    direct_probs = data.get("direct_probs", [])
    meta_probs = data.get("meta_probs", [])
    meta_resp = data.get("meta_responses", [])
    meta_mappings = data.get("meta_mappings") or [None] * len(questions)
    direct_metrics = data.get("direct_metrics", {})
    other_probs = (other_data or {}).get("other_probs", [])
    other_resp = (other_data or {}).get("other_responses", [])
    other_signals = (other_data or {}).get("other_signals")

    buf.write("=" * 80 + "\n")
    buf.write("INTROSPECTION RUN CONFIG\n")
    buf.write("=" * 80 + "\n")
    snapshot = {
        "BASE_MODEL_NAME":              _C.BASE_MODEL_NAME,
        "MODEL_NAME":                   _C.MODEL_NAME,
        "DATASET_NAME":                 _ctx.dataset_name,
        "NUM_QUESTIONS":                _ctx.num_questions,
        "SEED":                         _C.SEED,
        "META_TASK":                    _ctx.meta_task,
        "CONFIDENCE_SCALE":             _C.CONFIDENCE_SCALE,
        "DELEGATE_PROMPT_DESIGN":       _C.DELEGATE_PROMPT_DESIGN,
        "TEAMMATE_ACCURACY":            _C.TEAMMATE_ACCURACY,
        "REUSE_DIRECT_FROM_CONFIDENCE": _C.REUSE_DIRECT_FROM_CONFIDENCE,
        "is_base_model":                is_base,
        "use_chat_template":            use_chat_template,
        "FEW_SHOT_MODE":                _C.FEW_SHOT_MODE if is_base else None,
        "BASE_DELEGATE_MODE":           _C.BASE_DELEGATE_MODE if is_base else None,
        "BASE_DELEGATE_POOL_SOURCE":    _C.BASE_DELEGATE_POOL_SOURCE if is_base else None,
        "LOAD_IN_4BIT":                 _C.LOAD_IN_4BIT,
        "LOAD_IN_8BIT":                 _C.LOAD_IN_8BIT,
        "METRIC":                       _C.METRIC,
        "AVAILABLE_METRICS":            list(_C.AVAILABLE_METRICS),
        "UNCERTAINTY_METRICS":          sorted(_C.UNCERTAINTY_METRICS),
        "TRAIN_SPLIT":                  _C.TRAIN_SPLIT,
        "PROBE_ALPHA":                  _C.PROBE_ALPHA,
        "USE_PCA":                      _C.USE_PCA,
        "PCA_COMPONENTS":               _C.PCA_COMPONENTS,
        "EXTRACT_CONTRAST_DIRECTIONS":      _C.EXTRACT_CONTRAST_DIRECTIONS,
        "CONTRAST_PERCENT_ENTROPY":         _C.CONTRAST_PERCENT_ENTROPY,
        "CONTRAST_PERCENT_STATED_CONFIDENCE": _C.CONTRAST_PERCENT_STATED_CONFIDENCE,
        "CONTRAST_SAMPLES_PER_BIN":     _C.CONTRAST_SAMPLES_PER_BIN,
        "OUTPUTS_DIR":                  str(_C.OUTPUTS_DIR),
        "run_subfolder":                _run_subfolder_name(),
    }
    key_w = max(len(k) for k in snapshot)
    for k in snapshot:
        buf.write(f"  {k.ljust(key_w)} = {snapshot[k]!r}\n")
    buf.write("=" * 80 + "\n\n")
    buf.write(f"Showing first {n} of {len(questions)} examples.\n\n")

    for i in range(n):
        q = questions[i]
        buf.write("=" * 80 + "\n")
        buf.write(f"EXAMPLE {i + 1} / {n}   (id: {q.get('id', f'q_{i}')})\n")
        buf.write("=" * 80 + "\n")
        buf.write(f"Question: {q.get('question', '')}\n")
        for key, val in q.get("options", {}).items():
            buf.write(f"  {key}: {val}\n")
        buf.write(f"Correct answer: {q.get('correct_answer', '?')}\n\n")

        if is_base:
            direct_prompt, direct_opts = format_direct_prompt_base(q, mode=few_shot_mode)
        else:
            direct_prompt, direct_opts = format_direct_prompt(q, tokenizer, use_chat_template)

        buf.write("--- DIRECT PROMPT (sent to model) ---\n")
        buf.write(direct_prompt)
        buf.write("\n\n")

        buf.write("--- DIRECT RESPONSE ---\n")
        if i < len(direct_probs) and direct_probs[i]:
            p = direct_probs[i]
            probs_str = "  ".join(f"{L}={v:.3f}" for L, v in zip(direct_opts, p))
            argmax_letter = direct_opts[int(np.argmax(p))]
            is_correct_flag = argmax_letter == q.get("correct_answer")
            buf.write(f"Option probs: {probs_str}\n")
            buf.write(f"Chosen (argmax): {argmax_letter}   "
                      f"Correct? {'YES' if is_correct_flag else 'no'}\n")
        if direct_metrics and i < len(direct_metrics.get("entropy", [])):
            e = direct_metrics["entropy"][i]
            m = direct_metrics.get("margin", [float("nan")] * len(questions))[i]
            lg = direct_metrics.get("logit_gap", [float("nan")] * len(questions))[i]
            tp = direct_metrics.get("top_prob", [float("nan")] * len(questions))[i]
            buf.write(f"Uncertainty: entropy={float(e):.3f}  logit_gap={float(lg):.3f}  "
                      f"margin={float(m):.3f}  top_prob={float(tp):.3f}\n")
        buf.write("\n")

        if _ctx.meta_task == "delegate":
            meta_prompt, meta_opts, _mapping = format_delegate_prompt(
                q, tokenizer, use_chat_template, trial_index=i,
                is_base=is_base,
                few_shot_mode=_C.BASE_DELEGATE_MODE if is_base else "fixed",
                few_shot_pool=delegate_pool,
                teammate_accuracy=_C.TEAMMATE_ACCURACY,
                prompt_design=_C.DELEGATE_PROMPT_DESIGN,
            )
        else:
            if is_base:
                meta_prompt, meta_opts = format_meta_prompt_base(q, mode=few_shot_mode, scale=_C.CONFIDENCE_SCALE)
            else:
                meta_prompt, meta_opts = format_meta_prompt(q, tokenizer, use_chat_template, scale=_C.CONFIDENCE_SCALE)

        buf.write("--- META PROMPT (sent to model) ---\n")
        buf.write(meta_prompt)
        buf.write("\n\n")

        buf.write("--- META RESPONSE ---\n")
        if i < len(meta_probs) and meta_probs[i]:
            p = meta_probs[i]
            probs_str = "  ".join(f"{L}={v:.3f}" for L, v in zip(meta_opts, p))
            buf.write(f"Option probs: {probs_str}\n")
        if i < len(meta_resp):
            buf.write(f"Chosen (argmax): {meta_resp[i]}\n")
        if i < len(meta_probs) and meta_probs[i]:
            soft = response_to_confidence(
                meta_resp[i] if i < len(meta_resp) else "",
                np.asarray(meta_probs[i]),
                meta_mappings[i] if i < len(meta_mappings) else None,
                meta_task_type(meta_task=_ctx.meta_task, scale=_C.CONFIDENCE_SCALE),
            )
            buf.write(f"Soft confidence (probability-weighted): {float(soft):.3f}\n")
        buf.write("\n")

        if other_data is not None:
            other_prompt, other_opts = format_other_confidence_prompt(
                q, tokenizer, use_chat_template,
            )
            buf.write("--- OTHER-CONFIDENCE PROMPT (sent to model) ---\n")
            buf.write(other_prompt)
            buf.write("\n\n")
            buf.write("--- OTHER-CONFIDENCE RESPONSE ---\n")
            if i < len(other_probs) and other_probs[i]:
                p = other_probs[i]
                probs_str = "  ".join(f"{L}={v:.3f}" for L, v in zip(other_opts, p))
                buf.write(f"Option probs: {probs_str}\n")
            if i < len(other_resp):
                buf.write(f"Chosen (argmax): {other_resp[i]}\n")
            if other_signals is not None and i < len(other_signals):
                buf.write(f"Soft other-confidence (probability-weighted): "
                          f"{float(other_signals[i]):.3f}\n")
            buf.write("\n")

    buf.write("=" * 80 + "\n")
    buf.write("END\n")
    buf.write("=" * 80 + "\n")

    with open(output_path, "w") as f:
        f.write(buf.getvalue())
    print(f"Saved example prompts/responses to {output_path}")


def save_quick_summary_png(data: dict, questions: list, output_path: str) -> None:
    """3-panel diagnostic PNG: MC dist + meta-response dist + entropy-vs-confidence scatter.

    Runs BEFORE the slow probe / introspection analysis so distributions are
    visible while the rest of the pipeline finishes.
    """
    n = len(questions)
    direct_probs = data.get("direct_probs", [])
    options_per_q = [list(q.get("options", {}).keys()) for q in questions]
    direct_resp = [
        opts[int(np.argmax(p))] if opts and p else None
        for opts, p in zip(options_per_q, direct_probs)
    ]
    correct = [q.get("correct_answer") for q in questions]
    is_correct = [r == c for r, c in zip(direct_resp, correct)]
    acc = float(np.mean(is_correct)) if is_correct else float("nan")

    meta_probs = data.get("meta_probs", [])
    meta_resp = data.get("meta_responses", [])
    meta_mappings = data.get("meta_mappings", []) or [None] * n
    entropies = np.asarray(data.get("direct_metrics", {}).get("entropy", [np.nan] * n))

    if _ctx.meta_task == "delegate" and _C.DELEGATE_PROMPT_DESIGN == "mc_integrated":
        confidence_options = ANSWER_WITH_DELEGATE_OPTIONS
        t_idx = confidence_options.index("T")
        stated_num = np.full(n, np.nan)
        for i, p in enumerate(meta_probs):
            if not p:
                continue
            p_arr = np.asarray(p, dtype=float)
            if p_arr.shape[0] >= len(confidence_options):
                stated_num[i] = 1.0 - float(p_arr[t_idx])
        stated_ylabel = "P(Answer)"
    elif _ctx.meta_task == "delegate":
        confidence_options = ANSWER_OR_DELEGATE_OPTIONS
        stated_num = np.full(n, np.nan)
        for i, p in enumerate(meta_probs):
            if not p:
                continue
            p_arr = np.asarray(p, dtype=float)
            m = meta_mappings[i] if i < len(meta_mappings) and meta_mappings[i] else None
            if m:
                tokens = sorted(m.keys())
                ans_col = next((j for j, t in enumerate(tokens) if m.get(t) == "Answer"), None)
                if ans_col is not None and p_arr.shape[0] >= 2:
                    stated_num[i] = float(p_arr[ans_col])
            else:
                stated_num[i] = float(p_arr[0]) if p_arr.shape[0] >= 1 else np.nan
        stated_ylabel = "P(Answer)"
    else:
        confidence_options = list(_scale_options(_C.CONFIDENCE_SCALE).keys())
        stated_num = np.array(
            [get_meta_signal(np.asarray(p), scale=_C.CONFIDENCE_SCALE) if p else np.nan for p in meta_probs]
        )
        stated_ylabel = "stated_confidence_numeric (soft)"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    letters = ["A", "B", "C", "D"]
    chosen_counts = [direct_resp.count(L) for L in letters]
    correct_counts = [correct.count(L) for L in letters]
    x = np.arange(len(letters))
    w = 0.4
    ax.bar(x - w / 2, np.array(chosen_counts) / max(n, 1), width=w, label="chosen")
    ax.bar(x + w / 2, np.array(correct_counts) / max(n, 1), width=w, label="correct")
    ax.axhline(0.25, color="gray", linestyle=":", alpha=0.5, label="uniform")
    ax.set_xticks(x); ax.set_xticklabels(letters)
    ax.set_ylabel("fraction"); ax.set_title(f"MC answers  (acc = {acc:.1%}, n = {n})")
    ax.legend(fontsize=8); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    conf_counts = [meta_resp.count(k) for k in confidence_options]
    ax.bar(range(len(confidence_options)), np.array(conf_counts) / max(n, 1), color="tab:orange")
    ax.set_xticks(range(len(confidence_options)))
    ax.set_xticklabels(confidence_options, fontsize=9)
    ax.set_ylabel("fraction of responses")
    if _ctx.meta_task == "delegate" and _C.DELEGATE_PROMPT_DESIGN == "mc_integrated":
        panel2_title = "Answer-with-delegate distribution  (A/B/C/D/T)"
    elif _ctx.meta_task == "delegate":
        panel2_title = "Delegate digit distribution  ('1' vs '2')"
    else:
        panel2_title = f"Stated confidence  ({_C.CONFIDENCE_SCALE} scale)"
    ax.set_title(panel2_title); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis="y")

    valid_num = stated_num[~np.isnan(stated_num)]
    if len(valid_num):
        label = "P(Answer)" if _ctx.meta_task == "delegate" else "stated_num"
        ax.text(
            0.98, 0.95,
            f"mean {label} = {valid_num.mean():.3f}\nstd  = {valid_num.std():.3f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
        )

    ax = axes[2]
    mask = (~np.isnan(entropies)) & (~np.isnan(stated_num))
    if mask.sum() > 10:
        r, p_r = pearsonr(entropies[mask], stated_num[mask])
        rho, p_rho = spearmanr(entropies[mask], stated_num[mask])
    else:
        r = rho = float("nan")
        p_r = p_rho = float("nan")
    ax.scatter(entropies[mask], stated_num[mask], alpha=0.35, s=12)
    ax.set_xlabel("MC entropy (nats)")
    ax.set_ylabel(stated_ylabel)
    panel3_head = (
        "Entropy  vs  P(Answer)" if _ctx.meta_task == "delegate"
        else "Entropy  vs  stated confidence"
    )
    ax.set_title(
        f"{panel3_head}\n"
        f"Pearson r = {r:+.3f}  (p={p_r:.1e})\n"
        f"Spearman ρ = {rho:+.3f}  (p={p_rho:.1e})"
    )
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{get_model_display_label()}  /  {_ctx.dataset_name}  /  "
        f"{_ctx.meta_task}  /  scale={_C.CONFIDENCE_SCALE}",
        y=1.02, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Saved quick summary to {output_path}")
