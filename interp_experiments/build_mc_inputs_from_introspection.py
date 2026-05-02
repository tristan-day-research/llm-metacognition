"""
Build ablation inputs (_mc_stated_confidence_directions.npz + _mc_dataset.json)
from existing run_introspection_experiment.py outputs.

Why reuse introspection outputs instead of running identify_mc_correlate.py
--------------------------------------------------------------------------
identify_mc_correlate.py does a fresh MC forward pass to collect activations +
metrics, then fits directions. The introspection pipeline already saved the
same direct-pass activations and stated_confidence_numeric values for each run.
Reusing them avoids 6 GPU-hours of redundant MC passes and keeps the direction
fits on exactly the same trials the notebook figures were built from.

Output (per run, in outputs/):
  {base_name}_mc_stated_confidence_directions.npz
  {base_name}_mc_dataset.json
where base_name comes from ablation_run_configs.RunConfig.base_name.

Usage:
  python build_mc_inputs_from_introspection.py              # all 6 runs
  python build_mc_inputs_from_introspection.py base_SimpleMC instruct_TriviaMC  # subset
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

from ablation_run_configs import OUTPUT_DIR, RUNS, RunConfig
from core.directions import find_directions


# Match identify_mc_correlate.py defaults so the output is interchangeable
# with its format.
PROBE_ALPHA = 1000.0
PROBE_PCA_COMPONENTS = 100
PROBE_N_BOOTSTRAP = 100
PROBE_TRAIN_SPLIT = 0.8
MEAN_DIFF_QUANTILE = 0.25
SEED = 42

# Direction target — the new thing we add on top of entropy/margin/etc.
TARGET_KEY = "stated_confidence"       # used in output filename: _mc_{TARGET_KEY}_directions.npz
TARGET_FIELD = "stated_confidence_numeric"  # key in paired_data.json


def stack_direct_activations(npz_path: Path) -> Dict[int, np.ndarray]:
    """Same pattern as analyze_activations.ipynb's _stack_layer_npz.
    Returns {layer_idx: (n_samples, hidden_dim)} as float32."""
    with np.load(npz_path) as data:
        layer_keys = sorted(
            (k for k in data.files if k.startswith("layer_")),
            key=lambda k: int(k.split("_")[1]),
        )
        return {int(k.split("_")[1]): data[k].astype(np.float32) for k in layer_keys}


def build_dataset_items(paired: Dict) -> List[Dict]:
    """Per-question list that run_ablation_causality.py reads.
    Mirrors identify_mc_correlate.py:342-354 but pulls from paired_data.json
    rather than recomputing. Adds stated_confidence_numeric alongside entropy."""
    questions = paired["questions"]
    direct_probs = paired["direct_probs"]
    direct_logits = paired["direct_logits"]
    is_correct = paired["is_correct"]
    direct_metrics = paired["direct_metrics"]
    stated_conf = paired[TARGET_FIELD]

    n = len(questions)
    items = []
    for i in range(n):
        q = questions[i]
        probs = direct_probs[i] if direct_probs[i] else [0.25, 0.25, 0.25, 0.25]
        predicted_idx = int(np.argmax(probs))
        predicted_letter = "ABCD"[predicted_idx]

        item = {
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "predicted_answer": predicted_letter,
            "is_correct": bool(is_correct[i]),
            "options": q["options"],
            "probabilities": probs,
            "logits": direct_logits[i] if direct_logits[i] else [0.0, 0.0, 0.0, 0.0],
        }
        for metric_name, values in direct_metrics.items():
            item[metric_name] = float(values[i])
        v = stated_conf[i]
        item[TARGET_FIELD] = (
            float(v) if v is not None and not (isinstance(v, float) and np.isnan(v))
            else float("nan")
        )
        items.append(item)
    return items


def fit_stated_confidence_directions(
    activations_by_layer: Dict[int, np.ndarray],
    target_values: np.ndarray,
):
    """Drop NaN rows, call find_directions with identify_mc_correlate.py defaults."""
    valid = ~np.isnan(target_values)
    n_valid = int(valid.sum())
    n_total = len(target_values)
    if n_valid < 20:
        raise RuntimeError(
            f"Only {n_valid} non-NaN stated_confidence_numeric values out of {n_total} — "
            f"direction fit would be unreliable."
        )
    filtered = {layer: X[valid] for layer, X in activations_by_layer.items()}
    y = target_values[valid].astype(np.float64)
    results = find_directions(
        filtered, y,
        methods=["probe", "mean_diff"],
        probe_alpha=PROBE_ALPHA,
        probe_pca_components=PROBE_PCA_COMPONENTS,
        probe_n_bootstrap=PROBE_N_BOOTSTRAP,
        probe_train_split=PROBE_TRAIN_SPLIT,
        mean_diff_quantile=MEAN_DIFF_QUANTILE,
        seed=SEED,
        return_scaler=True,
    )
    return results, n_valid, n_total


def save_directions_npz(results, n_layers, dataset_name, model_name, out_path: Path):
    """Exact schema run_ablation_causality.py:load_directions expects.
    Mirrors identify_mc_correlate.py:475-493."""
    dir_save = {
        "_metadata_dataset": np.array(dataset_name),
        "_metadata_model": np.array(model_name),
        "_metadata_metric": np.array(TARGET_KEY),
    }
    for method in ("probe", "mean_diff"):
        for layer in range(n_layers):
            dir_save[f"{method}_layer_{layer}"] = results["directions"][method][layer]
            if method == "probe" and "scaler_scale" in results["fits"][method][layer]:
                dir_save[f"{method}_scaler_scale_{layer}"] = results["fits"][method][layer]["scaler_scale"]
                dir_save[f"{method}_scaler_mean_{layer}"] = results["fits"][method][layer]["scaler_mean"]
    np.savez(out_path, **dir_save)


def save_dataset_json(items, run: RunConfig, n_valid, n_total, out_path: Path):
    correct_count = sum(1 for it in items if it["is_correct"])
    accuracy = correct_count / len(items)

    stats = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(items),
        f"{TARGET_FIELD}_n_valid": n_valid,
        f"{TARGET_FIELD}_n_total": n_total,
    }
    metric_names = set()
    for it in items:
        for k, v in it.items():
            if isinstance(v, (int, float)) and k not in ("correct_answer", "predicted_answer"):
                metric_names.add(k)
    for m in sorted(metric_names):
        vals = np.array(
            [
                it[m] for it in items
                if isinstance(it[m], (int, float))
                and not (isinstance(it[m], float) and np.isnan(it[m]))
            ],
            dtype=np.float64,
        )
        if vals.size > 0:
            stats[f"{m}_mean"] = float(vals.mean())
            stats[f"{m}_std"] = float(vals.std())

    dataset_json = {
        "config": {
            "dataset": run.dataset,
            "num_questions": len(items),
            "base_model": run.base_model,
            "adapter": run.adapter,
            "seed": SEED,
            "source": "build_mc_inputs_from_introspection.py",
            "intro_stem": run.intro_stem,
        },
        "stats": stats,
        "data": items,
    }
    with open(out_path, "w") as f:
        json.dump(dataset_json, f, indent=2)


def process_run(run: RunConfig) -> bool:
    """Returns True on success, False on skip/failure. Prints progress inline."""
    if not run.direct_activations_path.exists():
        print(f"[{run.label}] MISSING {run.direct_activations_path} — skipping")
        return False
    if not run.paired_data_path.exists():
        print(f"[{run.label}] MISSING {run.paired_data_path} — skipping")
        return False

    print(f"\n=== [{run.label}] ===")
    print(f"  direct_acts: {run.direct_activations_path}")
    print(f"  paired:      {run.paired_data_path}")

    with open(run.paired_data_path, "r") as f:
        paired = json.load(f)
    activations = stack_direct_activations(run.direct_activations_path)
    n_layers = len(activations)
    n_samples = activations[0].shape[0]
    n_paired = len(paired["questions"])
    if n_samples != n_paired:
        print(f"[{run.label}] FAILED: sample mismatch acts={n_samples} paired={n_paired}")
        return False

    stated = np.array(
        [v if v is not None else np.nan for v in paired[TARGET_FIELD]],
        dtype=np.float64,
    )
    nan_frac = float(np.mean(np.isnan(stated)))
    print(f"  n_samples={n_samples}, n_layers={n_layers}, {TARGET_FIELD} NaN fraction={nan_frac:.1%}")
    if nan_frac > 0.20:
        print(f"  WARNING: >20% NaN — meta-task output parse may be degraded")

    results, n_valid, n_total = fit_stated_confidence_directions(activations, stated)

    probe_r2 = np.array([results["fits"]["probe"][L]["r2"] for L in range(n_layers)])
    md_r2 = np.array([results["fits"]["mean_diff"][L]["r2"] for L in range(n_layers)])
    print(
        f"  probe      best R²={probe_r2.max():.3f} at layer {int(np.argmax(probe_r2))} "
        f"(last={probe_r2[-1]:.3f})"
    )
    print(
        f"  mean_diff  best R²={md_r2.max():.3f} at layer {int(np.argmax(md_r2))} "
        f"(last={md_r2[-1]:.3f})"
    )

    items = build_dataset_items(paired)

    out_directions = OUTPUT_DIR / f"{run.base_name}_mc_{TARGET_KEY}_directions.npz"
    out_dataset = OUTPUT_DIR / f"{run.base_name}_mc_dataset.json"

    save_directions_npz(
        results=results,
        n_layers=n_layers,
        dataset_name=run.dataset,
        model_name=run.base_model if not run.adapter else f"{run.base_model}+{run.adapter}",
        out_path=out_directions,
    )
    save_dataset_json(items, run, n_valid, n_total, out_dataset)

    print(f"  wrote {out_directions.name}")
    print(f"  wrote {out_dataset.name}")
    print(f"  INPUT_BASE_NAME for ablation: {run.base_name!r}")
    return True


def main(argv: List[str]) -> int:
    selected = set(argv[1:]) if len(argv) > 1 else None
    n_success = 0
    n_total = 0
    for run in RUNS:
        if selected and run.label not in selected:
            continue
        n_total += 1
        try:
            if process_run(run):
                n_success += 1
        except Exception as exc:
            print(f"[{run.label}] FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\nBuild complete: {n_success}/{n_total} runs succeeded.")
    return 0 if n_success == n_total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
