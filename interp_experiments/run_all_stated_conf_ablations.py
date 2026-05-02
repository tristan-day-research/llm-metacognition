"""
Single-command driver for the stated-confidence direction ablation.

Does three things end-to-end:
  1. Builds the _mc_stated_confidence_directions.npz + _mc_dataset.json files
     for all 6 runs (base / instruct / finetuned × SimpleMC / TriviaMC) using
     existing introspection outputs — CPU only.
  2. Loads the model + adapter once per distinct (base_model, adapter) pair and
     runs the ablation for each matching dataset, mutating the module-level
     constants in run_ablation_causality so main() picks them up.
  3. Prints a one-line summary of each ablation's results JSON location.

Usage (remote):
  python run_all_stated_conf_ablations.py                 # all 6, full scale
  python run_all_stated_conf_ablations.py --dry-run       # tiny, ~minutes total
  python run_all_stated_conf_ablations.py --only base_TriviaMC instruct_SimpleMC

--dry-run sets LAYERS=[20], NUM_CONTROLS=2, NUM_QUESTIONS=20, BOOTSTRAP_N=50,
so you can confirm plumbing (direction load, dataset load, forward pass,
results JSON write) without waiting hours. Baseline corr should match the
corr(stated_conf, entropy) you see in the notebook.

Why mutate globals rather than refactor to a Config dataclass: the ablation
script has ~20 module-level constants read from many functions. A full refactor
is out of scope for this experiment. Module globals are plain attributes —
setting them before calling main() is equivalent to editing the file between
runs, just without manual text edits.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import gc
import sys
import traceback
from pathlib import Path
from typing import List

import build_mc_inputs_from_introspection as build_step
import run_ablation_causality as rac
from ablation_run_configs import RUNS, RunConfig

try:
    import torch
except ImportError:
    torch = None


def _apply_run_overrides(run: RunConfig, *, dry_run: bool) -> None:
    """Mutate run_ablation_causality module constants for this run."""
    rac.MODEL = run.base_model
    rac.ADAPTER = run.adapter
    rac.INPUT_BASE_NAME = run.base_name
    rac.DIRECTION_METRIC = "stated_confidence"
    rac.TARGET_METRIC = "entropy"
    rac.META_TASK = "confidence"  # emits stated_confidence_numeric signal
    # Must match the scale the introspection run used when producing the
    # direction file. run_introspection_for_ablation.py fixes this at "numeric".
    rac.CONFIDENCE_SCALE = "numeric"
    rac.BASE_CONFIDENCE_FEW_SHOT_MODE = "fixed"

    if dry_run:
        rac.LAYERS = [20]
        rac.NUM_CONTROLS = 2
        rac.NUM_CONTROLS_NONFINAL = 2
        rac.NUM_QUESTIONS = 20
        rac.BOOTSTRAP_N = 50
        rac.USE_TRANSFER_SPLIT = False  # lets NUM_QUESTIONS take effect
    else:
        # Full runs: rely on script defaults (all layers, USE_TRANSFER_SPLIT=True,
        # BOOTSTRAP_N=2000, NUM_CONTROLS=25) but make sure nothing from a prior
        # dry-run iteration leaks through.
        rac.LAYERS = None
        rac.NUM_CONTROLS = 25
        rac.NUM_QUESTIONS = 100
        rac.BOOTSTRAP_N = 2000
        rac.USE_TRANSFER_SPLIT = True


def _require_inputs(run: RunConfig) -> None:
    directions = rac.OUTPUT_DIR / f"{run.base_name}_mc_stated_confidence_directions.npz"
    dataset = rac.OUTPUT_DIR / f"{run.base_name}_mc_dataset.json"
    missing = [p for p in (directions, dataset) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[{run.label}] build step did not produce: {', '.join(str(p) for p in missing)}"
        )


def _select_runs(only: List[str]) -> List[RunConfig]:
    if not only:
        return list(RUNS)
    labels = set(only)
    unknown = labels - {r.label for r in RUNS}
    if unknown:
        raise SystemExit(f"Unknown --only labels: {sorted(unknown)}")
    return [r for r in RUNS if r.label in labels]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Small/fast settings for plumbing checks (1 layer, 2 controls, 20 questions, 50 bootstraps).",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Restrict to specific run labels (e.g. base_SimpleMC). Default: all 6.",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip the build step (assumes direction + dataset files already exist).",
    )
    args = parser.parse_args()

    runs = _select_runs(args.only or [])
    run_labels = [r.label for r in runs]
    print(f"Selected {len(runs)} run(s): {run_labels}")
    if args.dry_run:
        print("DRY-RUN mode: tiny settings (LAYERS=[20], NUM_CONTROLS=2, NUM_QUESTIONS=20, BOOTSTRAP_N=50)")

    # --- Step 1: build inputs (CPU only) ---
    if not args.skip_build:
        print("\n" + "=" * 70)
        print("STEP 1 / 2: build_mc_inputs_from_introspection")
        print("=" * 70)
        build_rc = build_step.main(["build_mc_inputs_from_introspection.py"] + run_labels)
        if build_rc != 0:
            print("\nBuild step reported failures; continuing to ablation only for runs whose inputs exist.")

    # --- Step 2: ablation per run ---
    print("\n" + "=" * 70)
    print("STEP 2 / 2: ablation")
    print("=" * 70)

    # Sort so runs sharing a (base_model, adapter) pair are back-to-back — minimizes
    # number of model loads if run_ablation_causality.main() is ever teach to cache.
    # Today each main() call reloads the model, so this is purely cosmetic, but it
    # also makes interleaved logs easier to scan.
    runs_sorted = sorted(runs, key=lambda r: (r.base_model, str(r.adapter), r.dataset))

    summary = []
    for run in runs_sorted:
        print("\n" + "#" * 70)
        print(f"# RUN: {run.label}  base={run.base_name}")
        print("#" * 70)
        try:
            _require_inputs(run)
        except FileNotFoundError as exc:
            print(exc)
            summary.append((run.label, "SKIPPED (no inputs)", None))
            continue

        _apply_run_overrides(run, dry_run=args.dry_run)

        try:
            rac.main()
        except Exception:
            traceback.print_exc()
            summary.append((run.label, "FAILED", None))
            continue
        finally:
            # Release the model + any tensors held inside rac.main's local scope
            # so the next iteration's load_model_and_tokenizer doesn't OOM.
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # run_ablation_causality writes outputs named via
        # f"{model_short}_{INPUT_BASE_NAME.split('_')[-1]}_ablation_{META_TASK}_{DIRECTION_METRIC}"
        # We reconstruct the results path for the summary.
        model_short = rac.get_model_short_name(rac.MODEL)
        base_output = (
            f"{model_short}_{rac.INPUT_BASE_NAME.split('_')[-1]}"
            f"_ablation_{rac.META_TASK}_{rac.DIRECTION_METRIC}"
        )
        results_path = rac.OUTPUT_DIR / f"{base_output}_results.json"
        summary.append((run.label, "OK" if results_path.exists() else "OK (no results file?)", results_path))

    print("\n" + "=" * 70)
    print("ALL RUNS SUMMARY")
    print("=" * 70)
    for label, status, path in summary:
        print(f"  {label:25s}  {status:25s}  {path if path else ''}")
    n_ok = sum(1 for _, s, _ in summary if s.startswith("OK"))
    print(f"\n{n_ok}/{len(summary)} runs completed successfully.")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
