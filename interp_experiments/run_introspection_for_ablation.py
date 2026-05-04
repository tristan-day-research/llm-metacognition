"""
Re-run run_introspection_experiment.py on the remote for the 3 model variants
(base / instruct / finetuned) × 2 datasets (SimpleMC / TriviaMC), producing
_direct_activations.npz + _paired_data.json that the stated-confidence ablation
build step consumes.

Only runs the CONFIDENCE meta-task (that's what produces stated_confidence_numeric).
Output files land in outputs/ directly — no subdirectory. That's what
ablation_run_configs.py will look for.

Usage on remote:
  python run_introspection_for_ablation.py                    # all 3 models
  python run_introspection_for_ablation.py --only base        # single model
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import argparse
import gc
import sys

import run_introspection_experiment as rie

try:
    import torch
except ImportError:
    torch = None


CONFIGS = [
    dict(
        label="base",
        BASE_MODEL_NAME="meta-llama/Llama-3.1-8B",
        MODEL_NAME="meta-llama/Llama-3.1-8B",
    ),
    dict(
        label="instruct",
        BASE_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct",
        MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct",
    ),
    dict(
        label="finetuned",
        BASE_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct",
        MODEL_NAME="Tristan-Day/ect_20251222_215412_v0uei7y1_2000",
    ),
]


def apply_overrides(cfg):
    for k, v in cfg.items():
        if k == "label":
            continue
        setattr(rie, k, v)
    # Confidence task gives us stated_confidence_numeric
    rie.DATASETS = ["SimpleMC", "TriviaMC"]
    rie.META_TASKS = ["confidence"]
    # 1-10 numeric scale. Base model uses fixed few-shot examples (the
    # numeric path only supports "fixed" or "none"; "fixed" matches the prior
    # base run's config.few_shot_mode). Ignored for instruct/finetuned.
    rie.CONFIDENCE_SCALE = "numeric"
    rie.FEW_SHOT_MODE = "fixed"
    # Fresh MC pass (no reuse — we want both direct + meta in one go)
    rie.REUSE_DIRECT_FROM_CONFIDENCE = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None,
                        help="Restrict to specific labels: base / instruct / finetuned")
    args = parser.parse_args()

    selected = [c for c in CONFIGS if not args.only or c["label"] in args.only]
    labels = [c["label"] for c in selected]
    print(f"Regenerating introspection for: {labels}")

    for cfg in selected:
        print("\n" + "#" * 70)
        print(f"# MODEL: {cfg['label']}  ({cfg['BASE_MODEL_NAME']} + {cfg['MODEL_NAME']})")
        print("#" * 70)
        apply_overrides(cfg)
        # run_introspection_experiment.main uses argparse; reset argv so its parser
        # uses defaults (not the flags we were given).
        sys.argv = ["run_introspection_experiment.py"]
        rie.main()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✓ All introspection runs complete. Files are in outputs/.")


if __name__ == "__main__":
    main()
