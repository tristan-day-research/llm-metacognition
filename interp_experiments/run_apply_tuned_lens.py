"""
Retroactively add tuned-lens NPZs to an EXISTING `run_collect_activations.py`
run subfolder, WITHOUT re-running the forward passes.

Use case: you ran `run_collect_activations.py` with the old vanilla-only
logit lens, and now want tuned-lens projections on the already-saved
activations. This script:

  1. Loads the model + (optional) adapter (you must point it at the SAME
     base/adapter that was used for the original run).
  2. Reads the saved `*_direct_activations.npz` and `*_meta_activations.npz`
     from a run subfolder.
  3. Trains a tuned lens on direct ∪ meta residuals (same calibration
     recipe the integrated runner uses).
  4. Applies the tuned lens and writes
       <prefix>_direct_tuned_logit_lens.npz
       <prefix>_meta_tuned_logit_lens.npz
       <prefix>_tuned_lens.pt
     alongside the existing files.

Usage:
    python interp_experiments/run_apply_tuned_lens.py \\
        outputs/activations_directions_logitlens/8b_finetuned_mixed_17173_all_test/

You can pass multiple subfolders; each is processed in turn. The model is
re-loaded per subfolder if BASE_MODEL_NAME / MODEL_NAME differ between them
(detected from each run's `_paired_data.json`).
"""

# --- repo path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _preflight import check_hf_login; check_hf_login()


import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.model_utils import DEVICE, load_model_and_tokenizer
from experiment_config import IntrospectionExperimentConfig as _C

from _runio import _atomic_savez_compressed, _pad_token_id_groups
from _tuned_lens import (
    apply_tuned_lens,
    load_tuned_lens,
    save_tuned_lens,
    train_tuned_lens,
)


def _find_prefix(folder: Path) -> str:
    """Locate the file prefix (everything before _paired_data.json) in a run folder."""
    candidates = list(folder.glob("*_paired_data.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected 1 *_paired_data.json in {folder}, got {len(candidates)}")
    return str(candidates[0])[:-len("_paired_data.json")]


def _load_activations_npz(path: Path):
    """Return {layer_idx: np.array} for layer_* keys in the NPZ, plus (option_strs, option_token_ids).

    The activation NPZ schema is: layer_0..layer_N + various scalar metric arrays.
    The lens NPZ schema is: option_strs + option_token_ids (we read those from the
    EXISTING vanilla-lens NPZ to keep the option set byte-identical).
    """
    with np.load(path, allow_pickle=True) as f:
        return {
            int(k.split("_")[1]): f[k].astype(np.float32)
            for k in f.files if k.startswith("layer_")
        }


def _load_option_metadata(lens_path: Path):
    """Pull option_strs + option_token_ids out of an existing vanilla-lens NPZ."""
    with np.load(lens_path, allow_pickle=True) as f:
        option_strs = [str(s) for s in f["option_strs"]]
        # option_token_ids was saved as padded 2D int64 with -1 sentinel.
        padded = f["option_token_ids"]
        groups = []
        for row in padded:
            row = [int(x) for x in row if int(x) >= 0]
            groups.append(row)
        return option_strs, groups, np.asarray(f.get("run_id", "")), np.asarray(f.get("question_ids", []))


def process_run_folder(folder: Path) -> None:
    folder = folder.resolve()
    prefix = _find_prefix(folder)
    print(f"\n{'=' * 80}")
    print(f"Processing: {folder}")
    print(f"Prefix:     {prefix}")
    print('=' * 80)

    # Read paired_data.json to find the model + adapter that produced this run.
    paired = json.load(open(f"{prefix}_paired_data.json"))
    cfg = paired.get("config", {}) or {}
    base_name = cfg.get("base_model_name") or _C.BASE_MODEL_NAME
    adapter = cfg.get("model_name")
    if adapter and adapter == base_name:
        adapter = None

    # Load the model.
    print(f"\nLoading model: {base_name}")
    if adapter:
        print(f"        adapter: {adapter}")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        base_name, adapter_path=adapter,
        load_in_4bit=_C.LOAD_IN_4BIT, load_in_8bit=_C.LOAD_IN_8BIT,
    )

    # Load the saved activations (direct + meta).
    print(f"\nLoading activations from {folder}/...")
    direct_acts = _load_activations_npz(Path(f"{prefix}_direct_activations.npz"))
    meta_acts = _load_activations_npz(Path(f"{prefix}_meta_activations.npz"))
    print(f"  direct: {len(direct_acts)} layers, n_questions={direct_acts[0].shape[0]}")
    print(f"  meta:   {len(meta_acts)} layers,  n_questions={meta_acts[0].shape[0]}")

    # Read option-set metadata from the existing VANILLA-lens NPZs (so the
    # tuned-lens output uses byte-identical option strings + token ids).
    direct_option_strs, direct_option_token_ids, run_id_arr, qid_arr_d = \
        _load_option_metadata(Path(f"{prefix}_direct_logit_lens.npz"))
    meta_option_strs, meta_option_token_ids, _, qid_arr_m = \
        _load_option_metadata(Path(f"{prefix}_meta_logit_lens.npz"))

    # Train (or load) the tuned lens.
    tuned_lens_path = f"{prefix}_tuned_lens.pt"
    if Path(tuned_lens_path).exists():
        print(f"\nLoading cached tuned lens from {tuned_lens_path}")
        tuned_lens = load_tuned_lens(tuned_lens_path, device=DEVICE)
    else:
        print("\nTraining tuned lens (per-layer affine, KL-to-final-logits)...")
        combined = {
            layer_idx: np.concatenate([direct_acts[layer_idx], meta_acts[layer_idx]], axis=0)
            for layer_idx in direct_acts
        }
        tuned_lens, train_losses = train_tuned_lens(
            model, combined,
            n_epochs=_C.TUNED_LENS_EPOCHS,
            lr=_C.TUNED_LENS_LR,
            batch_size=_C.TUNED_LENS_BATCH_SIZE,
            weight_decay=_C.TUNED_LENS_WEIGHT_DECAY,
        )
        save_tuned_lens(tuned_lens, tuned_lens_path)
        print(f"Saved tuned-lens affines → {tuned_lens_path}")
        print("Tuned-lens training (initial → final mean loss, every 4th layer):")
        for layer_idx in sorted(train_losses.keys())[::4]:
            losses = train_losses[layer_idx]
            if losses:
                print(f"  layer {layer_idx:>2d}: {losses[0]:.4f} → {losses[-1]:.4f}")

    # Apply to direct + meta and save the resulting NPZs.
    def _save(path, lens_out, opt_strs, opt_ids, qid_arr):
        _atomic_savez_compressed(
            path,
            option_logits=lens_out["option_logits"],
            top_k_ids=lens_out["top_k_ids"],
            top_k_logits=lens_out["top_k_logits"],
            layer_indices=lens_out["layer_indices"],
            option_strs=np.array(opt_strs, dtype=object),
            option_token_ids=_pad_token_id_groups(opt_ids),
            run_id=run_id_arr,
            question_ids=qid_arr,
        )

    print("\nApplying tuned lens (direct)...")
    direct_tuned = apply_tuned_lens(direct_acts, model, tuned_lens, direct_option_token_ids)
    _save(f"{prefix}_direct_tuned_logit_lens.npz",
          direct_tuned, direct_option_strs, direct_option_token_ids, qid_arr_d)

    print("Applying tuned lens (meta)...")
    meta_tuned = apply_tuned_lens(meta_acts, model, tuned_lens, meta_option_token_ids)
    _save(f"{prefix}_meta_tuned_logit_lens.npz",
          meta_tuned, meta_option_strs, meta_option_token_ids, qid_arr_m)

    print(f"\n✓ Done — wrote tuned-lens NPZs to {folder}")

    # Free GPU memory before the next folder.
    del model, tokenizer, tuned_lens, direct_acts, meta_acts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Retroactively add tuned-lens NPZs to existing run subfolders.",
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="One or more run subfolders to process.",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Run subfolders to process: {len(args.run_dirs)}")

    for folder in args.run_dirs:
        if not folder.is_dir():
            print(f"⚠ skipping {folder} — not a directory")
            continue
        try:
            process_run_folder(folder)
        except Exception as e:
            import traceback
            print(f"\n✗ {folder} failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
