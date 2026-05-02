#!/usr/bin/env python3
"""Analyze cross-layer similarity and decompose late directions into early-parallel vs orth.

This script implements the two diagnostics we discussed:

TEST 1 (cross-layer similarity):
  - Load per-layer direction vectors from an NPZ produced by identify_mc_correlate.py
    (default: outputs/{BASE_NAME}_mc_{METRIC}_directions.npz)
  - Compute a cosine-similarity matrix over layers of interest
  - Save a CSV + heatmap + print a compact summary

TEST 2 (late-layer decomposition):
  - Choose an "early reference" direction (by default, the normalized mean of EARLY_LAYERS)
  - For each L in LATE_LAYERS:
        d_L = parallel + orth
        parallel = (d_L · e) * e
        orth     = d_L - parallel
    and report how much of d_L's energy is parallel vs orth.
  - Optionally write NPZ files containing ONLY the parallel or orth component at each late layer,
    in the same key format as your ablation script expects (e.g., "mean_diff_layer_38").

Typical usage (no long CLI args; edit the CONFIG section below):
    python analyze_direction_mixture.py

Or override a few values from CLI:
    python analyze_direction_mixture.py --base_name ... --metric top_logit --method mean_diff \
        --early 30-33 --late 36-39 --write_npz

Notes:
- Directions are re-normalized within this script to be safe.
- If you write NPZ components, you can temporarily swap them into
  outputs/{BASE_NAME}_mc_{METRIC}_directions.npz (after backing up) and run your ablation script
  with LAYERS set to the corresponding late layer.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# matplotlib is available in your environment; keep it optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================
# CONFIG (edit these defaults)
# ============================

BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC"
METRIC = "top_logit"
METHOD = "mean_diff"  # "mean_diff" or "probe"

# Layers to compare
EARLY_LAYERS = "30-33"   # string spec: "30-33" or "30,31,33" etc.
LATE_LAYERS = "36-39"

# If set, compute cosine matrix across this union (EARLY ∪ LATE) by default.
# You can override with --layers.

OUTPUT_DIR = Path("outputs")

# If True, also compute similarity of each layer to the early reference direction.
REPORT_SIM_TO_EARLY_REF = True

# If True, save per-late-layer NPZ files for parallel/orth components.
WRITE_NPZ = False


# ============================
# Helpers
# ============================

def _parse_layers_spec(spec: str) -> List[int]:
    """Parse layer specs like "30-33,36,38-39" into a sorted unique list."""
    spec = (spec or "").strip()
    if not spec:
        return []
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip(); b = b.strip()
            if not a or not b:
                raise ValueError(f"Bad layer range: {part!r}")
            lo = int(a); hi = int(b)
            step = 1 if hi >= lo else -1
            out.extend(list(range(lo, hi + step, step)))
        else:
            out.append(int(part))
    return sorted(set(out))


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    u = _l2_normalize(u)
    v = _l2_normalize(v)
    return float(np.dot(u, v))


def _load_npz_directions(npz_path: Path) -> Dict[str, Dict[int, np.ndarray]]:
    """Load NPZ where keys look like "mean_diff_layer_38"."""
    data = np.load(npz_path)
    methods: Dict[str, Dict[int, np.ndarray]] = {}
    for key in data.files:
        if key.startswith("_"):
            continue
        parts = key.rsplit("_layer_", 1)
        if len(parts) != 2:
            continue
        method, layer_str = parts
        try:
            layer = int(layer_str)
        except ValueError:
            continue
        if method not in methods:
            methods[method] = {}
        methods[method][layer] = _l2_normalize(data[key])
    return methods


def _save_heatmap(mat: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(max(5, 0.35 * len(labels)), max(4, 0.35 * len(labels))))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


@dataclass
class DecompRow:
    layer: int
    cos_to_early_ref: float
    parallel_coeff: float
    parallel_energy_frac: float
    orth_energy_frac: float
    parallel_norm: float
    orth_norm: float


def main():
    ap = argparse.ArgumentParser(description="Analyze direction mixture across layers.")
    ap.add_argument("--base_name", type=str, default=BASE_NAME)
    ap.add_argument("--metric", type=str, default=METRIC)
    ap.add_argument("--method", type=str, default=METHOD)
    ap.add_argument("--directions_npz", type=str, default=None,
                    help="Override NPZ path. Default: outputs/{base_name}_mc_{metric}_directions.npz")
    ap.add_argument("--out_prefix", type=str, default=None,
                    help="Output prefix (without extension). Default: outputs/{base_name}_{metric}_{method}_mix")

    ap.add_argument("--early", type=str, default=EARLY_LAYERS, help="Early layers spec, e.g. 30-33")
    ap.add_argument("--late", type=str, default=LATE_LAYERS, help="Late layers spec, e.g. 36-39")
    ap.add_argument("--layers", type=str, default=None,
                    help="Explicit layers to use for cosine matrix (overrides union of early+late)")

    ap.add_argument("--early_ref", type=str, default="mean",
                    choices=["mean", "first", "median"],
                    help="How to define the early reference direction from EARLY layers")

    ap.add_argument("--write_npz", action="store_true", default=WRITE_NPZ,
                    help="Write NPZ files for parallel/orth components at late layers")
    args = ap.parse_args()

    base_name = args.base_name
    metric = args.metric
    method = args.method

    if args.directions_npz:
        npz_path = Path(args.directions_npz)
    else:
        npz_path = OUTPUT_DIR / f"{base_name}_mc_{metric}_directions.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Directions NPZ not found: {npz_path}")

    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        out_prefix = OUTPUT_DIR / f"{base_name}_{metric}_{method}_mix"

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Load directions
    methods = _load_npz_directions(npz_path)
    if method not in methods:
        raise KeyError(f"Method {method!r} not found in {npz_path.name}. Found: {sorted(methods.keys())}")

    dirs_by_layer = methods[method]
    available_layers = sorted(dirs_by_layer.keys())

    early_layers = _parse_layers_spec(args.early)
    late_layers = _parse_layers_spec(args.late)

    missing_early = [l for l in early_layers if l not in dirs_by_layer]
    missing_late = [l for l in late_layers if l not in dirs_by_layer]
    if missing_early:
        print(f"WARNING: early layers missing from NPZ for {method}: {missing_early}")
    if missing_late:
        print(f"WARNING: late layers missing from NPZ for {method}: {missing_late}")

    early_layers = [l for l in early_layers if l in dirs_by_layer]
    late_layers = [l for l in late_layers if l in dirs_by_layer]

    if not early_layers:
        raise ValueError("No valid EARLY layers after filtering. Check --early and the NPZ contents.")
    if not late_layers:
        raise ValueError("No valid LATE layers after filtering. Check --late and the NPZ contents.")

    # Define early reference direction e
    early_vecs = np.stack([dirs_by_layer[l] for l in early_layers], axis=0)
    if args.early_ref == "mean":
        e = _l2_normalize(np.mean(early_vecs, axis=0))
        e_desc = f"mean({early_layers[0]}..{early_layers[-1]})"
    elif args.early_ref == "first":
        e = _l2_normalize(dirs_by_layer[early_layers[0]])
        e_desc = f"layer {early_layers[0]}"
    else:  # median
        med_layer = early_layers[len(early_layers) // 2]
        e = _l2_normalize(dirs_by_layer[med_layer])
        e_desc = f"layer {med_layer}"

    print("=" * 80)
    print("DIRECTION MIXTURE ANALYSIS")
    print("=" * 80)
    print(f"NPZ:        {npz_path}")
    print(f"Method:     {method}")
    print(f"Available layers: {available_layers[0]}..{available_layers[-1]} (n={len(available_layers)})")
    print(f"EARLY:      {early_layers}")
    print(f"LATE:       {late_layers}")
    print(f"Early ref:  {e_desc}")

    # ---------------------
    # TEST 1: cosine matrix
    # ---------------------
    if args.layers:
        mat_layers = _parse_layers_spec(args.layers)
    else:
        mat_layers = sorted(set(early_layers + late_layers))

    mat_layers = [l for l in mat_layers if l in dirs_by_layer]
    if len(mat_layers) < 2:
        raise ValueError("Need at least 2 layers to build cosine matrix.")

    mat = np.zeros((len(mat_layers), len(mat_layers)), dtype=np.float32)
    for i, li in enumerate(mat_layers):
        vi = dirs_by_layer[li]
        for j, lj in enumerate(mat_layers):
            vj = dirs_by_layer[lj]
            mat[i, j] = np.dot(vi, vj)  # already normalized

    # Save CSV
    csv_path = Path(str(out_prefix) + "_cosines.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(str(l) for l in mat_layers) + "\n")
        for i, li in enumerate(mat_layers):
            f.write(str(li) + "," + ",".join(f"{mat[i, j]:.6f}" for j in range(len(mat_layers))) + "\n")

    # Save heatmap
    png_path = Path(str(out_prefix) + "_cosines.png")
    _save_heatmap(mat, [str(l) for l in mat_layers], png_path,
                  title=f"Cosine similarity: {base_name} {metric} {method}")

    # Print a compact summary
    print("\n[TEST 1] Cross-layer cosine similarity summary")
    def _mean_cos(A: List[int], B: List[int]) -> float:
        vals = []
        for la in A:
            for lb in B:
                vals.append(float(np.dot(dirs_by_layer[la], dirs_by_layer[lb])))
        return float(np.mean(vals)) if vals else float("nan")

    ee = _mean_cos(early_layers, early_layers)
    ll = _mean_cos(late_layers, late_layers)
    el = _mean_cos(early_layers, late_layers)
    # ee includes self-similarity; that's fine, it just anchors scale. Report also off-diagonal.
    def _mean_offdiag(layers: List[int]) -> float:
        vals = []
        for i, la in enumerate(layers):
            for j, lb in enumerate(layers):
                if i >= j:
                    continue
                vals.append(float(np.dot(dirs_by_layer[la], dirs_by_layer[lb])))
        return float(np.mean(vals)) if vals else float("nan")

    print(f"  mean cos(early,early): {ee:.4f}   (offdiag { _mean_offdiag(early_layers):.4f})")
    print(f"  mean cos(late,late):   {ll:.4f}   (offdiag { _mean_offdiag(late_layers):.4f})")
    print(f"  mean cos(early,late):  {el:.4f}")
    print(f"  wrote: {csv_path}")
    print(f"  wrote: {png_path}")

    if REPORT_SIM_TO_EARLY_REF:
        print("\n  cosine(layer, early_ref):")
        for l in mat_layers:
            print(f"    L {l:>3}: {float(np.dot(dirs_by_layer[l], e)):+.4f}")

    # -----------------------------------
    # TEST 2: decompose late into e// + e⊥
    # -----------------------------------
    print("\n[TEST 2] Late-layer decomposition into early-parallel vs orth")
    rows: List[DecompRow] = []
    for l in late_layers:
        d = dirs_by_layer[l]
        c = float(np.dot(d, e))
        parallel = c * e
        orth = d - parallel
        pnorm = float(np.linalg.norm(parallel))
        onorm = float(np.linalg.norm(orth))
        # Since d is unit, pnorm^2 + onorm^2 ~= 1
        pfrac = pnorm * pnorm
        ofrac = onorm * onorm
        rows.append(DecompRow(
            layer=l,
            cos_to_early_ref=c,
            parallel_coeff=c,
            parallel_energy_frac=pfrac,
            orth_energy_frac=ofrac,
            parallel_norm=pnorm,
            orth_norm=onorm,
        ))
        print(f"  L {l:>3}: cos(d,e)={c:+.4f}  parallel_energy={pfrac:.3f}  orth_energy={ofrac:.3f}  |orth|={onorm:.4f}")

    decomp_csv = Path(str(out_prefix) + "_late_decomp.csv")
    with open(decomp_csv, "w", encoding="utf-8") as f:
        f.write("layer,cos_to_early_ref,parallel_coeff,parallel_energy_frac,orth_energy_frac,parallel_norm,orth_norm\n")
        for r in rows:
            f.write(
                f"{r.layer},{r.cos_to_early_ref:.8f},{r.parallel_coeff:.8f},{r.parallel_energy_frac:.8f},"
                f"{r.orth_energy_frac:.8f},{r.parallel_norm:.8f},{r.orth_norm:.8f}\n"
            )
    print(f"  wrote: {decomp_csv}")

    if args.write_npz:
        # Write per-layer parallel/orth NPZs with the same key naming that run_ablation_causality expects.
        out_dir = out_prefix.parent
        print("\n  Writing NPZ component directions (one file per late layer):")
        for l in late_layers:
            d = dirs_by_layer[l]
            c = float(np.dot(d, e))
            parallel = c * e
            orth = d - parallel
            # Normalize components (ablation expects unit direction; magnitude is controlled separately by your alpha)
            p_unit = _l2_normalize(parallel)
            o_unit = _l2_normalize(orth)
            if float(np.linalg.norm(parallel)) == 0.0:
                print(f"    WARNING: L {l}: parallel component is zero; skipping parallel NPZ")
            else:
                p_path = out_dir / f"{base_name}_mc_{metric}_directions_TEST2_parallel_L{l}.npz"
                np.savez(p_path, **{f"{method}_layer_{l}": p_unit.astype(np.float32)})
                print(f"    wrote: {p_path}")
            if float(np.linalg.norm(orth)) < 1e-6:
                print(f"    WARNING: L {l}: orth component ~0; skipping orth NPZ")
            else:
                o_path = out_dir / f"{base_name}_mc_{metric}_directions_TEST2_orth_L{l}.npz"
                np.savez(o_path, **{f"{method}_layer_{l}": o_unit.astype(np.float32)})
                print(f"    wrote: {o_path}")

        print("\n  How to use these in your ablation script:")
        print("    1) Back up your original directions file:")
        print(f"         cp {OUTPUT_DIR}/{base_name}_mc_{metric}_directions.npz {OUTPUT_DIR}/{base_name}_mc_{metric}_directions.ORIG.npz")
        print("    2) Replace it with a component file (example for layer 38 parallel):")
        print(f"         cp {OUTPUT_DIR}/{base_name}_mc_{metric}_directions_TEST2_parallel_L38.npz {OUTPUT_DIR}/{base_name}_mc_{metric}_directions.npz")
        print("    3) In run_ablation_causality_bootstrap_ci_logitmargin.py set:")
        print("         LAYERS = [38]")
        print(f"         METHODS = ['{method}']")
        print("       then run ablation.")
        print("    4) Restore original:")
        print(f"         cp {OUTPUT_DIR}/{base_name}_mc_{metric}_directions.ORIG.npz {OUTPUT_DIR}/{base_name}_mc_{metric}_directions.npz")


if __name__ == "__main__":
    main()
