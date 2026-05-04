"""
Create a balanced finetune dataset from PopMC alone (max-size variant).

Sampling hierarchy:
  1. entropy_bin (low / med / high) — auto-sized to the smallest bin so each
     bin holds the same number of samples and the dataset is as large as
     possible while staying perfectly balanced. Override with --per-bin.
  2. PopMC prop — proportional to availability per bin, capped so no single
     prop exceeds MAX_PROP_FRACTION of the per-bin quota. Sub-binning by
     entropy still applies inside each prop group.
  3. entropy sub-bins — within each (prop, bin) cell, divide the bin's
     entropy range into N_SUB_BINS equal-width sub-bands and allocate the
     quota evenly across sub-bands (with overflow to denser sub-bands when
     a sparse one is exhausted). This prevents the "low" bin from
     collapsing to entropy ≈ 0.

Output format: every field from the source eval JSONL row (type, is_correct,
top_prob, probs_ABCD, model_answer, model_answer_position,
correct_answer_position, prob_gap, expected_confidence,
predicted_confidence_bin_index, predicted_other_confidence_bin_index,
expected_other_confidence, loss, entropy, etc.) plus distractors and
prop/s_pop/o_pop from the underlying dataset, and source="PopMC". Mirrors
finetune/create_balanced_dataset.py so analyze_performance.ipynb can plot
this split with the same tooling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
from collections import defaultdict

LN4 = math.log(4)
LOW_THRESHOLD  = LN4 / 3      # ~0.462
HIGH_THRESHOLD = 2 * LN4 / 3  # ~0.924
BIN_RANGES = {
    "low":  (0.0,            LOW_THRESHOLD),
    "med":  (LOW_THRESHOLD,  HIGH_THRESHOLD),
    "high": (HIGH_THRESHOLD, LN4 + 1e-9),
}

# No single PopMC prop may exceed this fraction of the per-bin quota.
MAX_PROP_FRACTION = 0.25

# Within each (prop, bin) cell, divide the bin's entropy range into this
# many equal-width sub-bands and try to sample evenly across them. Forces
# coverage of the upper-low region (entropy ~0.25-0.46) instead of letting
# the entropy~0 spike monopolize the low bin.
N_SUB_BINS = 3

EVAL_FILE = "outputs/evaluations/8b_instruct/2026-05-03-19-54-00_meta-llama-Llama-3.1-8B-Instruct_instruct_PopMC_n14267.jsonl"
DATA_FILE = "data/PopMC.jsonl"
SOURCE = "PopMC"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def entropy_bin(e: float) -> str:
    if e < LOW_THRESHOLD:
        return "low"
    if e < HIGH_THRESHOLD:
        return "med"
    return "high"


def load_eval_rows(path: str) -> dict[str, dict]:
    """Return qid -> full eval-sample row, dropping rows with missing/NaN entropy."""
    out: dict[str, dict] = {}
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "entropy" not in d or "qid" not in d:
                continue
            # Eval files mix sample and summary rows; only sample rows have qid.
            if not str(d.get("type", "")).endswith("eval_sample"):
                continue
            e = d["entropy"]
            # Skip NaN / non-finite — a few rows in the eval files have these.
            if not isinstance(e, (int, float)) or e != e or e in (float("inf"), float("-inf")):
                skipped += 1
                continue
            d["entropy"] = float(e)
            out[d["qid"]] = d
    if skipped:
        print(f"  (skipped {skipped} rows with missing/NaN entropy in {Path(path).name})")
    return out


def load_data_file(path: str) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = d.get("qid")
            if qid:
                out[qid] = d
    return out


# Eval-row fields the source-of-truth that must NOT be overwritten by
# raw-data merge. These are measured against the exact option order the
# model saw at eval time; clobbering any one of them would re-introduce
# the off-by-permutation bug.
_EVAL_PROTECTED_FIELDS = frozenset({
    "options", "correct_letter", "correct_answer", "correct_answer_text",
    "model_answer", "model_answer_text", "model_answer_position",
    "correct_answer_position", "probs_ABCD", "entropy",
    "is_correct", "top_prob", "second_prob", "prob_gap",
    "predicted_confidence_letter", "predicted_confidence_bin_index",
    "predicted_other_confidence_letter", "predicted_other_confidence_bin_index",
    "expected_confidence", "expected_other_confidence", "loss",
})


def build_cells(root: Path) -> dict[str, list[dict]]:
    """cells[bin] = list of enriched PopMC records.

    Strategy: the eval JSONL row is the source of truth for everything
    measured against the option order the model saw at eval time
    (options, correct_letter, model_answer, probs_ABCD, entropy, …). We
    take that record verbatim, then merge in raw-dataset metadata
    (distractors, prop/s_pop/o_pop, …) for fields the eval JSONL doesn't
    carry. Raw-data fields NEVER overwrite eval-row fields.
    """
    cells: dict[str, list[dict]] = {"low": [], "med": [], "high": []}
    eval_map = load_eval_rows(str(root / EVAL_FILE))
    data_map = load_data_file(str(root / DATA_FILE))
    skipped_legacy = 0
    for qid, eval_row in eval_map.items():
        row = data_map.get(qid)
        if row is None:
            continue
        if len(row.get("distractors", [])) < 3:
            continue
        # Hard requirement: eval row must record the exact options it
        # showed the model. Older eval files (pre-options-recording)
        # are unsafe for frozen training and are skipped with a warning.
        options = eval_row.get("options")
        correct_letter = eval_row.get("correct_letter")
        if not (
            isinstance(options, dict)
            and set(options.keys()) == set("ABCD")
            and correct_letter in {"A", "B", "C", "D"}
        ):
            skipped_legacy += 1
            continue
        # Merge raw-dataset fields the eval JSONL doesn't carry, but
        # never let them overwrite the eval-side source of truth.
        record = dict(eval_row)
        for key, value in row.items():
            if key in _EVAL_PROTECTED_FIELDS:
                continue
            if key == "correct_answer":
                # raw correct_answer is the TEXT; surface as
                # correct_answer_text only if eval didn't already.
                record.setdefault("correct_answer_text", value)
            elif key == "distractors":
                record.setdefault("distractors", list(value)[:3])
            elif key not in record:
                record[key] = value
        record["source"] = SOURCE
        record["entropy"] = round(eval_row["entropy"], 6)
        cells[entropy_bin(eval_row["entropy"])].append(record)
    if skipped_legacy:
        total = len(eval_map)
        pct = 100.0 * skipped_legacy / max(total, 1)
        print(
            f"  ⚠️  {SOURCE}: dropped {skipped_legacy}/{total} eval rows "
            f"({pct:.1f}%) that don't record their displayed options "
            f"(legacy eval file). Rerun run_evaluations.py and update EVAL_FILE."
        )
    return cells


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def _split_into_subbins(records: list[dict], bin_low: float, bin_high: float, n_subbins: int) -> list[list[dict]]:
    """Group records by equal-width entropy sub-band within [bin_low, bin_high)."""
    width = (bin_high - bin_low) / n_subbins
    buckets: list[list[dict]] = [[] for _ in range(n_subbins)]
    for r in records:
        e = r["entropy"]
        idx = int((e - bin_low) / width)
        if idx < 0:
            idx = 0
        elif idx >= n_subbins:
            idx = n_subbins - 1
        buckets[idx].append(r)
    return buckets


def _allocate_evenly(sizes: list[int], total: int) -> list[int]:
    """
    Split `total` evenly across len(sizes) buckets, capped by each bucket's
    availability. Deficit from small buckets rolls over to the larger ones.
    """
    n = len(sizes)
    quotas = [0] * n
    remaining = total
    free = list(range(n))
    while free and remaining > 0:
        share = remaining // len(free)
        if share == 0:
            # Distribute the last few one-by-one to buckets with capacity
            for i in sorted(free, key=lambda i: -sizes[i]):
                if remaining == 0:
                    break
                if quotas[i] < sizes[i]:
                    quotas[i] += 1
                    remaining -= 1
            break
        new_free = []
        for i in free:
            take = min(share, sizes[i] - quotas[i])
            quotas[i] += take
            remaining -= take
            if quotas[i] < sizes[i]:
                new_free.append(i)
        if len(new_free) == len(free):
            # No bucket capped this round — done
            break
        free = new_free
    return quotas


def systematic_sample_within_bin(
    records: list[dict],
    n: int,
    bin_low: float,
    bin_high: float,
    rng: random.Random,
) -> list[dict]:
    """
    Sample `n` records evenly across the entropy range [bin_low, bin_high).
    Splits the range into N_SUB_BINS equal-width bands; allocates n across
    the bands (capped by availability, deficit redistributed); within each
    band, samples randomly. This forces coverage of the upper-low region
    even when most data is concentrated near zero.
    """
    if n <= 0:
        return []
    if n >= len(records):
        return list(records)

    buckets = _split_into_subbins(records, bin_low, bin_high, N_SUB_BINS)
    sizes = [len(b) for b in buckets]
    quotas = _allocate_evenly(sizes, n)

    result: list[dict] = []
    for bucket, q in zip(buckets, quotas):
        if q <= 0:
            continue
        result.extend(rng.sample(bucket, q))
    return result


def allocate_quotas(counts: dict[str, int], total: int, max_frac: float) -> dict[str, int]:
    """
    Proportionally allocate `total` across groups given their counts,
    capping each group at min(max_frac*total, avail). Iterates until no
    group exceeds its limit, then uses largest-remainder for final rounding.
    """
    cap = max(1, int(total * max_frac))
    quotas: dict[str, int] = {}
    remaining = total
    free = set(counts.keys())

    # Each iteration: find groups that would exceed their limit at current proportions
    # and fix them at their limit; redistribute remaining budget to the others.
    for _ in range(len(counts) + 1):
        if not free or remaining <= 0:
            break
        total_free = sum(counts[g] for g in free)
        if total_free == 0:
            break

        # Which groups are constrained at this iteration?
        constrained = {
            g for g in free
            if remaining * counts[g] / total_free >= min(cap, counts[g])
        }
        if not constrained:
            # No group exceeds its limit — distribute proportionally with largest-remainder
            fair  = {g: remaining * counts[g] / total_free for g in free}
            floor = {g: int(f) for g, f in fair.items()}
            leftover = remaining - sum(floor.values())
            by_frac = sorted(free, key=lambda g: -(fair[g] - floor[g]))
            for i, g in enumerate(by_frac):
                quotas[g] = floor[g] + (1 if i < leftover else 0)
            remaining = 0
            break

        for g in constrained:
            limit = min(cap, counts[g])
            quotas[g] = limit
            remaining -= limit
            free.remove(g)

    return quotas


def sample_popmc_bin(
    records: list[dict],
    target: int,
    bin_low: float,
    bin_high: float,
    rng: random.Random,
) -> list[dict]:
    """
    Sample `target` records from a PopMC entropy bin.
    Stratifies by prop (proportional + capped); within each prop, samples
    evenly across the entropy range using sub-binning.
    """
    by_prop: dict[str, list] = defaultdict(list)
    for r in records:
        by_prop[r.get("prop", "unknown")].append(r)

    prop_counts = {p: len(recs) for p, recs in by_prop.items()}
    quotas = allocate_quotas(prop_counts, target, MAX_PROP_FRACTION)

    result = []
    for prop, q in quotas.items():
        if q <= 0:
            continue
        chosen = systematic_sample_within_bin(by_prop[prop], q, bin_low, bin_high, rng)
        result.extend(chosen)

    # If we're short (rounding), fill from remaining records using sub-bin
    # sampling on the leftover pool so coverage stays uniform.
    if len(result) < target:
        taken_qids = {r["qid"] for r in result}
        pool = [r for r in records if r["qid"] not in taken_qids]
        deficit = target - len(result)
        result.extend(systematic_sample_within_bin(pool, deficit, bin_low, bin_high, rng))

    return result


def sample_bin(
    cells: dict[str, list[dict]],
    bin_name: str,
    quota: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Sample one entropy bin using prop stratification + sub-bin coverage."""
    log: dict = {}
    bin_low, bin_high = BIN_RANGES[bin_name]
    pool = cells[bin_name]
    effective = min(quota, len(pool))
    if effective < quota:
        print(f"  WARNING: {bin_name}: requested {quota}, only {len(pool)} available")

    chosen = sample_popmc_bin(pool, effective, bin_low, bin_high, rng)
    log["requested"] = quota
    log["sampled"]   = len(chosen)

    sub_buckets = _split_into_subbins(chosen, bin_low, bin_high, N_SUB_BINS)
    log["subbins"] = [len(b) for b in sub_buckets]

    prop_counts: dict[str, int] = defaultdict(int)
    for r in chosen:
        prop_counts[r.get("prop", "unknown")] += 1
    log["props"] = dict(sorted(prop_counts.items(), key=lambda x: -x[1]))

    return chosen, log


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _stratified_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    """80/10/10 split that preserves entropy_bin composition.

    Within each entropy_bin: shuffle, then deal samples in a 8:1:1
    round-robin so even small cells split predictably. Without this,
    the random global shuffle can hand val a noticeably easier or
    harder mix than train.
    """
    rng = random.Random(seed + 1)
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cell[entropy_bin(r["entropy"])].append(r)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    test_ratio = 1.0 - train_ratio - val_ratio
    # Round-robin proportional dealing: build a length-10 pattern and walk it.
    pattern = (
        ["train"] * round(train_ratio * 10)
        + ["val"]   * round(val_ratio   * 10)
        + ["test"]  * round(test_ratio  * 10)
    )
    while len(pattern) < 10:
        pattern.append("train")
    pattern = pattern[:10]

    for cell, rows in by_cell.items():
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            splits[pattern[i % 10]].append(r)

    # Final shuffle within each split so rows aren't grouped by cell on disk.
    for split in splits.values():
        rng.shuffle(split)
    return splits


def split_and_write(
    records: list[dict],
    output_dir: Path,
    prefix: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    splits = _stratified_split(records, train_ratio, val_ratio, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        path = output_dir / f"{prefix}_{split_name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {split_name:5s}: {len(rows):>5d} → {path}")

    combined = output_dir / f"{prefix}.jsonl"
    all_rows = splits["train"] + splits["val"] + splits["test"]
    with open(combined, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {'all':5s}: {len(all_rows):>5d} → {combined}")

    # Per-split entropy_bin breakdown so drift between splits is obvious.
    print("\nSplit composition (entropy_bin):")
    cells = ("low", "med", "high")
    for split_name, rows in splits.items():
        from collections import Counter
        counts = Counter(entropy_bin(r["entropy"]) for r in rows)
        line = f"  {split_name:5s}  "
        for c in cells:
            line += f"  {c}={counts.get(c, 0):>5d}"
        print(line)


def audit_dataset(records: list[dict], splits: dict[str, list[dict]] | None = None) -> None:
    """Hard-checks every emitted row carries the contract its consumers rely on.

    Prints any violations and raises if a critical contract is broken so
    bad data never silently makes it into a training run.
    """
    problems: list[str] = []
    qid_to_split: dict[str, str] = {}

    def _check(row: dict, where: str) -> None:
        opts = row.get("options")
        if not (isinstance(opts, dict) and set(opts.keys()) == set("ABCD")):
            problems.append(f"{where} {row.get('qid')}: bad options dict")
            return
        cl = row.get("correct_letter")
        if cl not in {"A", "B", "C", "D"}:
            problems.append(f"{where} {row.get('qid')}: correct_letter={cl!r}")
        if "correct_answer_text" in row and row["correct_answer_text"] != opts.get(cl):
            problems.append(
                f"{where} {row.get('qid')}: correct_answer_text != options[correct_letter]"
            )
        ma = row.get("model_answer")
        if ma is not None and ma in opts:
            mat = row.get("model_answer_text")
            if mat is not None and mat != opts[ma]:
                problems.append(
                    f"{where} {row.get('qid')}: model_answer_text != options[model_answer]"
                )
        probs = row.get("probs_ABCD")
        if probs is not None and len(probs) != 4:
            problems.append(f"{where} {row.get('qid')}: probs_ABCD len={len(probs)}")

    for r in records:
        _check(r, "all")

    if splits is not None:
        for name, rows in splits.items():
            for r in rows:
                qid = r.get("qid")
                if qid in qid_to_split and qid_to_split[qid] != name:
                    problems.append(f"qid {qid} in both {qid_to_split[qid]} and {name}")
                else:
                    qid_to_split[qid] = name

    if problems:
        print("\n⚠️  AUDIT FAILURES (showing up to 10):")
        for p in problems[:10]:
            print(f"  {p}")
        print(f"  ({len(problems)} total)")
        raise SystemExit(1)
    print("\n✓ Audit passed — every row has options/correct_letter/probs_ABCD "
          "and no qid leaks across splits.")


def print_summary(records: list[dict]) -> None:
    from collections import Counter
    bin_counts: Counter = Counter(entropy_bin(r["entropy"]) for r in records)
    prop_counts: Counter = Counter(r.get("prop", "unknown") for r in records)

    print(f"\nFinal dataset: {len(records)} samples")
    print(f"\nBy entropy bin:  low={bin_counts['low']}, med={bin_counts['med']}, high={bin_counts['high']}")
    print(f"\nBy prop (top 15):")
    for p, c in prop_counts.most_common(15):
        print(f"  {p:20s}  {c:>5d}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",        default=".",    help="Project root (default: cwd)")
    parser.add_argument("--output-dir",  default="data", help="Output directory")
    parser.add_argument("--prefix",      default="balanced_popMC")
    parser.add_argument("--per-bin",     type=int, default=None,
                        help="Samples per entropy bin (default: min available across bins)")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rng  = random.Random(args.seed)

    print("Building entropy-binned cells...")
    cells = build_cells(root)

    print(f"\nAvailable per cell:")
    print(f"  {'low':>6s}  {'med':>6s}  {'high':>6s}  {'total':>6s}")
    lo, me, hi = len(cells["low"]), len(cells["med"]), len(cells["high"])
    print(f"  {lo:>6d}  {me:>6d}  {hi:>6d}  {lo+me+hi:>6d}")

    per_bin = args.per_bin if args.per_bin is not None else min(lo, me, hi)
    print(f"\nPer-bin quota: {per_bin}  (max balanced size = {per_bin * 3})")
    print(f"PopMC max prop fraction: {MAX_PROP_FRACTION:.0%}")
    print(f"Sub-bins per (prop, bin) cell: {N_SUB_BINS}\n")

    all_records = []
    for bin_name in ("low", "med", "high"):
        bin_low, bin_high = BIN_RANGES[bin_name]
        print(f"--- {bin_name} bin  (entropy [{bin_low:.3f}, {bin_high:.3f})) ---")
        chosen, log = sample_bin(cells, bin_name, per_bin, rng)
        all_records.extend(chosen)
        sub = log.get("subbins", [])
        sub_str = "/".join(str(x) for x in sub)
        print(f"  {log['sampled']:>5d} sampled   sub-bins: {sub_str}")
        top = list(log["props"].items())[:5]
        print(f"  top props: {top}")

    print_summary(all_records)

    print(f"\nWriting splits (train={args.train_ratio:.0%} / val={args.val_ratio:.0%} / test={1-args.train_ratio-args.val_ratio:.0%}):")
    split_and_write(all_records, root / args.output_dir, args.prefix, args.train_ratio, args.val_ratio, args.seed)

    # Re-read what we just wrote and audit it as a final tripwire.
    splits = {}
    for split_name in ("train", "val", "test"):
        path = root / args.output_dir / f"{args.prefix}_{split_name}.jsonl"
        with open(path) as f:
            splits[split_name] = [json.loads(line) for line in f]
    audit_dataset(all_records, splits)
    print("\nDone.")


if __name__ == "__main__":
    main()
