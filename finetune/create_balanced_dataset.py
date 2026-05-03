"""
Create a balanced finetune dataset from TriviaMC, PopMC, and SimpleMC.

Sampling hierarchy:
  1. entropy_bin (low / med / high) — 500 samples each, 1500 total
  2. source — fixed per-bin quotas: TriviaMC=300, PopMC=146, SimpleMC=54
  3. entropy sub-bins — within each (source, bin) cell, divide the bin's
     entropy range into N_SUB_BINS equal-width sub-bands and allocate the
     quota evenly across sub-bands (with overflow to denser sub-bands when
     a sparse one is exhausted). This prevents the "low" bin from collapsing
     to entropy ≈ 0 — it forces coverage of the upper-low region too.
  4. PopMC prop — proportional to availability per bin, capped so no single
     prop exceeds MAX_PROP_FRACTION of the PopMC quota. Sub-binning by
     entropy still applies inside each prop group.

Output format: every field from the source eval JSONL row (type, is_correct,
top_prob, probs_ABCD, model_answer, model_answer_position,
correct_answer_position, prob_gap, expected_confidence,
predicted_confidence_bin_index, predicted_other_confidence_bin_index,
expected_other_confidence, loss, entropy, etc.) plus distractors from the
underlying dataset and source/[prop, s_pop, o_pop]. Keeping the eval-row
fields lets analyze_performance.ipynb plot the balanced split with the
same tools it uses on raw evaluation files.
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

# Fixed per-bin source quotas (must sum to 500)
SOURCE_QUOTAS = {"TriviaMC": 300, "PopMC": 146, "SimpleMC": 54}
assert sum(SOURCE_QUOTAS.values()) == 500

# No single PopMC prop may exceed this fraction of the PopMC per-bin quota
MAX_PROP_FRACTION = 0.25

# Within each (source, bin) cell, divide the bin's entropy range into this
# many equal-width sub-bands and try to sample evenly across them. Forces
# coverage of the upper-low region (entropy ~0.25-0.46) instead of letting
# the entropy~0 spike monopolize the low bin.
N_SUB_BINS = 3

EVAL_FILES = {
    "TriviaMC": "outputs/evaluations/2026-05-03-18-00-04_meta-llama-Llama-3.1-8B-Instruct_instruct_TriviaMC_n2416.jsonl",
    "PopMC":    "outputs/evaluations/2026-05-03-17-49-00_meta-llama-Llama-3.1-8B-Instruct_instruct_PopMC_n14267.jsonl",
    "SimpleMC": "outputs/evaluations/2026-05-03-17-59-22_meta-llama-Llama-3.1-8B-Instruct_instruct_SimpleMC_n500.jsonl",
}
DATA_FILES = {
    "TriviaMC": "data/TriviaMC.jsonl",
    "PopMC":    "data/PopMC.jsonl",
    "SimpleMC": "data/SimpleMC.jsonl",
}


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


def _load_options_by_qid(raw_data_path: str) -> dict[str, dict]:
    """Run the canonical training loader on the raw data so the option order
    matches what the eval pass saw. Returns {qid: {options dict, correct_letter}}.

    This is what makes the balanced rows usable for frozen training: by
    freezing options now, the recorded ``model_answer`` letter and
    ``probs_ABCD`` indices stay aligned with what the finetuning model will
    later see in its prompt.
    """
    from data_handling import load_jsonl_dataset
    out = {}
    for row in load_jsonl_dataset(raw_data_path):
        qid = row.get("qid")
        if qid is not None:
            out[qid] = {
                "options": row["options"],
                "correct_letter": row["correct_letter"],
            }
    return out


def build_cells(root: Path) -> dict[str, dict[str, list[dict]]]:
    """cells[source][bin] = list of enriched records."""
    cells: dict[str, dict[str, list]] = {
        ds: {"low": [], "med": [], "high": []} for ds in EVAL_FILES
    }
    for ds in EVAL_FILES:
        eval_map = load_eval_rows(str(root / EVAL_FILES[ds]))
        data_map = load_data_file(str(root / DATA_FILES[ds]))
        opts_by_qid = _load_options_by_qid(str(root / DATA_FILES[ds]))
        for qid, eval_row in eval_map.items():
            row = data_map.get(qid)
            if row is None:
                continue
            if len(row.get("distractors", [])) < 3:
                continue
            opts_info = opts_by_qid.get(qid)
            if opts_info is None:
                # Couldn't reconstruct options for this qid — skip rather than
                # emit a row that can't be replayed at training time.
                continue
            # Sanity check: the eval row's recorded correct_answer letter must
            # match the letter we get from the deterministic shuffle. If it
            # doesn't, something has drifted and the row is unsafe to use.
            if eval_row.get("correct_answer") != opts_info["correct_letter"]:
                continue
            # Start from the full eval row (keeps type, is_correct, top_prob,
            # probs_ABCD, expected_confidence, model_answer_position, etc. so
            # downstream analysis can plot the balanced split with the same
            # tools used on raw eval files), then merge in EVERY field from
            # the raw data row so no source-side metadata is lost. Two fields
            # mean different things in each source and need explicit handling:
            #   - correct_answer: eval row stores the LETTER (A-D), raw stores
            #     the TEXT. Keep the letter under correct_answer and surface
            #     the text as correct_answer_text.
            #   - distractors: take the first 3 from the raw row (raw is the
            #     authoritative source).
            # All other raw-row fields (prop, s_pop, o_pop, and anything
            # source-specific that may exist now or be added later) are
            # copied verbatim if they don't already exist on the eval row.
            record = dict(eval_row)
            for key, value in row.items():
                if key == "correct_answer":
                    record["correct_answer_text"] = value
                elif key == "distractors":
                    record["distractors"] = value[:3]
                elif key not in record:
                    record[key] = value
            record["options"] = opts_info["options"]
            record["correct_letter"] = opts_info["correct_letter"]
            record["source"] = ds
            record["entropy"] = round(eval_row["entropy"], 6)
            cells[ds][entropy_bin(eval_row["entropy"])].append(record)
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
    cells: dict[str, dict[str, list[dict]]],
    bin_name: str,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Sample one entropy bin using fixed source quotas + sub-bin coverage."""
    result = []
    log: dict[str, dict] = {}
    bin_low, bin_high = BIN_RANGES[bin_name]

    for source, quota in SOURCE_QUOTAS.items():
        pool = cells[source][bin_name]
        effective = min(quota, len(pool))
        if effective < quota:
            print(f"  WARNING: {source} {bin_name}: requested {quota}, only {len(pool)} available")

        if source == "PopMC":
            chosen = sample_popmc_bin(pool, effective, bin_low, bin_high, rng)
        else:
            chosen = systematic_sample_within_bin(pool, effective, bin_low, bin_high, rng)

        result.extend(chosen)
        log[source] = {"requested": quota, "sampled": len(chosen)}

        # Sub-bin coverage breakdown (for any source)
        sub_buckets = _split_into_subbins(chosen, bin_low, bin_high, N_SUB_BINS)
        log[source]["subbins"] = [len(b) for b in sub_buckets]

        if source == "PopMC":
            prop_counts: dict[str, int] = defaultdict(int)
            for r in chosen:
                prop_counts[r.get("prop", "unknown")] += 1
            log[source]["props"] = dict(sorted(prop_counts.items(), key=lambda x: -x[1]))

    return result, log


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def split_and_write(
    records: list[dict],
    output_dir: Path,
    prefix: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed + 1)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    splits = {
        "train": shuffled[:train_end],
        "val":   shuffled[train_end:val_end],
        "test":  shuffled[val_end:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        path = output_dir / f"{prefix}_{split_name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {split_name:5s}: {len(rows):>5d} → {path}")

    combined = output_dir / f"{prefix}.jsonl"
    with open(combined, "w") as f:
        for r in shuffled:
            f.write(json.dumps(r) + "\n")
    print(f"  {'all':5s}: {n:>5d} → {combined}")


def print_summary(records: list[dict]) -> None:
    from collections import Counter
    bin_counts:  Counter = Counter(entropy_bin(r["entropy"]) for r in records)
    src_counts:  Counter = Counter(r["source"] for r in records)
    cell_counts: Counter = Counter((r["source"], entropy_bin(r["entropy"])) for r in records)

    print(f"\nFinal dataset: {len(records)} samples")
    print(f"\nBy entropy bin:  low={bin_counts['low']}, med={bin_counts['med']}, high={bin_counts['high']}")
    print(f"By source:       " + ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())))

    print(f"\n{'':12s}  {'low':>5s}  {'med':>5s}  {'high':>5s}  {'total':>5s}")
    for ds in ("TriviaMC", "PopMC", "SimpleMC"):
        lo = cell_counts.get((ds, "low"), 0)
        me = cell_counts.get((ds, "med"), 0)
        hi = cell_counts.get((ds, "high"), 0)
        print(f"  {ds:10s}  {lo:>5d}  {me:>5d}  {hi:>5d}  {lo+me+hi:>5d}")
    print(f"  {'total':10s}  {bin_counts['low']:>5d}  {bin_counts['med']:>5d}  {bin_counts['high']:>5d}  {len(records):>5d}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",        default=".",    help="Project root (default: cwd)")
    parser.add_argument("--output-dir",  default="data", help="Output directory")
    parser.add_argument("--prefix",      default="balanced_metacognition")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rng  = random.Random(args.seed)

    print("Building entropy-binned cells...")
    cells = build_cells(root)

    print(f"\nAvailable per cell:")
    print(f"\n{'':12s}  {'low':>6s}  {'med':>6s}  {'high':>6s}  {'total':>6s}")
    for ds in ("TriviaMC", "PopMC", "SimpleMC"):
        lo, me, hi = len(cells[ds]["low"]), len(cells[ds]["med"]), len(cells[ds]["high"])
        print(f"  {ds:10s}  {lo:>6d}  {me:>6d}  {hi:>6d}  {lo+me+hi:>6d}")

    print(f"\nTarget quotas per bin: {SOURCE_QUOTAS}")
    print(f"PopMC max prop fraction: {MAX_PROP_FRACTION:.0%}")
    print(f"Sub-bins per (source, bin) cell: {N_SUB_BINS}\n")

    all_records = []
    for bin_name in ("low", "med", "high"):
        bin_low, bin_high = BIN_RANGES[bin_name]
        print(f"--- {bin_name} bin  (entropy [{bin_low:.3f}, {bin_high:.3f})) ---")
        chosen, log = sample_bin(cells, bin_name, rng)
        all_records.extend(chosen)
        for source, info in log.items():
            sub = info.get("subbins", [])
            sub_str = "/".join(str(x) for x in sub)
            print(f"  {source:10s}: {info['sampled']:>3d} sampled   sub-bins: {sub_str}", end="")
            if "props" in info:
                top = list(info["props"].items())[:5]
                print(f"   top props: {top}", end="")
            print()

    print_summary(all_records)

    print(f"\nWriting splits (train={args.train_ratio:.0%} / val={args.val_ratio:.0%} / test={1-args.train_ratio-args.val_ratio:.0%}):")
    split_and_write(all_records, root / args.output_dir, args.prefix, args.train_ratio, args.val_ratio, args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
