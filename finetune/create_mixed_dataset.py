"""
Create a mixed finetune dataset from TriviaMC, PopMC, and SimpleMC where each
source contributes the same number of samples to every entropy bin (but the
sources may contribute different totals).

Sampling hierarchy:
  1. entropy_bin (low / med / high) — fixed quota per source per bin
  2. source — per-source per-bin quota set on the command line
  3. entropy sub-bins — within each (source, bin) cell, divide the bin's
     entropy range into N_SUB_BINS equal-width sub-bands and allocate the
     quota evenly across sub-bands (overflow to denser sub-bands when
     a sparse one is exhausted). Forces coverage of the upper-low region
     instead of letting the entropy ≈ 0 spike monopolize the low bin.
  4. PopMC prop — proportional to availability per bin, capped so no single
     prop exceeds MAX_PROP_FRACTION of the PopMC per-bin quota. Sub-binning
     by entropy still applies inside each prop group.

This is a parametrized variant of finetune/create_balanced_dataset.py. The
sampling code is identical — only SOURCE_QUOTAS becomes a flat per-source
quota that applies uniformly across low/med/high (so each source is itself
balanced across entropy bins, but sources need not be balanced against each
other).

Example invocations:
  # Mixed-2550-clean (TriviaMC=400, PopMC=400, SimpleMC=50 per bin)
  python finetune/create_mixed_dataset.py \
      --prefix mixed_2550_clean \
      --trivia-per-bin 400 --popmc-per-bin 400 --simple-per-bin 50

  # Mixed-3150-pop-heavy (TriviaMC=400, PopMC=600, SimpleMC=50 per bin)
  python finetune/create_mixed_dataset.py \
      --prefix mixed_3150_pop_heavy \
      --trivia-per-bin 400 --popmc-per-bin 600 --simple-per-bin 50
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

# No single PopMC prop may exceed this fraction of the PopMC per-bin quota
MAX_PROP_FRACTION = 0.25

# Within each (source, bin) cell, divide the bin's entropy range into this
# many equal-width sub-bands and try to sample evenly across them.
N_SUB_BINS = 3

EVAL_FILES = {
    "TriviaMC": "outputs/evaluations/8b_instruct/2026-05-03-20-04-26_meta-llama-Llama-3.1-8B-Instruct_instruct_TriviaMC_n2416.jsonl",
    "PopMC":    "outputs/evaluations/8b_instruct/2026-05-03-19-54-00_meta-llama-Llama-3.1-8B-Instruct_instruct_PopMC_n14267.jsonl",
    "SimpleMC": "outputs/evaluations/8b_instruct/2026-05-03-20-03-45_meta-llama-Llama-3.1-8B-Instruct_instruct_SimpleMC_n500.jsonl",
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
            if not str(d.get("type", "")).endswith("eval_sample"):
                continue
            e = d["entropy"]
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


_EVAL_PROTECTED_FIELDS = frozenset({
    "options", "correct_letter", "correct_answer", "correct_answer_text",
    "model_answer", "model_answer_text", "model_answer_position",
    "correct_answer_position", "probs_ABCD", "entropy",
    "is_correct", "top_prob", "second_prob", "prob_gap",
    "predicted_confidence_letter", "predicted_confidence_bin_index",
    "predicted_other_confidence_letter", "predicted_other_confidence_bin_index",
    "expected_confidence", "expected_other_confidence", "loss",
})


def build_cells(root: Path) -> dict[str, dict[str, list[dict]]]:
    """cells[source][bin] = list of enriched records.

    The eval JSONL row is the source of truth for everything measured against
    the option order the model saw at eval time. Raw-data fields fill in
    distractors / prop / s_pop / o_pop but never overwrite eval-row fields.
    """
    cells: dict[str, dict[str, list]] = {
        ds: {"low": [], "med": [], "high": []} for ds in EVAL_FILES
    }
    for ds in EVAL_FILES:
        eval_map = load_eval_rows(str(root / EVAL_FILES[ds]))
        data_map = load_data_file(str(root / DATA_FILES[ds]))
        skipped_legacy = 0
        for qid, eval_row in eval_map.items():
            row = data_map.get(qid)
            if row is None:
                continue
            if len(row.get("distractors", [])) < 3:
                continue
            options = eval_row.get("options")
            correct_letter = eval_row.get("correct_letter")
            if not (
                isinstance(options, dict)
                and set(options.keys()) == set("ABCD")
                and correct_letter in {"A", "B", "C", "D"}
            ):
                skipped_legacy += 1
                continue
            record = dict(eval_row)
            for key, value in row.items():
                if key in _EVAL_PROTECTED_FIELDS:
                    continue
                if key == "correct_answer":
                    record.setdefault("correct_answer_text", value)
                elif key == "distractors":
                    record.setdefault("distractors", list(value)[:3])
                elif key not in record:
                    record[key] = value
            record["source"] = ds
            record["entropy"] = round(eval_row["entropy"], 6)
            cells[ds][entropy_bin(eval_row["entropy"])].append(record)
        if skipped_legacy:
            total = len(eval_map)
            pct = 100.0 * skipped_legacy / max(total, 1)
            print(
                f"  ⚠️  {ds}: dropped {skipped_legacy}/{total} eval rows "
                f"({pct:.1f}%) that don't record their displayed options "
                f"(legacy eval file). Rerun run_evaluations.py on this "
                f"dataset and update EVAL_FILES."
            )
    return cells


# ---------------------------------------------------------------------------
# sampling  (identical to create_balanced_dataset.py)
# ---------------------------------------------------------------------------

def _split_into_subbins(records: list[dict], bin_low: float, bin_high: float, n_subbins: int) -> list[list[dict]]:
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
    n = len(sizes)
    quotas = [0] * n
    remaining = total
    free = list(range(n))
    while free and remaining > 0:
        share = remaining // len(free)
        if share == 0:
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
    cap = max(1, int(total * max_frac))
    quotas: dict[str, int] = {}
    remaining = total
    free = set(counts.keys())

    for _ in range(len(counts) + 1):
        if not free or remaining <= 0:
            break
        total_free = sum(counts[g] for g in free)
        if total_free == 0:
            break

        constrained = {
            g for g in free
            if remaining * counts[g] / total_free >= min(cap, counts[g])
        }
        if not constrained:
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

    if len(result) < target:
        taken_qids = {r["qid"] for r in result}
        pool = [r for r in records if r["qid"] not in taken_qids]
        deficit = target - len(result)
        result.extend(systematic_sample_within_bin(pool, deficit, bin_low, bin_high, rng))

    return result


def sample_bin(
    cells: dict[str, dict[str, list[dict]]],
    bin_name: str,
    source_quotas: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict], dict]:
    result = []
    log: dict[str, dict] = {}
    bin_low, bin_high = BIN_RANGES[bin_name]

    for source, quota in source_quotas.items():
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

        sub_buckets = _split_into_subbins(chosen, bin_low, bin_high, N_SUB_BINS)
        log[source]["subbins"] = [len(b) for b in sub_buckets]

        if source == "PopMC":
            prop_counts: dict[str, int] = defaultdict(int)
            for r in chosen:
                prop_counts[r.get("prop", "unknown")] += 1
            log[source]["props"] = dict(sorted(prop_counts.items(), key=lambda x: -x[1]))

    return result, log


# ---------------------------------------------------------------------------
# output  (identical to create_balanced_dataset.py)
# ---------------------------------------------------------------------------

def _stratified_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed + 1)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_cell[(r["source"], entropy_bin(r["entropy"]))].append(r)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    test_ratio = 1.0 - train_ratio - val_ratio
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

    print("\nSplit composition (source × entropy_bin):")
    cells = ("low", "med", "high")
    for split_name, rows in splits.items():
        from collections import Counter
        counts = Counter((r["source"], entropy_bin(r["entropy"])) for r in rows)
        print(f"  {split_name}:")
        for src in ("TriviaMC", "PopMC", "SimpleMC"):
            line = "    " + src.ljust(10)
            for c in cells:
                line += f"  {c}={counts.get((src, c), 0):>4d}"
            print(line)


def audit_dataset(records: list[dict], splits: dict[str, list[dict]] | None = None) -> None:
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
    parser.add_argument("--root",            default=".",    help="Project root (default: cwd)")
    parser.add_argument("--output-dir",      default="data", help="Output directory")
    parser.add_argument("--prefix",          required=True,  help="Output filename prefix (e.g. mixed_2550_clean)")
    parser.add_argument("--trivia-per-bin",  type=int, required=True, help="TriviaMC samples per entropy bin")
    parser.add_argument("--popmc-per-bin",   type=int, required=True, help="PopMC samples per entropy bin")
    parser.add_argument("--simple-per-bin",  type=int, required=True, help="SimpleMC samples per entropy bin")
    parser.add_argument("--train-ratio",     type=float, default=0.80)
    parser.add_argument("--val-ratio",       type=float, default=0.10)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    source_quotas = {
        "TriviaMC": args.trivia_per_bin,
        "PopMC":    args.popmc_per_bin,
        "SimpleMC": args.simple_per_bin,
    }

    root = Path(args.root).resolve()
    rng  = random.Random(args.seed)

    print("Building entropy-binned cells...")
    cells = build_cells(root)

    print(f"\nAvailable per cell:")
    print(f"\n{'':12s}  {'low':>6s}  {'med':>6s}  {'high':>6s}  {'total':>6s}")
    for ds in ("TriviaMC", "PopMC", "SimpleMC"):
        lo, me, hi = len(cells[ds]["low"]), len(cells[ds]["med"]), len(cells[ds]["high"])
        print(f"  {ds:10s}  {lo:>6d}  {me:>6d}  {hi:>6d}  {lo+me+hi:>6d}")

    per_bin_total = sum(source_quotas.values())
    print(f"\nTarget per-source per-bin quotas: {source_quotas}")
    print(f"Per-bin total: {per_bin_total}    Dataset total: {per_bin_total * 3}")
    print(f"PopMC max prop fraction: {MAX_PROP_FRACTION:.0%}")
    print(f"Sub-bins per (source, bin) cell: {N_SUB_BINS}\n")

    all_records = []
    for bin_name in ("low", "med", "high"):
        bin_low, bin_high = BIN_RANGES[bin_name]
        print(f"--- {bin_name} bin  (entropy [{bin_low:.3f}, {bin_high:.3f})) ---")
        chosen, log = sample_bin(cells, bin_name, source_quotas, rng)
        all_records.extend(chosen)
        for source, info in log.items():
            sub = info.get("subbins", [])
            sub_str = "/".join(str(x) for x in sub)
            print(f"  {source:10s}: {info['sampled']:>4d} sampled   sub-bins: {sub_str}", end="")
            if "props" in info:
                top = list(info["props"].items())[:5]
                print(f"   top props: {top}", end="")
            print()

    print_summary(all_records)

    print(f"\nWriting splits (train={args.train_ratio:.0%} / val={args.val_ratio:.0%} / test={1-args.train_ratio-args.val_ratio:.0%}):")
    split_and_write(all_records, root / args.output_dir, args.prefix, args.train_ratio, args.val_ratio, args.seed)

    splits = {}
    for split_name in ("train", "val", "test"):
        path = root / args.output_dir / f"{args.prefix}_{split_name}.jsonl"
        with open(path) as f:
            splits[split_name] = [json.loads(line) for line in f]
    audit_dataset(all_records, splits)
    print("\nDone.")


if __name__ == "__main__":
    main()
