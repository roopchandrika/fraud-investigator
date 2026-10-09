"""Build a stratified eval set from flagged transactions (ground truth comes from the generator).

    uv run python -m evals.build_dataset --per-pattern 8 --legit 18
Deterministic: same backfill seed -> same dataset.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from fraud_agent.db import pg

OUT = Path("evals/dataset.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pattern", type=int, default=8)
    ap.add_argument("--legit", type=int, default=18, help="false-positive flags (legit txns)")
    args = ap.parse_args()

    with pg() as conn:
        rows = conn.execute("""
            SELECT f.txn_id, t.is_fraud, coalesce(t.fraud_pattern, 'none') AS pattern, f.reasons
            FROM flags f JOIN transactions t USING (txn_id)""").fetchall()

    buckets = defaultdict(list)
    for r in rows:
        buckets[r["pattern"]].append(r)
    picked = []
    for pattern, items in sorted(buckets.items()):
        items.sort(key=lambda r: hashlib.md5(r["txn_id"].encode()).hexdigest())  # stable shuffle
        picked += items[: args.legit if pattern == "none" else args.per_pattern]

    with OUT.open("w") as f:
        for r in picked:
            f.write(json.dumps({"txn_id": r["txn_id"], "label_is_fraud": r["is_fraud"],
                                "label_pattern": r["pattern"], "detector_reasons": r["reasons"]}) + "\n")
    counts = {p: min(len(v), args.legit if p == "none" else args.per_pattern) for p, v in buckets.items()}
    print(f"Wrote {len(picked)} cases to {OUT}: {counts}")


if __name__ == "__main__":
    main()
