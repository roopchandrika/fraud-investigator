"""Deterministic backfill without Kafka: generate -> persist -> score, straight into the stores.

Used for quick local setup and in CI, so evals always run against the same data.
    uv run python -m fraud_agent.detector.backfill --count 20000
"""
import argparse
import json
from itertools import islice

from rich.console import Console
from rich.progress import track

from fraud_agent.config import get_settings
from fraud_agent.db import neo4j, pg
from fraud_agent.detector.consumer import INSERT_TXN
from fraud_agent.detector.features import FeatureState
from fraud_agent.detector.model import Detector
from fraud_agent.generator.producer import event_stream
from fraud_agent.generator.world import REFERENCE_TIME, build_world

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fraud-rate", type=float, default=0.002)
    args = ap.parse_args()

    detector = Detector.load(get_settings().detector_model_path)
    state = FeatureState()
    txns = list(islice(event_stream(build_world(), seed=args.seed, fraud_rate=args.fraud_rate,
                                    start=REFERENCE_TIME), args.count))
    results = detector.score_many([state.features(t) for t in txns])

    with pg() as conn, conn.cursor() as cur:
        cur.executemany(INSERT_TXN, [t.model_dump() for t in txns])
        cur.executemany("INSERT INTO flags (txn_id, score, reasons) VALUES (%s, %s, %s) "
                        "ON CONFLICT DO NOTHING",
                        [(t.txn_id, score, json.dumps(reasons))
                         for t, (flagged, score, reasons) in zip(txns, results) if flagged])

    with neo4j().session() as s:
        for i in track(range(0, len(txns), 1000), description="Writing graph"):
            batch = [{"payer": t.payer_id, "payee": t.payee_id, "txn_id": t.txn_id,
                      "amount": t.amount, "ts": t.ts.isoformat(), "device": t.device_id}
                     for t in txns[i:i + 1000]]
            s.run("""UNWIND $rows AS r
                     MERGE (a:Account {id: r.payer}) MERGE (b:Account {id: r.payee})
                     CREATE (a)-[:PAID {txn_id: r.txn_id, amount: r.amount,
                                        ts: datetime(r.ts), device: r.device}]->(b)""", rows=batch)

    n_flags = sum(f for f, _, _ in results)
    n_fraud = sum(t.is_fraud for t in txns)
    console.print(f"[green]Backfilled {len(txns)} txns ({n_fraud} fraudulent), {n_flags} flagged")


if __name__ == "__main__":
    main()
