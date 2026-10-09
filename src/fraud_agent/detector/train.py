"""Train the Isolation Forest on normal traffic, then measure detection on a labelled stream.

    uv run python -m fraud_agent.detector.train
Runs fully offline (no Kafka/DB needed).
"""
import argparse
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import IsolationForest

from fraud_agent.config import get_settings
from fraud_agent.detector.features import FEATURE_NAMES, FeatureState
from fraud_agent.detector.model import Detector
from fraud_agent.generator.producer import event_stream
from fraud_agent.generator.world import build_world

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=30000)
    ap.add_argument("--test-n", type=int, default=30000)
    args = ap.parse_args()
    s = get_settings()
    world = build_world()

    state = FeatureState()
    train_txns = list(islice(event_stream(world, seed=1, fraud_rate=0), args.train_n))
    X = np.array([[f[k] for k in FEATURE_NAMES] for f in (state.features(t) for t in train_txns)])
    model = IsolationForest(n_estimators=200, contamination=0.005, random_state=0).fit(X)
    det = Detector(model=model, threshold=s.iforest_threshold)
    Path(s.detector_model_path).parent.mkdir(parents=True, exist_ok=True)
    det.save(s.detector_model_path)
    console.print(f"[green]Saved detector to {s.detector_model_path}")

    # Evaluate on a fresh labelled stream that starts where training ended (state stays warm,
    # like a production consumer that has been running for a while)
    test_txns = list(islice(event_stream(world, seed=2, fraud_rate=0.002,
                                         start=train_txns[-1].ts), args.test_n))
    results = det.score_many([state.features(t) for t in test_txns])
    tp = fp = fn = 0
    recall_by_pattern: Counter = Counter()
    total_by_pattern: Counter = Counter()
    for t, (flagged, _, _) in zip(test_txns, results):
        tp += flagged and t.is_fraud
        fp += flagged and not t.is_fraud
        fn += (not flagged) and t.is_fraud
        if t.is_fraud:
            total_by_pattern[t.fraud_pattern] += 1
            recall_by_pattern[t.fraud_pattern] += flagged

    table = Table(title="Detector evaluation (transaction level)")
    for col in ("metric", "value"):
        table.add_column(col)
    table.add_row("precision", f"{tp / max(tp + fp, 1):.3f}")
    table.add_row("recall", f"{tp / max(tp + fn, 1):.3f}")
    table.add_row("flag rate (share of traffic sent to LLM)", f"{(tp + fp) / args.test_n:.3%}")
    for p, n in sorted(total_by_pattern.items()):
        table.add_row(f"recall: {p}", f"{recall_by_pattern[p] / n:.3f}  (n={n})")
    console.print(table)


if __name__ == "__main__":
    main()
