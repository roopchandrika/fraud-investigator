"""Online detector: consume transactions -> persist (Postgres + Neo4j) -> score -> publish flags.

    uv run python -m fraud_agent.detector.consumer
"""
import json

from confluent_kafka import Consumer, Producer
from rich.console import Console

from fraud_agent.config import get_settings
from fraud_agent.db import neo4j, pg
from fraud_agent.detector.features import FeatureState
from fraud_agent.detector.model import Detector
from fraud_agent.schemas import Flag, Transaction

console = Console()

INSERT_TXN = """
INSERT INTO transactions (txn_id, ts, payer_id, payee_id, merchant_id, amount, channel,
                          device_id, city, is_fraud, fraud_pattern)
VALUES (%(txn_id)s, %(ts)s, %(payer_id)s, %(payee_id)s, %(merchant_id)s, %(amount)s, %(channel)s,
        %(device_id)s, %(city)s, %(is_fraud)s, %(fraud_pattern)s)
ON CONFLICT DO NOTHING
"""

MERGE_EDGE = """
MERGE (a:Account {id: $payer}) MERGE (b:Account {id: $payee})
CREATE (a)-[:PAID {txn_id: $txn_id, amount: $amount, ts: datetime($ts), device: $device}]->(b)
"""


def persist(t: Transaction) -> None:
    with pg() as conn:
        conn.execute(INSERT_TXN, t.model_dump())
    with neo4j().session() as s:
        s.run(MERGE_EDGE, payer=t.payer_id, payee=t.payee_id, txn_id=t.txn_id,
              amount=t.amount, ts=t.ts.isoformat(), device=t.device_id)


def main() -> None:
    s = get_settings()
    detector = Detector.load(s.detector_model_path)
    state = FeatureState()
    consumer = Consumer({"bootstrap.servers": s.kafka_bootstrap, "group.id": "detector",
                         "auto.offset.reset": "earliest", "enable.auto.commit": True})
    producer = Producer({"bootstrap.servers": s.kafka_bootstrap})
    consumer.subscribe([s.topic_transactions])
    console.print("[green]Detector listening...")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                console.print(f"[red]{msg.error()}")
                continue
            t = Transaction.model_validate_json(msg.value())
            persist(t)
            flagged, score, reasons = detector.score(state.features(t))
            if not flagged:
                continue
            flag = Flag(txn_id=t.txn_id, score=score, reasons=reasons)
            with pg() as conn:
                conn.execute("INSERT INTO flags (txn_id, score, reasons) VALUES (%s, %s, %s) "
                             "ON CONFLICT DO NOTHING", (t.txn_id, score, json.dumps(reasons)))
            producer.produce(s.topic_flagged, key=t.txn_id, value=flag.model_dump_json())
            producer.poll(0)
            truth = f"[red]FRAUD:{t.fraud_pattern}" if t.is_fraud else "[dim]legit"
            console.print(f"FLAG {t.txn_id} score={score} {truth}[/] {reasons}")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    main()
