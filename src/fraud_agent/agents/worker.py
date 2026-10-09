"""Consume flagged transactions from Kafka and investigate each one.

    uv run python -m fraud_agent.agents.worker
Runs one case at a time; scale out by running more workers in the same consumer group.
"""
from confluent_kafka import Consumer
from rich.console import Console

from fraud_agent.agents.graph import case_id_for, investigate, open_graph, total_cost
from fraud_agent.config import get_settings
from fraud_agent.db import pg
from fraud_agent.schemas import Flag

console = Console()


def main() -> None:
    s = get_settings()
    consumer = Consumer({"bootstrap.servers": s.kafka_bootstrap, "group.id": "investigators",
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer.subscribe([s.topic_flagged])
    with open_graph() as graph:
        console.print("[green]Investigation worker listening...")
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            flag = Flag.model_validate_json(msg.value())
            case_id = case_id_for(flag.txn_id)
            with pg() as conn:
                exists = conn.execute("SELECT 1 FROM cases WHERE case_id = %s", (case_id,)).fetchone()
            if not exists:
                try:
                    result = investigate(graph, flag.txn_id, case_id)
                    cf = result.get("case_file", {})
                    paused = "PENDING REVIEW" if "__interrupt__" in result else "closed"
                    console.print(f"{case_id}: {cf.get('verdict')} / {cf.get('fraud_pattern')} -> "
                                  f"{cf.get('recommended_action')} [{paused}] ${total_cost(result)}")
                except Exception as e:  # noqa: BLE001 - one bad case must not stop the worker
                    console.print(f"[red]{case_id} failed: {e}")
            consumer.commit(msg)   # at-least-once; the case_id check makes reprocessing safe


if __name__ == "__main__":
    main()
