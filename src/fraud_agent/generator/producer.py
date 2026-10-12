"""Seed reference data and stream synthetic transactions to Kafka.

    uv run python -m fraud_agent.generator.producer --seed-db
    uv run python -m fraud_agent.generator.producer --rate 20 --fraud-rate 0.01 --count 5000
"""
import argparse
import heapq
import random
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from itertools import count

from confluent_kafka import Producer
from rich.console import Console

from fraud_agent.config import get_settings
from fraud_agent.db import init_neo4j, neo4j, pg
from fraud_agent.generator.patterns import FRAUD_PATTERNS, normal_txn
from fraud_agent.generator.world import World, build_world
from fraud_agent.schemas import Transaction

console = Console()


def event_stream(world: World, seed: int = 7, fraud_rate: float = 0.002,
                 start: datetime | None = None) -> Iterator[Transaction]:
    """Infinite, event-time-ordered stream of normal traffic with fraud patterns mixed in.

    Normal payments arrive about a minute apart in event time, which gives each of the ~400
    customers a realistic few payments per day (20k transactions ~ two weeks of history).
    Fraud patterns keep their own seconds-to-minutes timing, so bursts stand out against that.
    When producing live, event time runs faster than wall-clock time (an accelerated replay).
    """
    rng = random.Random(seed)
    clock = start or datetime.now(UTC)
    pending: list[tuple[datetime, int, Transaction]] = []
    tie = count()
    while True:
        clock += timedelta(seconds=rng.uniform(20, 100))
        heapq.heappush(pending, (clock, next(tie), normal_txn(world, rng, clock)))
        if rng.random() < fraud_rate:
            pattern = rng.choice(list(FRAUD_PATTERNS.values()))
            for t in pattern(world, rng, clock):
                heapq.heappush(pending, (t.ts, next(tie), t))
        while pending and pending[0][0] <= clock:
            yield heapq.heappop(pending)[2]


def seed_reference_data(world: World) -> None:
    with pg() as conn:
        conn.execute("TRUNCATE cases, flags, transactions, accounts, merchants")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO accounts (account_id, name, home_city, kyc_level, opened_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                [(a.account_id, a.name, a.home_city, a.kyc_level, a.opened_at)
                 for a in world.accounts.values()],
            )
            cur.executemany(
                "INSERT INTO merchants VALUES (%s, %s, %s, %s, %s)",
                [(m.merchant_id, m.name, m.category, m.risk_score, m.chargeback_rate)
                 for m in world.merchants.values()],
            )
    init_neo4j()
    with neo4j().session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run("UNWIND $ids AS id MERGE (:Account {id: id})", ids=list(world.accounts))
        s.run("UNWIND $ids AS id MERGE (:Account:Merchant {id: id})", ids=list(world.merchants))
    console.print(f"[green]Seeded {len(world.accounts)} accounts, {len(world.merchants)} merchants")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-db", action="store_true", help="(re)create accounts/merchants and exit")
    ap.add_argument("--rate", type=float, default=20, help="transactions per second")
    ap.add_argument("--fraud-rate", type=float, default=0.002, help="P(fraud pattern starts) per tick")
    ap.add_argument("--count", type=int, default=0, help="stop after N transactions (0 = forever)")
    ap.add_argument("--world-seed", type=int, default=42)
    args = ap.parse_args()

    world = build_world(seed=args.world_seed)
    if args.seed_db:
        seed_reference_data(world)
        return

    s = get_settings()
    producer = Producer({"bootstrap.servers": s.kafka_bootstrap, "linger.ms": 20})
    n_fraud = 0
    for i, txn in enumerate(event_stream(world, seed=int(time.time()), fraud_rate=args.fraud_rate), 1):
        # Key by payer so one account's events stay ordered within a partition
        producer.produce(s.topic_transactions, key=txn.payer_id, value=txn.model_dump_json())
        n_fraud += txn.is_fraud
        if i % 200 == 0:
            producer.poll(0)
            console.print(f"sent {i} txns ({n_fraud} fraudulent)")
        if args.count and i >= args.count:
            break
        time.sleep(1 / args.rate)
    producer.flush()


if __name__ == "__main__":
    main()
