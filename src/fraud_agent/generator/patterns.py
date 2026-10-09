"""Normal behaviour + four planted fraud patterns.

Each pattern returns a list of Transactions with absolute timestamps; the producer merges them
into the normal stream by event time.
"""
import random
import uuid
from datetime import datetime, timedelta
from itertools import pairwise

from fraud_agent.generator.world import CITIES, World
from fraud_agent.schemas import Transaction


def _txn(rng: random.Random, ts: datetime, payer: str, payee: str, amount: float,
         device: str, city: str,
         channel: str = "upi", merchant_id: str | None = None,
         pattern: str | None = None) -> Transaction:
    return Transaction(
        # Seeded ids: the same seed always produces the same dataset (reproducible evals)
        txn_id=f"TX{uuid.UUID(int=rng.getrandbits(128)).hex[:12].upper()}", ts=ts, payer_id=payer, payee_id=payee,
        merchant_id=merchant_id, amount=round(amount, 2), channel=channel, device_id=device,
        city=city, is_fraud=pattern is not None, fraud_pattern=pattern,
    )


def normal_txn(world: World, rng: random.Random, ts: datetime) -> Transaction:
    payer = rng.choice(world.regular)
    device = payer.primary_device if rng.random() < 0.95 else f"dev-{rng.getrandbits(32):08x}"
    city = payer.home_city if rng.random() < 0.97 else rng.choice(CITIES)
    amount = max(10.0, rng.gauss(payer.avg_amount, payer.avg_amount * 0.4))
    # Legit noise that looks scary: occasional rent / tuition / gadget purchase (5-12x avg)
    if rng.random() < 0.01:
        amount = payer.avg_amount * rng.uniform(5, 12)
    if rng.random() < 0.3:
        m = rng.choice(world.legit_merchants)
        return _txn(rng, ts, payer.account_id, m.merchant_id, amount, device, city,
                    channel=rng.choice(["upi", "card"]), merchant_id=m.merchant_id)
    return _txn(rng, ts, payer.account_id, rng.choice(payer.contacts), amount, device, city)


def velocity_burst(world: World, rng: random.Random, start: datetime) -> list[Transaction]:
    """Compromised account drains funds: many quick payments to new payees, just under 10k."""
    victim = rng.choice(world.regular)
    device = f"dev-{rng.getrandbits(32):08x}"
    payees = rng.sample(world.mules, k=min(len(world.mules), rng.randint(4, 7)))
    out, ts = [], start
    for _ in range(rng.randint(10, 16)):
        ts += timedelta(seconds=rng.randint(5, 20))
        out.append(_txn(rng, ts, victim.account_id, rng.choice(payees).account_id,
                        rng.uniform(2000, 9900), device, victim.home_city, pattern="velocity_burst"))
    return out


def mule_chain(world: World, rng: random.Random, start: datetime) -> list[Transaction]:
    """Victim -> mule1 -> mule2 -> mule3 -> cash-out, each hop forwarding 95-99% within minutes."""
    victim = rng.choice(world.regular)
    chain = rng.sample(world.mules, k=rng.randint(3, 4))
    amount = rng.uniform(25000, 90000)
    ts = start
    out = [_txn(rng, ts, victim.account_id, chain[0].account_id, amount,
                f"dev-{rng.getrandbits(32):08x}", rng.choice(CITIES), pattern="mule_chain")]
    for src, dst in pairwise(chain):
        ts += timedelta(minutes=rng.uniform(2, 8))
        amount *= rng.uniform(0.95, 0.99)
        out.append(_txn(rng, ts, src.account_id, dst.account_id, amount, src.primary_device,
                        src.home_city, pattern="mule_chain"))
    return out


def merchant_collusion(world: World, rng: random.Random, start: datetime) -> list[Transaction]:
    """Many fresh accounts pay one high-risk merchant round amounts; merchant pays out to one account."""
    merchant = rng.choice(world.colluding_merchants)
    payers = rng.sample(world.mules, k=rng.randint(6, 10))
    collector = rng.choice(world.mules)
    out, ts, total = [], start, 0.0
    for p in payers:
        ts += timedelta(minutes=rng.uniform(0.5, 3))
        amt = float(rng.choice([1000, 2000, 2500, 5000]))
        total += amt
        out.append(_txn(rng, ts, p.account_id, merchant.merchant_id, amt, p.primary_device, p.home_city,
                        channel="card", merchant_id=merchant.merchant_id, pattern="merchant_collusion"))
    ts += timedelta(minutes=rng.uniform(5, 15))
    out.append(_txn(rng, ts, merchant.merchant_id, collector.account_id, total * rng.uniform(0.85, 0.93),
                    "dev-merchant-pos", collector.home_city, pattern="merchant_collusion"))
    return out


def account_takeover(world: World, rng: random.Random, start: datetime) -> list[Transaction]:
    """New device + new city + one large transfer to a never-seen payee."""
    victim = rng.choice(world.regular)
    city = rng.choice([c for c in CITIES if c != victim.home_city])
    return [_txn(rng, start, victim.account_id, rng.choice(world.mules).account_id,
                 victim.avg_amount * rng.uniform(8, 20), f"dev-{rng.getrandbits(32):08x}", city,
                 pattern="account_takeover")]


FRAUD_PATTERNS = {
    "velocity_burst": velocity_burst,
    "mule_chain": mule_chain,
    "merchant_collusion": merchant_collusion,
    "account_takeover": account_takeover,
}
