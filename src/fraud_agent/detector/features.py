"""Streaming feature engineering, shared by offline training and the online detector.

State is kept in memory per account (fine for a demo; in production this would live in
Redis or a stream processor such as Flink).
"""
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fraud_agent.schemas import Transaction

FEATURE_NAMES = [
    "log_amount", "amount_ratio", "txns_5m", "distinct_payees_10m", "new_payee",
    "new_device", "new_city", "forward_ratio", "payee_fanin_30m", "round_amount",
]


@dataclass
class AccountState:
    sent: deque = field(default_factory=lambda: deque(maxlen=200))       # (ts, payee, amount)
    received: deque = field(default_factory=lambda: deque(maxlen=200))   # (ts, payer, amount)
    payees: set = field(default_factory=set)
    devices: Counter = field(default_factory=Counter)
    cities: Counter = field(default_factory=Counter)
    amount_sum: float = 0.0
    n_sent: int = 0


class FeatureState:
    def __init__(self) -> None:
        self.accounts: dict[str, AccountState] = defaultdict(AccountState)

    def features(self, t: Transaction) -> dict[str, float]:
        """Compute features for t using history *before* t, then update state."""
        payer, payee = self.accounts[t.payer_id], self.accounts[t.payee_id]
        has_history = payer.n_sent >= 5

        avg = payer.amount_sum / payer.n_sent if payer.n_sent else t.amount
        recv_20m = sum(a for ts, _, a in payer.received if _within(t.ts, ts, timedelta(minutes=20)))
        f = {
            "log_amount": math.log1p(t.amount),
            "amount_ratio": t.amount / avg if has_history else 1.0,
            "txns_5m": _count_since(payer.sent, t.ts, timedelta(minutes=5)),
            "distinct_payees_10m": len({p for ts, p, _ in payer.sent
                                        if _within(t.ts, ts, timedelta(minutes=10))}),
            "new_payee": float(has_history and t.payee_id not in payer.payees),
            "new_device": float(has_history and t.device_id not in payer.devices),
            "new_city": float(has_history and t.city not in payer.cities),
            # Only meaningful for sizeable inflows; small everyday amounts make this noisy
            "forward_ratio": t.amount / recv_20m if recv_20m >= 10000 else 0.0,
            "payee_fanin_30m": len({p for ts, p, _ in payee.received
                                    if _within(t.ts, ts, timedelta(minutes=30))}),
            "round_amount": float(t.amount >= 500 and t.amount % 500 == 0),
        }

        payer.sent.append((t.ts, t.payee_id, t.amount))
        payer.payees.add(t.payee_id)
        payer.devices[t.device_id] += 1
        payer.cities[t.city] += 1
        payer.amount_sum += t.amount
        payer.n_sent += 1
        payee.received.append((t.ts, t.payer_id, t.amount))
        return f


def _within(now: datetime, ts: datetime, window: timedelta) -> bool:
    return timedelta(0) <= now - ts <= window


def _count_since(events: deque, now: datetime, window: timedelta) -> int:
    return sum(1 for ts, *_ in events if _within(now, ts, window))


def rule_hits(f: dict[str, float]) -> list[str]:
    """Human-readable rules. These become the 'reasons' the triage agent sees."""
    reasons = []
    if f["txns_5m"] >= 6:
        reasons.append(f"velocity: {int(f['txns_5m'])} payments by payer in 5 min")
    if f["distinct_payees_10m"] >= 5 and f["new_payee"]:
        reasons.append(f"fan-out: {int(f['distinct_payees_10m'])} distinct payees in 10 min")
    if f["new_device"] and f["new_city"] and f["amount_ratio"] > 5:
        reasons.append("takeover signals: new device + new city + amount far above normal")
    if f["amount_ratio"] > 8:
        reasons.append(f"amount {f['amount_ratio']:.1f}x payer's average")
    if 0.8 <= f["forward_ratio"] <= 1.0:
        reasons.append(f"pass-through: forwards {f['forward_ratio']:.0%} of funds received in last 20 min")
    if f["payee_fanin_30m"] >= 5 and f["round_amount"]:
        reasons.append(f"fan-in: payee received from {int(f['payee_fanin_30m'])} payers in 30 min (round amounts)")
    return reasons
