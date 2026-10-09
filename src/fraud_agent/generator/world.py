"""Synthetic world: accounts, mule accounts and merchants with stable behavioural profiles.

Ids are random and share one format, so an id never leaks whether an account is a mule.
"""
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from faker import Faker

# Simulated "today" for the synthetic world. Account ages and the backfilled history are anchored
# here so the data is internally consistent (and reproducible) regardless of the wall clock.
REFERENCE_TIME = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata", "Jaipur"]
LEGIT_CATEGORIES = ["grocery", "fuel", "pharmacy", "electronics", "restaurants", "utilities", "travel"]


@dataclass
class Account:
    account_id: str
    name: str
    home_city: str
    kyc_level: str
    opened_at: datetime
    primary_device: str
    avg_amount: float
    contacts: list[str] = field(default_factory=list)
    is_mule: bool = False


@dataclass
class Merchant:
    merchant_id: str
    name: str
    category: str
    risk_score: float
    chargeback_rate: float
    colluding: bool = False


@dataclass
class World:
    accounts: dict[str, Account]
    merchants: dict[str, Merchant]

    @property
    def regular(self) -> list[Account]:
        return [a for a in self.accounts.values() if not a.is_mule]

    @property
    def mules(self) -> list[Account]:
        return [a for a in self.accounts.values() if a.is_mule]

    @property
    def legit_merchants(self) -> list[Merchant]:
        return [m for m in self.merchants.values() if not m.colluding]

    @property
    def colluding_merchants(self) -> list[Merchant]:
        return [m for m in self.merchants.values() if m.colluding]


def _id(rng: random.Random, prefix: str, used: set[str]) -> str:
    while True:
        candidate = f"{prefix}{rng.randint(100000, 999999)}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def build_world(seed: int = 42, n_accounts: int = 400, n_mules: int = 40,
                n_merchants: int = 30, n_colluding: int = 3) -> World:
    rng = random.Random(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)
    now = REFERENCE_TIME
    used: set[str] = set()

    accounts: dict[str, Account] = {}
    for i in range(n_accounts + n_mules):
        is_mule = i >= n_accounts
        acc = Account(
            account_id=_id(rng, "AC", used),
            name=fake.name(),
            home_city=rng.choice(CITIES),
            # Mules: newer accounts with minimal KYC (a real-world signal, but not a certainty)
            kyc_level="min" if (is_mule and rng.random() < 0.8) or rng.random() < 0.1 else "full",
            opened_at=now - timedelta(days=rng.randint(3, 40) if is_mule else rng.randint(200, 3000)),
            primary_device=f"dev-{rng.getrandbits(32):08x}",
            avg_amount=round(rng.lognormvariate(7.2, 0.7), 2),   # ~ INR 1,300 median
            is_mule=is_mule,
        )
        accounts[acc.account_id] = acc

    regular_ids = [a.account_id for a in accounts.values() if not a.is_mule]
    for acc in accounts.values():
        acc.contacts = rng.sample([x for x in regular_ids if x != acc.account_id], 6)

    merchants: dict[str, Merchant] = {}
    for i in range(n_merchants + n_colluding):
        colluding = i >= n_merchants
        m = Merchant(
            merchant_id=_id(rng, "MR", used),
            name=fake.company(),
            category=rng.choice(["electronics", "gift_cards", "crypto_otc"]) if colluding
            else rng.choice(LEGIT_CATEGORIES),
            risk_score=round(rng.uniform(0.55, 0.9) if colluding else rng.uniform(0.02, 0.35), 2),
            chargeback_rate=round(rng.uniform(0.03, 0.09) if colluding else rng.uniform(0.001, 0.01), 4),
            colluding=colluding,
        )
        merchants[m.merchant_id] = m

    return World(accounts=accounts, merchants=merchants)
