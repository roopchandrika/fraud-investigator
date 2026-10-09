"""LangChain tool wrappers. Docstrings are what the model sees, so they say *when* to use each tool."""
import json

from langchain_core.tools import tool

from fraud_agent.tools import core, retrieval


def _dump(x) -> str:
    return json.dumps(x, default=str)[:12000]   # hard cap: protects the context window


@tool
def get_transaction(txn_id: str) -> str:
    """Fetch a transaction and the anomaly detector's flag reasons. Start every investigation here."""
    return _dump(core.get_transaction(txn_id))


@tool
def get_account_profile(account_id: str) -> str:
    """KYC level, account age, status and 30-day behavioural baseline (usual device, city, average
    amount) for an account. Also works for merchant ids. Use to judge whether behaviour is unusual."""
    return _dump(core.get_account_profile(account_id))


@tool
def get_account_activity(account_id: str, around_ts: str, hours_before: float = 24,
                         hours_after: float = 2) -> str:
    """Payments sent and received by an account around an ISO-8601 timestamp. Use to check
    velocity, new payees, and whether received funds were quickly sent on."""
    return _dump(core.get_account_activity(account_id, around_ts, hours_before, hours_after))


@tool
def get_counterparty_graph(account_id: str, around_ts: str, window_minutes: int = 120) -> str:
    """Graph view (Neo4j) of who an account paid / was paid by around a time, with each
    neighbour's fan-in and fan-out. Use to detect mule hubs and collusion rings."""
    return _dump(core.get_counterparty_graph(account_id, around_ts, window_minutes))


@tool
def trace_money_flow(txn_id: str, max_hops: int = 5) -> str:
    """Follow the money forward from a transaction through onward payments of 80-100% of the
    amount within an hour. A multi-hop path is the signature of a mule chain."""
    return _dump(core.trace_money_flow(txn_id, max_hops))


@tool
def get_merchant_risk(merchant_id: str) -> str:
    """Merchant category, risk score, chargeback rate and 24h inflow pattern (distinct payers,
    share of round amounts). Use when a merchant is the payee or payer."""
    return _dump(core.get_merchant_risk(merchant_id))


@tool
def search_fraud_playbook(query: str) -> str:
    """Search the internal fraud playbook (typologies, red flags, false-positive guidance and the
    action policy). Cite the returned 'ref' values in your report."""
    return _dump(retrieval.search_playbook(query))


@tool
def search_similar_cases(query: str) -> str:
    """Search memory of previously investigated cases and how human reviewers decided them.
    Describe the pattern you are seeing in plain words."""
    return _dump(retrieval.search_cases(query))


INVESTIGATION_TOOLS = [
    get_transaction, get_account_profile, get_account_activity, get_counterparty_graph,
    trace_money_flow, get_merchant_risk, search_fraud_playbook, search_similar_cases,
]
