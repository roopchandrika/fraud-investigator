"""Investigation tools as plain Python functions.

These are framework-agnostic on purpose: the LangChain tools (tools/langchain_tools.py) and the
MCP server (mcp_server.py) are both thin wrappers over this module.

Guardrail: ground-truth columns (is_fraud, fraud_pattern) are never selected here.
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from fraud_agent.db import neo4j, pg

TXN_COLUMNS = "txn_id, ts, payer_id, payee_id, merchant_id, amount, channel, device_id, city"


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, datetime):
        return v.isoformat(timespec="seconds")
    if hasattr(v, "to_native"):           # neo4j DateTime
        return v.to_native().isoformat(timespec="seconds")
    return v


def get_transaction(txn_id: str) -> dict:
    """The transaction plus the detector's flag reasons (if flagged)."""
    with pg() as conn:
        txn = conn.execute(f"SELECT {TXN_COLUMNS} FROM transactions WHERE txn_id = %s",
                           (txn_id,)).fetchone()
        flag = conn.execute("SELECT score, reasons FROM flags WHERE txn_id = %s", (txn_id,)).fetchone()
    if not txn:
        return {"error": f"transaction {txn_id} not found"}
    return _jsonable({"transaction": txn, "flag": flag})


def get_account_profile(account_id: str) -> dict:
    """KYC/profile of an account (or merchant) plus 30-day behavioural baseline."""
    with pg() as conn:
        merchant = conn.execute("SELECT * FROM merchants WHERE merchant_id = %s",
                                (account_id,)).fetchone()
        if merchant:
            return _jsonable({"type": "merchant", **merchant})
        acc = conn.execute("SELECT account_id, home_city, kyc_level, opened_at, status "
                           "FROM accounts WHERE account_id = %s", (account_id,)).fetchone()
        if not acc:
            return {"error": f"account {account_id} not found"}
        stats = conn.execute(
            """SELECT count(*) AS n_sent_30d, round(avg(amount), 2) AS avg_amount,
                      round(max(amount), 2) AS max_amount,
                      count(DISTINCT payee_id) AS distinct_payees,
                      count(DISTINCT device_id) AS distinct_devices,
                      mode() WITHIN GROUP (ORDER BY device_id) AS usual_device,
                      mode() WITHIN GROUP (ORDER BY city) AS usual_city
               FROM transactions
               WHERE payer_id = %s AND ts > (SELECT max(ts) FROM transactions) - interval '30 days'""",
            (account_id,)).fetchone()
        received = conn.execute(
            "SELECT count(*) AS n_received_30d, count(DISTINCT payer_id) AS distinct_payers "
            "FROM transactions WHERE payee_id = %s AND ts > (SELECT max(ts) FROM transactions) - interval '30 days'",
            (account_id,)).fetchone()
        data_now = conn.execute("SELECT max(ts) AS t FROM transactions").fetchone()["t"]
    # Age as of the latest data, not the wall clock (the data may be simulated/backfilled)
    acc["account_age_days"] = ((data_now or acc["opened_at"]) - acc["opened_at"]).days
    return _jsonable({"type": "account", **acc, "baseline_30d": {**stats, **received}})


def get_account_activity(account_id: str, around_ts: str, hours_before: float = 24,
                         hours_after: float = 2, limit: int = 40) -> dict:
    """Payments sent and received by an account in a window around a timestamp (ISO-8601)."""
    center = datetime.fromisoformat(around_ts)
    lo, hi = center - timedelta(hours=hours_before), center + timedelta(hours=hours_after)
    with pg() as conn:
        rows = conn.execute(
            f"""SELECT {TXN_COLUMNS},
                       CASE WHEN payer_id = %(a)s THEN 'sent' ELSE 'received' END AS direction
                FROM transactions
                WHERE (payer_id = %(a)s OR payee_id = %(a)s) AND ts BETWEEN %(lo)s AND %(hi)s
                ORDER BY ts DESC LIMIT %(limit)s""",
            {"a": account_id, "lo": lo, "hi": hi, "limit": limit}).fetchall()
    sent = [r for r in rows if r["direction"] == "sent"]
    return _jsonable({
        "account_id": account_id, "window": [lo, hi], "n_sent": len(sent),
        "n_received": len(rows) - len(sent),
        "total_sent": sum(float(r["amount"]) for r in sent),
        "total_received": sum(float(r["amount"]) for r in rows if r["direction"] == "received"),
        "transactions": rows,
    })


def get_counterparty_graph(account_id: str, around_ts: str, window_minutes: int = 120) -> dict:
    """1-hop payment graph of an account (Neo4j), with each neighbour's own fan-in/fan-out
    in the same window. Useful for spotting mule hubs and collusion rings."""
    center = datetime.fromisoformat(around_ts)
    lo = (center - timedelta(minutes=window_minutes)).isoformat()
    hi = (center + timedelta(minutes=window_minutes)).isoformat()
    query = """
    MATCH (a:Account {id: $id})-[r:PAID]-(n:Account)
    WHERE r.ts >= datetime($lo) AND r.ts <= datetime($hi)
    WITH a, n, collect(r) AS rels
    OPTIONAL MATCH (n)-[o:PAID]->(x) WHERE o.ts >= datetime($lo) AND o.ts <= datetime($hi)
    WITH a, n, rels, count(DISTINCT x) AS n_fan_out
    OPTIONAL MATCH (y)-[i:PAID]->(n) WHERE i.ts >= datetime($lo) AND i.ts <= datetime($hi)
    RETURN n.id AS neighbour, 'Merchant' IN labels(n) AS is_merchant,
           [r IN rels | {txn_id: r.txn_id, amount: r.amount, ts: r.ts,
                         direction: CASE WHEN startNode(r) = a THEN 'out' ELSE 'in' END}] AS edges,
           n_fan_out, count(DISTINCT y) AS n_fan_in
    ORDER BY size(edges) DESC LIMIT 25
    """
    with neo4j().session() as s:
        rows = [r.data() for r in s.run(query, id=account_id, lo=lo, hi=hi)]
    return _jsonable({"account_id": account_id, "window_minutes": window_minutes,
                      "n_neighbours": len(rows), "neighbours": rows})


def trace_money_flow(txn_id: str, max_hops: int = 5, window_minutes: int = 60) -> dict:
    """Follow the money forward from a transaction: at each hop, find the largest onward payment
    by the recipient made shortly after receiving, carrying 80-100% of the amount."""
    with pg() as conn:
        txn = conn.execute(f"SELECT {TXN_COLUMNS} FROM transactions WHERE txn_id = %s",
                           (txn_id,)).fetchone()
    if not txn:
        return {"error": f"transaction {txn_id} not found"}
    hops = [txn]
    query = """
    MATCH (:Account {id: $holder})-[r:PAID]->(next:Account)
    WHERE r.ts > datetime($ts) AND r.ts <= datetime($ts) + duration({minutes: $win})
      AND r.amount >= $amount * 0.8 AND r.amount <= $amount * 1.0
    RETURN r.txn_id AS txn_id, r.ts AS ts, startNode(r).id AS payer_id, next.id AS payee_id,
           r.amount AS amount
    ORDER BY r.amount DESC LIMIT 1
    """
    with neo4j().session() as s:
        for _ in range(max_hops):
            cur = hops[-1]
            rec = s.run(query, holder=cur["payee_id"], ts=_jsonable(cur["ts"]),
                        amount=float(cur["amount"]), win=window_minutes).single()
            if rec is None:
                break
            hops.append(rec.data())
    return _jsonable({"origin_txn": txn_id, "n_onward_hops": len(hops) - 1, "path": hops,
                      "retained_share": float(hops[-1]["amount"]) / float(txn["amount"])})


def get_merchant_risk(merchant_id: str) -> dict:
    """Merchant category, vendor risk score and chargeback rate, plus last-24h inflow stats."""
    with pg() as conn:
        m = conn.execute("SELECT * FROM merchants WHERE merchant_id = %s", (merchant_id,)).fetchone()
        if not m:
            return {"error": f"merchant {merchant_id} not found"}
        inflow = conn.execute(
            """SELECT count(*) AS n_payments_24h, count(DISTINCT payer_id) AS distinct_payers_24h,
                      round(avg(amount), 2) AS avg_amount_24h,
                      round(avg(CASE WHEN amount %% 500 = 0 THEN 1 ELSE 0 END), 2)
                          AS share_round_amounts
               FROM transactions
               WHERE payee_id = %s AND ts > (SELECT max(ts) FROM transactions) - interval '24 hours'""",
            (merchant_id,)).fetchone()
    return _jsonable({**m, "inflow_24h": inflow})
