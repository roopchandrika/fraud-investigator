"""REST API for cases and human review.

    uv run uvicorn fraud_agent.api.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from fraud_agent.agents.graph import case_id_for, investigate, open_graph, resume, total_cost
from fraud_agent.db import pg
from fraud_agent.schemas import ReviewDecision

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    with open_graph() as graph:
        state["graph"] = graph
        yield


app = FastAPI(title="Fraud Investigator", lifespan=lifespan)


@app.get("/cases")
def list_cases(status: str | None = None, limit: int = 100) -> list[dict]:
    q = "SELECT case_id, txn_id, status, severity, decision, cost_usd, latency_s, updated_at, " \
        "case_file->>'verdict' AS verdict, case_file->>'fraud_pattern' AS pattern, " \
        "case_file->>'recommended_action' AS action FROM cases"
    params: list = []
    if status:
        q += " WHERE status = %s"
        params.append(status)
    q += " ORDER BY updated_at DESC LIMIT %s"
    with pg() as conn:
        return conn.execute(q, [*params, limit]).fetchall()


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    with pg() as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id = %s", (case_id,)).fetchone()
    if not row:
        raise HTTPException(404, "case not found")
    return row


@app.post("/investigate/{txn_id}")
async def start_investigation(txn_id: str, use_cache: bool = True) -> dict:
    result = await run_in_threadpool(investigate, state["graph"], txn_id, None, use_cache)
    return {"case_id": case_id_for(txn_id), "case_file": result.get("case_file"),
            "paused_for_review": "__interrupt__" in result, "cost_usd": total_cost(result)}


@app.post("/cases/{case_id}/decision")
async def decide(case_id: str, decision: ReviewDecision) -> dict:
    case = get_case(case_id)
    if case["status"] != "pending_review":
        raise HTTPException(409, f"case is {case['status']}, not pending_review")
    await run_in_threadpool(resume, state["graph"], case_id, decision.model_dump())
    return get_case(case_id)


@app.get("/metrics")
def metrics() -> dict:
    with pg() as conn:
        return conn.execute("""
            SELECT count(*) AS cases,
                   count(*) FILTER (WHERE status = 'pending_review') AS pending_review,
                   round(avg(cost_usd)::numeric, 4) AS avg_cost_usd,
                   round(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_s)::numeric, 1) AS p50_latency_s,
                   round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_s)::numeric, 1) AS p95_latency_s,
                   count(*) FILTER (WHERE decision = 'approved') AS approved,
                   count(*) FILTER (WHERE decision = 'rejected') AS rejected
            FROM cases""").fetchone()
