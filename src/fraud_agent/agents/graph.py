"""LangGraph investigation workflow.

    load_context -> triage --(benign)--> close_benign ------------------------------> finalize
                          \\--(investigate)--> investigator <-> tools                   ^
                                                   |                                     |
                                                reporter -> grounding_check --(no action)-+
                                                                  |
                                                    (freeze/block) -> human_review (interrupt)
                                                                          -> finalize

State is checkpointed in Postgres (thread_id = case_id), so a case can pause for human review
and be resumed later, from another process.
"""
import json
import operator
import time
from contextlib import contextmanager
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from rich.console import Console

from fraud_agent.agents import prompts
from fraud_agent.agents.llm import chat, is_refusal, usage_of
from fraud_agent.config import get_settings
from fraud_agent.db import pg
from fraud_agent.schemas import CaseFile, TriageResult
from fraud_agent.tools import core, retrieval
from fraud_agent.tools.langchain_tools import INVESTIGATION_TOOLS
from fraud_agent.tools.semantic_cache import SemanticCache, signature

console = Console()
ACTIONS_NEEDING_APPROVAL = {"freeze_account", "block_transaction"}


class InvestigationState(TypedDict, total=False):
    case_id: str
    txn_id: str
    use_cache: bool
    eval_mode: bool          # no DB writes, no memory writes: evals must not pollute production
    started_at: float
    context: dict
    triage: dict
    cache_hit: bool
    messages: Annotated[list[AnyMessage], add_messages]
    case_file: dict
    grounding: dict
    decision: dict
    usage: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------- persistence helpers

def save_case(case_id: str, txn_id: str, **fields: Any) -> None:
    cols = {"case_id": case_id, "txn_id": txn_id, **fields}
    if "case_file" in cols:
        cols["case_file"] = json.dumps(cols["case_file"])
    names = ", ".join(cols)
    placeholders = ", ".join(f"%({k})s" for k in cols)
    updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in cols if k != "case_id")
    with pg() as conn:
        conn.execute(f"INSERT INTO cases ({names}) VALUES ({placeholders}) "
                     f"ON CONFLICT (case_id) DO UPDATE SET {updates}, updated_at = now()", cols)


def _save(state: "InvestigationState", **fields: Any) -> None:
    if not state.get("eval_mode"):
        save_case(state["case_id"], state["txn_id"], **fields)


def total_cost(state: InvestigationState) -> float:
    return round(sum(u["cost_usd"] for u in state.get("usage", [])), 6)


def evidence_log(messages: list[AnyMessage]) -> str:
    """Render tool results with their ids, so the reporter can cite them and we can verify."""
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            parts.append(f"### tool_call_id={m.tool_call_id} tool={m.name}\n{m.content}")
    notes = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    if notes is not None:
        parts.append(f"### Investigator findings note\n{notes.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- nodes

def load_context(state: InvestigationState) -> dict:
    ctx = core.get_transaction(state["txn_id"])
    _save(state, status="investigating")
    return {"context": ctx, "started_at": time.time()}


def triage(state: InvestigationState) -> dict:
    s = get_settings()
    ctx = state["context"]
    flag = ctx.get("flag") or {"reasons": [], "score": 0}
    sig = signature(flag["reasons"], ctx["transaction"]["amount"])
    cache = SemanticCache() if state.get("use_cache", True) else None

    if cache and (hit := _safe(lambda: cache.get(sig))):
        return {"triage": hit, "cache_hit": True}

    llm = chat(s.triage_model, max_tokens=4000).with_structured_output(
        TriageResult, method="json_schema", include_raw=True)
    out = llm.invoke([SystemMessage(prompts.TRIAGE_SYSTEM),
                      HumanMessage(json.dumps(ctx, default=str))])
    result: TriageResult | None = out["parsed"]
    if result is None:   # parse failure or refusal: fail safe, i.e. escalate
        result = TriageResult(severity="high", suspected_pattern="none", investigate=True,
                              rationale="Triage output unavailable; escalated by default.")
    elif cache:
        _safe(lambda: cache.put(sig, result.model_dump()))
    return {"triage": result.model_dump(), "cache_hit": False,
            "usage": [usage_of(out["raw"], s.triage_model)]}


def close_benign(state: InvestigationState) -> dict:
    t = state["triage"]
    cf = CaseFile(txn_id=state["txn_id"], verdict="legitimate", fraud_pattern="none",
                  confidence=0.6, summary=f"Closed at triage: {t['rationale']}", evidence=[],
                  recommended_action="no_action", account_to_action=None, playbook_refs=[])
    return {"case_file": cf.model_dump(), "grounding": {"claims": 0, "grounded": 0, "dropped": 0}}


def investigator(state: InvestigationState) -> dict:
    s = get_settings()
    new: list[AnyMessage] = []
    if not state.get("messages"):
        brief = (f"Flagged transaction: {state['txn_id']}\n"
                 f"Detector output: {json.dumps(state['context'].get('flag'), default=str)}\n"
                 f"Triage: {json.dumps(state['triage'])}")
        new.append(HumanMessage(brief))
    llm = chat(s.investigator_model, effort="high").bind_tools(INVESTIGATION_TOOLS)
    response = llm.invoke([SystemMessage(prompts.INVESTIGATOR_SYSTEM),
                           *state.get("messages", []), *new])
    if is_refusal(response):
        console.print(f"[yellow]Investigator refused on {state['txn_id']}; continuing to reporter")
    return {"messages": [*new, response], "usage": [usage_of(response, s.investigator_model)]}


def route_investigator(state: InvestigationState) -> str:
    last = state["messages"][-1]
    rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    if isinstance(last, AIMessage) and last.tool_calls and rounds <= get_settings().max_investigator_steps:
        return "tools"
    return "reporter"


def reporter(state: InvestigationState) -> dict:
    s = get_settings()
    llm = chat(s.reporter_model).with_structured_output(
        CaseFile, method="json_schema", include_raw=True)
    out = llm.invoke([
        SystemMessage(prompts.REPORTER_SYSTEM),
        HumanMessage(f"txn_id: {state['txn_id']}\nTriage: {json.dumps(state['triage'])}\n\n"
                     f"# Evidence log\n{evidence_log(state['messages'])}"),
    ])
    cf: CaseFile | None = out["parsed"]
    if cf is None:
        cf = CaseFile(txn_id=state["txn_id"], verdict="suspicious", fraud_pattern="none",
                      confidence=0.0, summary="Report generation failed; needs manual review.",
                      evidence=[], recommended_action="monitor", account_to_action=None,
                      playbook_refs=[])
    return {"case_file": cf.model_dump(), "usage": [usage_of(out["raw"], s.reporter_model)]}


def grounding_check(state: InvestigationState) -> dict:
    """Deterministic guardrail: drop claims citing non-existent tool calls; enforce policy."""
    cf = dict(state["case_file"])
    valid_ids = {m.tool_call_id for m in state.get("messages", []) if isinstance(m, ToolMessage)}
    claims = cf["evidence"]
    grounded = [e for e in claims if e["tool_call_id"] in valid_ids]
    cf["evidence"] = grounded
    cf["confidence"] = min(1.0, max(0.0, cf["confidence"]))
    notes = []
    if cf["verdict"] == "fraud" and len(grounded) < 2:
        cf["verdict"] = "suspicious"
        notes.append("downgraded fraud->suspicious: fewer than 2 grounded evidence items")
    if cf["verdict"] == "legitimate" and cf["recommended_action"] in ACTIONS_NEEDING_APPROVAL:
        cf["recommended_action"] = "monitor"
        notes.append("action downgraded: legitimate verdict cannot freeze/block")
    return {"case_file": cf, "grounding": {
        "claims": len(claims), "grounded": len(grounded), "dropped": len(claims) - len(grounded),
        "notes": notes, "agent_latency_s": round(time.time() - state["started_at"], 2)}}


def route_after_grounding(state: InvestigationState) -> str:
    return ("human_review" if state["case_file"]["recommended_action"] in ACTIONS_NEEDING_APPROVAL
            else "finalize")


def human_review(state: InvestigationState) -> dict:
    cf = state["case_file"]
    _save(state, status="pending_review",
              severity=state["triage"]["severity"], case_file=cf,
              cost_usd=total_cost(state), latency_s=state["grounding"].get("agent_latency_s"))
    # Pauses the graph. Resumed via graph.invoke(Command(resume={...}), config)
    decision = interrupt({"case_id": state["case_id"], "case_file": cf})
    return {"decision": decision}


def finalize(state: InvestigationState) -> dict:
    cf, decision = state["case_file"], state.get("decision")
    if decision is None:
        outcome = "auto_closed"
    else:
        outcome = "approved" if decision.get("approved") else "rejected"
        if decision.get("approved") and cf["recommended_action"] == "freeze_account" \
                and cf.get("account_to_action"):
            with pg() as conn:
                conn.execute("UPDATE accounts SET status = 'frozen' WHERE account_id = %s",
                             (cf["account_to_action"],))
    if state.get("eval_mode"):
        return {}
    _save(state, status="closed",
              severity=state["triage"]["severity"], case_file=cf, decision=outcome,
              reviewer_note=(decision or {}).get("note"), cost_usd=total_cost(state),
              latency_s=state.get("grounding", {}).get("agent_latency_s"))
    # Long-term memory: future investigations can find this case and the human's decision
    if cf["verdict"] != "legitimate" or decision is not None:
        _safe(lambda: retrieval.remember_case(
            state["case_id"], cf["summary"],
            {"verdict": cf["verdict"], "fraud_pattern": cf["fraud_pattern"], "decision": outcome}))
    return {}


def _safe(fn):
    """Cache/memory are optimisations: never let them fail an investigation."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]non-fatal: {e}")
        return None


# ---------------------------------------------------------------- graph

def build_graph() -> StateGraph:
    g = StateGraph(InvestigationState)
    g.add_node("load_context", load_context)
    g.add_node("triage", triage)
    g.add_node("close_benign", close_benign)
    g.add_node("investigator", investigator)
    g.add_node("tools", ToolNode(INVESTIGATION_TOOLS, handle_tool_errors=True))
    g.add_node("reporter", reporter)
    g.add_node("grounding_check", grounding_check)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "triage")
    g.add_conditional_edges("triage", lambda s: "investigator" if s["triage"]["investigate"]
                            else "close_benign", ["investigator", "close_benign"])
    g.add_edge("close_benign", "finalize")
    g.add_conditional_edges("investigator", route_investigator, ["tools", "reporter"])
    g.add_edge("tools", "investigator")
    g.add_edge("reporter", "grounding_check")
    g.add_conditional_edges("grounding_check", route_after_grounding, ["human_review", "finalize"])
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g


@contextmanager
def open_graph():
    """Compiled graph with a Postgres checkpointer (needed for interrupt/resume across processes).
    A connection pool makes the compiled graph safe to call from several threads."""
    with ConnectionPool(get_settings().pg_dsn, max_size=10, kwargs={
            "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}) as pool:
        saver = PostgresSaver(pool)
        saver.setup()
        yield build_graph().compile(checkpointer=saver)


def case_id_for(txn_id: str) -> str:
    return f"CASE-{txn_id}"


def _config(case_id: str) -> dict:
    return {"configurable": {"thread_id": case_id}, "recursion_limit": 80,
            "metadata": {"prompt_version": prompts.PROMPT_VERSION}, "tags": ["investigation"]}


def investigate(graph, txn_id: str, case_id: str | None = None, use_cache: bool = True,
                eval_mode: bool = False) -> dict:
    case_id = case_id or case_id_for(txn_id)
    return graph.invoke({"txn_id": txn_id, "case_id": case_id, "use_cache": use_cache,
                         "eval_mode": eval_mode}, _config(case_id))


def resume(graph, case_id: str, decision: dict) -> dict:
    return graph.invoke(Command(resume=decision), _config(case_id))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Investigate one flagged transaction")
    ap.add_argument("txn_id")
    a = ap.parse_args()
    with open_graph() as graph:
        result = investigate(graph, a.txn_id)
        console.print_json(json.dumps(result.get("case_file"), default=str))
        console.print(f"grounding={result.get('grounding')} cost=${total_cost(result)}")
        if "__interrupt__" in result:
            console.print("[bold yellow]Paused for human review (approve in the UI or via the API)")
