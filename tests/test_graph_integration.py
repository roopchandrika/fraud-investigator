"""Graph integration test with a scripted fake LLM: exercises real tools, Postgres checkpointing,
the human-review interrupt, resume, and the freeze action - without any API cost.

Needs the docker stack + backfill (skipped otherwise):
    docker compose up -d --wait postgres neo4j redis
"""
import re

import pytest
from langchain_core.messages import AIMessage

from fraud_agent.agents import graph as g
from fraud_agent.db import pg
from fraud_agent.schemas import CaseFile, TriageResult


def _db_ready() -> bool:
    try:
        with pg() as conn:
            return conn.execute("SELECT count(*) AS n FROM flags").fetchone()["n"] > 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="needs docker stack + backfill")
USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


class _Structured:
    def __init__(self, schema, txn_id):
        self.schema, self.txn_id = schema, txn_id

    def invoke(self, messages):
        if self.schema is TriageResult:
            parsed = TriageResult(severity="high", suspected_pattern="mule_chain", investigate=True,
                                  rationale="pass-through")
        else:  # CaseFile: cite the real tool_call_ids present in the evidence log, plus one fake
            ids = re.findall(r"tool_call_id=(\S+) tool=(\S+)", messages[-1].content)
            evidence = [{"claim": f"fact from {name}", "source_tool": name, "tool_call_id": i}
                        for i, name in ids] + [{"claim": "made up", "source_tool": "x",
                                                "tool_call_id": "toolu_fabricated"}]
            parsed = CaseFile(txn_id=self.txn_id, verdict="fraud", fraud_pattern="mule_chain",
                              confidence=0.9, summary="Funds passed through a mule.",
                              evidence=evidence, recommended_action="freeze_account",
                              account_to_action="__PAYEE__", playbook_refs=[])
        return {"parsed": parsed, "raw": AIMessage("", usage_metadata=USAGE)}


class _Tools:
    def __init__(self, txn_id):
        self.txn_id = txn_id

    def invoke(self, messages):
        if not any(isinstance(m, AIMessage) for m in messages):
            return AIMessage("", usage_metadata=USAGE, tool_calls=[
                {"name": "get_transaction", "args": {"txn_id": self.txn_id}, "id": "toolu_1"},
                {"name": "trace_money_flow", "args": {"txn_id": self.txn_id}, "id": "toolu_2"}])
        return AIMessage("Findings: mule chain.", usage_metadata=USAGE)


class FakeChat:
    txn_id = ""

    def __init__(self, model, **_):
        pass

    def with_structured_output(self, schema, **_):
        return _Structured(schema, FakeChat.txn_id)

    def bind_tools(self, _tools):
        return _Tools(FakeChat.txn_id)


def test_full_flow_with_interrupt_and_resume(monkeypatch):
    with pg() as conn:
        row = conn.execute("""SELECT f.txn_id, t.payee_id FROM flags f JOIN transactions t
                              USING (txn_id) WHERE t.fraud_pattern = 'mule_chain' LIMIT 1""").fetchone()
        conn.execute("DELETE FROM cases WHERE case_id = %s", (f"TEST-{row['txn_id']}",))
        conn.execute("UPDATE accounts SET status = 'active' WHERE account_id = %s", (row["payee_id"],))
    FakeChat.txn_id = row["txn_id"]
    monkeypatch.setattr(g, "chat", FakeChat)
    monkeypatch.setattr(g.retrieval, "remember_case", lambda *a, **k: None)
    real_reporter = g.reporter

    def reporter_with_payee(state):
        out = real_reporter(state)
        out["case_file"]["account_to_action"] = row["payee_id"]
        return out
    monkeypatch.setattr(g, "reporter", reporter_with_payee)

    case_id = f"TEST-{row['txn_id']}"
    with g.open_graph() as graph:
        result = g.investigate(graph, row["txn_id"], case_id, use_cache=False)
        assert "__interrupt__" in result                         # paused for human review
        assert result["grounding"]["dropped"] == 1               # fabricated citation removed
        assert result["case_file"]["verdict"] == "fraud"         # 2 grounded items remain
        with pg() as conn:
            assert conn.execute("SELECT status FROM cases WHERE case_id = %s",
                                (case_id,)).fetchone()["status"] == "pending_review"

        g.resume(graph, case_id, {"approved": True, "reviewer": "test", "note": "ok"})

    with pg() as conn:
        case = conn.execute("SELECT * FROM cases WHERE case_id = %s", (case_id,)).fetchone()
        acct = conn.execute("SELECT status FROM accounts WHERE account_id = %s",
                            (row["payee_id"],)).fetchone()
        conn.execute("DELETE FROM cases WHERE case_id = %s", (case_id,))
        conn.execute("UPDATE accounts SET status = 'active' WHERE account_id = %s", (row["payee_id"],))
    assert case["status"] == "closed" and case["decision"] == "approved"
    assert case["cost_usd"] > 0
    assert acct["status"] == "frozen"
