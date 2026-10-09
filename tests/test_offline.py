"""Fast, offline unit tests (no Docker, no API key)."""
import random
from datetime import UTC, datetime, timedelta

from langchain_core.messages import ToolMessage

from fraud_agent.agents.graph import grounding_check, route_after_grounding
from fraud_agent.detector.features import FeatureState, rule_hits
from fraud_agent.generator.patterns import account_takeover, mule_chain, normal_txn, velocity_burst
from fraud_agent.generator.world import build_world
from fraud_agent.tools.semantic_cache import signature

T0 = datetime(2026, 9, 1, tzinfo=UTC)


def warm_state(world, rng, n=6000):
    state, ts = FeatureState(), T0
    for _ in range(n):
        ts += timedelta(seconds=2)
        state.features(normal_txn(world, rng, ts))
    return state, ts


def test_ids_are_deterministic():
    a = [normal_txn(build_world(), random.Random(1), T0).txn_id for _ in range(3)]
    b = [normal_txn(build_world(), random.Random(1), T0).txn_id for _ in range(3)]
    assert a == b


def test_velocity_burst_triggers_rules():
    world, rng = build_world(), random.Random(3)
    state, ts = warm_state(world, rng)
    reasons = [rule_hits(state.features(t)) for t in velocity_burst(world, rng, ts)]
    assert any(any(r.startswith("velocity") for r in rs) for rs in reasons)


def test_mule_chain_triggers_pass_through():
    world, rng = build_world(), random.Random(4)
    state, ts = warm_state(world, rng)
    reasons = [rule_hits(state.features(t)) for t in mule_chain(world, rng, ts)]
    assert any(any(r.startswith("pass-through") for r in rs) for rs in reasons[1:])


def test_takeover_triggers_rules():
    world, rng = build_world(), random.Random(5)
    state, ts = warm_state(world, rng)
    for _ in range(20):   # retry: the victim must have >=5 sends of history
        t = account_takeover(world, rng, ts)[0]
        if state.accounts[t.payer_id].n_sent >= 5:
            assert rule_hits(state.features(t))
            return


def test_normal_traffic_rarely_flags():
    world, rng = build_world(), random.Random(6)
    state, ts = warm_state(world, rng)
    flagged = 0
    for _ in range(3000):
        ts += timedelta(seconds=2)
        flagged += bool(rule_hits(state.features(normal_txn(world, rng, ts))))
    assert flagged / 3000 < 0.03


def _state(evidence, verdict="fraud", action="freeze_account"):
    return {
        "started_at": 0.0,
        "messages": [ToolMessage(content="{}", tool_call_id="toolu_real", name="get_transaction")],
        "case_file": {"txn_id": "TX1", "verdict": verdict, "fraud_pattern": "mule_chain",
                      "confidence": 1.4, "summary": "s", "evidence": evidence,
                      "recommended_action": action, "account_to_action": "AC1", "playbook_refs": []},
    }


def test_grounding_drops_fabricated_citations_and_downgrades():
    ev = [{"claim": "a", "source_tool": "get_transaction", "tool_call_id": "toolu_real"},
          {"claim": "b", "source_tool": "trace_money_flow", "tool_call_id": "toolu_FAKE"}]
    out = grounding_check(_state(ev))
    assert out["grounding"]["dropped"] == 1
    assert out["case_file"]["verdict"] == "suspicious"       # <2 grounded items
    assert out["case_file"]["confidence"] == 1.0              # clamped


def test_legitimate_verdict_cannot_freeze():
    out = grounding_check(_state([], verdict="legitimate", action="freeze_account"))
    assert out["case_file"]["recommended_action"] == "monitor"
    assert route_after_grounding(out) == "finalize"


def test_cache_signature_ignores_numbers():
    a = signature(["velocity: 7 payments by payer in 5 min"], 9000)
    b = signature(["velocity: 11 payments by payer in 5 min"], 8000)
    assert a == b
