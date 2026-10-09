"""End-to-end agent evals: verdict quality, grounding, LLM-as-judge, cost and latency.

    uv run python -m evals.run_evals --limit 10            # quick smoke run
    uv run python -m evals.run_evals --min-f1 0.8          # CI gate: exit 1 below threshold
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.table import Table

from fraud_agent.agents import prompts
from fraud_agent.agents.graph import evidence_log, investigate, open_graph, total_cost
from fraud_agent.agents.llm import chat, usage_of
from fraud_agent.config import get_settings
from fraud_agent.schemas import JudgeScore

console = Console()
RESULTS = Path("evals/results")


def judge(log: str, summary: str) -> tuple[JudgeScore | None, dict]:
    model = get_settings().judge_model
    llm = chat(model, max_tokens=4000).with_structured_output(
        JudgeScore, method="json_schema", include_raw=True)
    out = llm.invoke([SystemMessage(prompts.JUDGE_SYSTEM),
                      HumanMessage(f"# Evidence log\n{log}\n\n# Case summary\n{summary}")])
    return out["parsed"], usage_of(out["raw"], model)


def run_case(graph, case: dict, run_id: str, use_cache: bool) -> dict:
    t0 = time.time()
    try:
        state = investigate(graph, case["txn_id"], f"EVAL-{run_id}-{case['txn_id']}",
                            use_cache=use_cache, eval_mode=True)
    except Exception as e:  # noqa: BLE001
        return {**case, "error": str(e)}
    cf = state["case_file"]
    score, judge_usage = (None, {"cost_usd": 0})
    if state.get("messages"):   # skip judging cases closed at triage (no evidence log)
        score, judge_usage = judge(evidence_log(state["messages"]), cf["summary"])
    return {
        **case,
        "verdict": cf["verdict"], "pattern": cf["fraud_pattern"],
        "action": cf["recommended_action"], "confidence": cf["confidence"],
        "predicted_positive": cf["verdict"] in ("fraud", "suspicious"),
        "grounding": state.get("grounding", {}), "cache_hit": state.get("cache_hit", False),
        "tool_calls": sum(len(getattr(m, "tool_calls", []) or []) for m in state.get("messages", [])),
        "judge": score.model_dump() if score else None,
        "agent_cost_usd": total_cost(state), "judge_cost_usd": judge_usage["cost_usd"],
        "latency_s": round(time.time() - t0, 2),
    }


def summarise(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    tp = sum(r["predicted_positive"] and r["label_is_fraud"] for r in ok)
    fp = sum(r["predicted_positive"] and not r["label_is_fraud"] for r in ok)
    fn = sum(not r["predicted_positive"] and r["label_is_fraud"] for r in ok)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    fraud = [r for r in ok if r["label_is_fraud"]]
    legit = [r for r in ok if not r["label_is_fraud"]]
    claims = sum(r["grounding"].get("claims", 0) for r in ok)
    judged = [r["judge"] for r in ok if r["judge"]]
    lat = sorted(r["latency_s"] for r in ok)
    return {
        "n": len(results), "errors": len(results) - len(ok),
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
        "pattern_accuracy": round(sum(r["pattern"] == r["label_pattern"] for r in fraud)
                                  / max(len(fraud), 1), 3),
        "false_action_rate": round(sum(r["action"] in ("freeze_account", "block_transaction")
                                       for r in legit) / max(len(legit), 1), 3),
        "ungrounded_claim_rate": round(sum(r["grounding"].get("dropped", 0) for r in ok)
                                       / max(claims, 1), 3),
        "judge_faithfulness": round(statistics.mean(j["faithfulness"] for j in judged), 2) if judged else None,
        "judge_actionability": round(statistics.mean(j["actionability"] for j in judged), 2) if judged else None,
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in ok), 1) if ok else None,
        "avg_cost_usd": round(statistics.mean(r["agent_cost_usd"] for r in ok), 4) if ok else None,
        "p50_latency_s": lat[len(lat) // 2] if lat else None,
        "p95_latency_s": lat[int(len(lat) * 0.95) - 1] if lat else None,
        "cache_hits": sum(r["cache_hit"] for r in ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evals/dataset.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--use-cache", action="store_true", help="allow semantic-cache hits in triage")
    ap.add_argument("--min-f1", type=float, default=0.0, help="exit 1 if F1 is below this")
    args = ap.parse_args()

    cases = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    if args.limit:
        # Round-robin across labels so a small smoke run still covers every pattern
        by_label: dict[str, list] = {}
        for c in cases:
            by_label.setdefault(c["label_pattern"], []).append(c)
        mixed = [c for group in zip(*by_label.values()) for c in group]
        cases = (mixed + [c for c in cases if c not in mixed])[: args.limit]
    run_id = time.strftime("%Y%m%d-%H%M%S")
    console.print(f"Running {len(cases)} cases (run {run_id})...")

    with open_graph() as graph, ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda c: run_case(graph, c, run_id, args.use_cache), cases))

    summary = summarise(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{run_id}.json").write_text(json.dumps({"summary": summary, "results": results},
                                                       indent=2, default=str))
    table = Table(title=f"Agent eval - {run_id} - prompt {prompts.PROMPT_VERSION}")
    table.add_column("metric")
    table.add_column("value")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)
    (RESULTS / "latest.md").write_text("| metric | value |\n|---|---|\n" +
                                       "\n".join(f"| {k} | {v} |" for k, v in summary.items()) + "\n")

    if summary["f1"] < args.min_f1:
        console.print(f"[red]F1 {summary['f1']} < required {args.min_f1}")
        sys.exit(1)


if __name__ == "__main__":
    main()
