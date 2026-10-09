"""Analyst console: review the agent's case files and approve or reject actions.

    uv run streamlit run src/fraud_agent/ui/app.py
"""
import httpx
import pandas as pd
import streamlit as st

from fraud_agent.config import get_settings

API = get_settings().api_url
st.set_page_config(page_title="Fraud Analyst Console", layout="wide")
st.title("Fraud Analyst Console")

try:
    m = httpx.get(f"{API}/metrics", timeout=5).json()
except httpx.HTTPError:
    st.error(f"API not reachable at {API}. Start it with: uv run uvicorn fraud_agent.api.main:app")
    st.stop()

cols = st.columns(5)
cols[0].metric("Cases", m["cases"])
cols[1].metric("Awaiting review", m["pending_review"])
cols[2].metric("Avg cost / case", f"${m['avg_cost_usd'] or 0}")
cols[3].metric("p50 / p95 latency", f"{m['p50_latency_s'] or '-'}s / {m['p95_latency_s'] or '-'}s")
cols[4].metric("Approved / rejected", f"{m['approved']} / {m['rejected']}")

review_tab, all_tab = st.tabs(["Review queue", "All cases"])

with review_tab:
    pending = httpx.get(f"{API}/cases", params={"status": "pending_review"}).json()
    if not pending:
        st.info("Nothing awaiting review.")
    for c in pending:
        case = httpx.get(f"{API}/cases/{c['case_id']}").json()
        cf = case["case_file"]
        with st.expander(f"[{c['severity']}] {c['case_id']}: {cf['verdict'].upper()} / "
                         f"{cf['fraud_pattern']} -> {cf['recommended_action']} "
                         f"on {cf['account_to_action']}", expanded=True):
            st.write(cf["summary"])
            st.caption(f"confidence {cf['confidence']:.2f} · cost ${c['cost_usd']} · "
                       f"{c['latency_s']}s · playbook: {', '.join(cf['playbook_refs']) or '-'}")
            st.dataframe(pd.DataFrame(cf["evidence"]), use_container_width=True, hide_index=True)
            note = st.text_input("Reviewer note", key=f"note-{c['case_id']}")
            a, r, _ = st.columns([1, 1, 6])
            for col, approved, label in ((a, True, "Approve"), (r, False, "Reject")):
                if col.button(label, key=f"{label}-{c['case_id']}", type="primary" if approved else "secondary"):
                    httpx.post(f"{API}/cases/{c['case_id']}/decision",
                               json={"approved": approved, "note": note}, timeout=120)
                    st.rerun()

with all_tab:
    rows = httpx.get(f"{API}/cases", params={"limit": 500}).json()
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
