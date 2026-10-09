"""MCP server exposing the investigation tools to any MCP client (Claude Desktop, Claude Code, IDEs).

    uv run python -m fraud_agent.mcp_server          # stdio transport
Read-only by design: no tool here can freeze accounts or change data.
"""
from mcp.server.mcpserver import MCPServer

from fraud_agent.tools import core, retrieval

mcp = MCPServer(
    "fraud-investigation",
    instructions="Read-only tools for investigating flagged payment transactions: transaction "
                 "lookup, account baselines, payment graph, money-flow tracing, merchant risk, "
                 "and the fraud playbook.",
)


@mcp.tool()
def get_transaction(txn_id: str) -> dict:
    """Fetch a transaction and the anomaly detector's flag reasons."""
    return core.get_transaction(txn_id)


@mcp.tool()
def get_account_profile(account_id: str) -> dict:
    """KYC, account age, status and 30-day behavioural baseline for an account or merchant."""
    return core.get_account_profile(account_id)


@mcp.tool()
def get_account_activity(account_id: str, around_ts: str, hours_before: float = 24,
                         hours_after: float = 2) -> dict:
    """Payments sent/received by an account around an ISO-8601 timestamp."""
    return core.get_account_activity(account_id, around_ts, hours_before, hours_after)


@mcp.tool()
def get_counterparty_graph(account_id: str, around_ts: str, window_minutes: int = 120) -> dict:
    """1-hop payment graph with neighbours' fan-in/fan-out (mule hubs, collusion rings)."""
    return core.get_counterparty_graph(account_id, around_ts, window_minutes)


@mcp.tool()
def trace_money_flow(txn_id: str, max_hops: int = 5) -> dict:
    """Follow funds forward through onward payments of 80-100% of the amount."""
    return core.trace_money_flow(txn_id, max_hops)


@mcp.tool()
def get_merchant_risk(merchant_id: str) -> dict:
    """Merchant risk score, chargeback rate and 24h inflow pattern."""
    return core.get_merchant_risk(merchant_id)


@mcp.tool()
def search_fraud_playbook(query: str) -> list[dict]:
    """Search the fraud playbook (typologies, red flags, false positives, action policy)."""
    return retrieval.search_playbook(query)


if __name__ == "__main__":
    mcp.run()
