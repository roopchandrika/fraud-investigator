"""Shared data contracts. Everything crossing a boundary (Kafka, DB, LLM, API) is typed here."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Channel = Literal["upi", "card", "wallet"]
FraudPattern = Literal[
    "velocity_burst", "mule_chain", "merchant_collusion", "account_takeover", "none"
]


class Transaction(BaseModel):
    txn_id: str
    ts: datetime
    payer_id: str
    payee_id: str
    merchant_id: str | None = None
    amount: float
    channel: Channel
    device_id: str
    city: str
    # Ground truth (generator only). Stripped before anything reaches an agent.
    is_fraud: bool = False
    fraud_pattern: str | None = None


class Flag(BaseModel):
    txn_id: str
    score: float = Field(description="Combined anomaly score, 0..1, higher = more suspicious")
    reasons: list[str]


# ---------- LLM structured outputs ----------
# Every field is required (nullable where optional) and range checks live in code, not in the
# schema: strict JSON-schema decoding works best with plain, fully-required schemas.

class TriageResult(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    suspected_pattern: FraudPattern
    investigate: bool = Field(description="False only if the flag is clearly benign noise")
    rationale: str = Field(description="One or two sentences")


class Evidence(BaseModel):
    claim: str = Field(description="A single factual statement supporting the verdict")
    source_tool: str = Field(description="Name of the tool whose output supports the claim")
    tool_call_id: str = Field(description="The exact tool_call_id of that tool result")


class CaseFile(BaseModel):
    txn_id: str
    verdict: Literal["fraud", "suspicious", "legitimate"]
    fraud_pattern: FraudPattern
    confidence: float = Field(description="0.0 to 1.0")
    summary: str = Field(description="3-5 sentence analyst-ready summary")
    evidence: list[Evidence] = Field(description="Every claim must cite a real tool_call_id")
    recommended_action: Literal["freeze_account", "block_transaction", "monitor", "no_action"]
    account_to_action: str | None = Field(description="Account id the action applies to, or null")
    playbook_refs: list[str] = Field(description="Playbook 'ref' values relied on")


class ReviewDecision(BaseModel):
    approved: bool
    reviewer: str = "analyst"
    note: str = ""


class JudgeScore(BaseModel):
    faithfulness: int = Field(description="1-5")
    actionability: int = Field(description="1-5")
    reason: str = Field(description="One sentence")
