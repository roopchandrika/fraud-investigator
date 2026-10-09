"""Semantic cache for triage decisions (Redis).

Flags with near-identical signatures ("velocity: # payments by payer in # min" etc.) get the same
triage outcome, so we skip the LLM call. Linear scan is fine for a demo; at scale use a Redis
vector index (RediSearch) or pgvector.
"""
import json
import re

import numpy as np
import redis

from fraud_agent.config import get_settings
from fraud_agent.tools.retrieval import embeddings

KEY = "triage_cache:v1"


def signature(reasons: list[str], amount: float) -> str:
    """Normalise numbers away so structurally identical flags collide."""
    norm = sorted(re.sub(r"\d+(\.\d+)?", "#", r) for r in reasons)
    bucket = "small" if amount < 2000 else "medium" if amount < 10000 else "large"
    return f"amount={bucket}; " + "; ".join(norm)


class SemanticCache:
    def __init__(self) -> None:
        s = get_settings()
        self.r = redis.Redis.from_url(s.redis_url)
        self.threshold = s.semantic_cache_threshold

    def get(self, sig: str) -> dict | None:
        q = np.array(embeddings().embed_query(sig))
        best, best_sim = None, -1.0
        for raw in self.r.lrange(KEY, 0, 499):
            entry = json.loads(raw)
            v = np.array(entry["vec"])
            sim = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))
            if sim > best_sim:
                best, best_sim = entry, sim
        if best and best_sim >= self.threshold:
            return {**best["value"], "_cache_similarity": round(best_sim, 4)}
        return None

    def put(self, sig: str, value: dict) -> None:
        vec = embeddings().embed_query(sig)
        self.r.lpush(KEY, json.dumps({"sig": sig, "vec": vec, "value": value}))
        self.r.ltrim(KEY, 0, 499)
