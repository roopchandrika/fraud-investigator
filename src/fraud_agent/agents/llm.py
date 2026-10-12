"""Model factory + cost accounting. One place to change models, routing and pricing."""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from fraud_agent.config import get_settings

# USD per 1M tokens (input, output). Keep in sync with https://www.anthropic.com/pricing
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Models that support server-side refusal fallbacks (Claude API only)
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")


def chat(model: str, max_tokens: int = 16000, effort: str | None = None) -> ChatAnthropic:
    kwargs: dict = {}
    if model.startswith(_FALLBACK_PREFIXES):
        # If a safety classifier declines a request (e.g. a false positive on fraud language),
        # the API re-runs it on a recommended fallback model instead of returning a refusal.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["model_kwargs"] = {"fallbacks": "default"}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    if api_key := get_settings().anthropic_api_key:
        kwargs["api_key"] = api_key
    return ChatAnthropic(model=model, max_tokens=max_tokens, max_retries=3, **kwargs)


def usage_of(msg: AIMessage | None, model: str) -> dict:
    um = (msg.usage_metadata if msg is not None else None) or {}
    tin, tout = um.get("input_tokens", 0), um.get("output_tokens", 0)
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return {"model": model, "input_tokens": tin, "output_tokens": tout,
            "cost_usd": round((tin * pin + tout * pout) / 1e6, 6)}


def is_refusal(msg: AIMessage) -> bool:
    return (msg.response_metadata or {}).get("stop_reason") == "refusal"
