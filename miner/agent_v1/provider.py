from __future__ import annotations

import os


CHUTES_API_BASE = "https://llm.chutes.ai/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
CHA86_API_BASE = "https://api.cha86.dev/v1"

_OPENAI_COMPATIBLE = frozenset({"cha86", "openai", "openai-compatible"})


def normalize_provider(provider: str | None, *, api_base: str = "") -> str:
    normalized = str(provider or "").strip().lower()
    if normalized in {"openai-compatible", "openai_compatible"}:
        return "openai"
    if normalized:
        return normalized
    lowered_base = api_base.lower()
    if "cha86" in lowered_base:
        return "cha86"
    if "chutes.ai" in lowered_base:
        return "chutes"
    if "openrouter.ai" in lowered_base:
        return "openrouter"
    return "openrouter"


def provider_api_base(provider: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    resolved = normalize_provider(provider)
    if resolved == "chutes":
        return os.getenv("CHUTES_API_BASE", CHUTES_API_BASE).strip() or CHUTES_API_BASE
    if resolved in _OPENAI_COMPATIBLE or resolved == "cha86":
        if resolved == "cha86":
            return os.getenv("CHA86_API_BASE", CHA86_API_BASE).strip() or CHA86_API_BASE
        return (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("CHA86_API_BASE", CHA86_API_BASE)
        ).strip() or CHA86_API_BASE
    return os.getenv("OPENROUTER_API_BASE", OPENROUTER_API_BASE).strip() or OPENROUTER_API_BASE


def provider_api_key_env(provider: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    resolved = normalize_provider(provider)
    if resolved == "chutes":
        return "CHUTES_API_KEY"
    if resolved == "cha86":
        return "CHA86_API_KEY"
    if resolved in _OPENAI_COMPATIBLE:
        return "OPENAI_API_KEY"
    return "OPENROUTER_API_KEY"


def dspy_model_id(model: str, *, provider: str | None = None, api_base: str = "") -> str:
    normalized = model.strip()
    resolved_provider = normalize_provider(provider, api_base=api_base)
    if resolved_provider == "chutes":
        # DSPy routes custom OpenAI-compatible endpoints through LiteLLM's
        # openai provider. Chutes catalog IDs remain unchanged after the prefix.
        if normalized.startswith("chutes/"):
            normalized = normalized[len("chutes/") :]
        if normalized.startswith("openai/") and normalized.count("/") >= 2:
            return normalized
        return f"openai/{normalized}"
    if resolved_provider in _OPENAI_COMPATIBLE or resolved_provider == "cha86":
        if normalized.startswith("openai/"):
            return normalized
        if normalized.startswith("cha86/"):
            normalized = normalized[len("cha86/") :]
        return f"openai/{normalized}"
    if resolved_provider == "openrouter" and normalized and not normalized.startswith(
        "openrouter/"
    ):
        return f"openrouter/{normalized}"
    return normalized
