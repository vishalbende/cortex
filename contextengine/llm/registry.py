"""Auto-detect the right LLMClient for a given model string."""
from __future__ import annotations

from typing import Any

from contextengine.llm.anthropic import AnthropicClient
from contextengine.llm.base import LLMClient
from contextengine.llm.openai import OpenAIClient

_ANTHROPIC_PREFIXES = ("claude", "anthropic.", "anthropic/")
_OPENAI_PREFIXES = (
    "gpt",
    "openai.",
    "openai/",
    "o1",
    "o3",
    "o4",
    "chatgpt",
)


def detect_provider(model: str) -> str:
    """Return 'anthropic' | 'openai' | 'unknown' for a model id."""
    m = model.lower().strip()
    if m.startswith(_ANTHROPIC_PREFIXES):
        return "anthropic"
    if m.startswith(_OPENAI_PREFIXES):
        return "openai"
    return "unknown"


def client_for_model(
    model: str,
    *,
    anthropic_client: Any = None,
    openai_client: Any = None,
    default: str = "anthropic",
) -> LLMClient:
    """Build the appropriate LLMClient for the model.

    Detects the provider from the model prefix; caller may pass an
    already-constructed provider SDK client for either side. Unknown
    models fall back to `default` (anthropic unless overridden).
    """
    provider = detect_provider(model)
    if provider == "unknown":
        provider = default

    if provider == "anthropic":
        return AnthropicClient(client=anthropic_client)
    if provider == "openai":
        return OpenAIClient(client=openai_client)
    raise ValueError(f"Unsupported provider: {provider!r}")
