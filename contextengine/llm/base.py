"""LLMClient protocol — the minimal surface used by router, writer, catalog."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    """Return shape from LLMClient.complete().

    Future-proofs for usage/cache metadata without breaking call sites.
    """

    text: str


class LLMClient(Protocol):
    """Minimal, provider-agnostic completion API.

    `stable_prefix`: if the caller can split the prompt into a large
    invariant block plus a small variable suffix, `stable_prefix` holds
    the invariant block. Anthropic implementations should mark it with
    `cache_control: ephemeral` for prompt caching; OpenAI relies on
    automatic server-side prefix caching and concatenates instead.
    """

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stable_prefix: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse: ...
