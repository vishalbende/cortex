"""AnthropicClient — wraps `anthropic.AsyncAnthropic` behind the LLMClient protocol."""
from __future__ import annotations

from typing import Any

from contextengine.llm.base import LLMResponse


class AnthropicClient:
    """LLMClient for Claude models.

    Marks `stable_prefix` as `cache_control: ephemeral` so the invariant
    part of the prompt hits the prompt cache across repeated calls.
    """

    def __init__(self, client: Any = None) -> None:
        self._client = client

    async def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stable_prefix: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = await self._ensure_client()

        sys_blocks: list[dict[str, Any]] = []
        if system:
            sys_blocks.append({"type": "text", "text": system})
        if stable_prefix:
            sys_blocks.append(
                {
                    "type": "text",
                    "text": stable_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if sys_blocks:
            kwargs["system"] = sys_blocks

        response = await client.messages.create(**kwargs)
        text = "".join(
            getattr(b, "text", "") for b in response.content if getattr(b, "type", "text") == "text"
        )
        if not text and response.content:
            text = getattr(response.content[0], "text", "")
        return LLMResponse(text=text)
