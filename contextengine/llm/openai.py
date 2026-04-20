"""OpenAIClient — wraps `openai.AsyncOpenAI` behind the LLMClient protocol."""
from __future__ import annotations

from typing import Any

from contextengine.llm.base import LLMResponse


class OpenAIClient:
    """LLMClient for GPT / o-series models.

    OpenAI does not accept explicit cache-control markers. Instead it
    does automatic prefix caching (128+ tokens, ~5 minute TTL). This
    client concatenates `stable_prefix` *after* `system` so the
    invariant block sits in the natural cache position.

    When `json_mode=True`, passes `response_format={"type":"json_object"}`
    to force valid JSON output.
    """

    def __init__(self, client: Any = None) -> None:
        self._client = client

    async def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "OpenAIClient requires the `openai` package. "
                    "Install with: pip install 'contextengine[openai]' "
                    "or pip install openai"
                ) from exc
            self._client = openai.AsyncOpenAI()
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

        sys_parts: list[str] = []
        if system:
            sys_parts.append(system)
        if stable_prefix:
            sys_parts.append(stable_prefix)
        full_system = "\n\n".join(sys_parts)

        messages: list[dict[str, Any]] = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": user})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return LLMResponse(text=content)
