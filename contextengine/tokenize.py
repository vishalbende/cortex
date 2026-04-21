"""Token counting abstraction."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class CharEstimateTokenizer:
    """Fallback tokenizer: ~4 chars per token. Good enough for budget math."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


class TiktokenTokenizer:
    """Accurate tokenizer backed by tiktoken.

    Used for both OpenAI (native) and Claude (approximation). The `cl100k_base`
    encoding is a reasonable proxy for Claude tokenization — within ~5% for
    English text — and avoids a network call to Anthropic's server-side
    `count_tokens` endpoint.

    Install: `pip install tiktoken` (optional dep).
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "TiktokenTokenizer requires the `tiktoken` package. "
                "Install with: pip install tiktoken"
            ) from exc
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


class AnthropicTokenizer:
    """Native Anthropic server-side token counting via `messages.count_tokens`.

    Synchronous `count()` method calls the API through a thread-backed
    event loop so it can be dropped into the existing Tokenizer protocol.
    Caches results by text hash.

    Note: every call is a network request. Prefer `TiktokenTokenizer` for
    hot paths; use this for exactness on long stable prefixes.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        client: Any = None,
    ) -> None:
        self.model = model
        self._client = client
        self._cache: dict[int, int] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def count(self, text: str) -> int:
        if not text:
            return 0
        h = hash(text)
        if h in self._cache:
            return self._cache[h]
        client = self._get_client()
        result = client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        n = int(getattr(result, "input_tokens", 0) or result.get("input_tokens", 0))  # type: ignore[union-attr]
        self._cache[h] = n
        return n


class AsyncAnthropicTokenizer:
    """Async variant of AnthropicTokenizer for use in already-async code paths.

    `count(text)` returns an awaitable `int`. Calls go through
    `anthropic.AsyncAnthropic.messages.count_tokens` and are cached by
    text hash, so repeat content is free.

    This is an async-first type — it does NOT satisfy the sync Tokenizer
    protocol. Use it directly in router/writer hot paths via:
        counter = AsyncAnthropicTokenizer()
        n = await counter.count(prompt)
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        client: Any = None,
    ) -> None:
        self.model = model
        self._client = client
        self._cache: dict[int, int] = {}

    async def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def count(self, text: str) -> int:
        if not text:
            return 0
        h = hash(text)
        if h in self._cache:
            return self._cache[h]
        client = await self._get_client()
        result = await client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        n = int(getattr(result, "input_tokens", 0) or result.get("input_tokens", 0))  # type: ignore[union-attr]
        self._cache[h] = n
        return n

    async def count_many(self, texts: list[str]) -> list[int]:
        """Count many texts concurrently. Cached entries return immediately."""
        import asyncio

        return list(await asyncio.gather(*(self.count(t) for t in texts)))


@lru_cache(maxsize=1)
def _cached_default_tokenizer() -> Tokenizer:
    try:
        return TiktokenTokenizer()
    except ImportError:
        return CharEstimateTokenizer()


def get_tokenizer(model: str, *, anthropic_client: Any = None) -> Tokenizer:
    """Return the best available tokenizer for the given model.

    Order of preference:
      1. TiktokenTokenizer if tiktoken is installed (accurate for GPT,
         good approximation for Claude).
      2. CharEstimateTokenizer as always-available fallback.

    For exact Claude counting, instantiate `AnthropicTokenizer(model=...)`
    directly and pass it via `ContextEngine(tokenizer=...)`.
    """
    del model, anthropic_client
    return _cached_default_tokenizer()
