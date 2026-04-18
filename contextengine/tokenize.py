"""Token counting abstraction."""
from __future__ import annotations

from typing import Protocol


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


def get_tokenizer(model: str) -> Tokenizer:
    """Return the best available tokenizer for the given model.

    Order of preference:
      1. TiktokenTokenizer if tiktoken is installed (accurate for GPT,
         good approximation for Claude).
      2. CharEstimateTokenizer as always-available fallback.
    """
    del model
    try:
        return TiktokenTokenizer()
    except ImportError:
        return CharEstimateTokenizer()
