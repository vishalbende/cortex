"""Token counting abstraction."""
from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class CharEstimateTokenizer:
    """Fallback tokenizer: ~4 chars per token. Good enough for budget math at v1."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


def get_tokenizer(model: str) -> Tokenizer:
    """Return a tokenizer for the given model.

    v1 returns the char estimator for every model. Model-specific
    tokenizers (Anthropic's tokenizer lib, tiktoken for OpenAI) land
    in a later pass.
    """
    del model
    return CharEstimateTokenizer()
