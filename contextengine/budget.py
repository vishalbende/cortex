"""Token budget allocation and greedy packing."""
from __future__ import annotations

import json
from dataclasses import dataclass

from contextengine.tokenize import Tokenizer
from contextengine.types import Message, Tool


@dataclass(frozen=True)
class Budget:
    """Token budget for a single assemble() call."""

    total: int
    reserved_output: int = 4096

    @property
    def available(self) -> int:
        return max(0, self.total - self.reserved_output)


@dataclass(frozen=True)
class PackResult:
    tools: list[Tool]
    messages: list[Message]
    tokens_used: int
    tools_dropped: list[Tool]
    messages_dropped: list[Message]


def _message_tokens(m: Message, tokenizer: Tokenizer) -> int:
    if isinstance(m.content, str):
        return tokenizer.count(m.content)
    return tokenizer.count(json.dumps(m.content))


def pack(
    *,
    budget: Budget,
    system_tokens: int,
    memory_tokens: int,
    ranked_tools: list[Tool],
    history: list[Message],
    tokenizer: Tokenizer,
    required_tools: set[str] | None = None,
) -> PackResult:
    """Greedy-fit ranked tools + newest history within remaining budget.

    Tools occupy the stable prefix, so they are packed first. Required
    tools are always included even if they push us over. History is
    packed newest-first so the latest turns survive truncation.
    """
    required = required_tools or set()
    remaining = budget.available - system_tokens - memory_tokens
    if remaining < 0:
        remaining = 0

    tools_in: list[Tool] = []
    tools_tokens = 0

    by_name = {t.name: t for t in ranked_tools}
    for name in required:
        t = by_name.get(name)
        if t is not None and t not in tools_in:
            tools_in.append(t)
            tools_tokens += t.token_count

    tools_out: list[Tool] = []
    for t in ranked_tools:
        if t in tools_in:
            continue
        if tools_tokens + t.token_count <= remaining:
            tools_in.append(t)
            tools_tokens += t.token_count
        else:
            tools_out.append(t)

    remaining_after_tools = max(0, remaining - tools_tokens)

    messages_in: list[Message] = []
    messages_out: list[Message] = []
    messages_tokens = 0
    for m in reversed(history):
        mt = _message_tokens(m, tokenizer)
        if messages_tokens + mt <= remaining_after_tools:
            messages_in.insert(0, m)
            messages_tokens += mt
        else:
            messages_out.insert(0, m)

    return PackResult(
        tools=tools_in,
        messages=messages_in,
        tokens_used=system_tokens + memory_tokens + tools_tokens + messages_tokens,
        tools_dropped=tools_out,
        messages_dropped=messages_out,
    )
