"""Assemble a role-scoped memory block within a token budget."""
from __future__ import annotations

from contextengine.memory.store import MemoryStore
from contextengine.tokenize import Tokenizer


class MemoryAssembler:
    """Turns entity memory into an opaque string block for the prompt.

    Layout (newest events first, then facts):
        [memory]
        Facts:
          - key: value
        Recent events:
          - <ts> text
        [/memory]

    Filters by role visibility and truncates to fit `budget_tokens`.
    Events truncate before facts (facts are the durable signal).
    """

    def __init__(self, store: MemoryStore, tokenizer: Tokenizer) -> None:
        self.store = store
        self.tokenizer = tokenizer

    async def assemble(
        self,
        *,
        entity_id: str,
        role: str = "",
        budget_tokens: int = 4000,
    ) -> str:
        mem = await self.store.get(entity_id)
        if not mem.facts and not mem.events:
            return ""

        facts = sorted(
            (f for f in mem.facts if f.visible_to(role)),
            key=lambda f: (f.key, -f.version),
        )
        events = sorted(
            (e for e in mem.events if e.visible_to(role)),
            key=lambda e: e.ts,
            reverse=True,
        )

        fact_lines = [f"  - {f.key}: {f.value}" for f in facts]
        event_lines = [f"  - {e.text}" for e in events]

        def _assemble(ev_lines: list[str], fa_lines: list[str]) -> str:
            parts = ["[memory]"]
            if fa_lines:
                parts.append("Facts:")
                parts.extend(fa_lines)
            if ev_lines:
                parts.append("Recent events:")
                parts.extend(ev_lines)
            parts.append("[/memory]")
            return "\n".join(parts)

        text = _assemble(event_lines, fact_lines)
        if self.tokenizer.count(text) <= budget_tokens:
            return text

        while event_lines and self.tokenizer.count(_assemble(event_lines, fact_lines)) > budget_tokens:
            event_lines.pop()

        while fact_lines and self.tokenizer.count(_assemble(event_lines, fact_lines)) > budget_tokens:
            fact_lines.pop()

        if not event_lines and not fact_lines:
            return ""
        return _assemble(event_lines, fact_lines)
