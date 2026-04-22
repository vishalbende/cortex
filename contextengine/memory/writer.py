"""Post-turn extraction: assistant response + tool results → facts + events."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from contextengine._json import extract_json
from contextengine.llm.base import LLMClient
from contextengine.memory.policy import (
    AllowAllPolicy,
    PolicyViolation,
    WritePolicy,
    enforce_append,
    enforce_upsert,
)
from contextengine.memory.store import MemoryStore
from contextengine.memory.types import Event, Fact


@dataclass(frozen=True)
class WriteResult:
    facts_upserted: int
    events_appended: int
    facts_rejected: int = 0
    events_rejected: int = 0
    rationale: str = ""


class MemoryWriter:
    """Extracts durable facts and timeline events from a completed turn.

    Uses a cheap LLM (same model class as the router) to read the
    assistant's response and tool results, then emits structured updates
    to the store. Provider-agnostic via LLMClient — works with either
    Claude or GPT models.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        model: str,
        llm: LLMClient,
        policy: WritePolicy | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self._llm = llm
        self._policy: WritePolicy = policy or AllowAllPolicy()

    async def write(
        self,
        *,
        entity_id: str,
        user_message: str,
        assistant_response: str,
        tool_results: list[dict[str, Any]] | None = None,
        role: str = "",
    ) -> WriteResult:
        tool_block = ""
        if tool_results:
            lines = [f"- {r.get('name', '?')}: {str(r.get('result', ''))[:400]}" for r in tool_results]
            tool_block = "Tool results this turn:\n" + "\n".join(lines)

        prompt = (
            f"User message:\n{user_message}\n\n"
            f"Assistant response:\n{assistant_response}\n\n"
            f"{tool_block}\n\n"
            f"Extract durable memory updates for entity {entity_id!r}. "
            f"Return JSON:\n"
            f"  facts: array of {{key, value, source}} — only information "
            f"that remains true after this turn (preferences, identifiers, "
            f"constraints). Do NOT include transient conversation content.\n"
            f"  events: array of {{text, source}} — one-sentence "
            f"descriptions of what happened this turn.\n"
            f"Return ONLY the JSON, no preamble."
        )

        response = await self._llm.complete(
            model=self.model,
            system="",
            user=prompt,
            max_tokens=1024,
            json_mode=True,
        )
        data = extract_json(response.text)

        now = time.time()
        visibility = (role,) if role else ()

        fact_count = 0
        facts_rejected = 0
        for f in data.get("facts", []):
            key = f.get("key")
            value = f.get("value")
            if not key or value is None:
                continue
            fact = Fact(
                entity_id=entity_id,
                key=str(key),
                value=str(value),
                source=str(f.get("source", "")),
                ts=now,
                visibility=visibility,
            )
            try:
                enforce_upsert(self._policy, role, fact)
            except PolicyViolation:
                facts_rejected += 1
                continue
            await self.store.upsert_fact(fact)
            fact_count += 1

        event_count = 0
        events_rejected = 0
        for e in data.get("events", []):
            text = e.get("text")
            if not text:
                continue
            event = Event(
                entity_id=entity_id,
                text=str(text),
                source=str(e.get("source", "")),
                ts=now,
                visibility=visibility,
            )
            try:
                enforce_append(self._policy, role, event)
            except PolicyViolation:
                events_rejected += 1
                continue
            await self.store.append_event(event)
            event_count += 1

        return WriteResult(
            facts_upserted=fact_count,
            events_appended=event_count,
            facts_rejected=facts_rejected,
            events_rejected=events_rejected,
        )
