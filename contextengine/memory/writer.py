"""Post-turn extraction: assistant response + tool results → facts + events."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from contextengine._json import extract_json
from contextengine.memory.store import MemoryStore
from contextengine.memory.types import Event, Fact


@dataclass(frozen=True)
class WriteResult:
    facts_upserted: int
    events_appended: int
    rationale: str = ""


class MemoryWriter:
    """Extracts durable facts and timeline events from a completed turn.

    Uses a cheap LLM (same model class as the router) to read the
    assistant's response and tool results, then emit structured updates
    to the store. Intended to be called async/non-blocking after each
    turn via `ContextEngine.process_turn(...)`.

    Extraction prompt:
        "You are a memory writer. Read the turn and return JSON with two
        arrays: facts (entity-scoped key/value that is durably true now)
        and events (one-sentence descriptions of what happened this turn)."
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        model: str,
        anthropic_client: Any = None,
    ) -> None:
        self.store = store
        self.model = model
        self._client = anthropic_client

    async def _client_instance(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

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

        client = await self._client_instance()
        response = await client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        data = extract_json(response.content[0].text)

        now = time.time()
        visibility = (role,) if role else ()

        fact_count = 0
        for f in data.get("facts", []):
            key = f.get("key")
            value = f.get("value")
            if not key or value is None:
                continue
            await self.store.upsert_fact(
                Fact(
                    entity_id=entity_id,
                    key=str(key),
                    value=str(value),
                    source=str(f.get("source", "")),
                    ts=now,
                    visibility=visibility,
                )
            )
            fact_count += 1

        event_count = 0
        for e in data.get("events", []):
            text = e.get("text")
            if not text:
                continue
            await self.store.append_event(
                Event(
                    entity_id=entity_id,
                    text=str(text),
                    source=str(e.get("source", "")),
                    ts=now,
                    visibility=visibility,
                )
            )
            event_count += 1

        return WriteResult(facts_upserted=fact_count, events_appended=event_count)
