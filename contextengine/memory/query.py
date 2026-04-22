"""Framework-agnostic memory query API: search, list, export, delete (GDPR)."""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from typing import Any

from contextengine._json import extract_json
from contextengine.llm.base import LLMClient
from contextengine.memory.store import MemoryStore
from contextengine.memory.types import EntityMemory, Event, Fact


@dataclass(frozen=True)
class QueryResult:
    """Answer to a natural-language memory query."""

    answer: str
    facts: tuple[Fact, ...]
    events: tuple[Event, ...]


class MemoryQuery:
    """Search, filter, export, and delete entity memory.

    Programmatic methods (no LLM): `list_facts`, `list_events`, `export`,
    `erase`, `history`. LLM-backed method: `ask` — natural-language
    question answered from the entity's memory.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        llm: LLMClient | None = None,
        model: str | None = None,
    ) -> None:
        self.store = store
        self._llm = llm
        self._model = model

    async def list_facts(
        self,
        *,
        entity_id: str,
        key_pattern: str = "*",
        role: str = "",
    ) -> list[Fact]:
        mem = await self.store.get(entity_id)
        return [
            f
            for f in mem.facts
            if fnmatch.fnmatch(f.key, key_pattern) and f.visible_to(role)
        ]

    async def list_events(
        self,
        *,
        entity_id: str,
        since: float | None = None,
        until: float | None = None,
        role: str = "",
        source: str | None = None,
    ) -> list[Event]:
        mem = await self.store.get(entity_id)
        out: list[Event] = []
        for e in mem.events:
            if not e.visible_to(role):
                continue
            if since is not None and e.ts < since:
                continue
            if until is not None and e.ts > until:
                continue
            if source is not None and e.source != source:
                continue
            out.append(e)
        return out

    async def history(
        self, *, entity_id: str, key: str, role: str = ""
    ) -> list[Fact]:
        """Return versioned history for a single fact key (current only at v1)."""
        facts = await self.list_facts(entity_id=entity_id, key_pattern=key, role=role)
        return sorted(facts, key=lambda f: f.version, reverse=True)

    async def export(self, *, entity_id: str) -> dict[str, Any]:
        """Portable JSON-serializable snapshot of all memory for an entity (GDPR)."""
        mem = await self.store.get(entity_id)
        return {
            "entity_id": mem.entity_id,
            "facts": [
                {
                    "key": f.key,
                    "value": f.value,
                    "source": f.source,
                    "version": f.version,
                    "ts": f.ts,
                    "visibility": list(f.visibility),
                }
                for f in mem.facts
            ],
            "events": [
                {
                    "text": e.text,
                    "source": e.source,
                    "ts": e.ts,
                    "visibility": list(e.visibility),
                }
                for e in mem.events
            ],
        }

    async def export_json(self, *, entity_id: str) -> str:
        return json.dumps(await self.export(entity_id=entity_id), indent=2)

    async def erase(self, *, entity_id: str) -> None:
        """Hard-delete all memory for an entity (GDPR right-to-be-forgotten)."""
        await self.store.delete(entity_id)

    async def ask(
        self,
        *,
        entity_id: str,
        question: str,
        role: str = "",
        max_tokens: int = 512,
    ) -> QueryResult:
        """Answer a natural-language question from an entity's memory.

        Requires an `llm` + `model` passed at construction.
        """
        if self._llm is None or self._model is None:
            raise RuntimeError(
                "MemoryQuery.ask requires an LLMClient + model at construction"
            )

        mem: EntityMemory = await self.store.get(entity_id)
        facts = [f for f in mem.facts if f.visible_to(role)]
        events = [e for e in mem.events if e.visible_to(role)]

        facts_block = "\n".join(f"- {f.key}: {f.value}" for f in facts) or "(none)"
        events_block = (
            "\n".join(f"- [{e.source or 'event'}] {e.text}" for e in events) or "(none)"
        )

        prompt = (
            f"Entity: {entity_id}\n\n"
            f"Facts:\n{facts_block}\n\n"
            f"Events:\n{events_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer briefly using only the memory above. If the memory "
            f"doesn't contain the answer, say so."
        )

        response = await self._llm.complete(
            model=self._model,
            system="You answer questions about an entity from its stored memory.",
            user=prompt,
            max_tokens=max_tokens,
        )
        return QueryResult(
            answer=response.text.strip(),
            facts=tuple(facts),
            events=tuple(events),
        )
