"""Compact entity memory: fold stale facts + old events into a summary fact.

Complements `HistoryCompactor` (which compacts in-prompt conversation
history). `MemoryCompactor` compacts the persistent store itself.

Strategy:
  - Keep recent events verbatim (last `keep_recent_events`).
  - Keep facts that are still load-bearing (versioned ≥ `version_floor`
    OR referenced in the last `keep_recent_events` events).
  - Everything else is summarized via LLM into a single rolling fact
    keyed `__memory.summary` (versioned, replaces prior summary).
  - Original facts being summarized are deleted from the store.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from contextengine.llm.base import LLMClient
from contextengine.memory.store import MemoryStore
from contextengine.memory.types import Event, Fact


SUMMARY_KEY = "__memory.summary"


@dataclass(frozen=True)
class CompactResult:
    facts_before: int
    facts_after: int
    events_before: int
    events_after: int
    summary_written: bool


class MemoryCompactor:
    """Summarize and prune stale facts + events for a given entity.

    Triggered when entity memory exceeds `fact_threshold` facts or
    `event_threshold` events. Call `compact(entity_id)` directly, or
    check `should_compact(mem)` to gate.
    """

    def __init__(
        self,
        *,
        model: str,
        llm: LLMClient,
        fact_threshold: int = 50,
        event_threshold: int = 200,
        keep_recent_events: int = 20,
        version_floor: int = 2,
    ) -> None:
        self.model = model
        self._llm = llm
        self.fact_threshold = fact_threshold
        self.event_threshold = event_threshold
        self.keep_recent_events = keep_recent_events
        self.version_floor = version_floor

    def should_compact_counts(self, n_facts: int, n_events: int) -> bool:
        return n_facts > self.fact_threshold or n_events > self.event_threshold

    async def compact(self, store: MemoryStore, entity_id: str) -> CompactResult:
        mem = await store.get(entity_id)
        n_facts_before = len(mem.facts)
        n_events_before = len(mem.events)

        if not self.should_compact_counts(n_facts_before, n_events_before):
            return CompactResult(
                facts_before=n_facts_before,
                facts_after=n_facts_before,
                events_before=n_events_before,
                events_after=n_events_before,
                summary_written=False,
            )

        keep_events: list[Event] = sorted(mem.events, key=lambda e: e.ts)[-self.keep_recent_events:]
        drop_events: list[Event] = sorted(mem.events, key=lambda e: e.ts)[:-self.keep_recent_events]

        recent_tokens = " ".join(e.text for e in keep_events)
        keep_facts: list[Fact] = []
        drop_facts: list[Fact] = []
        for f in mem.facts:
            if f.key == SUMMARY_KEY:
                drop_facts.append(f)
                continue
            if f.version >= self.version_floor:
                keep_facts.append(f)
                continue
            if any(f.key.lower() in e.text.lower() or f.value.lower() in e.text.lower() for e in keep_events):
                keep_facts.append(f)
                continue
            drop_facts.append(f)

        if not drop_facts and not drop_events:
            return CompactResult(
                facts_before=n_facts_before,
                facts_after=n_facts_before,
                events_before=n_events_before,
                events_after=n_events_before,
                summary_written=False,
            )

        previous_summary = next((f for f in mem.facts if f.key == SUMMARY_KEY), None)

        fact_lines = [f"- {f.key}: {f.value}" for f in drop_facts if f.key != SUMMARY_KEY]
        event_lines = [f"- [{int(e.ts)}] {e.text}" for e in drop_events]

        prior = ""
        if previous_summary is not None:
            prior = f"\nExisting rolling summary (extend, don't repeat):\n{previous_summary.value}\n"

        prompt = (
            f"Compact this entity's memory into a durable summary.\n"
            f"Preserve anything that constrains future reasoning: identifiers, "
            f"preferences, constraints, unresolved threads. Drop transient "
            f"conversational details. Target 120–250 words, plain prose.{prior}\n\n"
            f"Facts to fold:\n" + ("\n".join(fact_lines) or "(none)") + "\n\n"
            f"Events to fold:\n" + ("\n".join(event_lines) or "(none)")
        )

        response = await self._llm.complete(
            model=self.model,
            system="You condense entity memory while preserving load-bearing facts.",
            user=prompt,
            max_tokens=768,
        )
        summary_text = response.text.strip()

        now = time.time()
        next_version = (previous_summary.version + 1) if previous_summary is not None else 1
        summary_fact = Fact(
            entity_id=entity_id,
            key=SUMMARY_KEY,
            value=summary_text,
            source="compactor",
            version=next_version,
            ts=now,
        )

        for f in drop_facts:
            await store.delete_fact(entity_id=entity_id, key=f.key)  # type: ignore[attr-defined]

        dropped_ids = {(e.entity_id, e.ts, e.text) for e in drop_events}
        if dropped_ids:
            await store.prune_events(entity_id=entity_id, keep=tuple(keep_events))  # type: ignore[attr-defined]

        await store.upsert_fact(summary_fact)

        return CompactResult(
            facts_before=n_facts_before,
            facts_after=len(keep_facts) + 1,
            events_before=n_events_before,
            events_after=len(keep_events),
            summary_written=True,
        )
