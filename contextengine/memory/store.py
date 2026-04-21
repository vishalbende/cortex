"""MemoryStore protocol and concrete implementations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from contextengine.memory.types import EntityMemory, Event, Fact


class MemoryStore(Protocol):
    """Async storage for entity memory."""

    async def get(self, entity_id: str) -> EntityMemory: ...

    async def upsert_fact(self, fact: Fact) -> None: ...

    async def append_event(self, event: Event) -> None: ...

    async def list_entities(self) -> list[str]: ...

    async def delete(self, entity_id: str) -> None: ...

    async def delete_fact(self, *, entity_id: str, key: str) -> None: ...

    async def prune_events(
        self, *, entity_id: str, keep: tuple[Event, ...]
    ) -> None: ...


class InMemoryStore:
    """Process-local dict-backed store. Useful for tests and prototypes."""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], Fact] = {}
        self._events: dict[str, list[Event]] = {}

    async def get(self, entity_id: str) -> EntityMemory:
        facts = tuple(f for (eid, _), f in self._facts.items() if eid == entity_id)
        events = tuple(self._events.get(entity_id, ()))
        return EntityMemory(entity_id=entity_id, facts=facts, events=events)

    async def upsert_fact(self, fact: Fact) -> None:
        key = (fact.entity_id, fact.key)
        existing = self._facts.get(key)
        if existing is not None:
            from dataclasses import replace

            fact = replace(fact, version=existing.version + 1)
        self._facts[key] = fact

    async def append_event(self, event: Event) -> None:
        self._events.setdefault(event.entity_id, []).append(event)

    async def list_entities(self) -> list[str]:
        ids = {eid for (eid, _) in self._facts} | set(self._events)
        return sorted(ids)

    async def delete(self, entity_id: str) -> None:
        self._facts = {k: v for k, v in self._facts.items() if k[0] != entity_id}
        self._events.pop(entity_id, None)

    async def delete_fact(self, *, entity_id: str, key: str) -> None:
        self._facts.pop((entity_id, key), None)

    async def prune_events(
        self, *, entity_id: str, keep: tuple[Event, ...]
    ) -> None:
        keep_keys = {(e.ts, e.text) for e in keep}
        events = self._events.get(entity_id, [])
        self._events[entity_id] = [e for e in events if (e.ts, e.text) in keep_keys]


class JSONStore:
    """File-backed store: one JSON file per entity at `root/{entity_id}.json`.

    Lossy on concurrent writes from multiple processes; intended for
    single-process local use. Persistent across restarts.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, entity_id: str) -> Path:
        safe = entity_id.replace("/", "_")
        return self.root / f"{safe}.json"

    def _load(self, entity_id: str) -> EntityMemory:
        path = self._path(entity_id)
        if not path.exists():
            return EntityMemory(entity_id=entity_id)
        data = json.loads(path.read_text())
        facts = tuple(
            Fact(
                entity_id=f["entity_id"],
                key=f["key"],
                value=f["value"],
                source=f.get("source", ""),
                version=f.get("version", 1),
                ts=f.get("ts", 0.0),
                visibility=tuple(f.get("visibility", [])),
            )
            for f in data.get("facts", [])
        )
        events = tuple(
            Event(
                entity_id=e["entity_id"],
                text=e["text"],
                source=e.get("source", ""),
                ts=e.get("ts", 0.0),
                visibility=tuple(e.get("visibility", [])),
            )
            for e in data.get("events", [])
        )
        return EntityMemory(entity_id=entity_id, facts=facts, events=events)

    def _save(self, mem: EntityMemory) -> None:
        payload = {
            "entity_id": mem.entity_id,
            "facts": [
                {
                    "entity_id": f.entity_id,
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
                    "entity_id": e.entity_id,
                    "text": e.text,
                    "source": e.source,
                    "ts": e.ts,
                    "visibility": list(e.visibility),
                }
                for e in mem.events
            ],
        }
        self._path(mem.entity_id).write_text(json.dumps(payload, indent=2))

    async def get(self, entity_id: str) -> EntityMemory:
        return self._load(entity_id)

    async def upsert_fact(self, fact: Fact) -> None:
        mem = self._load(fact.entity_id)
        existing_idx = next(
            (i for i, f in enumerate(mem.facts) if f.key == fact.key), None
        )
        from dataclasses import replace

        if existing_idx is not None:
            bumped = replace(fact, version=mem.facts[existing_idx].version + 1)
            new_facts = mem.facts[:existing_idx] + (bumped,) + mem.facts[existing_idx + 1 :]
        else:
            new_facts = mem.facts + (fact,)
        self._save(EntityMemory(entity_id=fact.entity_id, facts=new_facts, events=mem.events))

    async def append_event(self, event: Event) -> None:
        mem = self._load(event.entity_id)
        self._save(
            EntityMemory(
                entity_id=event.entity_id,
                facts=mem.facts,
                events=mem.events + (event,),
            )
        )

    async def list_entities(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    async def delete(self, entity_id: str) -> None:
        p = self._path(entity_id)
        if p.exists():
            p.unlink()

    async def delete_fact(self, *, entity_id: str, key: str) -> None:
        mem = self._load(entity_id)
        new_facts = tuple(f for f in mem.facts if f.key != key)
        if len(new_facts) == len(mem.facts):
            return
        self._save(
            EntityMemory(entity_id=entity_id, facts=new_facts, events=mem.events)
        )

    async def prune_events(
        self, *, entity_id: str, keep: tuple[Event, ...]
    ) -> None:
        mem = self._load(entity_id)
        keep_keys = {(e.ts, e.text) for e in keep}
        new_events = tuple(e for e in mem.events if (e.ts, e.text) in keep_keys)
        self._save(
            EntityMemory(entity_id=entity_id, facts=mem.facts, events=new_events)
        )
