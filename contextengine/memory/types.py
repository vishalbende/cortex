"""Memory data contracts: Fact (current truth), Event (append-only timeline)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fact:
    """A key/value assertion about an entity, versioned, visibility-scoped.

    `visibility` is a tuple of agent roles that can read this fact. An
    empty tuple means all roles can see it.
    """

    entity_id: str
    key: str
    value: str
    source: str = ""
    version: int = 1
    ts: float = field(default_factory=lambda: time.time())
    visibility: tuple[str, ...] = ()

    def visible_to(self, role: str) -> bool:
        return not self.visibility or role in self.visibility


@dataclass(frozen=True)
class Event:
    """A timestamped note in an entity's timeline."""

    entity_id: str
    text: str
    source: str = ""
    ts: float = field(default_factory=lambda: time.time())
    visibility: tuple[str, ...] = ()

    def visible_to(self, role: str) -> bool:
        return not self.visibility or role in self.visibility


@dataclass(frozen=True)
class EntityMemory:
    """All memory for one entity."""

    entity_id: str
    facts: tuple[Fact, ...] = ()
    events: tuple[Event, ...] = ()
