"""Handoff protocol: transfer a live conversation from one agent role to another."""
from __future__ import annotations

import time
from dataclasses import dataclass

from contextengine.memory.store import MemoryStore
from contextengine.memory.types import Event


@dataclass(frozen=True)
class Handoff:
    """One agent-to-agent transfer record.

    Stored as an Event in the shared memory store with source="handoff"
    and visibility scoped to (from_role, to_role) so both ends can see it.
    """

    entity_id: str
    from_role: str
    to_role: str
    reason: str
    summary: str
    ts: float


class HandoffProtocol:
    """Coordinates handoffs between agent views sharing a memory store.

    Writes a handoff Event to the store and returns a Handoff record.
    The receiving role's MemoryAssembler will pick it up next assemble().
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def handoff(
        self,
        *,
        entity_id: str,
        from_role: str,
        to_role: str,
        reason: str,
        summary: str = "",
    ) -> Handoff:
        ts = time.time()
        event = Event(
            entity_id=entity_id,
            text=f"Handoff {from_role} → {to_role}: {reason}"
            + (f" — {summary}" if summary else ""),
            source="handoff",
            ts=ts,
            visibility=tuple(sorted({from_role, to_role})),
        )
        await self.store.append_event(event)
        return Handoff(
            entity_id=entity_id,
            from_role=from_role,
            to_role=to_role,
            reason=reason,
            summary=summary,
            ts=ts,
        )

    async def list_handoffs(self, *, entity_id: str) -> list[Handoff]:
        mem = await self.store.get(entity_id)
        out: list[Handoff] = []
        for e in mem.events:
            if e.source != "handoff":
                continue
            body = e.text
            if "Handoff " not in body:
                continue
            try:
                header, _, rest = body.partition(":")
                arrow = header.replace("Handoff ", "", 1)
                from_role, _, to_role = arrow.partition(" → ")
                reason_part, _, summary = rest.strip().partition(" — ")
                out.append(
                    Handoff(
                        entity_id=entity_id,
                        from_role=from_role.strip(),
                        to_role=to_role.strip(),
                        reason=reason_part.strip(),
                        summary=summary.strip(),
                        ts=e.ts,
                    )
                )
            except (ValueError, AttributeError):
                continue
        return out
