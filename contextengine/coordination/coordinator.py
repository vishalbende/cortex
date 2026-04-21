"""MultiAgentCoordinator: multiple role-specific ContextEngines over one memory store.

Each agent gets its own ContextEngine instance (its own MCPs, system
prompt, router) but they all share the same MemoryStore — so facts,
events, and handoffs flow between roles.

Usage:
    coordinator = MultiAgentCoordinator(memory_store=InMemoryStore())
    coordinator.register("support", engine_support)
    coordinator.register("sales", engine_sales)

    # Support agent runs a turn...
    ctx = await coordinator.assemble("support", message=..., entity_id="c1")
    # ...decides to hand off to sales:
    await coordinator.handoff(
        entity_id="c1", from_role="support", to_role="sales",
        reason="Pricing question out of scope.",
    )

    # Sales agent picks up with the handoff note visible in its memory block:
    ctx = await coordinator.assemble("sales", message=..., entity_id="c1")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from contextengine.coordination.handoff import Handoff, HandoffProtocol
from contextengine.memory.store import InMemoryStore, MemoryStore
from contextengine.types import AssembleResult, Message

if TYPE_CHECKING:
    from contextengine.engine import ContextEngine


@dataclass
class AgentView:
    """One agent registered with the coordinator."""

    role: str
    engine: "ContextEngine"


class MultiAgentCoordinator:
    """Owns the shared store; routes assemble/handoff calls to role-specific engines."""

    def __init__(self, *, memory_store: MemoryStore | None = None) -> None:
        self._store: MemoryStore = memory_store or InMemoryStore()
        self._views: dict[str, AgentView] = {}
        self._handoff = HandoffProtocol(self._store)

    @property
    def memory(self) -> MemoryStore:
        return self._store

    def register(self, role: str, engine: "ContextEngine") -> None:
        """Register a role's engine. Rewrites engine's memory store to the shared one."""
        if role in self._views:
            raise ValueError(f"role {role!r} already registered")
        engine._memory_store = self._store  # noqa: SLF001
        from contextengine.memory.assembler import MemoryAssembler

        engine._memory_assembler = MemoryAssembler(self._store, engine.tokenizer)  # noqa: SLF001
        engine._memory_writer.store = self._store  # noqa: SLF001
        self._views[role] = AgentView(role=role, engine=engine)

    def get(self, role: str) -> "ContextEngine":
        if role not in self._views:
            raise KeyError(f"role {role!r} not registered")
        return self._views[role].engine

    def roles(self) -> list[str]:
        return list(self._views)

    async def assemble(
        self,
        role: str,
        *,
        message: str,
        history: list[Message] | None = None,
        entity_id: str | None = None,
        required_tools: tuple[str, ...] = (),
    ) -> AssembleResult:
        """Assemble context through the given role's engine, scoped to that role."""
        engine = self.get(role)
        return await engine.assemble(
            message=message,
            history=history,
            entity_id=entity_id,
            role=role,
            required_tools=required_tools,
        )

    async def handoff(
        self,
        *,
        entity_id: str,
        from_role: str,
        to_role: str,
        reason: str,
        summary: str = "",
    ) -> Handoff:
        if from_role not in self._views:
            raise KeyError(f"from_role {from_role!r} not registered")
        if to_role not in self._views:
            raise KeyError(f"to_role {to_role!r} not registered")
        return await self._handoff.handoff(
            entity_id=entity_id,
            from_role=from_role,
            to_role=to_role,
            reason=reason,
            summary=summary,
        )

    async def list_handoffs(self, *, entity_id: str) -> list[Handoff]:
        return await self._handoff.list_handoffs(entity_id=entity_id)

    async def close(self) -> None:
        for view in self._views.values():
            await view.engine.close()
