"""TraceRecorder + TraceRecord: capture context composition per assemble()."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextengine.telemetry.sinks import Sink
    from contextengine.types import AssembleResult


@dataclass(frozen=True)
class TraceEvent:
    """A single named span inside an assemble() call (router pass, budget pack, etc.)."""

    name: str
    elapsed_ms: float
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceRecord:
    """Everything we know about one assemble() invocation.

    Emitted attribute names follow the OpenTelemetry GenAI SIG
    convention (`gen_ai.*`) where applicable, plus contextengine-specific
    extensions for tool/memory composition.
    """

    trace_id: str
    ts: float
    model: str
    router_model: str
    entity_id: str | None
    role: str
    message_preview: str
    # gen_ai.* fields
    gen_ai_system: str
    gen_ai_request_model: str
    # context composition
    tokens_system: int
    tokens_memory: int
    tokens_tools: int
    tokens_history: int
    tokens_total: int
    tokens_budget: int
    tools_loaded: tuple[str, ...]
    tools_dropped: tuple[str, ...]
    mcps_represented: tuple[str, ...]
    elapsed_ms: float
    events: tuple[TraceEvent, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tools_loaded"] = list(self.tools_loaded)
        d["tools_dropped"] = list(self.tools_dropped)
        d["mcps_represented"] = list(self.mcps_represented)
        d["events"] = [asdict(e) for e in self.events]
        return d


class TraceRecorder:
    """Builds a TraceRecord for one assemble() call and dispatches to sinks."""

    def __init__(self, sinks: list["Sink"] | None = None) -> None:
        self._sinks: list[Sink] = list(sinks or [])
        self._start_ts: float = 0.0
        self._events: list[TraceEvent] = []

    def add_sink(self, sink: "Sink") -> None:
        self._sinks.append(sink)

    def start(self) -> None:
        self._start_ts = time.perf_counter()
        self._events = []

    def span(self, name: str, elapsed_ms: float, **attrs: Any) -> None:
        self._events.append(TraceEvent(name=name, elapsed_ms=elapsed_ms, attrs=dict(attrs)))

    async def emit(
        self,
        *,
        trace_id: str,
        model: str,
        router_model: str,
        entity_id: str | None,
        role: str,
        message: str,
        budget: int,
        result: "AssembleResult",
        extra: dict[str, Any] | None = None,
    ) -> TraceRecord:
        record = TraceRecord(
            trace_id=trace_id,
            ts=time.time(),
            model=model,
            router_model=router_model,
            entity_id=entity_id,
            role=role,
            message_preview=message[:120],
            gen_ai_system="anthropic",
            gen_ai_request_model=model,
            tokens_system=result.stats.tokens_system,
            tokens_memory=result.stats.tokens_memory,
            tokens_tools=result.stats.tokens_tools,
            tokens_history=result.stats.tokens_history,
            tokens_total=result.stats.tokens_total,
            tokens_budget=budget,
            tools_loaded=result.stats.tools_loaded,
            tools_dropped=result.stats.tools_dropped,
            mcps_represented=result.stats.mcps_represented,
            elapsed_ms=result.stats.elapsed_ms,
            events=tuple(self._events),
            extra=dict(extra or {}),
        )
        for sink in self._sinks:
            await sink.write(record)
        return record
