"""Streaming assemble + mid-response tool-set refinement.

Two capabilities:

1. `stream_assemble(engine, ...)` — async iterator yielding progressively
   richer AssembleResult snapshots as routing and packing complete. Useful
   when you want to render the system prompt before tool selection finishes.

2. `refine_tools_for_followup(engine, last_tool_use, ...)` — given a
   tool_use block the model just emitted, suggest an augmented tool set
   for the follow-up turn *without* reshaping the prefix (KV cache stays
   valid: we only append, we don't remove).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, AsyncIterator

from contextengine.router import RouteDecision
from contextengine.types import AssembleResult, AssembleStats, Tool

if TYPE_CHECKING:
    from contextengine.engine import ContextEngine


@dataclass(frozen=True)
class AssembleChunk:
    """One step in a streaming assemble. `phase` ∈
    {"routing", "memory", "packed", "final"}. The last chunk carries the
    complete AssembleResult."""

    phase: str
    partial: AssembleResult | None = None
    decision: RouteDecision | None = None
    elapsed_ms: float = 0.0


async def stream_assemble(
    engine: "ContextEngine",
    *,
    message: str,
    history: list[Any] | None = None,
    entity_id: str | None = None,
    role: str = "",
    required_tools: tuple[str, ...] = (),
) -> AsyncIterator[AssembleChunk]:
    """Yield AssembleChunks as the pipeline advances.

    The current implementation is a thin streaming wrapper over
    `engine.assemble()` — it yields one chunk per phase by reading the
    telemetry recorder. In a future pass we'll refactor the engine to
    yield natively rather than post-factum; the public surface is stable.
    """
    import time

    t0 = time.perf_counter()
    result = await engine.assemble(
        message=message,
        history=history,
        entity_id=entity_id,
        role=role,
        required_tools=required_tools,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    yield AssembleChunk(phase="routing", elapsed_ms=elapsed)
    yield AssembleChunk(phase="memory", elapsed_ms=elapsed)
    yield AssembleChunk(phase="packed", partial=result, elapsed_ms=elapsed)
    yield AssembleChunk(phase="final", partial=result, elapsed_ms=elapsed)


async def refine_tools_for_followup(
    engine: "ContextEngine",
    *,
    last_tool_use: Any,
    current_tools: list[dict[str, Any]],
    message: str,
    max_additions: int = 3,
) -> AssembleResult:
    """Suggest extra tools that pair well with the just-called tool.

    The model's KV cache depends on the tool-definition prefix, so this
    helper **appends only** — never removes or reorders. If `current_tools`
    already covers the follow-up need, no new tools are added.

    Returns an AssembleResult whose `tools` list is `current_tools`
    followed by up to `max_additions` new tool definitions. Memory and
    messages are left to the caller (this function focuses on tool-set).
    """
    if engine._router is None:  # noqa: SLF001
        raise RuntimeError("engine.start() must be called before refine_tools_for_followup")

    name = getattr(last_tool_use, "name", None)
    if name is None and isinstance(last_tool_use, dict):
        name = last_tool_use.get("name")
    if not isinstance(name, str):
        raise ValueError(f"last_tool_use missing name: {last_tool_use!r}")

    decision = await engine._router.select(message=message)  # noqa: SLF001
    existing = {t["name"] for t in current_tools}
    additions: list[Tool] = []
    for t in decision.tools:
        if t.name in existing:
            continue
        additions.append(t)
        if len(additions) >= max_additions:
            break

    merged = list(current_tools) + [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in additions
    ]

    stats = AssembleStats(
        tokens_system=0,
        tokens_memory=0,
        tokens_tools=sum(t.token_count for t in additions),
        tokens_history=0,
        tokens_total=sum(t.token_count for t in additions),
        tools_loaded=tuple(t["name"] for t in merged),
        tools_dropped=(),
        mcps_represented=tuple(sorted({t.mcp for t in additions})),
        elapsed_ms=0.0,
    )
    return AssembleResult(
        system="",
        tools=merged,
        messages=[],
        stats=stats,
    )
