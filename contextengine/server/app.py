"""Build an MCP server that proxies downstream MCPs and adds contextengine meta-tools."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from contextengine.coordination.handoff import HandoffProtocol
from contextengine.engine import ContextEngine
from contextengine.memory import Fact, MemoryQuery
from contextengine.server.meta_tools import META_TOOL_DEFS, META_TOOL_NAMES


@dataclass
class ContextEngineMCPServer:
    """Wraps a ContextEngine and exposes it as an MCP server.

    `list_tools()` returns the downstream MCP tool set (already
    namespaced: `<mcp>.<tool>`) concatenated with contextengine's
    meta-tools (`ce.*`).

    `call_tool(name, arguments)` dispatches:
      - `ce.*` → internal meta-tool handler
      - `<mcp>.<tool>` → `engine.execute(...)` which proxies to the
        owning downstream MCP.

    Transport is left to the caller — see `run_stdio()` for a stdio
    binding. The class itself is transport-agnostic so it can be wired
    to SSE / streamable-HTTP / in-process too.
    """

    engine: ContextEngine
    _query: MemoryQuery = field(init=False)
    _handoff: HandoffProtocol = field(init=False)

    def __post_init__(self) -> None:
        llm = getattr(self.engine, "_memory_llm", None)
        self._query = MemoryQuery(
            store=self.engine.memory,
            llm=llm,
            model=self.engine.memory_model,
        )
        self._handoff = HandoffProtocol(self.engine.memory)

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return [{"name", "description", "inputSchema"}, ...] — MCP protocol shape."""
        if self.engine.catalog is None:
            downstream: list[dict[str, Any]] = []
        else:
            downstream = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for mcp in self.engine.catalog.mcps
                for t in mcp.tools_flat
            ]
        return downstream + META_TOOL_DEFS

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Dispatch a tool call. Return MCP-shaped content list."""
        if name in META_TOOL_NAMES:
            result = await self._dispatch_meta(name, arguments)
        else:
            result = await self.engine.execute(
                {"name": name, "input": arguments or {}}
            )
        return _to_content(result)

    async def _dispatch_meta(
        self, name: str, arguments: dict[str, Any]
    ) -> Any:
        if name == "ce.remember":
            fact = Fact(
                entity_id=arguments["entity_id"],
                key=arguments["key"],
                value=str(arguments["value"]),
                source=arguments.get("source", "assistant"),
                visibility=tuple(arguments.get("visibility", [])),
            )
            await self.engine.memory.upsert_fact(fact)
            return {"ok": True, "key": fact.key}

        if name == "ce.recall":
            entity_id = arguments["entity_id"]
            role = arguments.get("role", "")
            mem = await self.engine.memory.get(entity_id)
            facts = [f for f in mem.facts if f.visible_to(role)]
            events = [e for e in mem.events if e.visible_to(role)]
            return {
                "facts": [{"key": f.key, "value": f.value} for f in facts],
                "events": [{"text": e.text, "ts": e.ts} for e in events],
            }

        if name == "ce.ask_memory":
            result = await self._query.ask(
                entity_id=arguments["entity_id"],
                question=arguments["question"],
                role=arguments.get("role", ""),
            )
            return {"answer": result.answer}

        if name == "ce.route":
            if self.engine._router is None:  # noqa: SLF001
                return {"error": "engine not started"}
            decision = await self.engine._router.select(  # noqa: SLF001
                message=arguments["message"]
            )
            return {
                "tools": [t.name for t in decision.tools],
                "mcps_selected": decision.mcps_selected,
            }

        if name == "ce.handoff":
            h = await self._handoff.handoff(
                entity_id=arguments["entity_id"],
                from_role=arguments["from_role"],
                to_role=arguments["to_role"],
                reason=arguments["reason"],
                summary=arguments.get("summary", ""),
            )
            return {"ok": True, "ts": h.ts}

        if name == "ce.export_memory":
            payload = await self._query.export(entity_id=arguments["entity_id"])
            return payload

        if name == "ce.erase_memory":
            await self._query.erase(entity_id=arguments["entity_id"])
            return {"ok": True}

        raise ValueError(f"Unknown meta-tool: {name!r}")


def _to_content(result: Any) -> list[dict[str, Any]]:
    """Wrap a Python result into MCP `content` blocks."""
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, default=str)
        except (TypeError, ValueError):
            text = str(result)
    return [{"type": "text", "text": text}]


def build_server(engine: ContextEngine) -> ContextEngineMCPServer:
    """Convenience factory."""
    return ContextEngineMCPServer(engine=engine)


async def run_stdio(engine: ContextEngine) -> None:
    """Bind the wrapper to an MCP stdio transport and serve forever.

    Requires the downstream engine to have called `start()` already.
    """
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    wrapper = build_server(engine)
    server = Server("contextengine")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        defs = await wrapper.list_tools()
        return [
            Tool(
                name=d["name"],
                description=d.get("description", ""),
                inputSchema=d.get("inputSchema", {}),
            )
            for d in defs
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        blocks = await wrapper.call_tool(name, arguments or {})
        return [TextContent(type="text", text=b["text"]) for b in blocks]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
