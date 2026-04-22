"""Tests for the contextengine-as-MCP-server wrapper (transport-agnostic)."""
import json

import pytest

from contextengine import ContextEngine, MCPServer
from contextengine.router import Router
from contextengine.server import ContextEngineMCPServer, META_TOOL_NAMES
from contextengine.types import Catalog, MCPCatalog, Tool, ToolCategory
from tests.fakes import FakeLLMClient


def _engine(llm: FakeLLMClient) -> ContextEngine:
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        llm_client=llm,
    )
    linear = MCPCatalog(
        name="linear",
        summary="Linear.",
        categories=(
            ToolCategory(
                name="issues",
                summary="",
                tools=(
                    Tool(
                        name="linear.create_issue",
                        mcp="linear",
                        description="Create",
                        input_schema={"type": "object"},
                    ),
                ),
            ),
        ),
    )
    catalog = Catalog(mcps=(linear,), version_hash="v1")
    engine._catalog = catalog
    engine._router = Router(catalog=catalog, router_model=engine.router_model, llm=llm)
    return engine


async def test_list_tools_includes_downstream_and_meta() -> None:
    engine = _engine(FakeLLMClient())
    server = ContextEngineMCPServer(engine=engine)
    tools = await server.list_tools()
    names = {t["name"] for t in tools}
    assert "linear.create_issue" in names
    assert META_TOOL_NAMES <= names


async def test_call_meta_remember_and_recall() -> None:
    engine = _engine(FakeLLMClient())
    server = ContextEngineMCPServer(engine=engine)

    blocks = await server.call_tool(
        "ce.remember",
        {"entity_id": "c1", "key": "tier", "value": "pro"},
    )
    payload = json.loads(blocks[0]["text"])
    assert payload == {"ok": True, "key": "tier"}

    blocks = await server.call_tool("ce.recall", {"entity_id": "c1"})
    recalled = json.loads(blocks[0]["text"])
    assert recalled["facts"] == [{"key": "tier", "value": "pro"}]


async def test_call_meta_route_returns_tool_list() -> None:
    llm = FakeLLMClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    engine = _engine(llm)
    server = ContextEngineMCPServer(engine=engine)
    blocks = await server.call_tool("ce.route", {"message": "make issue"})
    payload = json.loads(blocks[0]["text"])
    assert payload["tools"] == ["linear.create_issue"]


async def test_call_meta_ask_memory_uses_llm() -> None:
    llm = FakeLLMClient(responses=["Tier is pro."])
    engine = _engine(llm)
    server = ContextEngineMCPServer(engine=engine)
    # Seed fact first
    await engine.memory.upsert_fact(
        __import__("contextengine").Fact(entity_id="c1", key="tier", value="pro")
    )
    blocks = await server.call_tool(
        "ce.ask_memory",
        {"entity_id": "c1", "question": "what tier?"},
    )
    payload = json.loads(blocks[0]["text"])
    assert payload["answer"] == "Tier is pro."


async def test_call_meta_handoff_records_event() -> None:
    engine = _engine(FakeLLMClient())
    server = ContextEngineMCPServer(engine=engine)
    blocks = await server.call_tool(
        "ce.handoff",
        {
            "entity_id": "c1",
            "from_role": "support",
            "to_role": "sales",
            "reason": "pricing",
        },
    )
    payload = json.loads(blocks[0]["text"])
    assert payload["ok"] is True
    mem = await engine.memory.get("c1")
    assert mem.events[0].source == "handoff"


async def test_call_meta_export_and_erase() -> None:
    engine = _engine(FakeLLMClient())
    server = ContextEngineMCPServer(engine=engine)
    await engine.memory.upsert_fact(
        __import__("contextengine").Fact(entity_id="c1", key="k", value="v")
    )
    blocks = await server.call_tool("ce.export_memory", {"entity_id": "c1"})
    payload = json.loads(blocks[0]["text"])
    assert payload["entity_id"] == "c1"
    assert payload["facts"]

    await server.call_tool("ce.erase_memory", {"entity_id": "c1"})
    assert await engine.memory.list_entities() == []


async def test_call_downstream_tool_routes_to_engine_execute() -> None:
    engine = _engine(FakeLLMClient())
    server = ContextEngineMCPServer(engine=engine)

    class _FakeConnector:
        async def call_tool(self, name, args):
            return {"called": name, "args": args}

    engine._pool._connectors["linear"] = _FakeConnector()  # type: ignore[assignment]

    blocks = await server.call_tool(
        "linear.create_issue", {"title": "bug"}
    )
    payload = json.loads(blocks[0]["text"])
    assert payload == {"called": "create_issue", "args": {"title": "bug"}}


async def test_unknown_meta_tool_raises() -> None:
    engine = _engine(FakeLLMClient())
    server = ContextEngineMCPServer(engine=engine)
    with pytest.raises(ValueError, match="Unknown meta-tool"):
        await server._dispatch_meta("ce.nonexistent", {})
