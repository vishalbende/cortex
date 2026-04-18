"""Integration test: assemble with memory + telemetry + compaction together."""
import io
import json

import pytest

from contextengine import ContextEngine, MCPServer
from contextengine.memory import Fact
from contextengine.router import Router
from contextengine.telemetry import StdoutSink
from contextengine.types import Catalog, MCPCatalog, Message, Tool, ToolCategory
from tests.fakes import FakeAnthropicClient


def _seed_catalog(engine: ContextEngine, client: FakeAnthropicClient) -> None:
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
                        input_schema={},
                        token_count=30,
                    ),
                ),
            ),
        ),
    )
    catalog = Catalog(mcps=(linear,), version_hash="v1")
    engine._catalog = catalog
    engine._router = Router(
        catalog=catalog, router_model=engine.router_model, anthropic_client=client
    )


async def test_assemble_injects_memory_block_for_entity() -> None:
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        system_prompt="agent.",
    )
    await engine.memory.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))

    client = FakeAnthropicClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    _seed_catalog(engine, client)

    result = await engine.assemble(message="make an issue", entity_id="c1")
    assert "agent." in result.system
    assert "[memory]" in result.system
    assert "tier: pro" in result.system


async def test_telemetry_sink_receives_record() -> None:
    buf = io.StringIO()
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        telemetry_sinks=[StdoutSink(stream=buf)],
    )
    client = FakeAnthropicClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    _seed_catalog(engine, client)

    await engine.assemble(message="hi")
    output = buf.getvalue()
    assert "[contextengine] assemble" in output
    assert "mcps=['linear']" in output


async def test_compaction_triggered_on_long_history() -> None:
    client = FakeAnthropicClient(
        responses=[
            "Summary of prior conversation.",
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        compaction_threshold=4,
        compaction_keep_recent=2,
        anthropic_client=client,
    )
    _seed_catalog(engine, client)

    history = [Message(role="user", content=f"t{i}") for i in range(10)]
    result = await engine.assemble(message="now", history=history)

    assert any("compacted-summary" in str(m["content"]) for m in result.messages)


async def test_role_scoping_filters_memory() -> None:
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
    )
    await engine.memory.upsert_fact(
        Fact(entity_id="c1", key="margin", value="42%", visibility=("sales",))
    )
    await engine.memory.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))

    client = FakeAnthropicClient(
        responses=[
            json.dumps({"tools": ["linear.create_issue"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    _seed_catalog(engine, client)

    sales = await engine.assemble(message="q", entity_id="c1", role="sales")
    support = await engine.assemble(message="q", entity_id="c1", role="support")

    assert "margin: 42%" in sales.system
    assert "margin: 42%" not in support.system
    assert "tier: pro" in sales.system and "tier: pro" in support.system
