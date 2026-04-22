import json

import pytest

from contextengine import ContextEngine, MCPServer
from contextengine.router import Router
from contextengine.streaming import refine_tools_for_followup, stream_assemble
from contextengine.types import Catalog, MCPCatalog, Tool, ToolCategory
from tests.fakes import FakeLLMClient


def _seed(engine: ContextEngine, llm: FakeLLMClient) -> None:
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
                        token_count=20,
                    ),
                    Tool(
                        name="linear.comment",
                        mcp="linear",
                        description="Comment",
                        input_schema={},
                        token_count=15,
                    ),
                ),
            ),
        ),
    )
    catalog = Catalog(mcps=(linear,), version_hash="v1")
    engine._catalog = catalog
    engine._router = Router(catalog=catalog, router_model=engine.router_model, llm=llm)


async def test_stream_assemble_yields_chunks() -> None:
    llm = FakeLLMClient(
        responses=[json.dumps({"tools": ["linear.create_issue"]})]
    )
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        llm_client=llm,
    )
    _seed(engine, llm)

    phases: list[str] = []
    async for chunk in stream_assemble(engine, message="hi"):
        phases.append(chunk.phase)
    assert phases == ["routing", "memory", "packed", "final"]


async def test_refine_tools_appends_without_removing() -> None:
    llm = FakeLLMClient(
        responses=[
            json.dumps({"tools": ["linear.create_issue", "linear.comment"]}),
        ]
    )
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        llm_client=llm,
    )
    _seed(engine, llm)

    current = [
        {"name": "linear.create_issue", "description": "Create", "input_schema": {}},
    ]
    result = await refine_tools_for_followup(
        engine,
        last_tool_use={"name": "linear.create_issue", "input": {}},
        current_tools=current,
        message="now add a comment",
    )
    names = [t["name"] for t in result.tools]
    assert names[0] == "linear.create_issue"
    assert "linear.comment" in names
    assert len(names) == 2


async def test_refine_tools_requires_valid_name() -> None:
    llm = FakeLLMClient()
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        llm_client=llm,
    )
    _seed(engine, llm)
    with pytest.raises(ValueError, match="name"):
        await refine_tools_for_followup(
            engine,
            last_tool_use={"input": {}},
            current_tools=[],
            message="x",
        )
