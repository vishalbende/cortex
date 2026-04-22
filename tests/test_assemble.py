"""End-to-end assemble() test using injected fakes (no live LLM/MCP)."""
import json

import pytest

from contextengine import ContextEngine, MCPServer
from contextengine.router import Router
from contextengine.types import Catalog, MCPCatalog, Message, Tool, ToolCategory
from tests.fakes import FakeLLMClient


def _tool(name: str, mcp: str, tokens: int = 50) -> Tool:
    return Tool(
        name=f"{mcp}.{name}",
        mcp=mcp,
        description=f"Does {name}",
        input_schema={"type": "object"},
        category="general",
        token_count=tokens,
    )


def _seed_engine(e: ContextEngine, llm: FakeLLMClient) -> None:
    linear = MCPCatalog(
        name="linear",
        summary="Linear issue tracker.",
        categories=(
            ToolCategory(
                name="issues",
                summary="CRUD issues.",
                tools=(_tool("create_issue", "linear"), _tool("list_issues", "linear")),
            ),
        ),
    )
    stripe = MCPCatalog(
        name="stripe",
        summary="Stripe payments.",
        categories=(
            ToolCategory(
                name="refunds",
                summary="Refund flows.",
                tools=(_tool("create_refund", "stripe"),),
            ),
        ),
    )
    catalog = Catalog(mcps=(linear, stripe), version_hash="v1")
    e._catalog = catalog
    e._router = Router(catalog=catalog, router_model=e.router_model, llm=llm)


async def test_assemble_end_to_end_with_budget_pack() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"]), MCPServer(name="stripe", url="http://y")],
        model="claude-sonnet-4-5",
        system_prompt="You are a helpful agent.",
        budget=1000,
        reserved_output=0,
    )
    llm = FakeLLMClient(
        responses=[
            json.dumps({"mcps": ["stripe", "linear"]}),
            json.dumps({"tools": ["stripe.create_refund"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    _seed_engine(e, llm)

    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    result = await e.assemble(message="refund order then file a bug", history=history)

    assert result.system.startswith("You are a helpful agent.")
    tool_names = [t["name"] for t in result.tools]
    assert "stripe.create_refund" in tool_names
    assert "linear.create_issue" in tool_names
    assert set(result.stats.mcps_represented) == {"stripe", "linear"}
    assert result.messages[-1]["content"] == "refund order then file a bug"
    assert result.stats.tokens_total <= e.budget.available


async def test_assemble_with_memory_block() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        system_prompt="agent.",
    )
    llm = FakeLLMClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    _seed_engine(e, llm)

    result = await e.assemble(
        message="create an issue",
        memory="User prefers concise bug reports.",
    )
    assert "agent." in result.system
    assert "User prefers concise bug reports." in result.system
    assert result.stats.tokens_memory > 0


async def test_assemble_respects_required_tools() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"]), MCPServer(name="stripe", url="http://y")],
        model="claude-sonnet-4-5",
    )
    llm = FakeLLMClient(
        responses=[
            json.dumps({"mcps": ["linear"]}),
            json.dumps({"tools": []}),
        ]
    )
    _seed_engine(e, llm)

    result = await e.assemble(
        message="do anything",
        required_tools=("stripe.create_refund",),
    )
    assert "stripe.create_refund" in [t["name"] for t in result.tools]
