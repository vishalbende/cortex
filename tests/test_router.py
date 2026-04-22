import json

import pytest

from contextengine.router import Router
from contextengine.types import Catalog, MCPCatalog, Tool, ToolCategory
from tests.fakes import FakeLLMClient


def _tool(name: str, mcp: str, desc: str = "d") -> Tool:
    return Tool(name=f"{mcp}.{name}", mcp=mcp, description=desc, input_schema={})


def _catalog() -> Catalog:
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
    return Catalog(mcps=(linear, stripe), version_hash="v1")


def test_router_constructs() -> None:
    r = Router(catalog=_catalog(), router_model="m", llm=FakeLLMClient())
    assert r.router_model == "m"


async def test_select_single_mcp_skips_pass_one() -> None:
    single = Catalog(
        mcps=(
            MCPCatalog(
                name="linear",
                summary="Linear.",
                categories=(
                    ToolCategory(
                        name="issues",
                        summary="",
                        tools=(_tool("create_issue", "linear"),),
                    ),
                ),
            ),
        ),
        version_hash="v1",
    )
    llm = FakeLLMClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    r = Router(catalog=single, router_model="haiku", llm=llm)
    decision = await r.select(message="make an issue")
    assert [t.name for t in decision.tools] == ["linear.create_issue"]
    assert decision.mcps_selected == ["linear"]
    assert len(llm.calls) == 1


async def test_select_two_pass_multi_mcp() -> None:
    llm = FakeLLMClient(
        responses=[
            json.dumps({"mcps": ["stripe", "linear"]}),
            json.dumps({"tools": ["stripe.create_refund"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", llm=llm)
    decision = await r.select(message="refund then file an issue")
    assert [t.name for t in decision.tools] == [
        "stripe.create_refund",
        "linear.create_issue",
    ]
    assert decision.mcps_selected == ["stripe", "linear"]
    assert len(llm.calls) == 3


async def test_select_caches_by_message() -> None:
    llm = FakeLLMClient(
        responses=[
            json.dumps({"mcps": ["linear"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", llm=llm)
    a = await r.select(message="make issue")
    b = await r.select(message="make issue")
    assert a is b
    assert len(llm.calls) == 2


async def test_required_tools_always_included() -> None:
    llm = FakeLLMClient(
        responses=[
            json.dumps({"mcps": ["linear"]}),
            json.dumps({"tools": []}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", llm=llm)
    decision = await r.select(
        message="anything", required_tools=("stripe.create_refund",)
    )
    assert "stripe.create_refund" in [t.name for t in decision.tools]


async def test_stable_prefix_passed_to_llm() -> None:
    llm = FakeLLMClient(
        responses=[
            json.dumps({"mcps": ["linear"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", llm=llm)
    await r.select(message="hi")
    assert llm.calls[0]["stable_prefix"] is not None
    assert "Available MCP servers" in llm.calls[0]["stable_prefix"]
    assert llm.calls[0]["json_mode"] is True
