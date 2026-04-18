import json

import pytest

from contextengine.router import Router
from contextengine.types import Catalog, MCPCatalog, Tool, ToolCategory
from tests.fakes import FakeAnthropicClient


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
    r = Router(catalog=_catalog(), router_model="m")
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
    client = FakeAnthropicClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    r = Router(catalog=single, router_model="haiku", anthropic_client=client)
    decision = await r.select(message="make an issue")
    assert [t.name for t in decision.tools] == ["linear.create_issue"]
    assert decision.mcps_selected == ["linear"]
    assert len(client.messages.calls) == 1


async def test_select_two_pass_multi_mcp() -> None:
    client = FakeAnthropicClient(
        responses=[
            json.dumps({"mcps": ["stripe", "linear"]}),
            json.dumps({"tools": ["stripe.create_refund"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", anthropic_client=client)
    decision = await r.select(message="refund then file an issue")
    assert [t.name for t in decision.tools] == [
        "stripe.create_refund",
        "linear.create_issue",
    ]
    assert decision.mcps_selected == ["stripe", "linear"]
    assert len(client.messages.calls) == 3


async def test_select_caches_by_message() -> None:
    client = FakeAnthropicClient(
        responses=[
            json.dumps({"mcps": ["linear"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", anthropic_client=client)
    a = await r.select(message="make issue")
    b = await r.select(message="make issue")
    assert a is b
    assert len(client.messages.calls) == 2


async def test_required_tools_always_included() -> None:
    client = FakeAnthropicClient(
        responses=[
            json.dumps({"mcps": ["linear"]}),
            json.dumps({"tools": []}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", anthropic_client=client)
    decision = await r.select(
        message="anything", required_tools=("stripe.create_refund",)
    )
    assert "stripe.create_refund" in [t.name for t in decision.tools]


async def test_cache_control_applied_on_prefix() -> None:
    client = FakeAnthropicClient(
        responses=[
            json.dumps({"mcps": ["linear"]}),
            json.dumps({"tools": ["linear.create_issue"]}),
        ]
    )
    r = Router(catalog=_catalog(), router_model="haiku", anthropic_client=client)
    await r.select(message="hi")
    pass1_system = client.messages.calls[0]["system"]
    assert isinstance(pass1_system, list)
    assert any(b.get("cache_control", {}).get("type") == "ephemeral" for b in pass1_system)
