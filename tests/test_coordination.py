import json

import pytest

from contextengine import ContextEngine, MCPServer, MultiAgentCoordinator
from contextengine.coordination.handoff import HandoffProtocol
from contextengine.memory import Fact, InMemoryStore
from contextengine.router import Router
from contextengine.types import Catalog, MCPCatalog, Tool, ToolCategory
from tests.fakes import FakeLLMClient


def _make_engine(role: str, llm: FakeLLMClient) -> ContextEngine:
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
                        input_schema={},
                    ),
                ),
            ),
        ),
    )
    catalog = Catalog(mcps=(linear,), version_hash="v1")
    engine._catalog = catalog
    engine._router = Router(catalog=catalog, router_model=engine.router_model, llm=llm)
    return engine


async def test_handoff_writes_visible_event() -> None:
    store = InMemoryStore()
    proto = HandoffProtocol(store)
    h = await proto.handoff(
        entity_id="c1",
        from_role="support",
        to_role="sales",
        reason="pricing question",
        summary="customer asked about enterprise tier",
    )
    assert h.from_role == "support"
    mem = await store.get("c1")
    assert len(mem.events) == 1
    ev = mem.events[0]
    assert ev.source == "handoff"
    assert "support" in ev.text and "sales" in ev.text
    assert set(ev.visibility) == {"sales", "support"}


async def test_handoff_visible_to_both_roles_only() -> None:
    store = InMemoryStore()
    proto = HandoffProtocol(store)
    await proto.handoff(
        entity_id="c1",
        from_role="support",
        to_role="sales",
        reason="r",
    )
    mem = await store.get("c1")
    ev = mem.events[0]
    assert ev.visible_to("support")
    assert ev.visible_to("sales")
    assert not ev.visible_to("billing")


async def test_list_handoffs_parses_back() -> None:
    store = InMemoryStore()
    proto = HandoffProtocol(store)
    await proto.handoff(
        entity_id="c1", from_role="a", to_role="b", reason="r1", summary="s1"
    )
    await proto.handoff(
        entity_id="c1", from_role="b", to_role="a", reason="r2"
    )
    handoffs = await proto.list_handoffs(entity_id="c1")
    assert len(handoffs) == 2
    assert handoffs[0].from_role == "a"
    assert handoffs[0].to_role == "b"
    assert handoffs[0].reason == "r1"
    assert handoffs[0].summary == "s1"


async def test_coordinator_shares_memory_across_engines() -> None:
    coord = MultiAgentCoordinator()
    llm_a = FakeLLMClient()
    llm_b = FakeLLMClient()
    engine_a = _make_engine("support", llm_a)
    engine_b = _make_engine("sales", llm_b)

    coord.register("support", engine_a)
    coord.register("sales", engine_b)

    await engine_a.memory.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    mem_b = await engine_b.memory.get("c1")
    assert [f.key for f in mem_b.facts] == ["tier"]
    assert coord.memory is engine_a.memory is engine_b.memory


async def test_coordinator_handoff_visible_to_next_role() -> None:
    coord = MultiAgentCoordinator()
    llm_a = FakeLLMClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    llm_b = FakeLLMClient(responses=[json.dumps({"tools": ["linear.create_issue"]})])
    coord.register("support", _make_engine("support", llm_a))
    coord.register("sales", _make_engine("sales", llm_b))

    await coord.handoff(
        entity_id="c1",
        from_role="support",
        to_role="sales",
        reason="pricing",
        summary="wants enterprise",
    )
    result = await coord.assemble("sales", message="hi", entity_id="c1")
    assert "Handoff support → sales" in result.system
    assert "wants enterprise" in result.system


async def test_coordinator_rejects_unknown_roles() -> None:
    coord = MultiAgentCoordinator()
    with pytest.raises(KeyError):
        await coord.handoff(
            entity_id="c1", from_role="ghost", to_role="sales", reason="r"
        )
