import json
from pathlib import Path

import pytest

from contextengine.memory import (
    Event,
    Fact,
    InMemoryStore,
    MemoryQuery,
)
from tests.fakes import FakeLLMClient


async def _seed(store: InMemoryStore) -> None:
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    await store.upsert_fact(
        Fact(entity_id="c1", key="margin.q1", value="42%", visibility=("sales",))
    )
    await store.append_event(Event(entity_id="c1", text="paid invoice", ts=1.0))
    await store.append_event(Event(entity_id="c1", text="filed ticket", ts=2.0))


async def test_list_facts_filters_by_pattern_and_role() -> None:
    store = InMemoryStore()
    await _seed(store)
    q = MemoryQuery(store=store)
    all_sales = await q.list_facts(entity_id="c1", role="sales")
    assert {f.key for f in all_sales} == {"tier", "margin.q1"}

    all_support = await q.list_facts(entity_id="c1", role="support")
    assert {f.key for f in all_support} == {"tier"}

    margins = await q.list_facts(
        entity_id="c1", key_pattern="margin.*", role="sales"
    )
    assert [f.key for f in margins] == ["margin.q1"]


async def test_list_events_filters_by_window() -> None:
    store = InMemoryStore()
    await _seed(store)
    q = MemoryQuery(store=store)
    since_mid = await q.list_events(entity_id="c1", since=1.5)
    assert [e.text for e in since_mid] == ["filed ticket"]


async def test_export_shape() -> None:
    store = InMemoryStore()
    await _seed(store)
    q = MemoryQuery(store=store)
    snap = await q.export(entity_id="c1")
    assert snap["entity_id"] == "c1"
    assert len(snap["facts"]) == 2
    assert len(snap["events"]) == 2
    # must round-trip through JSON
    payload = await q.export_json(entity_id="c1")
    reloaded = json.loads(payload)
    assert reloaded["entity_id"] == "c1"


async def test_erase_hard_deletes_entity() -> None:
    store = InMemoryStore()
    await _seed(store)
    q = MemoryQuery(store=store)
    await q.erase(entity_id="c1")
    assert await store.list_entities() == []


async def test_ask_uses_llm_and_filters_by_role() -> None:
    store = InMemoryStore()
    await _seed(store)
    llm = FakeLLMClient(responses=["Tier is pro."])
    q = MemoryQuery(store=store, llm=llm, model="haiku")
    result = await q.ask(entity_id="c1", question="what tier?", role="support")
    assert result.answer == "Tier is pro."
    prompt = llm.calls[0]["user"]
    assert "tier: pro" in prompt
    assert "margin.q1" not in prompt


async def test_ask_requires_llm() -> None:
    store = InMemoryStore()
    q = MemoryQuery(store=store)
    with pytest.raises(RuntimeError, match="requires"):
        await q.ask(entity_id="c1", question="?")
