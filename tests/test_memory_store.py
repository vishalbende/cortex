from pathlib import Path

import pytest

from contextengine.memory import Event, Fact, InMemoryStore, JSONStore


async def test_in_memory_upsert_and_fetch() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    mem = await store.get("c1")
    assert [f.key for f in mem.facts] == ["tier"]
    assert mem.facts[0].value == "pro"
    assert mem.facts[0].version == 1


async def test_in_memory_upsert_bumps_version() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="enterprise"))
    mem = await store.get("c1")
    assert mem.facts[0].value == "enterprise"
    assert mem.facts[0].version == 2


async def test_in_memory_events_append() -> None:
    store = InMemoryStore()
    await store.append_event(Event(entity_id="c1", text="paid invoice", ts=1.0))
    await store.append_event(Event(entity_id="c1", text="requested refund", ts=2.0))
    mem = await store.get("c1")
    assert [e.text for e in mem.events] == ["paid invoice", "requested refund"]


async def test_in_memory_list_entities() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="a", key="x", value="1"))
    await store.append_event(Event(entity_id="b", text="x", ts=1.0))
    assert await store.list_entities() == ["a", "b"]


async def test_in_memory_delete() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="a", key="x", value="1"))
    await store.delete("a")
    assert await store.list_entities() == []


async def test_json_store_roundtrip(tmp_path: Path) -> None:
    store = JSONStore(tmp_path)
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    await store.append_event(Event(entity_id="c1", text="paid", ts=1.0))

    reopened = JSONStore(tmp_path)
    mem = await reopened.get("c1")
    assert [f.key for f in mem.facts] == ["tier"]
    assert [e.text for e in mem.events] == ["paid"]


async def test_json_store_version_bumps_on_update(tmp_path: Path) -> None:
    store = JSONStore(tmp_path)
    await store.upsert_fact(Fact(entity_id="c1", key="plan", value="basic"))
    await store.upsert_fact(Fact(entity_id="c1", key="plan", value="pro"))
    mem = await store.get("c1")
    assert [f.value for f in mem.facts if f.key == "plan"] == ["pro"]
    assert [f.version for f in mem.facts if f.key == "plan"] == [2]


async def test_visibility_filtering() -> None:
    fact = Fact(entity_id="c1", key="x", value="1", visibility=("sales",))
    assert fact.visible_to("sales")
    assert not fact.visible_to("support")
    assert not fact.visible_to("")
    unrestricted = Fact(entity_id="c1", key="y", value="2")
    assert unrestricted.visible_to("any")
