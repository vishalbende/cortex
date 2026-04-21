import pytest

from contextengine.memory import (
    Event,
    Fact,
    InMemoryStore,
    MemoryCompactor,
    SUMMARY_KEY,
)
from tests.fakes import FakeLLMClient


def _compactor(llm: FakeLLMClient) -> MemoryCompactor:
    return MemoryCompactor(
        model="m",
        llm=llm,
        fact_threshold=3,
        event_threshold=3,
        keep_recent_events=2,
        version_floor=2,
    )


async def test_noop_below_threshold() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="c1", key="a", value="1"))
    llm = FakeLLMClient()
    c = _compactor(llm)
    result = await c.compact(store, "c1")
    assert not result.summary_written
    assert result.facts_before == result.facts_after == 1
    assert llm.calls == []


async def test_compact_folds_stale_facts_into_summary() -> None:
    store = InMemoryStore()
    for i in range(5):
        await store.upsert_fact(Fact(entity_id="c1", key=f"k{i}", value=f"v{i}"))
    await store.upsert_fact(
        Fact(entity_id="c1", key="kept_high_version", value="v", version=5)
    )
    for i in range(5):
        await store.append_event(Event(entity_id="c1", text=f"e{i}", ts=float(i)))

    llm = FakeLLMClient(responses=["Rolling summary of prior state."])
    c = _compactor(llm)
    result = await c.compact(store, "c1")

    assert result.summary_written
    assert result.events_after == 2
    mem = await store.get("c1")
    keys = {f.key for f in mem.facts}
    assert SUMMARY_KEY in keys
    assert "kept_high_version" in keys
    assert llm.calls, "LLM should be invoked"


async def test_compact_versions_summary() -> None:
    store = InMemoryStore()
    for i in range(4):
        await store.upsert_fact(Fact(entity_id="c1", key=f"k{i}", value="v"))
    for i in range(4):
        await store.append_event(Event(entity_id="c1", text=f"e{i}", ts=float(i)))

    llm = FakeLLMClient(responses=["first summary", "second summary"])
    c = _compactor(llm)
    await c.compact(store, "c1")
    mem = await store.get("c1")
    summary = next(f for f in mem.facts if f.key == SUMMARY_KEY)
    assert summary.version == 1
    assert summary.value == "first summary"

    # Add more facts to trigger another compaction
    for i in range(4, 8):
        await store.upsert_fact(Fact(entity_id="c1", key=f"k{i}", value="v"))
    for i in range(4, 8):
        await store.append_event(Event(entity_id="c1", text=f"e{i}", ts=float(i)))

    await c.compact(store, "c1")
    mem = await store.get("c1")
    summary = next(f for f in mem.facts if f.key == SUMMARY_KEY)
    assert summary.version == 2
    assert summary.value == "second summary"


async def test_compact_keeps_recent_events() -> None:
    store = InMemoryStore()
    for i in range(5):
        await store.upsert_fact(Fact(entity_id="c1", key=f"k{i}", value="v"))
    for i in range(10):
        await store.append_event(Event(entity_id="c1", text=f"event-{i:02d}", ts=float(i)))

    c = _compactor(FakeLLMClient(responses=["s"]))
    await c.compact(store, "c1")
    mem = await store.get("c1")
    event_texts = [e.text for e in mem.events]
    assert event_texts == ["event-08", "event-09"]


async def test_store_delete_fact_and_prune_events() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="c1", key="a", value="1"))
    await store.upsert_fact(Fact(entity_id="c1", key="b", value="2"))
    await store.delete_fact(entity_id="c1", key="a")
    mem = await store.get("c1")
    assert {f.key for f in mem.facts} == {"b"}

    keep = Event(entity_id="c1", text="keep", ts=2.0)
    drop = Event(entity_id="c1", text="drop", ts=1.0)
    await store.append_event(keep)
    await store.append_event(drop)
    await store.prune_events(entity_id="c1", keep=(keep,))
    mem = await store.get("c1")
    assert [e.text for e in mem.events] == ["keep"]
