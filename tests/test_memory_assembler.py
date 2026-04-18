import pytest

from contextengine.memory import (
    Event,
    Fact,
    InMemoryStore,
    MemoryAssembler,
)
from contextengine.tokenize import CharEstimateTokenizer


async def test_assemble_empty_memory_returns_empty_string() -> None:
    store = InMemoryStore()
    a = MemoryAssembler(store, CharEstimateTokenizer())
    assert await a.assemble(entity_id="c1") == ""


async def test_assemble_formats_facts_and_events() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    await store.append_event(Event(entity_id="c1", text="paid invoice", ts=1.0))

    a = MemoryAssembler(store, CharEstimateTokenizer())
    block = await a.assemble(entity_id="c1", budget_tokens=10_000)
    assert "[memory]" in block
    assert "tier: pro" in block
    assert "paid invoice" in block
    assert block.endswith("[/memory]")


async def test_assemble_filters_by_role() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="c1", key="margin", value="42%", visibility=("sales",)))
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    await store.append_event(
        Event(entity_id="c1", text="internal note", visibility=("support",))
    )

    a = MemoryAssembler(store, CharEstimateTokenizer())
    sales_block = await a.assemble(entity_id="c1", role="sales", budget_tokens=10_000)
    assert "margin: 42%" in sales_block
    assert "tier: pro" in sales_block
    assert "internal note" not in sales_block

    support_block = await a.assemble(entity_id="c1", role="support", budget_tokens=10_000)
    assert "margin: 42%" not in support_block
    assert "internal note" in support_block


async def test_assemble_truncates_events_before_facts() -> None:
    store = InMemoryStore()
    await store.upsert_fact(Fact(entity_id="c1", key="tier", value="pro"))
    for i in range(20):
        await store.append_event(Event(entity_id="c1", text=f"event-{i:02d}", ts=float(i)))

    a = MemoryAssembler(store, CharEstimateTokenizer())
    tight_block = await a.assemble(entity_id="c1", budget_tokens=12)
    assert "tier: pro" in tight_block
