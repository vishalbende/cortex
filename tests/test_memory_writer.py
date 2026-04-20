import json

import pytest

from contextengine.memory import InMemoryStore, MemoryWriter
from tests.fakes import FakeLLMClient


async def test_writer_upserts_facts_and_appends_events() -> None:
    store = InMemoryStore()
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "facts": [
                        {"key": "tier", "value": "pro", "source": "assistant"},
                        {"key": "region", "value": "eu", "source": "tool"},
                    ],
                    "events": [
                        {"text": "user upgraded tier", "source": "assistant"},
                    ],
                }
            )
        ]
    )
    writer = MemoryWriter(store=store, model="haiku", llm=llm)
    result = await writer.write(
        entity_id="c1",
        user_message="upgrade me",
        assistant_response="Upgraded to pro.",
        tool_results=[{"name": "stripe.upgrade", "result": {"tier": "pro", "region": "eu"}}],
    )
    assert result.facts_upserted == 2
    assert result.events_appended == 1
    mem = await store.get("c1")
    assert {f.key for f in mem.facts} == {"tier", "region"}
    assert [e.text for e in mem.events] == ["user upgraded tier"]


async def test_writer_applies_role_visibility() -> None:
    store = InMemoryStore()
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "facts": [{"key": "x", "value": "1"}],
                    "events": [{"text": "evt"}],
                }
            )
        ]
    )
    writer = MemoryWriter(store=store, model="haiku", llm=llm)
    await writer.write(
        entity_id="c1",
        user_message="u",
        assistant_response="a",
        role="sales",
    )
    mem = await store.get("c1")
    assert mem.facts[0].visibility == ("sales",)
    assert mem.events[0].visibility == ("sales",)


async def test_writer_skips_malformed_entries() -> None:
    store = InMemoryStore()
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "facts": [
                        {"key": "", "value": "bad"},
                        {"value": "no-key"},
                        {"key": "good", "value": "yes"},
                    ],
                    "events": [{"text": ""}, {"text": "real event"}],
                }
            )
        ]
    )
    writer = MemoryWriter(store=store, model="haiku", llm=llm)
    result = await writer.write(
        entity_id="c1",
        user_message="u",
        assistant_response="a",
    )
    assert result.facts_upserted == 1
    assert result.events_appended == 1
