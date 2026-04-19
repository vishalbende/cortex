"""End-to-end process_turn test (memory writeback after a turn)."""
import json

import pytest

from contextengine import ContextEngine, MCPServer
from tests.fakes import FakeAnthropicClient


async def test_process_turn_writes_memory() -> None:
    client = FakeAnthropicClient(
        responses=[
            json.dumps(
                {
                    "facts": [{"key": "tier", "value": "pro"}],
                    "events": [{"text": "upgraded"}],
                }
            )
        ]
    )
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        anthropic_client=client,
    )
    result = await engine.process_turn(
        entity_id="c1",
        user_message="upgrade me",
        assistant_response="done",
    )
    assert result.facts_upserted == 1
    assert result.events_appended == 1
    mem = await engine.memory.get("c1")
    assert mem.facts[0].key == "tier"
    assert mem.facts[0].value == "pro"


async def test_process_turn_role_scoping() -> None:
    client = FakeAnthropicClient(
        responses=[
            json.dumps(
                {
                    "facts": [{"key": "margin", "value": "42%"}],
                    "events": [],
                }
            )
        ]
    )
    engine = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
        anthropic_client=client,
    )
    await engine.process_turn(
        entity_id="c1",
        user_message="u",
        assistant_response="a",
        role="sales",
    )
    mem = await engine.memory.get("c1")
    assert mem.facts[0].visibility == ("sales",)
