import json

import pytest

from contextengine.memory import (
    AllowAllPolicy,
    Event,
    Fact,
    InMemoryStore,
    MemoryWriter,
    PolicyViolation,
    RoleBasedWritePolicy,
    Rule,
)
from contextengine.memory.policy import enforce_append, enforce_upsert
from tests.fakes import FakeLLMClient


def _fact(key: str = "k", value: str = "v") -> Fact:
    return Fact(entity_id="c1", key=key, value=value)


def _event(text: str = "e") -> Event:
    return Event(entity_id="c1", text=text)


def test_allow_all_accepts_everything() -> None:
    p = AllowAllPolicy()
    assert p.can_upsert("any", _fact())
    assert p.can_append("any", _event())


def test_role_based_allows_matching_rule() -> None:
    p = RoleBasedWritePolicy.from_rules(
        [Rule(roles=("sales",), key_pattern="margin.*")]
    )
    assert p.can_upsert("sales", _fact(key="margin.q1"))
    assert not p.can_upsert("sales", _fact(key="other"))
    assert not p.can_upsert("support", _fact(key="margin.q1"))


def test_default_deny_when_no_rule_matches() -> None:
    p = RoleBasedWritePolicy.from_rules([Rule(roles=("sales",))])
    assert not p.can_upsert("support", _fact())


def test_default_allow_flag() -> None:
    p = RoleBasedWritePolicy(rules=[], default_allow=True)
    assert p.can_upsert("any", _fact())
    assert p.can_append("any", _event())


def test_wildcard_role_matches_any() -> None:
    p = RoleBasedWritePolicy.from_rules([Rule(roles=(), key_pattern="public.*")])
    assert p.can_upsert("sales", _fact(key="public.x"))
    assert p.can_upsert("support", _fact(key="public.y"))
    assert not p.can_upsert("sales", _fact(key="private"))


def test_events_allow_flag() -> None:
    p = RoleBasedWritePolicy.from_rules(
        [Rule(roles=("audit",), allow_events=False)]
    )
    assert not p.can_append("audit", _event())


def test_enforce_helpers_raise() -> None:
    p = RoleBasedWritePolicy.from_rules([])
    with pytest.raises(PolicyViolation):
        enforce_upsert(p, "x", _fact())
    with pytest.raises(PolicyViolation):
        enforce_append(p, "x", _event())


async def test_writer_enforces_policy() -> None:
    store = InMemoryStore()
    policy = RoleBasedWritePolicy.from_rules(
        [Rule(roles=("sales",), key_pattern="public.*")]
    )
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "facts": [
                        {"key": "public.a", "value": "1"},
                        {"key": "private.b", "value": "2"},
                    ],
                    "events": [{"text": "ev1"}],
                }
            )
        ]
    )
    writer = MemoryWriter(store=store, model="m", llm=llm, policy=policy)
    result = await writer.write(
        entity_id="c1",
        user_message="u",
        assistant_response="a",
        role="sales",
    )
    assert result.facts_upserted == 1
    assert result.facts_rejected == 1
    assert result.events_appended == 1
    mem = await store.get("c1")
    assert {f.key for f in mem.facts} == {"public.a"}


async def test_writer_rejects_events_when_policy_denies() -> None:
    store = InMemoryStore()
    policy = RoleBasedWritePolicy.from_rules(
        [Rule(roles=("audit",), allow_events=False)]
    )
    llm = FakeLLMClient(
        responses=[
            json.dumps({"facts": [], "events": [{"text": "ev1"}, {"text": "ev2"}]})
        ]
    )
    writer = MemoryWriter(store=store, model="m", llm=llm, policy=policy)
    result = await writer.write(
        entity_id="c1",
        user_message="u",
        assistant_response="a",
        role="audit",
    )
    assert result.events_appended == 0
    assert result.events_rejected == 2
