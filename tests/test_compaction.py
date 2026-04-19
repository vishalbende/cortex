import pytest

from contextengine.compaction import HistoryCompactor
from contextengine.types import Message
from tests.fakes import FakeAnthropicClient


def _make_history(n: int) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append(Message(role=role, content=f"turn-{i:02d}"))
    return out


def test_should_compact_threshold() -> None:
    c = HistoryCompactor(model="m", threshold=5, keep_recent=2)
    assert not c.should_compact(_make_history(5))
    assert c.should_compact(_make_history(6))


async def test_compact_summarizes_prefix_and_keeps_recent() -> None:
    client = FakeAnthropicClient(responses=["Prior conversation discussed onboarding."])
    c = HistoryCompactor(
        model="m", threshold=4, keep_recent=2, anthropic_client=client
    )
    hist = _make_history(10)
    out = await c.compact(hist)
    assert len(out) == 3
    assert out[0].content.startswith(HistoryCompactor.SENTINEL)
    assert "onboarding" in str(out[0].content)
    assert out[-2].content == "turn-08"
    assert out[-1].content == "turn-09"


async def test_compact_noop_below_threshold() -> None:
    c = HistoryCompactor(model="m", threshold=100)
    hist = _make_history(5)
    out = await c.compact(hist)
    assert out is hist


async def test_compact_extends_existing_summary() -> None:
    client = FakeAnthropicClient(responses=["Extended rolling summary."])
    c = HistoryCompactor(
        model="m", threshold=4, keep_recent=2, anthropic_client=client
    )
    hist: list[Message] = [
        Message(role="user", content=f"{HistoryCompactor.SENTINEL} prior rolling summary"),
        *_make_history(8),
    ]
    out = await c.compact(hist)
    assert out[0].content.startswith(HistoryCompactor.SENTINEL)
    assert "Extended" in str(out[0].content)
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "prior rolling summary" in prompt
