import pytest

from contextengine.compaction import HistoryCompactor
from contextengine.types import Message
from tests.fakes import FakeLLMClient


def _make_history(n: int) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append(Message(role=role, content=f"turn-{i:02d}"))
    return out


def test_should_compact_threshold() -> None:
    c = HistoryCompactor(model="m", llm=FakeLLMClient(), threshold=5, keep_recent=2)
    assert not c.should_compact(_make_history(5))
    assert c.should_compact(_make_history(6))


async def test_compact_summarizes_prefix_and_keeps_recent() -> None:
    llm = FakeLLMClient(responses=["Prior conversation discussed onboarding."])
    c = HistoryCompactor(model="m", llm=llm, threshold=4, keep_recent=2)
    hist = _make_history(10)
    out = await c.compact(hist)
    assert len(out) == 3
    assert out[0].content.startswith(HistoryCompactor.SENTINEL)
    assert "onboarding" in str(out[0].content)
    assert out[-2].content == "turn-08"
    assert out[-1].content == "turn-09"


async def test_compact_noop_below_threshold() -> None:
    c = HistoryCompactor(model="m", llm=FakeLLMClient(), threshold=100)
    hist = _make_history(5)
    out = await c.compact(hist)
    assert out is hist


async def test_compact_extends_existing_summary() -> None:
    llm = FakeLLMClient(responses=["Extended rolling summary."])
    c = HistoryCompactor(model="m", llm=llm, threshold=4, keep_recent=2)
    hist: list[Message] = [
        Message(role="user", content=f"{HistoryCompactor.SENTINEL} prior rolling summary"),
        *_make_history(8),
    ]
    out = await c.compact(hist)
    assert out[0].content.startswith(HistoryCompactor.SENTINEL)
    assert "Extended" in str(out[0].content)
    assert "prior rolling summary" in llm.calls[0]["user"]
