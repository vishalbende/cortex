import pytest

from contextengine.tokenize import AsyncAnthropicTokenizer


class _FakeResult:
    def __init__(self, n: int) -> None:
        self.input_tokens = n


class _FakeAsyncMessages:
    def __init__(self, n: int) -> None:
        self._n = n
        self.calls: list[dict] = []

    async def count_tokens(self, **kwargs) -> _FakeResult:
        self.calls.append(kwargs)
        return _FakeResult(self._n)


class _FakeAsyncAnthropic:
    def __init__(self, n: int) -> None:
        self.messages = _FakeAsyncMessages(n)


async def test_async_tokenizer_counts_via_async_api() -> None:
    client = _FakeAsyncAnthropic(n=21)
    tk = AsyncAnthropicTokenizer(model="claude-sonnet-4-5", client=client)
    assert await tk.count("hello") == 21


async def test_async_tokenizer_caches_by_hash() -> None:
    client = _FakeAsyncAnthropic(n=7)
    tk = AsyncAnthropicTokenizer(model="claude-sonnet-4-5", client=client)
    await tk.count("same text")
    await tk.count("same text")
    assert len(client.messages.calls) == 1


async def test_async_tokenizer_empty_short_circuits() -> None:
    client = _FakeAsyncAnthropic(n=99)
    tk = AsyncAnthropicTokenizer(model="claude-sonnet-4-5", client=client)
    assert await tk.count("") == 0
    assert len(client.messages.calls) == 0


async def test_count_many_concurrent() -> None:
    client = _FakeAsyncAnthropic(n=5)
    tk = AsyncAnthropicTokenizer(model="claude-sonnet-4-5", client=client)
    counts = await tk.count_many(["a", "b", "c"])
    assert counts == [5, 5, 5]
    # Distinct inputs → 3 API calls
    assert len(client.messages.calls) == 3
