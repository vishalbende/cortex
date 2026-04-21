from contextengine.tokenize import AnthropicTokenizer


class _FakeResult:
    def __init__(self, n: int) -> None:
        self.input_tokens = n


class _FakeMessages:
    def __init__(self, n: int) -> None:
        self._n = n
        self.calls: list[dict] = []

    def count_tokens(self, **kwargs) -> _FakeResult:
        self.calls.append(kwargs)
        return _FakeResult(self._n)


class _FakeAnthropic:
    def __init__(self, n: int) -> None:
        self.messages = _FakeMessages(n)


def test_anthropic_tokenizer_counts_via_api() -> None:
    client = _FakeAnthropic(n=42)
    tk = AnthropicTokenizer(model="claude-sonnet-4-5", client=client)
    assert tk.count("hello world") == 42
    assert len(client.messages.calls) == 1


def test_anthropic_tokenizer_caches_by_text() -> None:
    client = _FakeAnthropic(n=7)
    tk = AnthropicTokenizer(model="claude-sonnet-4-5", client=client)
    tk.count("same input")
    tk.count("same input")
    tk.count("same input")
    assert len(client.messages.calls) == 1


def test_anthropic_tokenizer_empty_short_circuits() -> None:
    client = _FakeAnthropic(n=100)
    tk = AnthropicTokenizer(model="claude-sonnet-4-5", client=client)
    assert tk.count("") == 0
    assert len(client.messages.calls) == 0
