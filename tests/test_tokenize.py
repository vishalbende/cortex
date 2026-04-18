import pytest

from contextengine.tokenize import CharEstimateTokenizer, TiktokenTokenizer, get_tokenizer


def test_char_estimate_empty() -> None:
    assert CharEstimateTokenizer().count("") == 0


def test_char_estimate_short() -> None:
    assert CharEstimateTokenizer().count("hi") == 1


def test_char_estimate_scales() -> None:
    assert CharEstimateTokenizer().count("a" * 400) == 100


def test_get_tokenizer_returns_some_tokenizer() -> None:
    t = get_tokenizer("claude-sonnet-4-5")
    assert t.count("hello world") >= 1


def test_tiktoken_tokenizer_counts_english() -> None:
    pytest.importorskip("tiktoken")
    t = TiktokenTokenizer()
    assert t.count("") == 0
    assert 1 <= t.count("hello world") <= 3
    assert t.count("the quick brown fox") >= 4
