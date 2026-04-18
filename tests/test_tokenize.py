from contextengine.tokenize import CharEstimateTokenizer, get_tokenizer


def test_char_estimate_empty() -> None:
    assert CharEstimateTokenizer().count("") == 0


def test_char_estimate_short() -> None:
    assert CharEstimateTokenizer().count("hi") == 1


def test_char_estimate_scales() -> None:
    assert CharEstimateTokenizer().count("a" * 400) == 100


def test_get_tokenizer_returns_estimator() -> None:
    t = get_tokenizer("claude-sonnet-4-5")
    assert t.count("hello world") >= 1
