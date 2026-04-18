import pytest

from contextengine._json import extract_json


def test_plain_json() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_json_markdown_fenced() -> None:
    text = '```json\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_json_plain_fenced() -> None:
    text = '```\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_json_with_leading_whitespace() -> None:
    assert extract_json('   \n\n  {"a": [1, 2]}   ') == {"a": [1, 2]}


def test_invalid_json_raises() -> None:
    with pytest.raises(ValueError):
        extract_json("not json at all")
