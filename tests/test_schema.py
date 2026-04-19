import pytest

from contextengine.mcp.schema import normalize_tool
from contextengine.tokenize import CharEstimateTokenizer


class _FakeRawTool:
    def __init__(self, name: str, description: str, inputSchema: dict) -> None:
        self.name = name
        self.description = description
        self.inputSchema = inputSchema


def test_normalize_from_object() -> None:
    raw = _FakeRawTool("create_issue", "Creates a Linear issue", {"type": "object"})
    t = normalize_tool(raw, mcp_name="linear", tokenizer=CharEstimateTokenizer())
    assert t.name == "linear.create_issue"
    assert t.mcp == "linear"
    assert t.description == "Creates a Linear issue"
    assert t.input_schema == {"type": "object"}
    assert t.token_count > 0


def test_normalize_from_dict() -> None:
    raw = {"name": "list_issues", "description": "", "inputSchema": {}}
    t = normalize_tool(raw, mcp_name="linear", tokenizer=CharEstimateTokenizer())
    assert t.name == "linear.list_issues"


def test_normalize_missing_name_raises() -> None:
    with pytest.raises(ValueError, match="missing 'name'"):
        normalize_tool({}, mcp_name="linear", tokenizer=CharEstimateTokenizer())
