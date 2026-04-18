from contextengine.budget import Budget, pack
from contextengine.tokenize import CharEstimateTokenizer
from contextengine.types import Message, Tool


def _tool(name: str, tokens: int) -> Tool:
    return Tool(
        name=name,
        mcp=name.split(".")[0],
        description="x",
        input_schema={},
        token_count=tokens,
    )


def test_budget_available() -> None:
    assert Budget(total=1000, reserved_output=100).available == 900


def test_budget_underflow_clamps_to_zero() -> None:
    assert Budget(total=100, reserved_output=500).available == 0


def test_pack_fits_all_when_room() -> None:
    tools = [_tool(f"m.t{i}", 10) for i in range(3)]
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    r = pack(
        budget=Budget(total=1000, reserved_output=0),
        system_tokens=50,
        memory_tokens=0,
        ranked_tools=tools,
        history=history,
        tokenizer=CharEstimateTokenizer(),
    )
    assert [t.name for t in r.tools] == ["m.t0", "m.t1", "m.t2"]
    assert len(r.messages) == 2
    assert r.tools_dropped == []


def test_pack_drops_low_ranked_tools() -> None:
    tools = [_tool(f"m.t{i}", 100) for i in range(5)]
    r = pack(
        budget=Budget(total=250, reserved_output=0),
        system_tokens=0,
        memory_tokens=0,
        ranked_tools=tools,
        history=[],
        tokenizer=CharEstimateTokenizer(),
    )
    assert [t.name for t in r.tools] == ["m.t0", "m.t1"]
    assert [t.name for t in r.tools_dropped] == ["m.t2", "m.t3", "m.t4"]


def test_pack_respects_required_tools() -> None:
    tools = [_tool(f"m.t{i}", 100) for i in range(5)]
    r = pack(
        budget=Budget(total=250, reserved_output=0),
        system_tokens=0,
        memory_tokens=0,
        ranked_tools=tools,
        history=[],
        tokenizer=CharEstimateTokenizer(),
        required_tools={"m.t4"},
    )
    names_in = {t.name for t in r.tools}
    assert "m.t4" in names_in


def test_pack_history_newest_first() -> None:
    history = [Message(role="user", content="old"), Message(role="user", content="newer")]
    r = pack(
        budget=Budget(total=20, reserved_output=0),
        system_tokens=0,
        memory_tokens=0,
        ranked_tools=[],
        history=history,
        tokenizer=CharEstimateTokenizer(),
    )
    assert [m.content for m in r.messages] == ["newer"]
    assert [m.content for m in r.messages_dropped] == ["old"]
