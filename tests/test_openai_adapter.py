from contextengine.adapters.openai import assemble_to_openai
from contextengine.types import AssembleResult, AssembleStats


def _stats() -> AssembleStats:
    return AssembleStats(
        tokens_system=0,
        tokens_memory=0,
        tokens_tools=0,
        tokens_history=0,
        tokens_total=0,
        tools_loaded=(),
        tools_dropped=(),
        mcps_represented=(),
        elapsed_ms=0.0,
    )


def test_converts_tool_shape() -> None:
    r = AssembleResult(
        system="You are helpful.",
        tools=[
            {
                "name": "linear.create_issue",
                "description": "Create an issue",
                "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
            }
        ],
        messages=[{"role": "user", "content": "hi"}],
        stats=_stats(),
    )
    openai = assemble_to_openai(r)
    assert openai["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "linear.create_issue",
                "description": "Create an issue",
                "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
            },
        }
    ]


def test_prepends_system_message() -> None:
    r = AssembleResult(
        system="You are concise.",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        stats=_stats(),
    )
    out = assemble_to_openai(r)
    assert out["messages"][0] == {"role": "system", "content": "You are concise."}
    assert out["messages"][1] == {"role": "user", "content": "hi"}


def test_omits_system_when_empty() -> None:
    r = AssembleResult(
        system="",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        stats=_stats(),
    )
    out = assemble_to_openai(r)
    assert out["messages"][0]["role"] == "user"


def test_converts_tool_use_block_to_tool_calls() -> None:
    r = AssembleResult(
        system="",
        tools=[],
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Refunding."},
                    {
                        "type": "tool_use",
                        "id": "abc",
                        "name": "stripe.create_refund",
                        "input": {"order_id": "o1"},
                    },
                ],
            }
        ],
        stats=_stats(),
    )
    out = assemble_to_openai(r)
    msg = out["messages"][0]
    assert msg["content"] == "Refunding."
    assert msg["tool_calls"] == [
        {
            "id": "abc",
            "type": "function",
            "function": {"name": "stripe.create_refund", "arguments": {"order_id": "o1"}},
        }
    ]


def test_to_openai_method_available_on_result() -> None:
    r = AssembleResult(
        system="s",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        stats=_stats(),
    )
    assert r.to_openai()["messages"][0]["role"] == "system"
    assert r.to_anthropic()["system"] == "s"
