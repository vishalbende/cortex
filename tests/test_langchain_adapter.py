from contextengine.adapters.langchain import (
    assemble_to_langchain,
    messages_to_langchain_dicts,
    tools_to_langchain_schemas,
)
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


def _result() -> AssembleResult:
    return AssembleResult(
        system="You are helpful.",
        tools=[
            {
                "name": "linear.create_issue",
                "description": "Create",
                "input_schema": {"type": "object"},
            }
        ],
        messages=[{"role": "user", "content": "hi"}],
        stats=_stats(),
    )


def test_tools_to_langchain_schemas() -> None:
    schemas = tools_to_langchain_schemas(_result())
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "linear.create_issue",
                "description": "Create",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_messages_prepend_system() -> None:
    msgs = messages_to_langchain_dicts(_result())
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_assemble_to_langchain_combined_shape() -> None:
    out = assemble_to_langchain(_result())
    assert set(out) == {"messages", "tools"}
    assert len(out["messages"]) == 2
    assert len(out["tools"]) == 1
