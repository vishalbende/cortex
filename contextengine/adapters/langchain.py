"""LangChain / LangGraph adapter: expose contextengine results as LangChain inputs.

LangChain is an optional dependency — the adapter fails with a clear
ImportError when langchain-core is missing, but the non-LangChain helpers
(`tools_to_langchain_schemas`, `messages_to_langchain_dicts`) work unconditionally.
"""
from __future__ import annotations

from typing import Any

from contextengine.types import AssembleResult


def tools_to_langchain_schemas(result: AssembleResult) -> list[dict[str, Any]]:
    """Return tool schemas in the shape LangChain's `bind_tools` expects
    (OpenAI-compatible function spec).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in result.tools
    ]


def messages_to_langchain_dicts(
    result: AssembleResult,
) -> list[dict[str, Any]]:
    """Return messages in the {role, content} dict shape LangChain accepts."""
    out: list[dict[str, Any]] = []
    if result.system:
        out.append({"role": "system", "content": result.system})
    for m in result.messages:
        out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return out


def assemble_to_langchain(result: AssembleResult) -> dict[str, Any]:
    """Combined shape: {messages, tools} for direct use with a LangChain model.

    Example:
        from langchain_anthropic import ChatAnthropic

        ctx = await engine.assemble(message="...")
        payload = assemble_to_langchain(ctx)
        model = ChatAnthropic(model="claude-sonnet-4-5").bind_tools(payload["tools"])
        response = await model.ainvoke(payload["messages"])
    """
    return {
        "messages": messages_to_langchain_dicts(result),
        "tools": tools_to_langchain_schemas(result),
    }


def langgraph_context_node(engine: Any, *, entity_id_key: str = "entity_id"):
    """Factory that returns a LangGraph node function for context assembly.

    The returned coroutine reads state[entity_id_key] and state["message"],
    calls `engine.assemble`, and writes `system`, `tools`, `messages`
    back into state. Designed to slot into LangGraph StateGraph.
    """

    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        result = await engine.assemble(
            message=state.get("message", ""),
            entity_id=state.get(entity_id_key),
            role=state.get("role", ""),
            history=state.get("history", []),
        )
        return {
            "system": result.system,
            "tools": result.tools,
            "messages": result.messages,
            "stats": result.stats,
        }

    return _node
