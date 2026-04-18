"""Convert an Anthropic-shaped AssembleResult to OpenAI's chat.completions shape."""
from __future__ import annotations

from typing import Any

from contextengine.types import AssembleResult


def assemble_to_openai(result: AssembleResult) -> dict[str, Any]:
    """Return {model-agnostic messages, tools} ready for OpenAI chat.completions.

    Transformations:
      - Anthropic tool: {name, description, input_schema}
        → OpenAI tool:  {type: "function", function: {name, description, parameters}}
      - Anthropic system (top-level string) → OpenAI system message injected
        at the head of `messages`.
      - Anthropic user/assistant text content passes through; tool_use
        blocks are converted to OpenAI's `tool_calls` shape when present.
    """
    tools = [
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

    messages: list[dict[str, Any]] = []
    if result.system:
        messages.append({"role": "system", "content": result.system})

    for m in result.messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
        tool_calls = [
            {
                "id": b.get("id", ""),
                "type": "function",
                "function": {
                    "name": b.get("name", ""),
                    "arguments": b.get("input", {}),
                },
            }
            for b in content
            if b.get("type") == "tool_use"
        ]
        entry: dict[str, Any] = {"role": role, "content": "".join(text_parts)}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        messages.append(entry)

    return {"messages": messages, "tools": tools}
