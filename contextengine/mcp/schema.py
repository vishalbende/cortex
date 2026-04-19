"""Convert MCP tool definitions to the internal Tool type."""
from __future__ import annotations

from typing import Any

from contextengine.tokenize import Tokenizer
from contextengine.types import Tool


def _get(raw: Any, key: str, default: Any = None) -> Any:
    if hasattr(raw, key):
        return getattr(raw, key)
    if isinstance(raw, dict):
        return raw.get(key, default)
    return default


def normalize_tool(raw: Any, *, mcp_name: str, tokenizer: Tokenizer) -> Tool:
    """Convert an MCP tool object (from the `mcp` package or a dict) into a Tool.

    Accepts either an attribute-bearing object (mcp.types.Tool) or a dict
    with keys: name, description, inputSchema.
    """
    name = _get(raw, "name")
    if not name:
        raise ValueError("MCP tool is missing 'name'")
    description = _get(raw, "description", "") or ""
    schema = _get(raw, "inputSchema", {}) or {}

    namespaced = f"{mcp_name}.{name}"
    token_count = tokenizer.count(namespaced) + tokenizer.count(description)

    return Tool(
        name=namespaced,
        mcp=mcp_name,
        description=description,
        input_schema=schema,
        token_count=token_count,
    )
