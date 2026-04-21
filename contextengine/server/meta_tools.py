"""Meta-tools the contextengine MCP server exposes alongside downstream MCPs.

These are the tools that make contextengine *different* from a plain
MCP gateway: memory query, remember/recall, handoffs, route hints, and
GDPR export/erase. They're stable across downstream MCP sets.
"""
from __future__ import annotations

from typing import Any


def _entity_schema(*, required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "description": "Entity identifier."},
            "role": {"type": "string", "description": "Agent role for visibility scoping."},
        },
        "required": required,
    }


META_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "ce.remember",
        "description": (
            "Persist a durable fact for an entity. Use for identifiers, "
            "preferences, constraints that stay true after this turn."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "key": {"type": "string"},
                "value": {"type": "string"},
                "source": {"type": "string", "default": "assistant"},
                "role": {"type": "string", "default": ""},
                "visibility": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Roles that may read. Empty = all roles.",
                },
            },
            "required": ["entity_id", "key", "value"],
        },
    },
    {
        "name": "ce.recall",
        "description": (
            "Return the entity's full memory block scoped to `role`. Use "
            "when you need the raw facts + recent events."
        ),
        "inputSchema": _entity_schema(required=["entity_id"]),
    },
    {
        "name": "ce.ask_memory",
        "description": (
            "Natural-language query over entity memory. Returns a short "
            "answer grounded in stored facts + events."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "question": {"type": "string"},
                "role": {"type": "string", "default": ""},
            },
            "required": ["entity_id", "question"],
        },
    },
    {
        "name": "ce.route",
        "description": (
            "Return the recommended tool subset for a user message. Useful "
            "when the client wants to prune its own tool list mid-session."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "ce.handoff",
        "description": (
            "Record an agent-to-agent handoff. The note becomes visible "
            "to both roles' memory assembly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "from_role": {"type": "string"},
                "to_role": {"type": "string"},
                "reason": {"type": "string"},
                "summary": {"type": "string", "default": ""},
            },
            "required": ["entity_id", "from_role", "to_role", "reason"],
        },
    },
    {
        "name": "ce.export_memory",
        "description": "Export an entity's full memory as a JSON string (GDPR).",
        "inputSchema": _entity_schema(required=["entity_id"]),
    },
    {
        "name": "ce.erase_memory",
        "description": "Hard-delete an entity's memory (GDPR right-to-be-forgotten).",
        "inputSchema": _entity_schema(required=["entity_id"]),
    },
]


META_TOOL_NAMES: set[str] = {t["name"] for t in META_TOOL_DEFS}
