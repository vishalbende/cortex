"""Run contextengine itself as an MCP server.

Exposes downstream MCP tools (namespaced, proxied) plus memory +
routing meta-tools so any MCP client — Claude Code, Cursor, Claude
Desktop, Copilot — can use contextengine without a Python SDK.
"""

from contextengine.server.app import ContextEngineMCPServer, build_server
from contextengine.server.meta_tools import META_TOOL_DEFS, META_TOOL_NAMES

__all__ = [
    "ContextEngineMCPServer",
    "build_server",
    "META_TOOL_DEFS",
    "META_TOOL_NAMES",
]
