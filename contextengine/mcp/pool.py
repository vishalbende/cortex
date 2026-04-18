"""Lifecycle management for a set of MCP connectors."""
from __future__ import annotations

from contextengine.mcp.connector import MCPConnector
from contextengine.types import MCPServer, Tool


class MCPPool:
    """Owns the MCPConnector for each configured MCP server."""

    def __init__(self, servers: list[MCPServer]) -> None:
        names = [s.name for s in servers]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate MCPServer names: {names}")
        self._servers = servers
        self._connectors: dict[str, MCPConnector] = {}

    async def start(self) -> None:
        for s in self._servers:
            c = MCPConnector(s)
            await c.connect()
            self._connectors[s.name] = c

    async def close(self) -> None:
        for c in self._connectors.values():
            await c.close()
        self._connectors.clear()

    def get(self, mcp_name: str) -> MCPConnector:
        try:
            return self._connectors[mcp_name]
        except KeyError:
            raise KeyError(f"Unknown MCP: {mcp_name!r}") from None

    async def list_all_tools(self) -> dict[str, list[Tool]]:
        out: dict[str, list[Tool]] = {}
        for name, c in self._connectors.items():
            out[name] = await c.list_tools()
        return out
