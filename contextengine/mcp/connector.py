"""Single MCP server connection wrapper."""
from __future__ import annotations

from typing import Any

from contextengine.types import MCPServer, Tool


class MCPConnector:
    """Wraps an MCP client session for one server (stdio or SSE/HTTP).

    v1 is stubbed — real wiring uses the official `mcp` package:
    `mcp.client.stdio.stdio_client` for command-based servers and
    `mcp.client.sse.sse_client` / HTTP client for URL-based servers.
    """

    def __init__(self, config: MCPServer) -> None:
        self.config = config
        self._session: Any = None

    async def connect(self) -> None:
        raise NotImplementedError("MCPConnector.connect is stubbed")

    async def list_tools(self) -> list[Tool]:
        raise NotImplementedError("MCPConnector.list_tools is stubbed")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        del name, arguments
        raise NotImplementedError("MCPConnector.call_tool is stubbed")

    async def close(self) -> None:
        raise NotImplementedError("MCPConnector.close is stubbed")
