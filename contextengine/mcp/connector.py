"""Single MCP server connection wrapper."""
from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from contextengine.mcp.schema import normalize_tool
from contextengine.tokenize import CharEstimateTokenizer, Tokenizer
from contextengine.types import MCPServer, Tool


class MCPConnector:
    """Wraps an MCP client session for one server (stdio or SSE/HTTP).

    Uses the official `mcp` Python SDK. The session and its underlying
    transport are held open via an AsyncExitStack for the lifetime of the
    connector.
    """

    def __init__(self, config: MCPServer, *, tokenizer: Tokenizer | None = None) -> None:
        self.config = config
        self._tokenizer = tokenizer or CharEstimateTokenizer()
        self._session: Any = None
        self._stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        from mcp import ClientSession

        stack = AsyncExitStack()
        try:
            if self.config.command is not None:
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(
                    command=self.config.command[0],
                    args=list(self.config.command[1:]),
                    env=self.config.env or None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            else:
                from mcp.client.sse import sse_client

                assert self.config.url is not None
                read, write = await stack.enter_async_context(sse_client(self.config.url))

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session

    async def list_tools(self) -> list[Tool]:
        if self._session is None:
            raise RuntimeError(f"MCPConnector({self.config.name!r}) not connected")
        result = await self._session.list_tools()
        raw_tools = getattr(result, "tools", None) or result
        return [
            normalize_tool(t, mcp_name=self.config.name, tokenizer=self._tokenizer)
            for t in raw_tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError(f"MCPConnector({self.config.name!r}) not connected")
        return await self._session.call_tool(name, arguments)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None
