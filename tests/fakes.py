"""Test doubles for Anthropic client and MCP session."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FakeBlock:
    text: str


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]


@dataclass
class FakeMessages:
    """Replays a queue of canned text responses in order."""

    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeMessages ran out of responses")
        text = self.responses.pop(0)
        return _FakeResponse(content=[_FakeBlock(text=text)])


class FakeAnthropicClient:
    """Minimal stand-in for anthropic.AsyncAnthropic in router/catalog tests."""

    def __init__(self, responses: list[str]) -> None:
        self.messages = FakeMessages(responses=list(responses))


@dataclass
class FakeMCPSession:
    """Minimal stand-in for mcp.ClientSession."""

    tools: list[Any]
    call_results: dict[str, Any] = field(default_factory=dict)
    call_log: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Any:
        return type("R", (), {"tools": self.tools})()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.call_log.append((name, dict(arguments)))
        return self.call_results.get(name, {"ok": True, "name": name})
