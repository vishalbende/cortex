"""Test doubles for LLM clients, Anthropic SDK, and MCP sessions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contextengine.llm.base import LLMResponse


@dataclass
class FakeLLMClient:
    """In-memory LLMClient that replays a queue of canned text responses.

    Records every `complete()` call in `.calls` for assertions.
    """

    responses: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stable_prefix: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(
            dict(
                model=model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                stable_prefix=stable_prefix,
                json_mode=json_mode,
            )
        )
        if not self.responses:
            raise AssertionError("FakeLLMClient ran out of responses")
        return LLMResponse(text=self.responses.pop(0))


@dataclass
class _FakeBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]


@dataclass
class FakeMessages:
    """Replays canned responses in the shape of anthropic.AsyncAnthropic.messages."""

    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeMessages ran out of responses")
        text = self.responses.pop(0)
        return _FakeResponse(content=[_FakeBlock(text=text)])


class FakeAnthropicClient:
    """Minimal stand-in for anthropic.AsyncAnthropic — still used by
    tests that want to exercise the real AnthropicClient wrapper."""

    def __init__(self, responses: list[str]) -> None:
        self.messages = FakeMessages(responses=list(responses))


@dataclass
class _FakeChatCompletionMessage:
    content: str


@dataclass
class _FakeChatCompletionChoice:
    message: _FakeChatCompletionMessage


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChatCompletionChoice]


@dataclass
class FakeChatCompletions:
    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> _FakeChatCompletion:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeChatCompletions ran out of responses")
        text = self.responses.pop(0)
        return _FakeChatCompletion(
            choices=[
                _FakeChatCompletionChoice(
                    message=_FakeChatCompletionMessage(content=text)
                )
            ]
        )


class FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = FakeChatCompletions(responses=responses)


class FakeOpenAIClient:
    """Minimal stand-in for openai.AsyncOpenAI."""

    def __init__(self, responses: list[str]) -> None:
        self.chat = FakeChat(responses=list(responses))


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
