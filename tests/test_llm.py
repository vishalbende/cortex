"""Tests for LLMClient implementations and auto-detection registry."""
import pytest

from contextengine.llm import (
    AnthropicClient,
    LLMResponse,
    OpenAIClient,
    client_for_model,
)
from contextengine.llm.registry import detect_provider
from tests.fakes import FakeAnthropicClient, FakeOpenAIClient


def test_detect_provider_anthropic() -> None:
    assert detect_provider("claude-sonnet-4-5") == "anthropic"
    assert detect_provider("claude-haiku-4-5") == "anthropic"
    assert detect_provider("anthropic.claude-3-opus") == "anthropic"


def test_detect_provider_openai() -> None:
    assert detect_provider("gpt-4o") == "openai"
    assert detect_provider("gpt-4o-mini") == "openai"
    assert detect_provider("o1-preview") == "openai"
    assert detect_provider("o3-mini") == "openai"
    assert detect_provider("openai/gpt-4o") == "openai"


def test_detect_provider_unknown() -> None:
    assert detect_provider("llama-3-70b") == "unknown"
    assert detect_provider("some-random-model") == "unknown"


def test_client_for_model_anthropic() -> None:
    c = client_for_model("claude-sonnet-4-5")
    assert isinstance(c, AnthropicClient)


def test_client_for_model_openai() -> None:
    c = client_for_model("gpt-4o-mini")
    assert isinstance(c, OpenAIClient)


def test_client_for_model_unknown_uses_default() -> None:
    c = client_for_model("llama-3", default="anthropic")
    assert isinstance(c, AnthropicClient)
    c2 = client_for_model("llama-3", default="openai")
    assert isinstance(c2, OpenAIClient)


async def test_anthropic_client_wraps_sdk() -> None:
    fake = FakeAnthropicClient(responses=["hello from claude"])
    client = AnthropicClient(client=fake)
    response = await client.complete(
        model="claude-haiku-4-5",
        system="sys",
        user="hi",
        max_tokens=100,
    )
    assert isinstance(response, LLMResponse)
    assert response.text == "hello from claude"
    assert fake.messages.calls[0]["model"] == "claude-haiku-4-5"
    assert fake.messages.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


async def test_anthropic_client_emits_cache_control_for_stable_prefix() -> None:
    fake = FakeAnthropicClient(responses=["ok"])
    client = AnthropicClient(client=fake)
    await client.complete(
        model="claude-haiku-4-5",
        system="sys",
        user="q",
        max_tokens=100,
        stable_prefix="big stable prefix",
    )
    system_blocks = fake.messages.calls[0]["system"]
    assert len(system_blocks) == 2
    assert system_blocks[0] == {"type": "text", "text": "sys"}
    assert system_blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert system_blocks[1]["text"] == "big stable prefix"


async def test_anthropic_client_no_system_if_both_empty() -> None:
    fake = FakeAnthropicClient(responses=["ok"])
    client = AnthropicClient(client=fake)
    await client.complete(model="claude-haiku-4-5", system="", user="q", max_tokens=50)
    assert "system" not in fake.messages.calls[0]


async def test_openai_client_wraps_sdk() -> None:
    fake = FakeOpenAIClient(responses=["hello from gpt"])
    client = OpenAIClient(client=fake)
    response = await client.complete(
        model="gpt-4o-mini",
        system="sys",
        user="hi",
        max_tokens=100,
    )
    assert response.text == "hello from gpt"
    call = fake.chat.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["max_completion_tokens"] == 100
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


async def test_openai_client_concatenates_stable_prefix() -> None:
    fake = FakeOpenAIClient(responses=["ok"])
    client = OpenAIClient(client=fake)
    await client.complete(
        model="gpt-4o-mini",
        system="sys",
        user="q",
        max_tokens=50,
        stable_prefix="big stable prefix",
    )
    call = fake.chat.completions.calls[0]
    system_msg = call["messages"][0]["content"]
    assert "sys" in system_msg
    assert "big stable prefix" in system_msg
    assert system_msg.index("sys") < system_msg.index("big stable prefix")


async def test_openai_client_json_mode() -> None:
    fake = FakeOpenAIClient(responses=["{}"])
    client = OpenAIClient(client=fake)
    await client.complete(
        model="gpt-4o-mini",
        system="",
        user="q",
        max_tokens=50,
        json_mode=True,
    )
    assert fake.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


async def test_openai_client_omits_system_if_empty() -> None:
    fake = FakeOpenAIClient(responses=["ok"])
    client = OpenAIClient(client=fake)
    await client.complete(model="gpt-4o-mini", system="", user="q", max_tokens=50)
    messages = fake.chat.completions.calls[0]["messages"]
    assert messages[0]["role"] == "user"
