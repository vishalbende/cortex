"""LLM provider abstraction — Anthropic, OpenAI, or a user-supplied client."""

from contextengine.llm.anthropic import AnthropicClient
from contextengine.llm.base import LLMClient, LLMResponse
from contextengine.llm.openai import OpenAIClient
from contextengine.llm.registry import client_for_model

__all__ = [
    "LLMClient",
    "LLMResponse",
    "AnthropicClient",
    "OpenAIClient",
    "client_for_model",
]
