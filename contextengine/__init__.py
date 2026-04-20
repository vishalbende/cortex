"""contextengine — MCP-aware context orchestration for LLM agents."""

from contextengine.engine import ContextEngine
from contextengine.llm import AnthropicClient, LLMClient, LLMResponse, OpenAIClient
from contextengine.memory import (
    EntityMemory,
    Event,
    Fact,
    InMemoryStore,
    JSONStore,
    MemoryAssembler,
    MemoryStore,
    MemoryWriter,
    WriteResult,
)
from contextengine.telemetry import FileSink, Sink, StdoutSink, TraceRecord, TraceRecorder
from contextengine.types import (
    AssembleResult,
    AssembleStats,
    Catalog,
    MCPCatalog,
    MCPServer,
    Message,
    Tool,
    ToolCategory,
)

__all__ = [
    "ContextEngine",
    "LLMClient",
    "LLMResponse",
    "AnthropicClient",
    "OpenAIClient",
    "MCPServer",
    "Tool",
    "ToolCategory",
    "MCPCatalog",
    "Catalog",
    "Message",
    "AssembleResult",
    "AssembleStats",
    "EntityMemory",
    "Event",
    "Fact",
    "MemoryStore",
    "InMemoryStore",
    "JSONStore",
    "MemoryAssembler",
    "MemoryWriter",
    "WriteResult",
    "TraceRecord",
    "TraceRecorder",
    "Sink",
    "FileSink",
    "StdoutSink",
]

__version__ = "0.1.0"
