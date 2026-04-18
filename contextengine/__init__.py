"""contextengine — MCP-aware context orchestration for LLM agents."""

from contextengine.engine import ContextEngine
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
    "MCPServer",
    "Tool",
    "ToolCategory",
    "MCPCatalog",
    "Catalog",
    "Message",
    "AssembleResult",
    "AssembleStats",
]

__version__ = "0.0.1"
