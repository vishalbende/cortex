"""Data contracts for contextengine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class MCPServer:
    """Configuration for a single MCP server connection."""

    name: str
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.command is None and self.url is None:
            raise ValueError(f"MCPServer {self.name!r} requires either command or url")
        if self.command is not None and self.url is not None:
            raise ValueError(f"MCPServer {self.name!r} must set only one of command/url")


@dataclass(frozen=True)
class Tool:
    """A single tool exposed by an MCP server, namespaced by server name."""

    name: str
    mcp: str
    description: str
    input_schema: dict[str, Any]
    category: str = ""
    token_count: int = 0


@dataclass(frozen=True)
class ToolCategory:
    """A cluster of related tools within a single MCP, with an LLM-generated summary."""

    name: str
    summary: str
    tools: tuple[Tool, ...]


@dataclass(frozen=True)
class MCPCatalog:
    """All tools exposed by a single MCP, grouped into categories."""

    name: str
    summary: str
    categories: tuple[ToolCategory, ...]

    @property
    def tools_flat(self) -> tuple[Tool, ...]:
        return tuple(t for c in self.categories for t in c.tools)


@dataclass(frozen=True)
class Catalog:
    """Full hierarchical catalog across all connected MCPs."""

    mcps: tuple[MCPCatalog, ...]
    version_hash: str


@dataclass(frozen=True)
class Message:
    """A conversation message, framework-agnostic."""

    role: Literal["user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


@dataclass(frozen=True)
class AssembleStats:
    """Diagnostics emitted for each assemble() call."""

    tokens_system: int
    tokens_memory: int
    tokens_tools: int
    tokens_history: int
    tokens_total: int
    tools_loaded: tuple[str, ...]
    tools_dropped: tuple[str, ...]
    mcps_represented: tuple[str, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class AssembleResult:
    """Output of ContextEngine.assemble() — ready to pass to any LLM SDK."""

    system: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    stats: AssembleStats
