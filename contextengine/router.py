"""Vectorless tool routing via two-pass LLM traversal over the MCP catalog."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextengine.types import Catalog, Tool


@dataclass(frozen=True)
class RouteDecision:
    """Output of the router — ranked tools plus rationale for explainability."""

    tools: list[Tool]
    mcps_selected: list[str]
    rationale: str = ""


class Router:
    """Two-pass LLM traversal over the MCP catalog.

    Pass 1 scores MCPs against the message using their summaries.
    Pass 2 ranks tools within each selected MCP via category summaries
    and tool descriptions. Decisions are memoized by
    (message_hash, catalog.version_hash).
    """

    def __init__(
        self,
        *,
        catalog: Catalog,
        router_model: str,
        anthropic_client: Any = None,
    ) -> None:
        self.catalog = catalog
        self.router_model = router_model
        self._client = anthropic_client
        self._cache: dict[tuple[str, str], RouteDecision] = {}

    async def select(
        self,
        *,
        message: str,
        required_tools: tuple[str, ...] = (),
    ) -> RouteDecision:
        """Return ranked tools for the given message.

        Stub at v1 — the two-pass LLM traversal lands in the next pass.
        """
        del message, required_tools
        raise NotImplementedError("Router.select is stubbed — implement in next pass")
