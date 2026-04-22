"""Vectorless tool routing via two-pass LLM traversal over the MCP catalog."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from contextengine._json import extract_json
from contextengine.llm.base import LLMClient
from contextengine.types import Catalog, MCPCatalog, Tool


@dataclass(frozen=True)
class RouteDecision:
    """Output of the router — ranked tools plus the MCPs that contributed."""

    tools: list[Tool]
    mcps_selected: list[str]
    rationale: str = ""


class Router:
    """Two-pass LLM traversal over the MCP catalog.

    Pass 1 scores MCPs against the message using their summaries.
    Pass 2 ranks tools within each selected MCP via category summaries
    and tool descriptions. Decisions are memoized by
    (message_hash, catalog.version_hash).

    Provider-agnostic via LLMClient: pass an AnthropicClient, OpenAIClient,
    or anything implementing the `complete()` protocol. The router hands
    the stable catalog prefix to the client as `stable_prefix=...` so the
    provider can apply its own caching policy.
    """

    def __init__(
        self,
        *,
        catalog: Catalog,
        router_model: str,
        llm: LLMClient,
    ) -> None:
        self.catalog = catalog
        self.router_model = router_model
        self._llm = llm
        self._cache: dict[tuple[str, str], RouteDecision] = {}

    async def select(
        self,
        *,
        message: str,
        required_tools: tuple[str, ...] = (),
    ) -> RouteDecision:
        key = (
            hashlib.sha256(message.encode()).hexdigest()[:16],
            self.catalog.version_hash,
        )
        if key in self._cache:
            return self._cache[key]

        mcp_names = await self._select_mcps(message)

        selected: list[Tool] = []
        seen: set[str] = set()
        for mcp_name in mcp_names:
            mcp = next((m for m in self.catalog.mcps if m.name == mcp_name), None)
            if mcp is None or not mcp.categories:
                continue
            picked = await self._select_tools(message=message, mcp=mcp)
            for t in picked:
                if t.name not in seen:
                    seen.add(t.name)
                    selected.append(t)

        by_name = {t.name: t for m in self.catalog.mcps for t in m.tools_flat}
        for name in required_tools:
            t = by_name.get(name)
            if t is not None and t.name not in seen:
                seen.add(t.name)
                selected.append(t)

        decision = RouteDecision(
            tools=selected,
            mcps_selected=list(mcp_names),
        )
        self._cache[key] = decision
        return decision

    async def _select_mcps(self, message: str) -> list[str]:
        if not self.catalog.mcps:
            return []
        if len(self.catalog.mcps) == 1:
            return [self.catalog.mcps[0].name]

        mcps_block = "\n".join(f"- {m.name}: {m.summary}" for m in self.catalog.mcps)
        system = (
            "You route user messages to MCP tool servers. Given a list of "
            "MCPs and their summaries, return the subset relevant to a "
            "user message, ranked by relevance. Respond with JSON only."
        )
        stable_prefix = (
            f"Available MCP servers:\n{mcps_block}\n\n"
            f"Return JSON: {{\"mcps\": [\"name1\", ...]}} — ordered by "
            f"relevance. Include an MCP only if at least one of its tools "
            f"could plausibly help."
        )

        response = await self._llm.complete(
            model=self.router_model,
            system=system,
            stable_prefix=stable_prefix,
            user=f"User message: {message}",
            max_tokens=512,
            json_mode=True,
        )
        data = extract_json(response.text)
        mcps = data.get("mcps", [])
        return [m for m in mcps if isinstance(m, str)]

    async def _select_tools(
        self,
        *,
        message: str,
        mcp: MCPCatalog,
    ) -> list[Tool]:
        parts: list[str] = [f"MCP: {mcp.name} — {mcp.summary}", ""]
        for c in mcp.categories:
            parts.append(f"Category '{c.name}' — {c.summary}")
            for t in c.tools:
                parts.append(f"  - {t.name}: {t.description[:200]}")
            parts.append("")
        catalog_block = "\n".join(parts).rstrip()

        system = (
            "You pick the minimal tool subset needed to serve a user "
            "message from one MCP. Respond with JSON only."
        )
        stable_prefix = (
            f"{catalog_block}\n\n"
            f"Return JSON: {{\"tools\": [\"full.namespaced.name\", ...]}} — "
            f"tools ranked by relevance. Include only tools that could "
            f"meaningfully help. Prefer fewer."
        )

        response = await self._llm.complete(
            model=self.router_model,
            system=system,
            stable_prefix=stable_prefix,
            user=f"User message: {message}",
            max_tokens=1024,
            json_mode=True,
        )
        data = extract_json(response.text)
        by_name = {t.name: t for t in mcp.tools_flat}
        return [by_name[n] for n in data.get("tools", []) if n in by_name]
