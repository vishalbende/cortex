"""Top-level orchestration: start, assemble, execute, close."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from contextengine.budget import Budget, pack
from contextengine.catalog import build_catalog
from contextengine.mcp.pool import MCPPool
from contextengine.router import Router
from contextengine.tokenize import Tokenizer, get_tokenizer
from contextengine.types import (
    AssembleResult,
    AssembleStats,
    Catalog,
    MCPServer,
    Message,
)


class ContextEngine:
    """MCP-aware context orchestration — budget-packed tool routing + proxying."""

    def __init__(
        self,
        *,
        mcps: list[MCPServer],
        model: str,
        router_model: str = "claude-haiku-4-5",
        budget: int = 80_000,
        reserved_output: int = 4096,
        system_prompt: str = "",
        cache_dir: str | Path = ".contextengine",
        anthropic_client: Any = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if not mcps:
            raise ValueError("ContextEngine requires at least one MCPServer")

        self.model = model
        self.router_model = router_model
        self.system_prompt = system_prompt
        self.cache_dir = Path(cache_dir)
        self.budget = Budget(total=budget, reserved_output=reserved_output)
        self.tokenizer = tokenizer or get_tokenizer(model)
        self._client = anthropic_client

        self._pool = MCPPool(mcps)
        self._catalog: Catalog | None = None
        self._router: Router | None = None

    async def start(self) -> None:
        """Connect to all MCPs, enumerate tools, build the hierarchical catalog."""
        await self._pool.start()
        tools_by_mcp = await self._pool.list_all_tools()
        self._catalog = await build_catalog(
            tools_by_mcp=tools_by_mcp,
            router_model=self.router_model,
            anthropic_client=self._client,
            cache_dir=self.cache_dir,
        )
        self._router = Router(
            catalog=self._catalog,
            router_model=self.router_model,
            anthropic_client=self._client,
        )

    async def assemble(
        self,
        *,
        message: str,
        history: list[Message] | None = None,
        memory: str = "",
        required_tools: tuple[str, ...] = (),
    ) -> AssembleResult:
        """Return system/tools/messages packed within budget, cache-friendly ordered."""
        if self._router is None:
            raise RuntimeError("ContextEngine.start() must be called before assemble()")

        t0 = time.perf_counter()
        history = list(history or [])

        decision = await self._router.select(message=message, required_tools=required_tools)

        system_final = "\n\n".join(x for x in (self.system_prompt, memory) if x)
        system_tokens = self.tokenizer.count(self.system_prompt)
        memory_tokens = self.tokenizer.count(memory)

        user_msg = Message(role="user", content=message)
        packed = pack(
            budget=self.budget,
            system_tokens=system_tokens,
            memory_tokens=memory_tokens,
            ranked_tools=decision.tools,
            history=[*history, user_msg],
            tokenizer=self.tokenizer,
            required_tools=set(required_tools),
        )

        tool_payload = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in packed.tools
        ]
        message_payload = [{"role": m.role, "content": m.content} for m in packed.messages]
        tools_tokens = sum(t.token_count for t in packed.tools)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = AssembleStats(
            tokens_system=system_tokens,
            tokens_memory=memory_tokens,
            tokens_tools=tools_tokens,
            tokens_history=packed.tokens_used - system_tokens - memory_tokens - tools_tokens,
            tokens_total=packed.tokens_used,
            tools_loaded=tuple(t.name for t in packed.tools),
            tools_dropped=tuple(t.name for t in packed.tools_dropped),
            mcps_represented=tuple(sorted({t.mcp for t in packed.tools})),
            elapsed_ms=elapsed_ms,
        )

        return AssembleResult(
            system=system_final,
            tools=tool_payload,
            messages=message_payload,
            stats=stats,
        )

    async def execute(self, tool_use: Any) -> Any:
        """Proxy a tool_use block from the model to the owning MCP server.

        Stub at v1 — dispatches based on `tool_use.name` (namespaced
        `<mcp>.<tool>`) to `MCPPool.get(mcp).call_tool(...)`.
        """
        del tool_use
        raise NotImplementedError("execute is stubbed — implement in next pass")

    async def close(self) -> None:
        """Tear down MCP connections."""
        await self._pool.close()
