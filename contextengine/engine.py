"""Top-level orchestration: start, assemble, execute, process_turn, close."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from contextengine.budget import Budget, pack
from contextengine.catalog import build_catalog
from contextengine.compaction import HistoryCompactor
from contextengine.llm.base import LLMClient
from contextengine.llm.registry import client_for_model
from contextengine.mcp.pool import MCPPool
from contextengine.memory.assembler import MemoryAssembler
from contextengine.memory.store import InMemoryStore, MemoryStore
from contextengine.memory.writer import MemoryWriter, WriteResult
from contextengine.router import Router
from contextengine.telemetry.recorder import TraceRecorder
from contextengine.telemetry.sinks import Sink
from contextengine.tokenize import Tokenizer, get_tokenizer
from contextengine.types import (
    AssembleResult,
    AssembleStats,
    Catalog,
    MCPServer,
    Message,
)


class ContextEngine:
    """MCP-aware context orchestration — routing, memory, telemetry, proxying.

    Supports Anthropic and OpenAI models for internal LLM calls (routing,
    categorization, memory writeback, compaction). The provider is
    auto-detected from the model string — models starting with `claude*`
    use Anthropic, `gpt*`/`o1`/`o3`/`o4` use OpenAI. Override by passing
    an explicit `llm_client`, `anthropic_client`, or `openai_client`.
    """

    def __init__(
        self,
        *,
        mcps: list[MCPServer],
        model: str,
        router_model: str = "claude-haiku-4-5",
        memory_model: str | None = None,
        budget: int = 80_000,
        reserved_output: int = 4096,
        memory_budget: int = 4_000,
        system_prompt: str = "",
        cache_dir: str | Path = ".contextengine",
        anthropic_client: Any = None,
        openai_client: Any = None,
        llm_client: LLMClient | None = None,
        tokenizer: Tokenizer | None = None,
        memory_store: MemoryStore | None = None,
        telemetry_sinks: list[Sink] | None = None,
        compaction_threshold: int = 40,
        compaction_keep_recent: int = 10,
    ) -> None:
        if not mcps:
            raise ValueError("ContextEngine requires at least one MCPServer")

        self.model = model
        self.router_model = router_model
        self.memory_model = memory_model or router_model
        self.system_prompt = system_prompt
        self.cache_dir = Path(cache_dir)
        self.budget = Budget(total=budget, reserved_output=reserved_output)
        self.memory_budget = memory_budget
        self.tokenizer = tokenizer or get_tokenizer(model)

        self._anthropic_client = anthropic_client
        self._openai_client = openai_client

        def _resolve(model_id: str) -> LLMClient:
            if llm_client is not None:
                return llm_client
            return client_for_model(
                model_id,
                anthropic_client=anthropic_client,
                openai_client=openai_client,
            )

        self._router_llm = _resolve(self.router_model)
        self._memory_llm = _resolve(self.memory_model)

        self._pool = MCPPool(mcps, tokenizer=self.tokenizer)
        self._catalog: Catalog | None = None
        self._router: Router | None = None

        self._memory_store: MemoryStore = memory_store or InMemoryStore()
        self._memory_assembler = MemoryAssembler(self._memory_store, self.tokenizer)
        self._memory_writer = MemoryWriter(
            store=self._memory_store,
            model=self.memory_model,
            llm=self._memory_llm,
        )

        self._telemetry = TraceRecorder(sinks=list(telemetry_sinks or []))

        self._compactor = HistoryCompactor(
            model=self.memory_model,
            llm=self._memory_llm,
            threshold=compaction_threshold,
            keep_recent=compaction_keep_recent,
        )

    @property
    def memory(self) -> MemoryStore:
        return self._memory_store

    @property
    def catalog(self) -> Catalog | None:
        return self._catalog

    @property
    def telemetry(self) -> TraceRecorder:
        return self._telemetry

    async def start(self) -> None:
        """Connect MCPs, enumerate tools, build the hierarchical catalog."""
        await self._pool.start()
        await self._rebuild_catalog()

    async def _rebuild_catalog(self) -> None:
        tools_by_mcp = await self._pool.list_all_tools()
        self._catalog = await build_catalog(
            tools_by_mcp=tools_by_mcp,
            router_model=self.router_model,
            llm=self._router_llm,
            cache_dir=self.cache_dir,
        )
        self._router = Router(
            catalog=self._catalog,
            router_model=self.router_model,
            llm=self._router_llm,
        )

    async def add_mcp(self, server: MCPServer) -> None:
        """Hot-add an MCP server. Rebuilds catalog; router cache is reset."""
        if server.name in self._pool._connectors:  # noqa: SLF001
            raise ValueError(f"MCP {server.name!r} is already connected")
        self._pool._servers.append(server)  # noqa: SLF001
        from contextengine.mcp.connector import MCPConnector

        connector = MCPConnector(server, tokenizer=self.tokenizer)
        await connector.connect()
        self._pool._connectors[server.name] = connector  # noqa: SLF001
        await self._rebuild_catalog()

    async def remove_mcp(self, name: str) -> None:
        """Hot-remove an MCP server. Rebuilds catalog; router cache is reset."""
        connector = self._pool._connectors.pop(name, None)  # noqa: SLF001
        if connector is None:
            raise KeyError(f"Unknown MCP: {name!r}")
        self._pool._servers = [  # noqa: SLF001
            s for s in self._pool._servers if s.name != name  # noqa: SLF001
        ]
        await connector.close()
        await self._rebuild_catalog()

    async def assemble(
        self,
        *,
        message: str,
        history: list[Message] | None = None,
        memory: str = "",
        entity_id: str | None = None,
        role: str = "",
        required_tools: tuple[str, ...] = (),
    ) -> AssembleResult:
        """Return system/tools/messages packed within budget, cache-friendly ordered.

        If `entity_id` is provided and no `memory` override is passed, the
        engine assembles a role-scoped memory block from the memory store.
        """
        if self._router is None:
            raise RuntimeError("ContextEngine.start() must be called before assemble()")

        self._telemetry.start()
        trace_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        history = list(history or [])

        if self._compactor.should_compact(history):
            t_compact = time.perf_counter()
            history = await self._compactor.compact(history)
            self._telemetry.span(
                "compaction",
                (time.perf_counter() - t_compact) * 1000,
                kept=len(history),
            )

        memory_block = memory
        if not memory_block and entity_id is not None:
            t_mem = time.perf_counter()
            memory_block = await self._memory_assembler.assemble(
                entity_id=entity_id,
                role=role,
                budget_tokens=self.memory_budget,
            )
            self._telemetry.span(
                "memory_assemble",
                (time.perf_counter() - t_mem) * 1000,
                entity_id=entity_id,
                role=role,
            )

        t_route = time.perf_counter()
        decision = await self._router.select(
            message=message, required_tools=required_tools
        )
        self._telemetry.span(
            "routing",
            (time.perf_counter() - t_route) * 1000,
            mcps_selected=list(decision.mcps_selected),
            tools_selected=len(decision.tools),
        )

        system_final = "\n\n".join(x for x in (self.system_prompt, memory_block) if x)
        system_tokens = self.tokenizer.count(self.system_prompt)
        memory_tokens = self.tokenizer.count(memory_block)

        user_msg = Message(role="user", content=message)
        t_pack = time.perf_counter()
        packed = pack(
            budget=self.budget,
            system_tokens=system_tokens,
            memory_tokens=memory_tokens,
            ranked_tools=decision.tools,
            history=[*history, user_msg],
            tokenizer=self.tokenizer,
            required_tools=set(required_tools),
        )
        self._telemetry.span(
            "budget_pack",
            (time.perf_counter() - t_pack) * 1000,
            tools_loaded=len(packed.tools),
            tools_dropped=len(packed.tools_dropped),
            messages_dropped=len(packed.messages_dropped),
        )

        tool_payload = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
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
        result = AssembleResult(
            system=system_final,
            tools=tool_payload,
            messages=message_payload,
            stats=stats,
        )

        await self._telemetry.emit(
            trace_id=trace_id,
            model=self.model,
            router_model=self.router_model,
            entity_id=entity_id,
            role=role,
            message=message,
            budget=self.budget.total,
            result=result,
        )
        return result

    async def execute(self, tool_use: Any) -> Any:
        """Proxy a tool_use block (dict or object form) to the owning MCP server.

        The tool name must be namespaced as `<mcp>.<tool>`.
        """
        name = getattr(tool_use, "name", None)
        if name is None and isinstance(tool_use, dict):
            name = tool_use.get("name")
        if not isinstance(name, str):
            raise ValueError(f"tool_use is missing a string 'name': {tool_use!r}")

        arguments = getattr(tool_use, "input", None)
        if arguments is None and isinstance(tool_use, dict):
            arguments = tool_use.get("input")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError(f"tool_use 'input' must be a dict, got {type(arguments)}")

        if "." not in name:
            raise ValueError(
                f"Tool name must be namespaced as '<mcp>.<tool>', got {name!r}"
            )
        mcp_name, tool_name = name.split(".", 1)
        connector = self._pool.get(mcp_name)
        return await connector.call_tool(tool_name, arguments)

    async def process_turn(
        self,
        *,
        entity_id: str,
        user_message: str,
        assistant_response: str,
        tool_results: list[dict[str, Any]] | None = None,
        role: str = "",
    ) -> WriteResult:
        """Write durable facts + events extracted from a completed turn."""
        return await self._memory_writer.write(
            entity_id=entity_id,
            user_message=user_message,
            assistant_response=assistant_response,
            tool_results=tool_results,
            role=role,
        )

    async def close(self) -> None:
        """Tear down MCP connections."""
        await self._pool.close()
