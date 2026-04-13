"""
Cortex Engine — top-level orchestrator that wires everything together.

This is the single entry point for running Cortex. It:
  1. Reads config and builds all subsystems via dependency injection
  2. Registers domain agents
  3. Accepts user intents
  4. Drives the Planner → Agent → PageStore → MistakeTracker loop
  5. Optionally spawns agents in tmux panes for visual parallelism
  6. Produces the final structured output per the Cortex spec
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from cortex.config import CortexConfig
from cortex.models import (
    AgentOutput,
    PageIndex,
    PageType,
    Plan,
    PlanStep,
    Quality,
    StepStatus,
)
from cortex.agents.registry import AgentRegistry
from cortex.agents.ai_agent import AIAgent
from cortex.agents.design_agent import DesignAgent
from cortex.agents.test_writer_agent import TestWriterAgent
from cortex.planner.planner import Planner
from cortex.context.page_store import PageStore
from cortex.context.rag_bridge import PageIndexRAG
from cortex.mistakes.tracker import MistakeTracker
from cortex.permissions.resolver import PermissionResolver
from cortex.tmux_runner import TmuxRunner

logger = logging.getLogger(__name__)


class CortexEngine:
    """
    Main orchestrator. Accepts intents, decomposes them into plans,
    executes through agents, and returns structured results.
    """

    def __init__(self, config: CortexConfig | None = None) -> None:
        self.config = config or CortexConfig()
        warnings = self.config.validate()
        for w in warnings:
            logger.warning("Config: %s", w)

        # ── Build subsystems (Dependency Inversion) ──────────────────
        self.page_store = PageStore(max_tokens=self.config.max_context_tokens)
        self.mistakes = MistakeTracker()
        self.permissions = PermissionResolver(
            granted_permissions=self.config.default_permissions
        )
        self.registry = AgentRegistry()
        self.rag = PageIndexRAG(
            model=self.config.default_model,
            data_dir=self.config.rag_data_dir,
        )
        self.tmux = TmuxRunner(session_name=self.config.tmux_session_name)

        # ── Build planner ────────────────────────────────────────────
        self.planner = Planner(
            registry=self.registry,
            page_store=self.page_store,
            mistake_tracker=self.mistakes,
            permissions=self.permissions,
            api_key=self.config.anthropic_api_key or None,
            model=self.config.default_model,
        )

        # ── Register default agents ──────────────────────────────────
        self._register_default_agents()

        # ── Event callbacks (TUI hooks into these) ───────────────────
        self._on_intent: list[Any] = []
        self._on_complete: list[Any] = []

    def _register_default_agents(self) -> None:
        """Register the built-in domain agents."""
        # AI Agent — fully wired if API key present
        if self.config.anthropic_api_key:
            self.registry.register(AIAgent(
                api_key=self.config.anthropic_api_key,
                default_model=self.config.default_model,
            ))
        else:
            logger.info("Skipping AIAgent registration (no API key).")

        # Design Agent (skeleton)
        self.registry.register(DesignAgent())

        # Test Writer Agent (skeleton)
        self.registry.register(TestWriterAgent())

        logger.info("Registered agents: %s", self.registry.list_agents())

    # ── Public API ───────────────────────────────────────────────────

    async def run(self, intent: str) -> dict[str, Any]:
        """
        Main entry point. Accepts a user intent string and returns
        the full structured Cortex output.
        """
        logger.info("═" * 60)
        logger.info("CORTEX — New intent: %s", intent[:100])
        logger.info("═" * 60)

        # Store the conversation page
        conv_page = PageIndex(
            type=PageType.CONVERSATION,
            summary=f"User: {intent[:60]}",
            token_count=len(intent) // 4,
            tags=["intent:user"],
            content=intent,
        )
        self.page_store.add(conv_page)

        # Optional: query RAG for relevant context
        rag_pages = await self.rag.query(intent, top_k=3)
        for page in rag_pages:
            self.page_store.add(page)

        # Decompose intent into a plan
        plan = await self.planner.decompose(intent)
        logger.info("Plan: %d steps", len(plan.steps))

        # Optional: set up tmux for visual parallelism
        if self.config.use_tmux:
            self.tmux.setup()

        # Execute the plan
        outputs = await self.planner.execute(plan, max_retries=self.config.max_retries)

        # Teardown tmux
        if self.config.use_tmux:
            self.tmux.teardown()

        # Build final output
        return self._build_output(plan, outputs)

    async def run_with_tmux(self, intent: str) -> dict[str, Any]:
        """
        Run with tmux panes visible — each parallel step group
        gets its own pane. Good for demos and debugging.
        """
        self.config.use_tmux = True
        return await self.run(intent)

    async def index_document(self, path: str, doc_id: str | None = None) -> str:
        """Index a document into the RAG bridge for future queries."""
        return await self.rag.index_document(path, doc_id)

    # ── Output builder ───────────────────────────────────────────────

    def _build_output(self, plan: Plan, outputs: list[AgentOutput]) -> dict[str, Any]:
        """
        Assemble the final structured output per the Cortex spec.
        """
        # Collect skills used
        skills_used = set()
        for step in plan.steps:
            if step.agent == "ai_agent":
                skills_used.add("ai")
            elif step.agent == "design_agent":
                skills_used.add("design")
            elif step.agent == "test_writer_agent":
                skills_used.add("test_writer")
            skills_used.add("planner")
        if self.mistakes.count() > 0:
            skills_used.add("mistake_marker")

        # Collect the primary result (from the last successful output)
        primary_result = None
        for output in reversed(outputs):
            if output.quality != Quality.ERROR and output.result:
                primary_result = output.result
                break

        # Pages loaded
        pages_loaded = [p.id for p in self.page_store.all_pages()]

        # Steps executed
        steps_executed = []
        for step in plan.steps:
            steps_executed.append({
                "agent": step.agent,
                "action": step.action,
                "status": step.status.value,
                "quality": step.quality.value,
            })

        output = {
            "plan_id": plan.plan_id,
            "intent_summary": plan.intent[:200],
            "skills_used": sorted(skills_used),
            "pages_loaded": pages_loaded,
            "steps_executed": steps_executed,
            "result": primary_result,
            "mistakes": [m.to_dict() for m in self.mistakes.all],
            "lessons_learned": self.mistakes.lessons_learned(),
            "memory_updates": [],
        }

        logger.info("═" * 60)
        logger.info("CORTEX — Complete. Steps: %d, Mistakes: %d",
                     len(plan.steps), self.mistakes.count())
        logger.info("═" * 60)

        return output

    # ── Convenience ──────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return current engine status for the TUI."""
        return {
            "agents": self.registry.list_agents(),
            "pages": len(self.page_store),
            "total_tokens": self.page_store.total_tokens,
            "mistakes": self.mistakes.count(),
            "permissions": self.permissions.granted_permissions,
            "rag_available": self.rag.is_available(),
            "tmux_panes": {
                name: info.status
                for name, info in self.tmux.list_panes().items()
            },
        }
