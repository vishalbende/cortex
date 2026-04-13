"""
Cortex Engine — top-level orchestrator that wires everything together.

Runs on the local Claude Code CLI. No API keys needed.

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
from typing import Any

from cortex.config import CortexConfig
from cortex.claude_code import ClaudeCode
from cortex.models import (
    AgentOutput,
    PageIndex,
    PageType,
    Plan,
    Quality,
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
from cortex.sessions.manager import SessionManager
from cortex.sessions.model import Session, SessionStatus
from cortex.tmux_runner import TmuxRunner

logger = logging.getLogger(__name__)


class CortexEngine:
    """
    Main orchestrator. Accepts intents, decomposes them into plans,
    executes through agents, and returns structured results.

    Powered by the local Claude Code CLI — no API keys required.
    """

    def __init__(self, config: CortexConfig | None = None) -> None:
        self.config = config or CortexConfig()
        warnings = self.config.validate()
        for w in warnings:
            logger.warning("Config: %s", w)

        # ── Claude Code CLI bridge ───────────────────────────────────
        self.claude = ClaudeCode(
            model=self.config.default_model,
            cwd=self.config.cwd,
        )

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
        self.session_manager = SessionManager(
            sessions_dir=self.config.sessions_dir,
        )

        # ── Active session tracking ──────────────────────────────────
        self._current_session: Session | None = None

        # ── Build planner (uses Claude Code CLI) ─────────────────────
        self.planner = Planner(
            registry=self.registry,
            page_store=self.page_store,
            mistake_tracker=self.mistakes,
            permissions=self.permissions,
            claude=self.claude if ClaudeCode.is_installed() else None,
        )

        # ── Register default agents ──────────────────────────────────
        self._register_default_agents()

    def _register_default_agents(self) -> None:
        """Register the built-in domain agents."""
        # AI Agent — uses local Claude Code CLI
        if ClaudeCode.is_installed():
            self.registry.register(AIAgent(
                default_model=self.config.default_model,
                cwd=self.config.cwd,
            ))
        else:
            logger.warning("Claude Code CLI not found. AI agent disabled.")

        # Design Agent (skeleton)
        self.registry.register(DesignAgent())

        # Test Writer Agent (skeleton)
        self.registry.register(TestWriterAgent())

        logger.info("Registered agents: %s", self.registry.list_agents())

    # ── Public API ───────────────────────────────────────────────────

    async def run(self, intent: str, session: Session | None = None) -> dict[str, Any]:
        """
        Main entry point. Accepts a user intent string and returns
        the full structured Cortex output.

        If a session is provided, resumes that session. Otherwise
        creates a new one and auto-saves throughout.
        """
        logger.info("=" * 60)
        logger.info("CORTEX — New intent: %s", intent[:100])
        logger.info("=" * 60)

        # ── Session setup ────────────────────────────────────────────
        if session:
            self._current_session = session
            session.status = SessionStatus.ACTIVE
            session.add_event("resumed", {"intent": intent})
        else:
            self._current_session = Session(
                intent=intent,
                model=self.config.default_model,
                status=SessionStatus.ACTIVE,
            )
            self._current_session.add_event("started", {"intent": intent})

        self.session_manager.save(self._current_session)

        try:
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

            # Save plan to session
            self._current_session.plan_steps = [
                {"id": s.id, "agent": s.agent, "action": s.action,
                 "status": s.status.value, "quality": s.quality.value}
                for s in plan.steps
            ]
            self._current_session.add_event("planned", {
                "step_count": len(plan.steps),
            })
            self.session_manager.save(self._current_session)

            # Optional: set up tmux for visual parallelism
            if self.config.use_tmux:
                self.tmux.setup()

            # Execute the plan
            outputs = await self.planner.execute(plan, max_retries=self.config.max_retries)

            # Teardown tmux
            if self.config.use_tmux:
                self.tmux.teardown()

            # Build final output
            result = self._build_output(plan, outputs)

            # ── Auto-save completed session ──────────────────────────
            self._current_session.plan_steps = result.get("steps_executed", [])
            self._current_session.pages = [
                p.to_dict() for p in self.page_store.all_pages()
            ]
            self._current_session.mistakes = result.get("mistakes", [])
            self._current_session.outputs = [
                {"agent": s.get("agent"), "status": s.get("status")}
                for s in result.get("steps_executed", [])
            ]
            self._current_session.result = result.get("result")

            if any(s.get("status") == "failed" for s in result.get("steps_executed", [])):
                self._current_session.mark_failed("One or more steps failed")
            else:
                self._current_session.mark_completed(result.get("result"))

            self.session_manager.save(self._current_session)
            return result

        except Exception as e:
            # Save failure state
            if self._current_session:
                self._current_session.mark_failed(str(e))
                self._current_session.pages = [
                    p.to_dict() for p in self.page_store.all_pages()
                ]
                self._current_session.mistakes = [
                    m.to_dict() for m in self.mistakes.all
                ]
                self.session_manager.save(self._current_session)
            raise

    @property
    def current_session(self) -> Session | None:
        return self._current_session

    async def resume(self, session_id: str) -> dict[str, Any]:
        """Resume a previously saved session."""
        session = self.session_manager.load(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        logger.info("Resuming session %s: %s", session.id, session.intent[:60])

        # Restore pages from session
        for page_data in session.pages:
            page = PageIndex(
                id=page_data.get("id", ""),
                type=PageType(page_data.get("type", "memory")),
                summary=page_data.get("summary", ""),
                token_count=page_data.get("token_count", 0),
                tags=page_data.get("tags", []),
                content=page_data.get("content", ""),
            )
            self.page_store.add(page)

        # Restore mistakes
        for m_data in session.mistakes:
            from cortex.models import MistakeRecord, MistakeType
            m = MistakeRecord(
                step_id=m_data.get("step_id", ""),
                type=MistakeType(m_data.get("type", "plan_error")),
                description=m_data.get("description", ""),
                correction=m_data.get("correction", ""),
                learned=m_data.get("learned", ""),
            )
            self.mistakes.record(m)

        # Re-run with the original intent
        return await self.run(session.intent, session=session)

    async def index_document(self, path: str, doc_id: str | None = None) -> str:
        """Index a document into the RAG bridge for future queries."""
        return await self.rag.index_document(path, doc_id)

    # ── Output builder ───────────────────────────────────────────────

    def _build_output(self, plan: Plan, outputs: list[AgentOutput]) -> dict[str, Any]:
        """Assemble the final structured output per the Cortex spec."""
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

        primary_result = None
        for output in reversed(outputs):
            if output.quality != Quality.ERROR and output.result:
                primary_result = output.result
                break

        return {
            "plan_id": plan.plan_id,
            "intent_summary": plan.intent[:200],
            "skills_used": sorted(skills_used),
            "pages_loaded": [p.id for p in self.page_store.all_pages()],
            "steps_executed": [
                {
                    "agent": step.agent,
                    "action": step.action,
                    "status": step.status.value,
                    "quality": step.quality.value,
                }
                for step in plan.steps
            ],
            "result": primary_result,
            "mistakes": [m.to_dict() for m in self.mistakes.all],
            "lessons_learned": self.mistakes.lessons_learned(),
            "memory_updates": [],
        }

    # ── Convenience ──────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return current engine status for the TUI."""
        return {
            "claude_code_installed": ClaudeCode.is_installed(),
            "claude_code_version": ClaudeCode.version(),
            "model": self.config.default_model,
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
