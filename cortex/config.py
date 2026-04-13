"""
Cortex configuration — centralises all tunables.

All dependencies are injected at runtime (Dependency Inversion Principle).
This module provides the default configuration and a factory for building
the full Cortex stack from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class CortexConfig:
    """Top-level configuration for a Cortex session."""

    # ── API ──────────────────────────────────────────────────────────
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    default_model: str = "claude-sonnet-4-6"

    # ── Context ──────────────────────────────────────────────────────
    max_context_tokens: int = 80_000
    similarity_threshold: float = 0.75

    # ── Planner ──────────────────────────────────────────────────────
    confidence_threshold: float = 0.6
    max_retries: int = 1

    # ── Permissions ──────────────────────────────────────────────────
    default_permissions: list[str] = field(default_factory=lambda: [
        "read", "write", "design:read", "design:write",
        "test:generate", "ai:reason", "rag:query",
    ])

    # ── Tmux ─────────────────────────────────────────────────────────
    tmux_session_name: str = "cortex"
    use_tmux: bool = True

    # ── TUI ──────────────────────────────────────────────────────────
    tui_refresh_rate: float = 0.5  # seconds

    # ── RAG (VectifyAI PageIndex) ────────────────────────────────────
    rag_data_dir: str = field(
        default_factory=lambda: os.environ.get("CORTEX_RAG_DIR", "/tmp/cortex_rag")
    )

    def validate(self) -> list[str]:
        """Return a list of configuration warnings."""
        warnings = []
        if not self.anthropic_api_key:
            warnings.append(
                "ANTHROPIC_API_KEY not set. AI agent and LLM planning will be disabled."
            )
        return warnings
