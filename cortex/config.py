"""
Cortex configuration — centralises all tunables.

Cortex runs on the local Claude Code CLI. No API keys needed.
All dependencies are injected at runtime (Dependency Inversion Principle).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class CortexConfig:
    """Top-level configuration for a Cortex session."""

    # ── Claude Code ──────────────────────────────────────────────────
    default_model: str = "sonnet"  # claude CLI model names: haiku, sonnet, opus

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

    # ── Sessions ─────────────────────────────────────────────────────
    sessions_dir: str | None = None  # defaults to .cortex/sessions/

    # ── Working directory ────────────────────────────────────────────
    cwd: str = field(default_factory=os.getcwd)

    def validate(self) -> list[str]:
        """Return a list of configuration warnings."""
        from cortex.claude_code import ClaudeCode

        warnings = []
        if not ClaudeCode.is_installed():
            warnings.append(
                "Claude Code CLI (`claude`) not found on PATH. "
                "Install it: https://docs.anthropic.com/en/docs/claude-code"
            )
        return warnings
