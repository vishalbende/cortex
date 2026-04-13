"""
Claude Code CLI bridge — talks to the local `claude` command.

Cortex runs on top of Claude Code (the local CLI tool), NOT the
Anthropic API. This module wraps the `claude` command so the rest
of Cortex can call it without knowing subprocess details.

Claude Code supports several invocation modes:
  claude "prompt"                     # one-shot
  claude -p "prompt"                  # print mode (no interactive UI)
  claude -p "prompt" --output-format json  # structured JSON output
  claude --model sonnet              # model selection

This bridge provides:
  - ClaudeCode.run(prompt)            → raw text response
  - ClaudeCode.run_json(prompt)       → parsed JSON response
  - ClaudeCode.stream(prompt)         → streaming line-by-line
  - ClaudeCode.is_installed()         → check if claude CLI exists
  - ClaudeCode.version()              → installed version string
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    """Parsed response from a claude CLI invocation."""

    text: str = ""
    exit_code: int = 0
    model: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0
    session_id: str = ""
    is_error: bool = False

    @classmethod
    def from_json(cls, raw: str, exit_code: int = 0) -> "ClaudeResponse":
        """Parse a --output-format json response."""
        try:
            data = json.loads(raw)
            # Claude Code JSON output has a `result` field with the text
            text = data.get("result", "")
            if not text:
                # Some versions nest it differently
                text = data.get("text", data.get("content", raw))
            return cls(
                text=str(text),
                exit_code=exit_code,
                model=data.get("model", ""),
                cost_usd=data.get("cost_usd", 0.0),
                duration_ms=data.get("duration_ms", 0),
                session_id=data.get("session_id", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return cls(text=raw, exit_code=exit_code)

    @classmethod
    def error(cls, message: str, exit_code: int = 1) -> "ClaudeResponse":
        return cls(text=message, exit_code=exit_code, is_error=True)


class ClaudeCode:
    """
    Bridge to the locally installed Claude Code CLI.

    All Cortex LLM operations go through this instead of the
    Anthropic Python SDK. No API key needed — Claude Code handles
    authentication via its own local session.
    """

    DEFAULT_TIMEOUT = 120  # seconds

    def __init__(
        self,
        model: str = "sonnet",
        cwd: str | None = None,
        allowed_tools: list[str] | None = None,
        max_turns: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.cwd = cwd or str(Path.cwd())
        self.allowed_tools = allowed_tools or []
        self.max_turns = max_turns
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._claude_bin = shutil.which("claude") or "claude"

    # ── Core execution ───────────────────────────────────────────────

    async def run(self, prompt: str, system_prompt: str | None = None) -> ClaudeResponse:
        """
        Run a prompt through Claude Code in print mode (-p).
        Returns the full text response. Times out after self.timeout seconds.
        """
        cmd = self._build_command(prompt, system_prompt=system_prompt)
        logger.debug("claude cmd: %s", " ".join(cmd[:6]) + "...")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("claude timed out after %ds", self.timeout)
            return ClaudeResponse.error(
                f"Claude Code timed out after {self.timeout}s", exit_code=124
            )

        text = stdout.decode("utf-8", errors="replace").strip()
        exit_code = proc.returncode or 0

        if exit_code != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            logger.warning("claude exited %d: %s", exit_code, err_text[:200])
            return ClaudeResponse.error(err_text or text, exit_code)

        return ClaudeResponse(text=text, exit_code=0, model=self.model)

    async def run_json(self, prompt: str, system_prompt: str | None = None) -> ClaudeResponse:
        """
        Run a prompt with --output-format json for structured output.
        Times out after self.timeout seconds.
        """
        cmd = self._build_command(
            prompt, system_prompt=system_prompt, output_format="json"
        )
        logger.debug("claude json cmd: %s", " ".join(cmd[:6]) + "...")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("claude json timed out after %ds", self.timeout)
            return ClaudeResponse.error(
                f"Claude Code timed out after {self.timeout}s", exit_code=124
            )

        raw = stdout.decode("utf-8", errors="replace").strip()
        exit_code = proc.returncode or 0

        if exit_code != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            return ClaudeResponse.error(err or raw, exit_code)

        return ClaudeResponse.from_json(raw, exit_code)

    async def stream(self, prompt: str, system_prompt: str | None = None) -> AsyncIterator[str]:
        """
        Stream output line-by-line from Claude Code.
        Useful for long-running tasks displayed in the TUI.
        """
        cmd = self._build_command(prompt, system_prompt=system_prompt, stream=True)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        async for line in proc.stdout:
            yield line.decode("utf-8", errors="replace").rstrip("\n")

        await proc.wait()

    # ── Command builder ──────────────────────────────────────────────

    def _build_command(
        self,
        prompt: str,
        system_prompt: str | None = None,
        output_format: str | None = None,
        stream: bool = False,
    ) -> list[str]:
        """Assemble the claude CLI command."""
        cmd = [self._claude_bin, "-p", prompt]

        # Model
        cmd.extend(["--model", self.model])

        # System prompt
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # Output format
        if output_format:
            cmd.extend(["--output-format", output_format])

        # Allowed tools
        for tool in self.allowed_tools:
            cmd.extend(["--allowedTools", tool])

        # Max turns
        if self.max_turns:
            cmd.extend(["--max-turns", str(self.max_turns)])

        return cmd

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def is_installed() -> bool:
        """Check if the claude CLI is available on PATH."""
        return shutil.which("claude") is not None

    @staticmethod
    def version() -> str:
        """Return the installed Claude Code version string."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    def with_model(self, model: str) -> "ClaudeCode":
        """Return a new ClaudeCode instance with a different model."""
        return ClaudeCode(
            model=model,
            cwd=self.cwd,
            allowed_tools=self.allowed_tools,
            max_turns=self.max_turns,
            timeout=self.timeout,
        )

    def with_tools(self, tools: list[str]) -> "ClaudeCode":
        """Return a new ClaudeCode instance with specific allowed tools."""
        return ClaudeCode(
            model=self.model,
            cwd=self.cwd,
            allowed_tools=tools,
            max_turns=self.max_turns,
            timeout=self.timeout,
        )

    def __repr__(self) -> str:
        installed = "yes" if self.is_installed() else "no"
        return f"<ClaudeCode model={self.model} installed={installed}>"
