"""
Tmux Runner — spawns each agent in its own tmux pane for true parallelism.

When running multiple agents concurrently, each gets a dedicated tmux pane
inside a Cortex session. This gives:
  - Visual isolation: each agent's output streams in its own pane
  - True parallelism: separate Python processes, not just async tasks
  - Easy debugging: attach to any pane to inspect a stuck agent
  - Clean teardown: kill the tmux session to stop everything

Layout:
  ┌──────────────┬──────────────┐
  │  ai_agent    │  design_agent│
  ├──────────────┼──────────────┤
  │  test_writer │  planner     │
  └──────────────┴──────────────┘

Usage:
  runner = TmuxRunner(session_name="cortex")
  runner.setup()
  runner.spawn_agent("ai_agent", "python -m cortex.run_agent ai_agent --task '...'")
  runner.teardown()
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PaneInfo:
    """Tracks a single tmux pane running an agent."""
    agent_name: str
    pane_id: str
    pid: int | None = None
    status: str = "running"  # running | done | error


class TmuxRunner:
    """
    Manages a tmux session where each agent gets its own pane.
    Falls back to subprocess-based execution if tmux is not available.
    """

    def __init__(self, session_name: str = "cortex", work_dir: str | None = None) -> None:
        self.session_name = session_name
        self.work_dir = work_dir or str(Path.cwd())
        self._panes: dict[str, PaneInfo] = {}
        self._tmux_available = shutil.which("tmux") is not None
        self._fallback_procs: dict[str, subprocess.Popen] = {}

    # ── Session lifecycle ────────────────────────────────────────────

    def setup(self) -> bool:
        """Create the tmux session. Returns True if tmux is available."""
        if not self._tmux_available:
            logger.warning("tmux not found. Falling back to subprocess mode.")
            return False

        # Kill existing session if any
        subprocess.run(
            ["tmux", "kill-session", "-t", self.session_name],
            capture_output=True,
        )

        # Create new detached session
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.session_name, "-x", "200", "-y", "50"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("Failed to create tmux session: %s", result.stderr)
            return False

        # Rename the first window
        subprocess.run(
            ["tmux", "rename-window", "-t", f"{self.session_name}:0", "cortex"],
            capture_output=True,
        )
        logger.info("Tmux session '%s' created.", self.session_name)
        return True

    def teardown(self) -> None:
        """Kill the tmux session and all agent panes."""
        if self._tmux_available:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name],
                capture_output=True,
            )
            logger.info("Tmux session '%s' killed.", self.session_name)

        # Also clean up any fallback subprocesses
        for name, proc in self._fallback_procs.items():
            if proc.poll() is None:
                proc.terminate()
                logger.info("Terminated fallback subprocess for %s", name)

        self._panes.clear()
        self._fallback_procs.clear()

    # ── Agent pane management ────────────────────────────────────────

    def spawn_agent(self, agent_name: str, command: str) -> str | None:
        """
        Spawn an agent in a new tmux pane (or subprocess fallback).
        Returns the pane ID or PID as a string.
        """
        if self._tmux_available:
            return self._spawn_tmux_pane(agent_name, command)
        else:
            return self._spawn_subprocess(agent_name, command)

    def _spawn_tmux_pane(self, agent_name: str, command: str) -> str | None:
        """Create a new tmux pane for this agent."""
        # Split the current window to create a new pane
        result = subprocess.run(
            [
                "tmux", "split-window", "-t", self.session_name,
                "-h",  # horizontal split
                "-P", "-F", "#{pane_id}",  # print pane ID
                command,
            ],
            capture_output=True, text=True, cwd=self.work_dir,
        )

        if result.returncode != 0:
            # If split fails (too small), try a new window instead
            result = subprocess.run(
                [
                    "tmux", "new-window", "-t", self.session_name,
                    "-n", agent_name,
                    "-P", "-F", "#{pane_id}",
                    command,
                ],
                capture_output=True, text=True, cwd=self.work_dir,
            )

        if result.returncode != 0:
            logger.error("Failed to spawn pane for %s: %s", agent_name, result.stderr)
            return None

        pane_id = result.stdout.strip()
        self._panes[agent_name] = PaneInfo(agent_name=agent_name, pane_id=pane_id)

        # Re-tile for clean layout
        subprocess.run(
            ["tmux", "select-layout", "-t", self.session_name, "tiled"],
            capture_output=True,
        )

        logger.info("Agent '%s' spawned in pane %s", agent_name, pane_id)
        return pane_id

    def _spawn_subprocess(self, agent_name: str, command: str) -> str:
        """Fallback: run agent as a subprocess."""
        log_file = Path(tempfile.mkdtemp()) / f"{agent_name}.log"
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=self.work_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
        )
        self._fallback_procs[agent_name] = proc
        self._panes[agent_name] = PaneInfo(
            agent_name=agent_name,
            pane_id=f"pid:{proc.pid}",
            pid=proc.pid,
        )
        logger.info("Agent '%s' spawned as PID %d (log: %s)", agent_name, proc.pid, log_file)
        return f"pid:{proc.pid}"

    # ── Status and output ────────────────────────────────────────────

    def get_pane_output(self, agent_name: str, lines: int = 50) -> str:
        """Capture recent output from an agent's tmux pane."""
        info = self._panes.get(agent_name)
        if not info:
            return ""

        if self._tmux_available and info.pane_id and not info.pane_id.startswith("pid:"):
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", info.pane_id, "-p", "-S", f"-{lines}"],
                capture_output=True, text=True,
            )
            return result.stdout if result.returncode == 0 else ""
        return ""

    def is_agent_running(self, agent_name: str) -> bool:
        """Check if an agent's pane/process is still alive."""
        info = self._panes.get(agent_name)
        if not info:
            return False

        if agent_name in self._fallback_procs:
            return self._fallback_procs[agent_name].poll() is None

        if self._tmux_available and info.pane_id:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", self.session_name, "-F", "#{pane_id} #{pane_dead}"],
                capture_output=True, text=True,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2 and parts[0] == info.pane_id:
                    return parts[1] == "0"
        return False

    def send_to_pane(self, agent_name: str, text: str) -> None:
        """Send keystrokes to an agent's pane (for interactive agents)."""
        info = self._panes.get(agent_name)
        if info and self._tmux_available:
            subprocess.run(
                ["tmux", "send-keys", "-t", info.pane_id, text, "Enter"],
                capture_output=True,
            )

    def list_panes(self) -> dict[str, PaneInfo]:
        return dict(self._panes)

    @property
    def active_count(self) -> int:
        return sum(1 for name in self._panes if self.is_agent_running(name))
