"""
Cortex TUI — Textual-based terminal dashboard with real-time updates.

Layout:
  ┌──────────────────┬──────────────────┐
  │  PLAN            │  PAGES LOADED    │
  │  step 1 ✓        │  page:001 (rag)  │
  │  step 2 ⟳        │  page:002 (plan) │
  ├──────────────────┼──────────────────┤
  │  ACTIVE AGENTS   │  MISTAKE LOG     │
  │  ▶ ai_agent      │  ⚠ step_2 warn  │
  ├──────────────────┴──────────────────┤
  │  OUTPUT LOG                         │
  │  ⟳ Decomposing intent...           │
  │  ▶ step_1 [ai_agent] starting...   │
  │  ✓ step_1 done (0.8s)              │
  ├─────────────────────────────────────┤
  │ cortex> Type your instruction…      │
  └─────────────────────────────────────┘
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
)

from cortex.models import PlanStep, Quality, StepStatus


# ── Status icons ─────────────────────────────────────────────────────

STATUS_ICONS = {
    StepStatus.PENDING: "○",
    StepStatus.RUNNING: "⟳",
    StepStatus.DONE: "✓",
    StepStatus.FAILED: "✗",
    StepStatus.SKIPPED: "–",
}

STATUS_COLORS = {
    StepStatus.PENDING: "dim",
    StepStatus.RUNNING: "cyan",
    StepStatus.DONE: "green",
    StepStatus.FAILED: "red",
    StepStatus.SKIPPED: "dim",
}

QUALITY_STYLE = {
    Quality.OK: "green",
    Quality.WARNING: "yellow",
    Quality.ERROR: "red",
}


# ── Custom Textual Messages for real-time updates ────────────────────

class EngineLog(Message):
    """Push a log line to the output panel from inside a worker."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PlanUpdated(Message):
    """Plan steps changed — refresh the Plan panel."""

    def __init__(self, steps: list[dict]) -> None:
        super().__init__()
        self.steps = steps


class AgentsUpdated(Message):
    """Active agents changed — refresh the Agents panel."""

    def __init__(self, agents: list[dict]) -> None:
        super().__init__()
        self.agents = agents


class PagesUpdated(Message):
    """Context pages changed — refresh the Pages panel."""

    def __init__(self, pages: list[dict]) -> None:
        super().__init__()
        self.pages = pages


class MistakesUpdated(Message):
    """Mistakes changed — refresh the Mistake panel."""

    def __init__(self, mistakes: list[dict]) -> None:
        super().__init__()
        self.mistakes = mistakes


# ── Custom Widgets ───────────────────────────────────────────────────

class PlanPanel(Static):
    """Displays the current execution plan with live step statuses."""

    steps: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]PLAN[/bold cyan]")
        if not self.steps:
            yield Label("  [dim]No plan yet.[/dim]")
        for s in self.steps:
            status_str = s.get("status", "pending")
            try:
                status = StepStatus(status_str)
            except ValueError:
                status = StepStatus.PENDING
            icon = STATUS_ICONS.get(status, "?")
            color = STATUS_COLORS.get(status, "white")
            quality_str = s.get("quality", "ok")
            try:
                quality = Quality(quality_str)
            except ValueError:
                quality = Quality.OK
            q_color = QUALITY_STYLE.get(quality, "white")
            yield Label(
                f"  {icon} [{color}]{s.get('id', '?')}[/{color}] "
                f"[{q_color}]{s.get('agent', '')}[/{q_color}] "
                f"[dim]{s.get('action', '')}[/dim]"
            )


class PagesPanel(Static):
    """Shows loaded context pages with type badges."""

    pages: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]PAGES LOADED[/bold cyan]")
        if not self.pages:
            yield Label("  [dim]No pages loaded.[/dim]")
        for p in self.pages[:15]:
            badge = p.get("type", "?")[:4].upper()
            yield Label(f"  [{badge}] {p.get('id', '?')} — {p.get('summary', '')}")


class AgentsPanel(Static):
    """Shows active agents and their current status."""

    agents: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]ACTIVE AGENTS[/bold cyan]")
        if not self.agents:
            yield Label("  [dim]No agents running.[/dim]")
        for a in self.agents:
            status = a.get("status", "idle")
            icon = "▶" if status == "running" else "■" if status == "done" else "○"
            color = "green" if status == "running" else "cyan" if status == "done" else "dim"
            step = a.get("step", "")
            yield Label(
                f"  {icon} [{color}]{a.get('name', '?')}[/{color}] "
                f"[dim]{step}[/dim]"
            )


class MistakePanel(Static):
    """Filterable mistake log."""

    mistakes: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold red]MISTAKE LOG[/bold red]")
        if not self.mistakes:
            yield Label("  [dim]No mistakes recorded.[/dim]")
        for m in self.mistakes:
            icon = "✗" if m.get("type") in ("tool_failure", "plan_error") else "⚠"
            yield Label(
                f"  {icon} [yellow]{m.get('step_id', '?')}[/yellow] "
                f"[dim]{m.get('type', '')}[/dim]: {m.get('description', '')}"
            )


# ── Markdown detection helper ────────────────────────────────────────

def _looks_like_markdown(text: str) -> bool:
    """Heuristic: returns True if text appears to contain Markdown formatting."""
    import re
    indicators = [
        re.search(r'^#{1,6}\s', text, re.MULTILINE),   # headings
        re.search(r'\*\*[^*]+\*\*', text),               # bold
        re.search(r'^[-*]\s', text, re.MULTILINE),        # unordered lists
        re.search(r'^\d+\.\s', text, re.MULTILINE),       # ordered lists
        re.search(r'^```', text, re.MULTILINE),            # code fences
        re.search(r'^---+$', text, re.MULTILINE),          # horizontal rules
        re.search(r'\[.+\]\(.+\)', text),                  # links
    ]
    # If two or more indicators match, it's likely markdown
    return sum(1 for i in indicators if i) >= 2


# ── Main TUI App ─────────────────────────────────────────────────────

class CortexTUI(App):
    """
    Cortex terminal dashboard — observe plan execution in real time.
    Type instructions in the input box at the bottom and press Enter.
    """

    TITLE = "Cortex — Agent Orchestration"
    SUB_TITLE = "Enter instruction below | quit to exit"

    CSS = """
    Screen {
        background: #0F172A;
        color: #F8FAFC;
        layout: vertical;
    }

    #top-row { height: 1fr; min-height: 6; }
    #mid-row { height: 1fr; min-height: 6; }

    #output-log {
        height: 2fr;
        min-height: 6;
        border: solid #6366F1;
        padding: 0 1;
    }

    #intent-input { height: 3; margin: 0 1; }

    PlanPanel   { width: 1fr; height: 100%; border: solid #6366F1; padding: 0 1; }
    PagesPanel  { width: 1fr; height: 100%; border: solid #6366F1; padding: 0 1; }
    AgentsPanel { width: 1fr; height: 100%; border: solid #6366F1; padding: 0 1; }
    MistakePanel{ width: 1fr; height: 100%; border: solid #EF4444; padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=False),
        Binding("escape", "cancel_task", "Cancel", priority=True),
        Binding("p", "show_plan", "Plan", priority=False),
        Binding("m", "show_mistakes", "Mistakes", priority=False),
        Binding("c", "show_context", "Context", priority=False),
        Binding("r", "rerun", "Re-run", priority=False),
        Binding("s", "show_sessions", "Sessions", priority=False),
    ]

    def __init__(self, engine=None, **kwargs):
        super().__init__(**kwargs)
        self._engine = engine
        self._last_intent: str | None = None
        self._current_worker: Worker | None = None
        self._active_agents: dict[str, dict] = {}
        self._step_timers: dict[str, float] = {}

        # Detect repo info for display
        self._repo_info = self._detect_repo()

    @property
    def is_busy(self) -> bool:
        if self._current_worker is None:
            return False
        return self._current_worker.state == WorkerState.RUNNING

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            yield PlanPanel(id="plan-panel")
            yield PagesPanel(id="pages-panel")
        with Horizontal(id="mid-row"):
            yield AgentsPanel(id="agents-panel")
            yield MistakePanel(id="mistake-panel")
        yield RichLog(id="output-log", highlight=True, markup=True)
        yield Input(
            placeholder="cortex> Type your instruction and press Enter…",
            id="intent-input",
        )
        yield Footer()

    def _detect_repo(self) -> dict:
        """Detect git repo name, branch, remotes, and all branches with tracking info."""
        cwd = self._engine.config.cwd if self._engine else str(Path.cwd())
        info: dict = {
            "cwd": cwd,
            "name": Path(cwd).name,
            "branch": None,
            "is_git": False,
            "remotes": [],      # list of {"name": "origin", "url": "..."}
            "branches": [],     # list of {"name": "main", "tracking": "origin/main", "current": True}
        }
        try:
            # Get repo root name
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, cwd=cwd, timeout=3,
            )
            if result.returncode == 0:
                repo_root = result.stdout.strip()
                info["name"] = Path(repo_root).name
                info["cwd"] = repo_root
                info["is_git"] = True

            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=cwd, timeout=3,
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()

            # Get all remotes with URLs
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True, text=True, cwd=cwd, timeout=3,
            )
            if result.returncode == 0:
                seen = set()
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] not in seen:
                        seen.add(parts[0])
                        info["remotes"].append({"name": parts[0], "url": parts[1]})

            # Get all local branches with tracking info
            result = subprocess.run(
                ["git", "branch", "-vv", "--no-color"],
                capture_output=True, text=True, cwd=cwd, timeout=3,
            )
            if result.returncode == 0:
                import re
                for line in result.stdout.strip().splitlines():
                    is_current = line.startswith("*")
                    line = line.lstrip("* ").strip()
                    parts = line.split(None, 2)
                    if not parts:
                        continue
                    branch_name = parts[0]
                    # Extract tracking remote from [origin/main] or [origin/main: ahead 1]
                    tracking = None
                    match = re.search(r'\[([^\]:]+)', line)
                    if match:
                        tracking = match.group(1).strip()
                    info["branches"].append({
                        "name": branch_name,
                        "tracking": tracking,
                        "current": is_current,
                    })
        except Exception:
            pass
        return info

    def on_mount(self) -> None:
        # Set subtitle with repo info
        repo = self._repo_info
        if repo["is_git"]:
            self.sub_title = f"{repo['name']} ({repo['branch']}) — {repo['cwd']}"
        else:
            self.sub_title = repo["cwd"]

        self.log_output("[bold green]Cortex TUI started.[/bold green]")
        if repo["is_git"]:
            self.log_output(
                f"[bold cyan]Repo:[/bold cyan] {repo['name']}  "
                f"[bold cyan]Branch:[/bold cyan] {repo['branch']}  "
                f"[bold cyan]Path:[/bold cyan] {repo['cwd']}"
            )
            # Show remotes
            if repo["remotes"]:
                self.log_output("[bold cyan]Remotes:[/bold cyan]")
                for r in repo["remotes"]:
                    self.log_output(f"  [green]{r['name']}[/green]  {r['url']}")

            # Show all branches with tracking info
            if repo["branches"]:
                self.log_output("[bold cyan]Branches:[/bold cyan]")
                for b in repo["branches"]:
                    marker = "[bold green]* [/bold green]" if b["current"] else "  "
                    name_color = "bold green" if b["current"] else "white"
                    tracking = f" [dim]→ {b['tracking']}[/dim]" if b["tracking"] else " [dim](no remote)[/dim]"
                    self.log_output(f"  {marker}[{name_color}]{b['name']}[/{name_color}]{tracking}")
        else:
            self.log_output(f"[bold cyan]Working directory:[/bold cyan] {repo['cwd']}")

        if self._engine:
            self.log_output("[dim]Engine connected. Type an instruction below to begin.[/dim]")
        else:
            self.log_output("[dim]No engine connected. Launch with: cortex tui[/dim]")
        self.query_one("#intent-input", Input).focus()

    # ── Message handlers for real-time updates ───────────────────────

    def on_engine_log(self, message: EngineLog) -> None:
        self.log_output(message.text)

    def on_plan_updated(self, message: PlanUpdated) -> None:
        panel = self.query_one("#plan-panel", PlanPanel)
        panel.steps = message.steps

    def on_agents_updated(self, message: AgentsUpdated) -> None:
        panel = self.query_one("#agents-panel", AgentsPanel)
        panel.agents = message.agents

    def on_pages_updated(self, message: PagesUpdated) -> None:
        panel = self.query_one("#pages-panel", PagesPanel)
        panel.pages = message.pages

    def on_mistakes_updated(self, message: MistakesUpdated) -> None:
        panel = self.query_one("#mistake-panel", MistakePanel)
        panel.mistakes = message.mistakes

    # ── Input handling ───────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        intent = event.value.strip()
        input_widget = self.query_one("#intent-input", Input)
        input_widget.value = ""

        if not intent:
            return

        # Built-in commands
        if intent in ("quit", "exit", "/quit", "/exit", "q"):
            self.exit()
            return
        if intent in ("/status", "status"):
            if self._engine:
                import json
                self.log_output(f"[cyan]{json.dumps(self._engine.status(), indent=2)}[/cyan]")
            return
        if intent in ("/agents", "agents"):
            if self._engine:
                self.log_output(f"[cyan]Agents: {', '.join(self._engine.registry.list_agents())}[/cyan]")
            return
        if intent in ("/help", "help"):
            self.log_output(
                "[bold]Commands:[/bold] /status, /agents, /sessions, /resume <id>, /help, quit\n"
                "[bold]Keys:[/bold] q=quit, Esc=cancel, s=sessions, p=plan, m=mistakes, c=context, r=re-run"
            )
            return
        if intent in ("/sessions", "sessions", "/ss"):
            self._show_sessions()
            return
        if intent.startswith(("/resume ", "resume ")):
            self._resume_session(intent.split(maxsplit=1)[1].strip())
            return
        if intent.startswith(("/show ", "show ")):
            self._show_session_detail(intent.split(maxsplit=1)[1].strip())
            return

        # Run intent
        if not self._engine:
            self.log_output("[red]No engine connected.[/red]")
            return
        if self.is_busy:
            self.log_output("[yellow]Task in progress. Press Escape to cancel.[/yellow]")
            return

        self._last_intent = intent
        self._active_agents.clear()
        self._step_timers.clear()
        self.log_output(f"\n[bold]▶ Intent:[/bold] {intent}")

        self._current_worker = self.run_worker(
            self._execute_intent(intent),
            name="intent_runner",
            exclusive=True,
        )

    # ── Engine execution with live callbacks ─────────────────────────

    async def _execute_intent(self, intent: str) -> dict:
        """Execute an intent with real-time TUI updates via callbacks."""
        engine = self._engine
        app = self

        # ── Wire up planner callbacks for live updates ───────────────

        async def on_step_start(step: PlanStep) -> None:
            """Called when a step begins execution."""
            app._step_timers[step.id] = time.monotonic()
            app._active_agents[step.agent] = {
                "name": step.agent, "status": "running", "step": step.action,
            }
            # Post messages to update TUI (thread-safe via Textual messages)
            app.post_message(EngineLog(
                f"  [cyan]⟳[/cyan] [bold]{step.id}[/bold] "
                f"[cyan][{step.agent}][/cyan] {step.action}"
            ))
            app.post_message(AgentsUpdated(list(app._active_agents.values())))
            # Update plan panel with current step statuses
            steps = [
                {"id": s.id, "agent": s.agent, "action": s.action,
                 "status": s.status.value, "quality": s.quality.value}
                for s in engine.planner._current_plan_steps
            ] if hasattr(engine.planner, '_current_plan_steps') else []
            if steps:
                app.post_message(PlanUpdated(steps))

        async def on_step_done(step: PlanStep, output) -> None:
            """Called when a step completes."""
            elapsed = time.monotonic() - app._step_timers.get(step.id, time.monotonic())
            status_icon = "✓" if step.status == StepStatus.DONE else "✗"
            status_color = "green" if step.status == StepStatus.DONE else "red"
            app._active_agents[step.agent] = {
                "name": step.agent, "status": "done", "step": step.action,
            }
            app.post_message(EngineLog(
                f"  [{status_color}]{status_icon}[/{status_color}] [bold]{step.id}[/bold] "
                f"[{status_color}][{step.agent}][/{status_color}] "
                f"done ({elapsed:.1f}s) — quality: {step.quality.value}"
            ))
            app.post_message(AgentsUpdated(list(app._active_agents.values())))
            # Update pages
            app.post_message(PagesUpdated(
                [p.to_dict() for p in engine.page_store.all_pages()]
            ))
            # Update mistakes
            app.post_message(MistakesUpdated(
                [m.to_dict() for m in engine.mistakes.all]
            ))

        async def on_replan(plan) -> None:
            """Called when the planner re-plans after a failure."""
            app.post_message(EngineLog(
                "[yellow]⟳ Re-planning after failure…[/yellow]"
            ))

        # Hook callbacks into the planner
        engine.planner.on_step_start = on_step_start
        engine.planner.on_step_done = on_step_done
        engine.planner.on_replan = on_replan

        # ── Run the engine ───────────────────────────────────────────
        self.post_message(EngineLog("[dim]⟳ Decomposing intent into plan…[/dim]"))

        result = await engine.run(intent)

        # Update plan panel with final state
        steps_executed = result.get("steps_executed", [])
        if steps_executed:
            self.post_message(PlanUpdated(steps_executed))

        # Final pages + mistakes update
        self.post_message(PagesUpdated(
            [p.to_dict() for p in engine.page_store.all_pages()]
        ))
        self.post_message(MistakesUpdated(
            [m.to_dict() for m in engine.mistakes.all]
        ))

        return result

    # ── Worker lifecycle ─────────────────────────────────────────────

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "intent_runner":
            return

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            self._current_worker = None
            self._active_agents.clear()
            self.post_message(AgentsUpdated([]))

            if not isinstance(result, dict):
                self.log_output("[green]✓ Done.[/green]")
                return

            output = result.get("result", "")
            if output:
                output_str = str(output)
                # Detect markdown content and render it properly
                if _looks_like_markdown(output_str):
                    self.log_output("[green]✓ Result:[/green]")
                    self.log_markdown(output_str)
                else:
                    self.log_output(f"\n[green]✓ Result:[/green] {output_str[:500]}")
            else:
                self.log_output("[green]✓ Done.[/green]")

            mistakes = result.get("mistakes", [])
            if mistakes:
                self.log_output(f"[yellow]⚠ {len(mistakes)} mistake(s) recorded[/yellow]")

            lessons = result.get("lessons_learned", [])
            if lessons:
                self.log_output(f"[dim]Lessons: {'; '.join(lessons)}[/dim]")

            if self._engine and self._engine.current_session:
                self.log_output(f"[dim]Session: {self._engine.current_session.id}[/dim]")

        elif event.state == WorkerState.ERROR:
            self.log_output(f"[red]✗ Error: {event.worker.error}[/red]")
            self._current_worker = None
            self._active_agents.clear()

        elif event.state == WorkerState.CANCELLED:
            self.log_output("[yellow]✗ Task cancelled.[/yellow]")
            self._current_worker = None
            self._active_agents.clear()

    def action_cancel_task(self) -> None:
        if self.is_busy and self._current_worker:
            self._current_worker.cancel()
            self._current_worker = None
            self.log_output("[yellow]Cancelling task…[/yellow]")
        else:
            self.query_one("#intent-input", Input).focus()

    # ── Public update methods ────────────────────────────────────────

    def update_plan(self, steps: list[dict]) -> None:
        self.post_message(PlanUpdated(steps))

    def update_pages(self, pages: list[dict]) -> None:
        self.post_message(PagesUpdated(pages))

    def update_agents(self, agents: list[dict]) -> None:
        self.post_message(AgentsUpdated(agents))

    def update_mistakes(self, mistakes: list[dict]) -> None:
        self.post_message(MistakesUpdated(mistakes))

    def log_output(self, message: str) -> None:
        log_widget = self.query_one("#output-log", RichLog)
        log_widget.write(message)

    def log_markdown(self, text: str) -> None:
        """Render markdown content in the output log using Rich's Markdown."""
        log_widget = self.query_one("#output-log", RichLog)
        log_widget.write(RichMarkdown(text))

    # ── Session helpers ──────────────────────────────────────────────

    def _show_sessions(self) -> None:
        if not self._engine:
            self.log_output("[red]No engine connected.[/red]")
            return
        entries = self._engine.session_manager.list_sessions(limit=10)
        if not entries:
            self.log_output("[dim]No saved sessions.[/dim]")
            return
        icons = {"active": "⟳", "completed": "✓", "failed": "✗", "cancelled": "–", "paused": "⏸"}
        self.log_output("\n[bold cyan]Saved Sessions:[/bold cyan]")
        for e in entries:
            icon = icons.get(e.get("status", ""), "?")
            sid = e.get("id", "?")[:12]
            intent = e.get("intent", "")[:45]
            status = e.get("status", "?")
            steps = e.get("step_count", 0)
            done = e.get("completed_steps", 0)
            self.log_output(
                f"  {icon} [bold]{sid}[/bold]  {status:<10}  "
                f"{done}/{steps} steps  [dim]{intent}[/dim]"
            )
        self.log_output("[dim]Type /resume <id> to resume, /show <id> for details[/dim]")

    def _show_session_detail(self, session_id: str) -> None:
        if not self._engine:
            return
        session = self._engine.session_manager.load(session_id)
        if not session:
            self.log_output(f"[red]Session not found: {session_id}[/red]")
            return
        self.log_output(f"\n[bold cyan]Session: {session.id}[/bold cyan]")
        self.log_output(f"  Intent:  {session.intent}")
        self.log_output(f"  Status:  {session.status.value}")
        self.log_output(f"  Model:   {session.model}")
        self.log_output(f"  Steps:   {session.completed_steps}/{session.step_count}")
        self.log_output(f"  Pages:   {len(session.pages)}")
        self.log_output(f"  Mistakes:{len(session.mistakes)}")
        if session.plan_steps:
            self.log_output("  [bold]Plan:[/bold]")
            for s in session.plan_steps:
                icon = {"done": "✓", "failed": "✗", "running": "⟳", "pending": "○"}.get(s.get("status", ""), "?")
                self.log_output(f"    {icon} {s.get('id', '?')} [{s.get('agent', '')}] {s.get('action', '')}")
        if session.result:
            result_str = str(session.result)
            if _looks_like_markdown(result_str):
                self.log_output("  [green]Result:[/green]")
                self.log_markdown(result_str)
            else:
                self.log_output(f"  [green]Result:[/green] {result_str[:200]}")

    def _resume_session(self, session_id: str) -> None:
        if not self._engine:
            self.log_output("[red]No engine connected.[/red]")
            return
        if self.is_busy:
            self.log_output("[yellow]Task in progress. Press Escape to cancel first.[/yellow]")
            return
        if session_id == "latest":
            latest = self._engine.session_manager.get_latest()
            if not latest:
                self.log_output("[dim]No sessions to resume.[/dim]")
                return
            session_id = latest.id
        session = self._engine.session_manager.load(session_id)
        if not session:
            self.log_output(f"[red]Session not found: {session_id}[/red]")
            return
        self._last_intent = session.intent
        self.log_output(f"\n[bold]▶ Resuming:[/bold] {session.intent}")
        self.log_output(f"[dim]Session {session.id} — {session.completed_steps}/{session.step_count} steps done[/dim]")
        self._current_worker = self.run_worker(
            self._engine.resume(session_id),
            name="intent_runner",
            exclusive=True,
        )

    # ── Keyboard actions ─────────────────────────────────────────────

    def action_show_plan(self) -> None:
        if self._engine and self._engine.current_session:
            steps = self._engine.current_session.plan_steps
            if steps:
                self.log_output("\n[bold cyan]Current Plan:[/bold cyan]")
                for s in steps:
                    icon = {"done": "✓", "failed": "✗", "running": "⟳", "pending": "○"}.get(s.get("status", ""), "?")
                    self.log_output(f"  {icon} {s.get('id', '?')} [{s.get('agent', '')}] {s.get('action', '')}")
                return
        self.log_output("[dim]No active plan.[/dim]")

    def action_show_mistakes(self) -> None:
        if self._engine and self._engine.mistakes.count() > 0:
            self.log_output("\n[bold red]Mistakes:[/bold red]")
            for m in self._engine.mistakes.all:
                self.log_output(f"  ⚠ {m.step_id}: {m.description}")
                self.log_output(f"    [dim]Learned: {m.learned}[/dim]")
        else:
            self.log_output("[dim]No mistakes recorded.[/dim]")

    def action_show_context(self) -> None:
        if self._engine:
            pages = self._engine.page_store.all_pages()
            if pages:
                self.log_output(f"\n[bold cyan]Context Pages ({len(pages)}):[/bold cyan]")
                for p in pages:
                    self.log_output(f"  [{p.type.value[:4].upper()}] {p.id} — {p.summary[:40]}")
                return
        self.log_output("[dim]No context pages loaded.[/dim]")

    def action_show_sessions(self) -> None:
        self._show_sessions()

    def action_rerun(self) -> None:
        if self._last_intent and self._engine and not self.is_busy:
            self.log_output(f"[yellow]Re-running:[/yellow] {self._last_intent}")
            input_widget = self.query_one("#intent-input", Input)
            input_widget.value = self._last_intent
            input_widget.action_submit()
        else:
            self.log_output("[yellow]Nothing to re-run.[/yellow]")
