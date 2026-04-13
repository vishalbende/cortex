"""
Cortex TUI — Textual-based terminal dashboard.

Designed with tui.studio aesthetics: clean panels, live-updating status,
and keyboard shortcuts for power users.

Layout:
  ┌─────────────────────────────────────┐
  │  PLAN           │  PAGES LOADED     │
  │  step 1 ✓       │  page:001 (rag)   │
  │  step 2 ⟳       │  page:002 (plan)  │
  │  step 3 ○       │  page:003 (mem)   │
  ├─────────────────────────────────────┤
  │  ACTIVE AGENTS  │  MISTAKE LOG      │
  │  ▶ ai_agent     │  ⚠ step_2 warn   │
  │  ▶ design_agent │                   │
  ├─────────────────────────────────────┤
  │  OUTPUT STREAM                      │
  │  > Generating component spec...     │
  └─────────────────────────────────────┘

Keyboard shortcuts: p=plan, m=mistakes, c=context, r=re-run, q=quit
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Log,
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

QUALITY_STYLE = {
    Quality.OK: "green",
    Quality.WARNING: "yellow",
    Quality.ERROR: "red",
}


# ── Custom Widgets ───────────────────────────────────────────────────

class PlanPanel(Static):
    """Displays the current execution plan with live step statuses."""

    DEFAULT_CSS = """
    PlanPanel {
        border: solid #6366F1;
        padding: 1;
        height: 100%;
    }
    """

    steps: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]PLAN[/bold cyan]")
        for s in self.steps:
            icon = STATUS_ICONS.get(StepStatus(s.get("status", "pending")), "?")
            color = QUALITY_STYLE.get(Quality(s.get("quality", "ok")), "white")
            yield Label(
                f"  {icon} [{color}]{s.get('id', '?')}[/{color}] "
                f"[dim]{s.get('agent', '')}[/dim] — {s.get('action', '')[:40]}"
            )

    def update_steps(self, steps: list[dict]) -> None:
        self.steps = steps


class PagesPanel(Static):
    """Shows loaded context pages with type badges."""

    DEFAULT_CSS = """
    PagesPanel {
        border: solid #6366F1;
        padding: 1;
        height: 100%;
    }
    """

    pages: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]PAGES LOADED[/bold cyan]")
        for p in self.pages[:15]:
            badge = p.get("type", "?")[:4]
            yield Label(f"  [{badge}] {p.get('id', '?')} — {p.get('summary', '')[:35]}")

    def update_pages(self, pages: list[dict]) -> None:
        self.pages = pages


class AgentsPanel(Static):
    """Shows active agents and their tmux pane status."""

    DEFAULT_CSS = """
    AgentsPanel {
        border: solid #6366F1;
        padding: 1;
        height: 100%;
    }
    """

    agents: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]ACTIVE AGENTS[/bold cyan]")
        for a in self.agents:
            status = a.get("status", "idle")
            icon = "▶" if status == "running" else "■"
            color = "green" if status == "running" else "dim"
            pane = a.get("pane_id", "")
            yield Label(
                f"  {icon} [{color}]{a.get('name', '?')}[/{color}] "
                f"[dim]{pane}[/dim]"
            )

    def update_agents(self, agents: list[dict]) -> None:
        self.agents = agents


class MistakePanel(Static):
    """Filterable mistake log."""

    DEFAULT_CSS = """
    MistakePanel {
        border: solid #EF4444;
        padding: 1;
        height: 100%;
    }
    """

    mistakes: reactive[list[dict]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold red]MISTAKE LOG[/bold red]")
        if not self.mistakes:
            yield Label("  [dim]No mistakes recorded.[/dim]")
        for m in self.mistakes:
            icon = "✗" if m.get("type") in ("tool_failure", "plan_error") else "⚠"
            yield Label(
                f"  {icon} [yellow]{m.get('step_id', '?')}[/yellow] "
                f"[dim]{m.get('type', '')}[/dim]: {m.get('description', '')[:40]}"
            )

    def update_mistakes(self, mistakes: list[dict]) -> None:
        self.mistakes = mistakes


# ── Main TUI App ─────────────────────────────────────────────────────

class CortexTUI(App):
    """
    Cortex terminal dashboard — observe plan execution in real time.
    """

    TITLE = "Cortex — Agent Orchestration"
    SUB_TITLE = "Press q to quit, p=plan, m=mistakes, c=context, r=re-run"

    CSS = """
    Screen {
        background: #0F172A;
        color: #F8FAFC;
    }

    #top-row {
        height: 40%;
    }

    #mid-row {
        height: 30%;
    }

    #output-log {
        border: solid #6366F1;
        padding: 1;
        height: 30%;
    }

    PlanPanel { width: 1fr; }
    PagesPanel { width: 1fr; }
    AgentsPanel { width: 1fr; }
    MistakePanel { width: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "show_plan", "Plan"),
        Binding("m", "show_mistakes", "Mistakes"),
        Binding("c", "show_context", "Context"),
        Binding("r", "rerun", "Re-run"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            yield PlanPanel(id="plan-panel")
            yield PagesPanel(id="pages-panel")
        with Horizontal(id="mid-row"):
            yield AgentsPanel(id="agents-panel")
            yield MistakePanel(id="mistake-panel")
        yield RichLog(id="output-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.log_output("[bold green]Cortex TUI started.[/bold green] Waiting for engine events...")

    # ── Public methods for the engine to push updates ────────────────

    def update_plan(self, steps: list[dict]) -> None:
        panel = self.query_one("#plan-panel", PlanPanel)
        panel.update_steps(steps)

    def update_pages(self, pages: list[dict]) -> None:
        panel = self.query_one("#pages-panel", PagesPanel)
        panel.update_pages(pages)

    def update_agents(self, agents: list[dict]) -> None:
        panel = self.query_one("#agents-panel", AgentsPanel)
        panel.update_agents(agents)

    def update_mistakes(self, mistakes: list[dict]) -> None:
        panel = self.query_one("#mistake-panel", MistakePanel)
        panel.update_mistakes(mistakes)

    def log_output(self, message: str) -> None:
        log_widget = self.query_one("#output-log", RichLog)
        log_widget.write(message)

    # ── Actions ──────────────────────────────────────────────────────

    def action_show_plan(self) -> None:
        self.log_output("[cyan]Showing full plan...[/cyan]")

    def action_show_mistakes(self) -> None:
        self.log_output("[red]Showing mistake log...[/red]")

    def action_show_context(self) -> None:
        self.log_output("[cyan]Showing context pages...[/cyan]")

    def action_rerun(self) -> None:
        self.log_output("[yellow]Re-running failed step...[/yellow]")
