"""
Cortex CLI — the direct `cortex` command.

Install with:
  uv tool install .          # global command
  uv pip install -e .        # editable dev install

Usage:
  cortex run "Design a login component"
  cortex run --tui "Analyze auth module"
  cortex run --tmux "Build and test user profiles"
  cortex interactive
  cortex index report.pdf
  cortex agents
  cortex status
  cortex version
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import textwrap

from pathlib import Path

from cortex import __version__
from cortex.config import CortexConfig
from cortex.engine import CortexEngine

# ── Logging ──────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s"
LOG_FORMAT_SHORT = "%(levelname)-7s │ %(message)s"


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = LOG_FORMAT if verbose else LOG_FORMAT_SHORT
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


# ── Engine factory ───────────────────────────────────────────────────

def _build_engine(args: argparse.Namespace) -> CortexEngine:
    model = getattr(args, "model", "sonnet")
    use_tmux = getattr(args, "tmux", False)

    config = CortexConfig(
        default_model=model,
        use_tmux=use_tmux,
    )
    return CortexEngine(config=config)


# ── Subcommands ──────────────────────────────────────────────────────

async def _cmd_run(args: argparse.Namespace) -> None:
    """Execute a single intent."""
    engine = _build_engine(args)

    if args.index:
        doc_id = await engine.index_document(args.index)
        logging.getLogger("cortex").info("Indexed document: %s", doc_id)

    if args.tui:
        _cmd_tui(engine, args.intent)
        return

    result = await engine.run(args.intent)
    print(json.dumps(result, indent=2, default=str))


async def _cmd_interactive(args: argparse.Namespace) -> None:
    """Launch the interactive REPL."""
    engine = _build_engine(args)

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Cortex Interactive Mode                            ║")
    print("║  Type your intent and press Enter.                  ║")
    print("║  Commands: /status  /agents  /pages  /quit          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            intent = input("cortex> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not intent:
            continue
        if intent in ("/quit", "quit", "exit", "/exit", "q"):
            print("Goodbye.")
            break
        elif intent == "/status":
            print(json.dumps(engine.status(), indent=2))
            continue
        elif intent == "/agents":
            print("Registered agents:", engine.registry.list_agents())
            continue
        elif intent == "/pages":
            for p in engine.page_store.all_pages():
                print(f"  [{p.type.value}] {p.id} — {p.summary}")
            continue

        result = await engine.run(intent)
        if result.get("result"):
            print()
            if isinstance(result["result"], str):
                print(result["result"])
            else:
                print(json.dumps(result["result"], indent=2, default=str))
            print()
        if result.get("mistakes"):
            print(f"  ⚠ {len(result['mistakes'])} mistake(s) recorded.")
        if result.get("lessons_learned"):
            print("  Lessons:", "; ".join(result["lessons_learned"]))
        print()


def _cmd_tui(engine_or_args, intent: str | None = None) -> None:
    """Launch the Textual TUI dashboard."""
    from cortex.tui.app import CortexTUI

    if isinstance(engine_or_args, argparse.Namespace):
        engine = _build_engine(engine_or_args)
        intent = getattr(engine_or_args, "intent", None)
    else:
        engine = engine_or_args

    tui_app = CortexTUI(engine=engine)

    if intent:
        # Pre-fill the input so it runs through the normal worker path
        def _queue_intent():
            input_widget = tui_app.query_one("#intent-input")
            input_widget.value = intent
            input_widget.action_submit()

        tui_app.call_after_refresh(_queue_intent)

    tui_app.run()


async def _cmd_index(args: argparse.Namespace) -> None:
    """Index a document into the RAG store."""
    engine = _build_engine(args)
    for filepath in args.files:
        doc_id = await engine.index_document(filepath)
        print(f"Indexed: {filepath} → {doc_id}")


def _cmd_agents(args: argparse.Namespace) -> None:
    """List all registered agents."""
    engine = _build_engine(args)
    agents = engine.registry.list_agents()
    print(f"Registered agents ({len(agents)}):")
    for name in agents:
        agent = engine.registry.get(name)
        desc = agent.description if agent else ""
        print(f"  • {name:25s} {desc}")


def _cmd_status(args: argparse.Namespace) -> None:
    """Show engine status."""
    engine = _build_engine(args)
    print(json.dumps(engine.status(), indent=2))


def _cmd_version(_args: argparse.Namespace) -> None:
    """Print version."""
    print(f"cortex {__version__}")


def _cmd_setup(args: argparse.Namespace) -> None:
    """One-command install: scaffold .cortex/ and verify the environment."""
    from cortex.scaffold.init import CortexInit

    target = getattr(args, "dir", ".")
    minimal = getattr(args, "minimal", False)

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Cortex Setup                                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 1. Scaffold
    init = CortexInit(target_dir=target, minimal=minimal, force=False)
    if init.is_initialized(target):
        print("  ✓  .cortex/ already exists — skipping scaffold")
    else:
        summary = init.run()
        print(f"  ✓  Scaffolded .cortex/ ({len(summary['created'])} files)")

    # 2. Check Claude Code CLI
    from cortex.claude_code import ClaudeCode

    if ClaudeCode.is_installed():
        ver = ClaudeCode.version()
        print(f"  ✓  Claude Code CLI found ({ver})")
    else:
        print("  ⚠  Claude Code CLI (`claude`) not found on PATH")
        print("     Install: npm install -g @anthropic-ai/claude-code")
        print("     Docs:    https://docs.anthropic.com/en/docs/claude-code")

    # 3. Print summary
    print()
    print("  Ready. Try:")
    print(f'    cortex run "Hello from {Path(target).resolve().name}"')
    print("    cortex interactive")
    print()


def _cmd_init(args: argparse.Namespace) -> None:
    """Scaffold the .cortex/ directory structure."""
    from cortex.scaffold.init import CortexInit

    target = getattr(args, "dir", ".")
    minimal = getattr(args, "minimal", False)
    force = getattr(args, "force", False)
    dry_run = getattr(args, "dry_run", False)

    init = CortexInit(
        target_dir=target,
        minimal=minimal,
        force=force,
        dry_run=dry_run,
    )

    if init.is_initialized(target) and not force:
        print(f"⚠  .cortex/ already exists in {init.target}")
        print("   Use --force to overwrite existing files.")
        print()
        print(init.tree(target))
        return

    summary = init.run()

    # Print results
    print()
    print(f"✓  Cortex initialized in {summary['target']}")
    print()

    if summary["created"]:
        print(f"   Created ({len(summary['created'])}):")
        for f in summary["created"]:
            print(f"     + {f}")

    if summary["skipped"]:
        print(f"   Skipped ({len(summary['skipped'])}):")
        for f in summary["skipped"]:
            print(f"     · {f}")

    if summary["overwritten"]:
        print(f"   Overwritten ({len(summary['overwritten'])}):")
        for f in summary["overwritten"]:
            print(f"     ! {f}")

    print()
    print("   Directory structure:")
    print()
    for line in init.tree(target).split("\n"):
        print(f"     {line}")
    print()
    print("   Next steps:")
    print("     1. Edit CORTEX.md with your project conventions")
    print("     2. Review .cortex/settings.json permissions")
    print("     3. Customize rules in .cortex/rules/")
    print("     4. Run: cortex run \"Your first intent\"")
    print()


def _cmd_sessions(args: argparse.Namespace) -> None:
    """List all saved sessions."""
    from cortex.sessions.manager import SessionManager
    from cortex.sessions.model import SessionStatus

    mgr = SessionManager()
    status_filter = None
    if hasattr(args, "filter") and args.filter:
        try:
            status_filter = SessionStatus(args.filter)
        except ValueError:
            print(f"Unknown status: {args.filter}")
            return

    limit = getattr(args, "limit", 20)
    entries = mgr.list_sessions(status=status_filter, limit=limit)

    if not entries:
        print("No sessions found.")
        return

    status_icons = {
        "active": "⟳", "completed": "✓", "failed": "✗",
        "cancelled": "–", "paused": "⏸",
    }

    print(f"Sessions ({len(entries)}):\n")
    for e in entries:
        icon = status_icons.get(e.get("status", ""), "?")
        sid = e.get("id", "?")[:12]
        intent = e.get("intent", "")[:50]
        status = e.get("status", "?")
        steps = e.get("step_count", 0)
        done = e.get("completed_steps", 0)
        created = e.get("created_at", "")[:19].replace("T", " ")
        print(f"  {icon} {sid}  {status:<10}  {done}/{steps} steps  {created}")
        print(f"    {intent}")
        print()


async def _cmd_resume(args: argparse.Namespace) -> None:
    """Resume a previous session."""
    engine = _build_engine(args)
    session_id = args.session_id

    # Support "latest" shorthand
    if session_id == "latest":
        latest = engine.session_manager.get_latest()
        if not latest:
            print("No sessions found.")
            return
        session_id = latest.id

    print(f"Resuming session: {session_id}")
    try:
        result = await engine.resume(session_id)
        print(json.dumps(result, indent=2, default=str))
    except ValueError as e:
        print(f"Error: {e}")


def _cmd_session_show(args: argparse.Namespace) -> None:
    """Show details of a specific session."""
    from cortex.sessions.manager import SessionManager

    mgr = SessionManager()
    session = mgr.load(args.session_id)

    if not session:
        print(f"Session not found: {args.session_id}")
        return

    print(f"Session: {session.id}")
    print(f"Intent:  {session.intent}")
    print(f"Status:  {session.status.value}")
    print(f"Model:   {session.model}")
    print(f"Created: {session.created_at[:19].replace('T', ' ')}")
    print(f"Updated: {session.updated_at[:19].replace('T', ' ')}")
    print(f"Steps:   {session.completed_steps}/{session.step_count}")
    print(f"Pages:   {len(session.pages)}")
    print(f"Mistakes:{len(session.mistakes)}")
    print()

    if session.plan_steps:
        print("Plan:")
        for s in session.plan_steps:
            icon = {"done": "✓", "failed": "✗", "running": "⟳", "pending": "○"}.get(
                s.get("status", ""), "?"
            )
            print(f"  {icon} {s.get('id', '?')} [{s.get('agent', '')}] {s.get('action', '')[:50]}")
        print()

    if session.events:
        print("Timeline:")
        for ev in session.events[-10:]:
            ts = ev.timestamp[:19].replace("T", " ") if ev.timestamp else ""
            print(f"  {ts}  {ev.type}  {json.dumps(ev.data)[:60] if ev.data else ''}")
        print()

    if session.result:
        print(f"Result: {str(session.result)[:200]}")


def _cmd_session_delete(args: argparse.Namespace) -> None:
    """Delete a session."""
    from cortex.sessions.manager import SessionManager

    mgr = SessionManager()
    if mgr.delete(args.session_id):
        print(f"Deleted session: {args.session_id}")
    else:
        print(f"Session not found: {args.session_id}")


def _cmd_tree(args: argparse.Namespace) -> None:
    """Show the .cortex/ directory tree."""
    from cortex.scaffold.init import CortexInit

    target = getattr(args, "dir", ".")
    if not CortexInit.is_initialized(target):
        print("No .cortex/ directory found. Run `cortex init` first.")
        return

    print(CortexInit.tree(target))


# ── Argument parser ──────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="cortex",
        description="Cortex — Intelligent Agent Orchestration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            One-line install:
              uvx --from git+https://github.com/anthropics/cortex cortex setup

            Quick start:
              cortex setup                       # install + scaffold in one shot
              cortex run "Design a login component"
              cortex run --tui "Analyze the auth module"
              cortex interactive

            Install methods:
              uv tool install cortex-ai          # from PyPI (global CLI)
              uv tool install .                  # from source (global CLI)
              uv pip install -e ".[dev]"         # editable dev install
              pipx install cortex-ai             # pipx alternative
        """),
    )

    # Global flags
    root.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    root.add_argument("--model", default="sonnet", help="Claude Code model (haiku, sonnet, opus)")

    subs = root.add_subparsers(dest="command", title="commands")

    # ── cortex run ───────────────────────────────────────────────────
    p_run = subs.add_parser("run", help="Execute a single intent")
    p_run.add_argument("intent", help="The intent / task to execute")
    p_run.add_argument("--tui", action="store_true", help="Open the Textual TUI dashboard")
    p_run.add_argument("--tmux", action="store_true", help="Spawn agents in tmux panes")
    p_run.add_argument("--index", metavar="FILE", help="Index a document before running")
    p_run.set_defaults(func=_cmd_run, is_async=True)

    # ── cortex interactive ───────────────────────────────────────────
    p_int = subs.add_parser("interactive", aliases=["i", "repl"], help="Interactive REPL")
    p_int.set_defaults(func=_cmd_interactive, is_async=True)

    # ── cortex tui ───────────────────────────────────────────────────
    p_tui = subs.add_parser("tui", help="Launch TUI dashboard")
    p_tui.add_argument("intent", nargs="?", help="Optional intent to auto-run")
    p_tui.set_defaults(func=_cmd_tui, is_async=False)

    # ── cortex index ─────────────────────────────────────────────────
    p_idx = subs.add_parser("index", help="Index documents into the RAG store")
    p_idx.add_argument("files", nargs="+", help="Files to index (PDF, text, etc.)")
    p_idx.set_defaults(func=_cmd_index, is_async=True)

    # ── cortex setup ─────────────────────────────────────────────────
    p_setup = subs.add_parser(
        "setup",
        help="One-command install: scaffold .cortex/ and verify env",
        description="Scaffolds .cortex/ and checks your environment. The only command you need to get started.",
    )
    p_setup.add_argument("dir", nargs="?", default=".", help="Target directory (default: current)")
    p_setup.add_argument("--minimal", action="store_true", help="Only create CORTEX.md + settings.json")
    p_setup.set_defaults(func=_cmd_setup, is_async=False)

    # ── cortex init ──────────────────────────────────────────────────
    p_init = subs.add_parser(
        "init",
        help="Scaffold the .cortex/ directory in a project",
        description="Creates the .cortex/ directory structure with rules, skills, agents, and settings.",
    )
    p_init.add_argument("dir", nargs="?", default=".", help="Target directory (default: current)")
    p_init.add_argument("--minimal", action="store_true", help="Only create CORTEX.md + settings.json")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files")
    p_init.add_argument("--dry-run", action="store_true", help="Show what would be created without writing")
    p_init.set_defaults(func=_cmd_init, is_async=False)

    # ── cortex tree ──────────────────────────────────────────────────
    p_tree = subs.add_parser("tree", help="Show the .cortex/ directory tree")
    p_tree.add_argument("dir", nargs="?", default=".", help="Target directory (default: current)")
    p_tree.set_defaults(func=_cmd_tree, is_async=False)

    # ── cortex sessions ─────────────────────────────────────────────
    p_sess = subs.add_parser("sessions", aliases=["ss"], help="List saved sessions")
    p_sess.add_argument("--filter", choices=["active", "completed", "failed", "cancelled", "paused"],
                        help="Filter by status")
    p_sess.add_argument("--limit", type=int, default=20, help="Max sessions to show")
    p_sess.set_defaults(func=_cmd_sessions, is_async=False)

    # ── cortex resume ────────────────────────────────────────────────
    p_resume = subs.add_parser("resume", help="Resume a previous session")
    p_resume.add_argument("session_id", help="Session ID to resume, or 'latest'")
    p_resume.set_defaults(func=_cmd_resume, is_async=True)

    # ── cortex session show ──────────────────────────────────────────
    p_show = subs.add_parser("session-show", aliases=["show"], help="Show session details")
    p_show.add_argument("session_id", help="Session ID to inspect")
    p_show.set_defaults(func=_cmd_session_show, is_async=False)

    # ── cortex session delete ────────────────────────────────────────
    p_sdel = subs.add_parser("session-delete", help="Delete a saved session")
    p_sdel.add_argument("session_id", help="Session ID to delete")
    p_sdel.set_defaults(func=_cmd_session_delete, is_async=False)

    # ── cortex agents ────────────────────────────────────────────────
    p_agents = subs.add_parser("agents", help="List registered agents")
    p_agents.set_defaults(func=_cmd_agents, is_async=False)

    # ── cortex status ────────────────────────────────────────────────
    p_status = subs.add_parser("status", help="Show engine status")
    p_status.set_defaults(func=_cmd_status, is_async=False)

    # ── cortex version ───────────────────────────────────────────────
    p_ver = subs.add_parser("version", help="Print version")
    p_ver.set_defaults(func=_cmd_version, is_async=False)

    return root


# ── Entry point ──────────────────────────────────────────────────────

def app() -> None:
    """
    Main entry point registered as the `cortex` console script.
    Called by: `cortex run ...`, `cortex interactive`, etc.
    """
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    if getattr(args, "is_async", False):
        asyncio.run(args.func(args))
    else:
        args.func(args)


# Allow `python -m cortex.cli` as well
if __name__ == "__main__":
    app()
