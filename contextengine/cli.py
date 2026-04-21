"""Command-line entry point: `contextengine` (see pyproject [project.scripts])."""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from typing import Any

from contextengine import ContextEngine, MCPServer
from contextengine.telemetry.sinks import StdoutSink


def _parse_mcp_spec(spec: str) -> MCPServer:
    """Parse `name=...` spec. Examples:

      filesystem=npx -y @modelcontextprotocol/server-filesystem /tmp
      stripe=https://mcp.stripe.com/sse
    """
    if "=" not in spec:
        raise SystemExit(f"--mcp must be 'name=command-or-url', got: {spec}")
    name, rest = spec.split("=", 1)
    name = name.strip()
    rest = rest.strip()
    if rest.startswith("http://") or rest.startswith("https://"):
        return MCPServer(name=name, url=rest)
    return MCPServer(name=name, command=shlex.split(rest))


def _build_engine(args: argparse.Namespace) -> ContextEngine:
    if not args.mcp:
        raise SystemExit("at least one --mcp is required")
    return ContextEngine(
        mcps=[_parse_mcp_spec(s) for s in args.mcp],
        model=args.model,
        router_model=args.router_model,
        budget=args.budget,
        system_prompt=args.system or "",
        telemetry_sinks=[StdoutSink()] if args.verbose else [],
    )


async def _cmd_run(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    await engine.start()
    try:
        result = await engine.assemble(message=args.message)
        out: dict[str, Any] = {
            "system": result.system,
            "tools": result.tools,
            "messages": result.messages,
            "stats": {
                "tokens_system": result.stats.tokens_system,
                "tokens_memory": result.stats.tokens_memory,
                "tokens_tools": result.stats.tokens_tools,
                "tokens_history": result.stats.tokens_history,
                "tokens_total": result.stats.tokens_total,
                "tools_loaded": list(result.stats.tools_loaded),
                "tools_dropped": list(result.stats.tools_dropped),
                "mcps_represented": list(result.stats.mcps_represented),
                "elapsed_ms": result.stats.elapsed_ms,
            },
        }
        print(json.dumps(out, indent=2))
        return 0
    finally:
        await engine.close()


async def _cmd_catalog(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    await engine.start()
    try:
        catalog = engine.catalog
        if catalog is None:
            print("(no catalog)")
            return 1
        print(f"catalog version: {catalog.version_hash}")
        for mcp in catalog.mcps:
            print(f"\n- {mcp.name}: {mcp.summary}")
            for cat in mcp.categories:
                print(f"  └ {cat.name}: {cat.summary}")
                for tool in cat.tools:
                    print(f"    · {tool.name}")
        return 0
    finally:
        await engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contextengine")
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--router-model", default="claude-haiku-4-5")
    parser.add_argument("--budget", type=int, default=80_000)
    parser.add_argument("--system", default="", help="system prompt text")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--mcp",
        action="append",
        default=[],
        help="MCP spec: name=command-or-url (repeatable)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="assemble context for one message")
    run.add_argument("message", help="user message")

    sub.add_parser("catalog", help="print the hierarchical MCP catalog")

    dash = sub.add_parser("dashboard", help="summarize a JSONL telemetry file")
    dash.add_argument("traces", help="path to JSONL traces")
    dash.add_argument("--format", choices=["text", "html"], default="text")
    dash.add_argument("--output", "-o", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return asyncio.run(_cmd_run(args))
    if args.cmd == "catalog":
        return asyncio.run(_cmd_catalog(args))
    if args.cmd == "dashboard":
        from contextengine.dashboard import main as dashboard_main

        passthrough = [args.traces, "--format", args.format]
        if args.output:
            passthrough += ["-o", args.output]
        return dashboard_main(passthrough)
    return 1


if __name__ == "__main__":
    sys.exit(main())
