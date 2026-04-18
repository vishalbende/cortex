"""End-to-end demo: connect MCPs → assemble → print context composition.

Prereqs:
    uv sync --extra dev   (or equivalent)
    export ANTHROPIC_API_KEY=sk-...
    npx is on PATH (for the stdio MCP)

Run:
    python examples/demo.py "list the first 5 files in my home directory"
"""
from __future__ import annotations

import asyncio
import os
import sys

from contextengine import ContextEngine, MCPServer


async def main(message: str) -> None:
    home = os.path.expanduser("~")
    engine = ContextEngine(
        mcps=[
            MCPServer(
                name="filesystem",
                command=["npx", "-y", "@modelcontextprotocol/server-filesystem", home],
            ),
        ],
        model="claude-sonnet-4-5",
        router_model="claude-haiku-4-5",
        budget=80_000,
        system_prompt="You are a concise assistant.",
    )

    await engine.start()
    try:
        result = await engine.assemble(message=message)
    finally:
        await engine.close()

    print(f"message: {message!r}")
    print(f"system  : {result.system[:60]!r}...")
    print(f"loaded  : {result.stats.tools_loaded}")
    print(f"dropped : {result.stats.tools_dropped}")
    print(f"mcps    : {result.stats.mcps_represented}")
    print(
        f"tokens  : system={result.stats.tokens_system} "
        f"memory={result.stats.tokens_memory} "
        f"tools={result.stats.tokens_tools} "
        f"history={result.stats.tokens_history} "
        f"total={result.stats.tokens_total}"
    )
    print(f"elapsed : {result.stats.elapsed_ms:.0f}ms")


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "list files in my home directory"
    asyncio.run(main(msg))
