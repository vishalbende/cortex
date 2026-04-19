"""Demo: use contextengine to assemble context and call OpenAI's chat.completions.

Run:
    pip install openai
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-...   # router + categorizer still use Claude Haiku
    python examples/demo_openai.py
"""
from __future__ import annotations

import asyncio
import os

from contextengine import ContextEngine, MCPServer


async def main() -> None:
    engine = ContextEngine(
        mcps=[
            MCPServer(
                name="filesystem",
                command=[
                    "npx", "-y", "@modelcontextprotocol/server-filesystem",
                    os.path.expanduser("~"),
                ],
            )
        ],
        model="gpt-4o",
        router_model="claude-haiku-4-5",
        budget=80_000,
        system_prompt="You are a concise assistant.",
    )
    await engine.start()
    try:
        ctx = await engine.assemble(message="list files in my home directory")
        openai_kwargs = ctx.to_openai()

        from openai import AsyncOpenAI

        oai = AsyncOpenAI()
        response = await oai.chat.completions.create(
            model=engine.model,
            **openai_kwargs,
        )
        msg = response.choices[0].message
        print(f"\n[openai] {msg.content or ''}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  tool_call: {tc.function.name}({tc.function.arguments})")
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
