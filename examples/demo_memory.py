"""Full-loop demo: assemble → model call → process_turn writeback.

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/demo_memory.py
"""
from __future__ import annotations

import asyncio
import os

from anthropic import AsyncAnthropic

from contextengine import ContextEngine, JSONStore, MCPServer
from contextengine.telemetry import StdoutSink


async def main() -> None:
    client = AsyncAnthropic()
    home = os.path.expanduser("~")
    engine = ContextEngine(
        mcps=[
            MCPServer(
                name="filesystem",
                command=["npx", "-y", "@modelcontextprotocol/server-filesystem", home],
            )
        ],
        model="claude-sonnet-4-5",
        router_model="claude-haiku-4-5",
        budget=80_000,
        system_prompt="You are a concise assistant.",
        anthropic_client=client,
        memory_store=JSONStore(".contextengine/memory"),
        telemetry_sinks=[StdoutSink()],
    )

    await engine.start()
    try:
        entity_id = "demo-user"

        msg = "I prefer concise answers. What files are in my home directory?"
        ctx = await engine.assemble(message=msg, entity_id=entity_id)

        response = await client.messages.create(
            model=engine.model,
            max_tokens=1024,
            system=ctx.system,
            messages=ctx.messages,
            tools=ctx.tools,
        )
        assistant_text = "".join(b.text for b in response.content if b.type == "text")
        print(f"\n[assistant] {assistant_text[:200]}")

        write = await engine.process_turn(
            entity_id=entity_id,
            user_message=msg,
            assistant_response=assistant_text,
        )
        print(f"[memory] wrote {write.facts_upserted} facts, {write.events_appended} events")

        print("\n--- Next turn (memory should surface) ---")
        ctx2 = await engine.assemble(
            message="Remind me what I just asked about.", entity_id=entity_id
        )
        if "[memory]" in ctx2.system:
            print("[memory block present in system prompt]")
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
