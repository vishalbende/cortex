"""
Standalone agent runner — executed inside a tmux pane.

Usage (called by TmuxRunner):
  python -m cortex.run_agent <agent_name> --task "Do something"

Each agent runs in its own process inside its own tmux pane, providing
true parallelism and visual isolation. Uses the local Claude Code CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("cortex.runner")


def build_agent(agent_name: str):
    """Instantiate the requested agent."""
    if agent_name == "ai_agent":
        from cortex.agents.ai_agent import AIAgent
        from cortex.claude_code import ClaudeCode
        if not ClaudeCode.is_installed():
            logger.error("ai_agent requires Claude Code CLI (`claude`) on PATH")
            sys.exit(1)
        return AIAgent()

    elif agent_name == "design_agent":
        from cortex.agents.design_agent import DesignAgent
        return DesignAgent()

    elif agent_name == "test_writer_agent":
        from cortex.agents.test_writer_agent import TestWriterAgent
        return TestWriterAgent()

    else:
        logger.error("Unknown agent: %s", agent_name)
        sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description="Cortex standalone agent runner")
    parser.add_argument("agent", help="Agent name (ai_agent, design_agent, test_writer_agent)")
    parser.add_argument("--task", required=True, help="The task/intent for the agent")
    parser.add_argument("--constraints", default="{}", help="JSON constraints dict")
    parser.add_argument("--output", default=None, help="Output file path (JSON)")
    args = parser.parse_args()

    from cortex.models import AgentInput, PlanStep

    agent = build_agent(args.agent)
    constraints = json.loads(args.constraints)

    step = PlanStep(id="standalone", agent=args.agent, action=args.task)
    agent_input = AgentInput(
        intent=args.task,
        constraints=constraints,
        step=step,
    )

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Agent: %-33s║", args.agent)
    logger.info("║  Task:  %-33s║", args.task[:33])
    logger.info("╚══════════════════════════════════════════╝")

    output = await agent.run(agent_input)

    result_dict = {
        "agent": args.agent,
        "quality": output.quality.value,
        "confidence": output.confidence,
        "result": output.result if isinstance(output.result, (str, dict, list)) else str(output.result),
        "mistakes": [m.to_dict() for m in output.mistakes],
    }

    print(json.dumps(result_dict, indent=2, default=str))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        logger.info("Output saved to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
