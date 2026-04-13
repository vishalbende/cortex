"""
Abstract base agent — every domain agent inherits from this.

Follows the Interface Segregation and Liskov Substitution principles:
any agent can be swapped for another of the same type without breaking
the planner, and interfaces are small and focused.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from cortex.models import AgentInput, AgentOutput, Quality, MistakeRecord, MistakeType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Contract every Cortex agent must satisfy.

    Agents are stateless — all persistent state lives in the PageIndex
    managed by the context layer. The planner injects context via AgentInput
    and reads results via AgentOutput.
    """

    name: str = "base_agent"
    description: str = "Abstract base agent"
    required_permissions: list[str] = []

    # ── Core contract ────────────────────────────────────────────────

    @abstractmethod
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Run the agent's primary action. Must be implemented by subclasses."""
        ...

    # ── Lifecycle hooks (optional overrides) ─────────────────────────

    async def validate_input(self, agent_input: AgentInput) -> bool:
        """Pre-execution validation. Return False to abort the step."""
        if not agent_input.intent:
            logger.warning("[%s] Received empty intent", self.name)
            return False
        return True

    async def on_error(self, error: Exception, agent_input: AgentInput) -> AgentOutput:
        """
        Default error handler. Subclasses can override for custom recovery.
        Returns an AgentOutput with quality=ERROR and a MistakeRecord.
        """
        mistake = MistakeRecord(
            step_id=agent_input.step.id if agent_input.step else "unknown",
            type=MistakeType.TOOL_FAILURE,
            description=f"[{self.name}] {error}",
            correction="Returned error output for re-planning.",
            learned=f"Agent '{self.name}' failed on intent: {agent_input.intent[:80]}",
        )
        logger.error("[%s] %s", self.name, error)
        return AgentOutput(
            result=None,
            confidence=0.0,
            quality=Quality.ERROR,
            mistakes=[mistake],
        )

    # ── Safe runner (planner calls this, not execute directly) ───────

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Wrapper the planner calls. Handles validation, execution, and
        error recovery so individual agents don't have to.
        """
        if not await self.validate_input(agent_input):
            return AgentOutput(
                result=None,
                confidence=0.0,
                quality=Quality.ERROR,
                mistakes=[
                    MistakeRecord(
                        step_id=agent_input.step.id if agent_input.step else "unknown",
                        type=MistakeType.PLAN_ERROR,
                        description=f"[{self.name}] Input validation failed.",
                        correction="Step skipped due to bad input.",
                        learned="Ensure intent and required inputs are present.",
                    )
                ],
            )
        try:
            output = await self.execute(agent_input)
            return output
        except Exception as exc:
            return await self.on_error(exc, agent_input)

    def __repr__(self) -> str:
        return f"<Agent:{self.name}>"
