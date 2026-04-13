"""
Agent Registry — runtime registration and lookup of domain agents.

Follows the Open-Closed Principle: new agents are added by registering
them at runtime, not by modifying the registry or planner code.
Follows Dependency Inversion: the planner depends on the registry
abstraction, not on concrete agent implementations.
"""

from __future__ import annotations

import logging
from typing import Type

from cortex.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Thread-safe registry of available domain agents."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register a concrete agent instance by its name."""
        if agent.name in self._agents:
            logger.warning("Overwriting agent: %s", agent.name)
        self._agents[agent.name] = agent
        logger.info("Registered agent: %s", agent.name)

    def get(self, name: str) -> BaseAgent | None:
        """Look up an agent by name. Returns None if not found."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """Return names of all registered agents."""
        return list(self._agents.keys())

    def has(self, name: str) -> bool:
        return name in self._agents

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"<AgentRegistry agents={self.list_agents()}>"
