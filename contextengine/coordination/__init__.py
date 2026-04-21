"""Multi-agent coordination: handoffs + role-specific engine views on shared memory."""

from contextengine.coordination.handoff import Handoff, HandoffProtocol
from contextengine.coordination.coordinator import MultiAgentCoordinator, AgentView

__all__ = [
    "Handoff",
    "HandoffProtocol",
    "MultiAgentCoordinator",
    "AgentView",
]
