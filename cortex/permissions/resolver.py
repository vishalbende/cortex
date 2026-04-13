"""
Permission Resolver — semantic permission checking for agent actions.

Rules from the Cortex spec:
  - Resolve permissions semantically via embedding similarity
  - Threshold: 0.75 cosine similarity against declared permission scopes
  - Destructive actions always require explicit user confirmation
  - Never infer permission from context — always resolve explicitly
  - Permission denial → MistakeRecord type: permission_violation
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from cortex.models import MistakeRecord, MistakeType

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75

# Actions that always require explicit user confirmation
DESTRUCTIVE_ACTIONS = frozenset({
    "delete", "remove", "drop", "truncate", "overwrite",
    "destroy", "purge", "reset", "format", "wipe",
})


class PermissionResolver:
    """
    Resolves whether an agent has permission to perform an action.

    Uses semantic similarity (SequenceMatcher as a lightweight proxy;
    swap in embedding cosine similarity for production) to match
    requested actions against declared permission scopes.
    """

    def __init__(self, granted_permissions: list[str] | None = None) -> None:
        self._granted: list[str] = granted_permissions or []
        self._denied_log: list[MistakeRecord] = []

    # ── Grant / revoke ───────────────────────────────────────────────

    def grant(self, permission: str) -> None:
        if permission not in self._granted:
            self._granted.append(permission)
            logger.info("Permission granted: %s", permission)

    def revoke(self, permission: str) -> None:
        self._granted = [p for p in self._granted if p != permission]

    # ── Resolution ───────────────────────────────────────────────────

    def check(self, requested: str, step_id: str = "") -> bool:
        """
        Check whether `requested` action is permitted.
        Returns True if allowed, False otherwise.
        Records a MistakeRecord on denial.
        """
        # Destructive actions always need explicit confirmation
        if self._is_destructive(requested):
            logger.warning("Destructive action requires user confirmation: %s", requested)
            self._record_denial(requested, step_id, "Destructive action needs user confirmation.")
            return False

        # Check against granted scopes via semantic similarity
        for scope in self._granted:
            sim = _similarity(requested.lower(), scope.lower())
            if sim >= SIMILARITY_THRESHOLD:
                logger.debug("Permission OK: '%s' matched '%s' (%.2f)", requested, scope, sim)
                return True

        # No match → denied
        self._record_denial(requested, step_id, "No matching permission scope found.")
        return False

    def check_agent(self, agent_name: str, required: list[str], step_id: str = "") -> bool:
        """Check all permissions required by an agent. Returns True if all pass."""
        return all(self.check(perm, step_id) for perm in required)

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _is_destructive(action: str) -> bool:
        action_lower = action.lower()
        return any(d in action_lower for d in DESTRUCTIVE_ACTIONS)

    def _record_denial(self, requested: str, step_id: str, reason: str) -> None:
        m = MistakeRecord(
            step_id=step_id,
            type=MistakeType.PERMISSION_VIOLATION,
            description=f"Permission denied for '{requested}': {reason}",
            correction="Action blocked. Awaiting explicit user grant.",
            learned=f"Action '{requested}' requires explicit permission.",
        )
        self._denied_log.append(m)
        logger.warning("Permission DENIED: %s — %s", requested, reason)

    @property
    def denied(self) -> list[MistakeRecord]:
        return list(self._denied_log)

    @property
    def granted_permissions(self) -> list[str]:
        return list(self._granted)


def _similarity(a: str, b: str) -> float:
    """
    Lightweight semantic similarity (SequenceMatcher).
    For production, replace with embedding cosine similarity.
    """
    return SequenceMatcher(None, a, b).ratio()
