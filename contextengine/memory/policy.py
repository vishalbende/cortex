"""Write policies: govern which roles may write which keys to memory."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Protocol

from contextengine.memory.types import Event, Fact


class WritePolicy(Protocol):
    """Decide whether a role may upsert a fact / append an event."""

    def can_upsert(self, role: str, fact: Fact) -> bool: ...

    def can_append(self, role: str, event: Event) -> bool: ...


class AllowAllPolicy:
    """Default policy — unrestricted writes (back-compat)."""

    def can_upsert(self, role: str, fact: Fact) -> bool:
        del role, fact
        return True

    def can_append(self, role: str, event: Event) -> bool:
        del role, event
        return True


@dataclass(frozen=True)
class Rule:
    """One permission rule.

    `roles` is a tuple of role names (empty tuple = applies to any role).
    `key_pattern` is an fnmatch-style glob applied to Fact.key. Ignored
    for events (use `allow_events=False` to deny events for a role).
    """

    roles: tuple[str, ...] = ()
    key_pattern: str = "*"
    allow_write: bool = True
    allow_events: bool = True


@dataclass
class RoleBasedWritePolicy:
    """Rule-list policy. First matching rule wins; default is deny.

    Examples:
        RoleBasedWritePolicy.from_rules([
            Rule(roles=("sales",), key_pattern="margin.*"),
            Rule(roles=("support",), key_pattern="ticket.*"),
            # support cannot touch margin.* — no matching allow rule → deny.
        ])
    """

    rules: list[Rule] = field(default_factory=list)
    default_allow: bool = False

    @classmethod
    def from_rules(cls, rules: list[Rule]) -> "RoleBasedWritePolicy":
        return cls(rules=list(rules))

    def _role_matches(self, rule: Rule, role: str) -> bool:
        return not rule.roles or role in rule.roles

    def can_upsert(self, role: str, fact: Fact) -> bool:
        for rule in self.rules:
            if not self._role_matches(rule, role):
                continue
            if not fnmatch.fnmatch(fact.key, rule.key_pattern):
                continue
            return rule.allow_write
        return self.default_allow

    def can_append(self, role: str, event: Event) -> bool:
        del event
        for rule in self.rules:
            if not self._role_matches(rule, role):
                continue
            return rule.allow_events
        return self.default_allow


class PolicyViolation(PermissionError):
    """Raised when a write is rejected by the active policy."""


def enforce_upsert(policy: WritePolicy, role: str, fact: Fact) -> None:
    if not policy.can_upsert(role, fact):
        raise PolicyViolation(
            f"role {role!r} is not allowed to write fact {fact.key!r}"
        )


def enforce_append(policy: WritePolicy, role: str, event: Event) -> None:
    if not policy.can_append(role, event):
        raise PolicyViolation(f"role {role!r} is not allowed to append events")
