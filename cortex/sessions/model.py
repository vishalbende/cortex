"""
Session data model — captures the full state of a Cortex run.

A Session stores everything needed to resume work:
  - Intent and plan steps with their statuses
  - Context pages loaded during execution
  - Mistakes recorded
  - Agent outputs
  - Timestamps for tracking
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Lifecycle state of a session."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class SessionEvent:
    """A single event in the session timeline."""
    timestamp: str
    type: str  # "intent", "plan", "step_start", "step_done", "error", "cancel", "resume"
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "type": self.type, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEvent":
        return cls(
            timestamp=d.get("timestamp", ""),
            type=d.get("type", ""),
            data=d.get("data", {}),
        )


@dataclass
class Session:
    """
    A complete Cortex session with full state for save/resume.

    Sessions are stored as JSON files in .cortex/sessions/.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    intent: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    model: str = "sonnet"

    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Plan state
    plan_steps: list[dict] = field(default_factory=list)

    # Context pages
    pages: list[dict] = field(default_factory=list)

    # Mistakes
    mistakes: list[dict] = field(default_factory=list)

    # Agent outputs
    outputs: list[dict] = field(default_factory=list)

    # Event timeline
    events: list[SessionEvent] = field(default_factory=list)

    # Final result
    result: Any = None

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    # ── Timeline helpers ────────────────────────────────────────────

    def add_event(self, event_type: str, data: dict | None = None) -> None:
        """Append an event to the timeline."""
        self.events.append(SessionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            type=event_type,
            data=data or {},
        ))
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, result: Any = None) -> None:
        self.status = SessionStatus.COMPLETED
        self.result = result
        self.add_event("completed", {"result_preview": str(result)[:200] if result else ""})

    def mark_failed(self, error: str = "") -> None:
        self.status = SessionStatus.FAILED
        self.add_event("failed", {"error": error})

    def mark_cancelled(self) -> None:
        self.status = SessionStatus.CANCELLED
        self.add_event("cancelled")

    def mark_paused(self) -> None:
        self.status = SessionStatus.PAUSED
        self.add_event("paused")

    # ── Derived properties ──────────────────────────────────────────

    @property
    def summary(self) -> str:
        """One-line summary for listing."""
        status_icon = {
            SessionStatus.ACTIVE: "⟳",
            SessionStatus.COMPLETED: "✓",
            SessionStatus.FAILED: "✗",
            SessionStatus.CANCELLED: "–",
            SessionStatus.PAUSED: "⏸",
        }.get(self.status, "?")
        intent_short = self.intent[:60] + ("…" if len(self.intent) > 60 else "")
        return f"{status_icon} [{self.id}] {intent_short}"

    @property
    def duration_seconds(self) -> float:
        """Time between created_at and updated_at."""
        try:
            created = datetime.fromisoformat(self.created_at)
            updated = datetime.fromisoformat(self.updated_at)
            return (updated - created).total_seconds()
        except (ValueError, TypeError):
            return 0.0

    @property
    def step_count(self) -> int:
        return len(self.plan_steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.plan_steps if s.get("status") == "done")

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.plan_steps if s.get("status") == "failed")

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "intent": self.intent,
            "status": self.status.value,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "plan_steps": self.plan_steps,
            "pages": self.pages,
            "mistakes": self.mistakes,
            "outputs": self.outputs,
            "events": [e.to_dict() for e in self.events],
            "result": self.result if isinstance(self.result, (str, int, float, bool, list, dict, type(None))) else str(self.result),
            "tags": self.tags,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        events = [SessionEvent.from_dict(e) for e in d.get("events", [])]
        try:
            status = SessionStatus(d.get("status", "active"))
        except ValueError:
            status = SessionStatus.ACTIVE

        return cls(
            id=d.get("id", uuid.uuid4().hex[:12]),
            intent=d.get("intent", ""),
            status=status,
            model=d.get("model", "sonnet"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            plan_steps=d.get("plan_steps", []),
            pages=d.get("pages", []),
            mistakes=d.get("mistakes", []),
            outputs=d.get("outputs", []),
            events=events,
            result=d.get("result"),
            tags=d.get("tags", []),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Session":
        return cls.from_dict(json.loads(raw))
