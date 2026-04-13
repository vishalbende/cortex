"""
Core data models for Cortex.

All shared types live here: PageIndex entries, Plans, Steps, MistakeRecords,
and the Agent input/output contracts. Agents are stateless — all state lives
in PageIndex entries managed by the context layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────────────────

class Quality(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class PageType(str, Enum):
    MEMORY = "memory"
    RAG = "rag"
    TOOL_RESULT = "tool_result"
    PLAN = "plan"
    CONVERSATION = "conversation"
    MISTAKE = "mistake"


class MistakeType(str, Enum):
    HALLUCINATION = "hallucination"
    TOOL_FAILURE = "tool_failure"
    PLAN_ERROR = "plan_error"
    PERMISSION_VIOLATION = "permission_violation"
    LOW_CONFIDENCE = "low_confidence"


# ── PageIndex ────────────────────────────────────────────────────────────

@dataclass
class PageIndex:
    """A single page of context managed by the context layer."""

    id: str = field(default_factory=lambda: f"page:{uuid.uuid4().hex[:8]}")
    type: PageType = PageType.CONVERSATION
    summary: str = ""
    token_count: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tags: list[str] = field(default_factory=list)
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "summary": self.summary,
            "token_count": self.token_count,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "content": self.content,
        }


# ── Mistake Record ───────────────────────────────────────────────────────

@dataclass
class MistakeRecord:
    """Tracks a single mistake observed during execution."""

    mistake_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    step_id: str = ""
    type: MistakeType = MistakeType.PLAN_ERROR
    description: str = ""
    correction: str = ""
    learned: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mistake_id": self.mistake_id,
            "step_id": self.step_id,
            "type": self.type.value,
            "description": self.description,
            "correction": self.correction,
            "learned": self.learned,
        }


# ── Plan / Step ──────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single atomic step inside a Plan."""

    id: str = ""
    agent: str = ""
    action: str = ""
    depends_on: list[str] = field(default_factory=list)
    parallel: bool = False
    confidence: float = 0.9
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    status: StepStatus = StepStatus.PENDING
    quality: Quality = Quality.OK
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "action": self.action,
            "depends_on": self.depends_on,
            "parallel": self.parallel,
            "confidence": self.confidence,
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "status": self.status.value,
            "quality": self.quality.value,
        }


@dataclass
class Plan:
    """A dependency-ordered execution plan for an intent."""

    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    intent: str = ""
    steps: list[PlanStep] = field(default_factory=list)

    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def runnable_steps(self) -> list[PlanStep]:
        """Steps whose dependencies are all DONE."""
        done_ids = {s.id for s in self.steps if s.status == StepStatus.DONE}
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in done_ids for dep in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps
        )

    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "intent": self.intent,
            "steps": [s.to_dict() for s in self.steps],
        }


# ── Agent Input / Output contracts ──────────────────────────────────────

@dataclass
class AgentInput:
    """Standard input every agent receives."""

    intent: str = ""
    relevant_pages: list[PageIndex] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    active_skills: list[str] = field(default_factory=list)
    step: PlanStep | None = None


@dataclass
class AgentOutput:
    """Standard output every agent returns."""

    result: Any = None
    pages_to_add: list[PageIndex] = field(default_factory=list)
    pages_to_update: list[str] = field(default_factory=list)
    confidence: float = 1.0
    quality: Quality = Quality.OK
    mistakes: list[MistakeRecord] = field(default_factory=list)
