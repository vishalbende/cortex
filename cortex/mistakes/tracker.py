"""
Mistake Tracker — records, annotates, and summarises errors during execution.

Rules from the Cortex spec:
  - Tag any step output with quality: ok | warning | error
  - On warning: annotate the issue inline and continue
  - On error: halt the step, emit a mistake record, re-plan
  - Maintain a mistake log per session as a special PageIndex page
  - Mistake pages are NEVER evicted from context
  - Never silently recover from an error. Always mark it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from cortex.models import (
    MistakeRecord,
    MistakeType,
    PageIndex,
    PageType,
    Quality,
)

logger = logging.getLogger(__name__)


class MistakeTracker:
    """Session-scoped mistake log."""

    def __init__(self) -> None:
        self._mistakes: list[MistakeRecord] = []

    # ── Recording ────────────────────────────────────────────────────

    def record(self, mistake: MistakeRecord) -> None:
        """Add a mistake to the session log."""
        self._mistakes.append(mistake)
        logger.warning(
            "MISTAKE [%s] step=%s: %s",
            mistake.type.value,
            mistake.step_id,
            mistake.description,
        )

    def record_from_output(
        self,
        step_id: str,
        quality: Quality,
        description: str,
        correction: str = "",
        learned: str = "",
        mistake_type: MistakeType = MistakeType.PLAN_ERROR,
    ) -> MistakeRecord | None:
        """Convenience: create and record a mistake if quality is not OK."""
        if quality == Quality.OK:
            return None
        m = MistakeRecord(
            step_id=step_id,
            type=mistake_type,
            description=description,
            correction=correction,
            learned=learned,
        )
        self.record(m)
        return m

    # ── Querying ─────────────────────────────────────────────────────

    @property
    def all(self) -> list[MistakeRecord]:
        return list(self._mistakes)

    @property
    def errors(self) -> list[MistakeRecord]:
        return [
            m
            for m in self._mistakes
            if m.type
            in (MistakeType.TOOL_FAILURE, MistakeType.PLAN_ERROR, MistakeType.HALLUCINATION)
        ]

    @property
    def warnings_only(self) -> list[MistakeRecord]:
        return [m for m in self._mistakes if m.type == MistakeType.LOW_CONFIDENCE]

    def count(self) -> int:
        return len(self._mistakes)

    # ── Export as PageIndex page ─────────────────────────────────────

    def to_page(self) -> PageIndex:
        """
        Serialize the full mistake log as a single PageIndex page
        of type MISTAKE. This page is NEVER evicted.
        """
        lines = ["# Mistake Log", ""]
        for m in self._mistakes:
            lines.append(f"## [{m.type.value.upper()}] {m.mistake_id}")
            lines.append(f"- **Step:** {m.step_id}")
            lines.append(f"- **Description:** {m.description}")
            lines.append(f"- **Correction:** {m.correction}")
            lines.append(f"- **Learned:** {m.learned}")
            lines.append("")

        content = "\n".join(lines)
        return PageIndex(
            id="page:mistake_log",
            type=PageType.MISTAKE,
            summary=f"Mistake log ({len(self._mistakes)} entries)",
            token_count=len(content) // 4,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tags=["system:mistakes"],
            content=content,
        )

    # ── Lessons learned summary ──────────────────────────────────────

    def lessons_learned(self) -> list[str]:
        """One-liner lessons from every mistake, for the final output."""
        return [m.learned for m in self._mistakes if m.learned]

    def __len__(self) -> int:
        return len(self._mistakes)
