"""History compaction: summarize older turns via LLM when history grows long."""
from __future__ import annotations

import json

from contextengine.llm.base import LLMClient
from contextengine.types import Message


class HistoryCompactor:
    """Summarize older messages into a single rolling summary turn.

    Strategy:
      - Keep the most recent `keep_recent` turns verbatim.
      - Summarize everything before that in one cheap LLM call.
      - Replace the prefix with a single Message(role="user", content="<summary>")
        marked with a sentinel so future compactions can re-compact cleanly.

    Triggered when `len(history) > threshold`. Provider-agnostic.
    """

    SENTINEL = "<compacted-summary>"

    def __init__(
        self,
        *,
        model: str,
        llm: LLMClient,
        threshold: int = 40,
        keep_recent: int = 10,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.keep_recent = keep_recent
        self._llm = llm

    def should_compact(self, history: list[Message]) -> bool:
        return len(history) > self.threshold

    async def compact(self, history: list[Message]) -> list[Message]:
        if not self.should_compact(history):
            return history

        prefix = history[: -self.keep_recent]
        recent = history[-self.keep_recent :]

        existing_summary = ""
        turns: list[Message] = []
        for m in prefix:
            if (
                isinstance(m.content, str)
                and m.role == "user"
                and m.content.startswith(self.SENTINEL)
            ):
                existing_summary = m.content[len(self.SENTINEL) :].strip()
            else:
                turns.append(m)

        transcript_lines = []
        for m in turns:
            body = m.content if isinstance(m.content, str) else json.dumps(m.content)
            transcript_lines.append(f"[{m.role}] {body[:600]}")
        transcript = "\n".join(transcript_lines)

        prior = f"\nExisting rolling summary (extend, do not repeat):\n{existing_summary}" if existing_summary else ""
        prompt = (
            f"Summarize this conversation into durable facts, open questions, "
            f"and decisions. Preserve anything load-bearing for later turns. "
            f"Target 120-200 words. Plain prose, no bullet lists.{prior}\n\n"
            f"Transcript:\n{transcript}"
        )

        response = await self._llm.complete(
            model=self.model,
            system="",
            user=prompt,
            max_tokens=512,
        )
        summary = response.text.strip()

        summary_turn = Message(role="user", content=f"{self.SENTINEL} {summary}")
        return [summary_turn, *recent]
