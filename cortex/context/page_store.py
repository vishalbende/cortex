"""
PageStore — in-memory context page management.

Handles storage, retrieval, eviction, and priority-based selection of
context pages. Mistake pages are NEVER evicted per the Cortex spec.

Selection priority per turn:
  1. Active plan page (always)
  2. RAG pages matching intent tags
  3. Recent tool_result pages (same session)
  4. Mistake pages (always include if any exist)
  5. Memory pages from prior sessions
  6. Conversation pages (last N turns)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from cortex.models import PageIndex, PageType

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 80_000


class PageStore:
    """In-memory page index for a single Cortex session."""

    def __init__(self, max_tokens: int = MAX_CONTEXT_TOKENS) -> None:
        self._pages: dict[str, PageIndex] = {}
        self.max_tokens = max_tokens

    # ── CRUD ─────────────────────────────────────────────────────────

    def add(self, page: PageIndex) -> None:
        self._pages[page.id] = page
        logger.debug("Added page %s (%s)", page.id, page.type.value)
        self._maybe_evict()

    def get(self, page_id: str) -> PageIndex | None:
        return self._pages.get(page_id)

    def update(self, page_id: str, content: str, summary: str = "") -> None:
        page = self._pages.get(page_id)
        if page:
            page.content = content
            page.summary = summary or page.summary
            page.timestamp = datetime.now(timezone.utc).isoformat()
            page.token_count = len(content) // 4  # rough estimate

    def remove(self, page_id: str) -> None:
        self._pages.pop(page_id, None)

    def all_pages(self) -> list[PageIndex]:
        return list(self._pages.values())

    # ── Selection (priority-ordered) ─────────────────────────────────

    def select_for_turn(self, intent_tags: list[str] | None = None) -> list[PageIndex]:
        """
        Return pages in priority order for a single planner turn.
        """
        intent_tags = intent_tags or []
        buckets: dict[int, list[PageIndex]] = {i: [] for i in range(1, 7)}

        for page in self._pages.values():
            if page.type == PageType.PLAN:
                buckets[1].append(page)
            elif page.type == PageType.RAG and _tags_match(page.tags, intent_tags):
                buckets[2].append(page)
            elif page.type == PageType.TOOL_RESULT:
                buckets[3].append(page)
            elif page.type == PageType.MISTAKE:
                buckets[4].append(page)
            elif page.type == PageType.MEMORY:
                buckets[5].append(page)
            elif page.type == PageType.CONVERSATION:
                buckets[6].append(page)

        selected: list[PageIndex] = []
        for priority in sorted(buckets):
            selected.extend(
                sorted(buckets[priority], key=lambda p: p.timestamp, reverse=True)
            )
        return selected

    # ── Token accounting & eviction ──────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return sum(p.token_count for p in self._pages.values())

    def _maybe_evict(self) -> None:
        """Compress lowest-relevance pages when context exceeds the budget."""
        if self.total_tokens <= self.max_tokens:
            return
        # Sort by eviction priority: conversation first, then memory, etc.
        # Mistake pages are NEVER evicted.
        evictable = [
            p
            for p in self._pages.values()
            if p.type != PageType.MISTAKE
        ]
        evictable.sort(key=lambda p: (_type_priority(p.type), p.timestamp))

        while self.total_tokens > self.max_tokens and evictable:
            victim = evictable.pop(0)
            logger.info("Evicting page %s (%s)", victim.id, victim.type.value)
            self._pages.pop(victim.id, None)

    def __len__(self) -> int:
        return len(self._pages)


# ── Helpers ──────────────────────────────────────────────────────────────

def _tags_match(page_tags: list[str], intent_tags: list[str]) -> bool:
    return bool(set(page_tags) & set(intent_tags))


def _type_priority(ptype: PageType) -> int:
    """Lower number → more likely to be evicted first."""
    return {
        PageType.CONVERSATION: 0,
        PageType.MEMORY: 1,
        PageType.TOOL_RESULT: 2,
        PageType.RAG: 3,
        PageType.PLAN: 4,
        PageType.MISTAKE: 99,  # never evicted
    }.get(ptype, 5)
