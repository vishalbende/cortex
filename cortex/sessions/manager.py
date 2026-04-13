"""
SessionManager — persistence layer for Cortex sessions.

Sessions are stored as JSON files under .cortex/sessions/:
  .cortex/sessions/
    a1b2c3d4e5f6.json
    f6e5d4c3b2a1.json
    _index.json           ← lightweight index for fast listing

The manager handles:
  - save(session)         → write to disk
  - load(session_id)      → read from disk
  - list_sessions()       → list all with filters
  - delete(session_id)    → remove from disk
  - get_active()          → return the currently active session
  - auto_save(session)    → save only if changed (debounced)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cortex.sessions.model import Session, SessionStatus

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages session persistence in .cortex/sessions/.

    Thread-safe for single-writer (the engine). The TUI reads
    through the manager but never writes concurrently.
    """

    INDEX_FILE = "_index.json"

    def __init__(self, sessions_dir: str | Path | None = None) -> None:
        if sessions_dir:
            self._dir = Path(sessions_dir)
        else:
            # Default: .cortex/sessions/ in current working directory
            self._dir = Path.cwd() / ".cortex" / "sessions"

        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()

    # ── Core operations ─────────────────────────────────────────────

    def save(self, session: Session) -> Path:
        """Save a session to disk and update the index."""
        session.updated_at = datetime.now(timezone.utc).isoformat()

        filepath = self._dir / f"{session.id}.json"
        filepath.write_text(session.to_json(), encoding="utf-8")

        # Update index
        self._index[session.id] = {
            "id": session.id,
            "intent": session.intent[:100],
            "status": session.status.value,
            "model": session.model,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "step_count": session.step_count,
            "completed_steps": session.completed_steps,
            "tags": session.tags,
        }
        self._save_index()

        logger.debug("Session saved: %s", session.id)
        return filepath

    def load(self, session_id: str) -> Session | None:
        """Load a session from disk by ID."""
        filepath = self._dir / f"{session_id}.json"
        if not filepath.exists():
            logger.warning("Session not found: %s", session_id)
            return None

        try:
            raw = filepath.read_text(encoding="utf-8")
            return Session.from_json(raw)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None

    def delete(self, session_id: str) -> bool:
        """Delete a session from disk."""
        filepath = self._dir / f"{session_id}.json"
        if filepath.exists():
            filepath.unlink()
            self._index.pop(session_id, None)
            self._save_index()
            logger.info("Session deleted: %s", session_id)
            return True
        return False

    # ── Queries ─────────────────────────────────────────────────────

    def list_sessions(
        self,
        status: SessionStatus | None = None,
        limit: int = 20,
        tag: str | None = None,
    ) -> list[dict]:
        """
        List sessions from the index, newest first.
        Optionally filter by status or tag.
        """
        entries = list(self._index.values())

        # Filter by status
        if status:
            entries = [e for e in entries if e.get("status") == status.value]

        # Filter by tag
        if tag:
            entries = [e for e in entries if tag in e.get("tags", [])]

        # Sort by updated_at descending
        entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)

        return entries[:limit]

    def get_active(self) -> Session | None:
        """Return the most recently active session, if any."""
        active = self.list_sessions(status=SessionStatus.ACTIVE, limit=1)
        if active:
            return self.load(active[0]["id"])
        return None

    def get_latest(self) -> Session | None:
        """Return the most recent session regardless of status."""
        entries = self.list_sessions(limit=1)
        if entries:
            return self.load(entries[0]["id"])
        return None

    def find_by_intent(self, query: str, limit: int = 5) -> list[dict]:
        """Search sessions by intent text (case-insensitive substring)."""
        query_lower = query.lower()
        matches = [
            e for e in self._index.values()
            if query_lower in e.get("intent", "").lower()
        ]
        matches.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
        return matches[:limit]

    def count(self, status: SessionStatus | None = None) -> int:
        """Count sessions, optionally filtered by status."""
        if status:
            return sum(
                1 for e in self._index.values()
                if e.get("status") == status.value
            )
        return len(self._index)

    # ── Index management ────────────────────────────────────────────

    def _load_index(self) -> dict[str, dict]:
        """Load the index file, or rebuild from session files."""
        index_path = self._dir / self.INDEX_FILE
        if index_path.exists():
            try:
                raw = index_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, KeyError):
                pass

        # Rebuild index from files
        return self._rebuild_index()

    def _rebuild_index(self) -> dict[str, dict]:
        """Scan session files and rebuild the index."""
        index: dict[str, dict] = {}
        for filepath in self._dir.glob("*.json"):
            if filepath.name == self.INDEX_FILE:
                continue
            try:
                raw = filepath.read_text(encoding="utf-8")
                data = json.loads(raw)
                sid = data.get("id", filepath.stem)
                index[sid] = {
                    "id": sid,
                    "intent": data.get("intent", "")[:100],
                    "status": data.get("status", "active"),
                    "model": data.get("model", "sonnet"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "step_count": len(data.get("plan_steps", [])),
                    "completed_steps": sum(
                        1 for s in data.get("plan_steps", [])
                        if s.get("status") == "done"
                    ),
                    "tags": data.get("tags", []),
                }
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping corrupt session file: %s", filepath)
                continue

        self._index = index
        self._save_index()
        return index

    def _save_index(self) -> None:
        """Write the index to disk."""
        index_path = self._dir / self.INDEX_FILE
        index_path.write_text(
            json.dumps(self._index, indent=2, default=str),
            encoding="utf-8",
        )
