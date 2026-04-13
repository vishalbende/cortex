"""Tests for the session management system."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from cortex.sessions.model import Session, SessionEvent, SessionStatus
from cortex.sessions.manager import SessionManager


# ── Session model tests ─────────────────────────────────────────────


class TestSession:
    def test_create_session(self):
        s = Session(intent="Build a login page")
        assert s.intent == "Build a login page"
        assert s.status == SessionStatus.ACTIVE
        assert len(s.id) == 12
        assert s.created_at
        assert s.events == []

    def test_add_event(self):
        s = Session(intent="test")
        s.add_event("started", {"intent": "test"})
        assert len(s.events) == 1
        assert s.events[0].type == "started"
        assert s.events[0].data["intent"] == "test"

    def test_mark_completed(self):
        s = Session(intent="test")
        s.mark_completed(result="Done!")
        assert s.status == SessionStatus.COMPLETED
        assert s.result == "Done!"
        assert len(s.events) == 1
        assert s.events[0].type == "completed"

    def test_mark_failed(self):
        s = Session(intent="test")
        s.mark_failed("something broke")
        assert s.status == SessionStatus.FAILED
        assert s.events[0].data["error"] == "something broke"

    def test_mark_cancelled(self):
        s = Session(intent="test")
        s.mark_cancelled()
        assert s.status == SessionStatus.CANCELLED

    def test_mark_paused(self):
        s = Session(intent="test")
        s.mark_paused()
        assert s.status == SessionStatus.PAUSED

    def test_summary(self):
        s = Session(intent="Build a login page with OAuth")
        assert s.id in s.summary
        assert "Build a login" in s.summary

    def test_step_counts(self):
        s = Session(intent="test", plan_steps=[
            {"id": "s1", "status": "done"},
            {"id": "s2", "status": "done"},
            {"id": "s3", "status": "failed"},
            {"id": "s4", "status": "pending"},
        ])
        assert s.step_count == 4
        assert s.completed_steps == 2
        assert s.failed_steps == 1

    def test_serialization_roundtrip(self):
        s = Session(intent="Build auth", model="opus", tags=["urgent"])
        s.add_event("started", {"intent": "Build auth"})
        s.plan_steps = [{"id": "s1", "agent": "ai_agent", "action": "build", "status": "done"}]
        s.mark_completed("Auth module built")

        # Serialize
        data = s.to_dict()
        json_str = s.to_json()

        # Deserialize
        s2 = Session.from_dict(data)
        s3 = Session.from_json(json_str)

        assert s2.id == s.id
        assert s2.intent == "Build auth"
        assert s2.status == SessionStatus.COMPLETED
        assert s2.model == "opus"
        assert s2.tags == ["urgent"]
        assert len(s2.events) == 2  # started + completed
        assert s2.result == "Auth module built"
        assert s2.plan_steps == s.plan_steps

        assert s3.id == s.id
        assert s3.status == SessionStatus.COMPLETED

    def test_from_dict_with_unknown_status(self):
        data = {"id": "abc123", "intent": "test", "status": "unknown_status"}
        s = Session.from_dict(data)
        assert s.status == SessionStatus.ACTIVE  # fallback


class TestSessionEvent:
    def test_event_roundtrip(self):
        e = SessionEvent(timestamp="2025-01-01T00:00:00Z", type="started", data={"x": 1})
        d = e.to_dict()
        e2 = SessionEvent.from_dict(d)
        assert e2.timestamp == e.timestamp
        assert e2.type == "started"
        assert e2.data == {"x": 1}


# ── SessionManager tests ────────────────────────────────────────────


class TestSessionManager:
    @pytest.fixture
    def tmp_sessions_dir(self, tmp_path):
        return tmp_path / "sessions"

    @pytest.fixture
    def manager(self, tmp_sessions_dir):
        return SessionManager(sessions_dir=tmp_sessions_dir)

    def test_save_and_load(self, manager):
        s = Session(intent="Build a component")
        s.add_event("started")
        manager.save(s)

        loaded = manager.load(s.id)
        assert loaded is not None
        assert loaded.id == s.id
        assert loaded.intent == "Build a component"
        assert len(loaded.events) == 1

    def test_load_nonexistent(self, manager):
        assert manager.load("nonexistent") is None

    def test_delete(self, manager):
        s = Session(intent="delete me")
        manager.save(s)
        assert manager.load(s.id) is not None

        result = manager.delete(s.id)
        assert result is True
        assert manager.load(s.id) is None

    def test_delete_nonexistent(self, manager):
        assert manager.delete("nonexistent") is False

    def test_list_sessions(self, manager):
        s1 = Session(intent="First task")
        s2 = Session(intent="Second task")
        s3 = Session(intent="Third task")
        s3.mark_completed("done")

        manager.save(s1)
        manager.save(s2)
        manager.save(s3)

        entries = manager.list_sessions()
        assert len(entries) == 3

    def test_list_sessions_filter_by_status(self, manager):
        s1 = Session(intent="Active task")
        s2 = Session(intent="Done task")
        s2.mark_completed("done")

        manager.save(s1)
        manager.save(s2)

        active = manager.list_sessions(status=SessionStatus.ACTIVE)
        assert len(active) == 1
        assert active[0]["intent"] == "Active task"

        completed = manager.list_sessions(status=SessionStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0]["intent"] == "Done task"

    def test_list_sessions_limit(self, manager):
        for i in range(10):
            manager.save(Session(intent=f"Task {i}"))

        entries = manager.list_sessions(limit=3)
        assert len(entries) == 3

    def test_get_active(self, manager):
        s1 = Session(intent="Old active")
        s2 = Session(intent="Completed")
        s2.mark_completed()

        manager.save(s1)
        manager.save(s2)

        active = manager.get_active()
        assert active is not None
        assert active.intent == "Old active"

    def test_get_latest(self, manager):
        s1 = Session(intent="First")
        s2 = Session(intent="Second")

        manager.save(s1)
        manager.save(s2)

        latest = manager.get_latest()
        assert latest is not None
        # Should be s2 since it was saved last (most recent updated_at)

    def test_find_by_intent(self, manager):
        manager.save(Session(intent="Build a login page"))
        manager.save(Session(intent="Design the API"))
        manager.save(Session(intent="Login form validation"))

        results = manager.find_by_intent("login")
        assert len(results) == 2

    def test_count(self, manager):
        manager.save(Session(intent="Task 1"))
        s2 = Session(intent="Task 2")
        s2.mark_completed()
        manager.save(s2)

        assert manager.count() == 2
        assert manager.count(status=SessionStatus.ACTIVE) == 1
        assert manager.count(status=SessionStatus.COMPLETED) == 1

    def test_index_rebuild(self, tmp_sessions_dir):
        # Create a manager and save sessions
        mgr1 = SessionManager(sessions_dir=tmp_sessions_dir)
        mgr1.save(Session(intent="Rebuild test"))

        # Delete the index file
        index_path = tmp_sessions_dir / "_index.json"
        index_path.unlink()

        # New manager should rebuild from files
        mgr2 = SessionManager(sessions_dir=tmp_sessions_dir)
        entries = mgr2.list_sessions()
        assert len(entries) == 1
        assert entries[0]["intent"] == "Rebuild test"

    def test_save_updates_existing(self, manager):
        s = Session(intent="Evolving task")
        manager.save(s)

        s.mark_completed("All done")
        manager.save(s)

        loaded = manager.load(s.id)
        assert loaded.status == SessionStatus.COMPLETED
        assert loaded.result == "All done"

        # Only one entry in index
        assert manager.count() == 1
