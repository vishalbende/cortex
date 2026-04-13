"""Tests for the MistakeTracker."""

import pytest
from cortex.mistakes.tracker import MistakeTracker
from cortex.models import MistakeRecord, MistakeType, PageType, Quality


class TestMistakeTracker:
    def test_record_and_count(self):
        tracker = MistakeTracker()
        m = MistakeRecord(
            step_id="s1",
            type=MistakeType.TOOL_FAILURE,
            description="Something broke",
            learned="Fix it",
        )
        tracker.record(m)
        assert tracker.count() == 1
        assert len(tracker.all) == 1

    def test_record_from_output_ok_ignored(self):
        tracker = MistakeTracker()
        result = tracker.record_from_output("s1", Quality.OK, "All good")
        assert result is None
        assert tracker.count() == 0

    def test_record_from_output_error(self):
        tracker = MistakeTracker()
        result = tracker.record_from_output(
            "s1", Quality.ERROR, "Broken",
            correction="Fixed", learned="Check inputs",
        )
        assert result is not None
        assert tracker.count() == 1

    def test_to_page(self):
        tracker = MistakeTracker()
        tracker.record(MistakeRecord(
            step_id="s1", type=MistakeType.HALLUCINATION,
            description="Bad claim", learned="Verify",
        ))
        page = tracker.to_page()
        assert page.type == PageType.MISTAKE
        assert page.id == "page:mistake_log"
        assert "Bad claim" in page.content

    def test_lessons_learned(self):
        tracker = MistakeTracker()
        tracker.record(MistakeRecord(step_id="s1", type=MistakeType.PLAN_ERROR, learned="Lesson A"))
        tracker.record(MistakeRecord(step_id="s2", type=MistakeType.TOOL_FAILURE, learned="Lesson B"))
        lessons = tracker.lessons_learned()
        assert lessons == ["Lesson A", "Lesson B"]

    def test_empty_tracker(self):
        tracker = MistakeTracker()
        assert tracker.count() == 0
        assert tracker.lessons_learned() == []
        assert tracker.errors == []
