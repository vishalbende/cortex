"""Tests for core data models."""

import pytest
from cortex.models import (
    AgentInput,
    AgentOutput,
    MistakeRecord,
    MistakeType,
    PageIndex,
    PageType,
    Plan,
    PlanStep,
    Quality,
    StepStatus,
)


class TestPageIndex:
    def test_creation_defaults(self):
        page = PageIndex()
        assert page.id.startswith("page:")
        assert page.type == PageType.CONVERSATION
        assert page.token_count == 0

    def test_to_dict(self):
        page = PageIndex(id="page:test", type=PageType.RAG, summary="Test", content="Hello")
        d = page.to_dict()
        assert d["id"] == "page:test"
        assert d["type"] == "rag"
        assert d["summary"] == "Test"

    def test_empty_content(self):
        page = PageIndex()
        assert page.content == ""
        assert page.tags == []


class TestPlan:
    def _make_plan(self):
        return Plan(
            intent="Test intent",
            steps=[
                PlanStep(id="s1", agent="a", action="first"),
                PlanStep(id="s2", agent="b", action="second", depends_on=["s1"]),
                PlanStep(id="s3", agent="c", action="third", depends_on=["s1"], parallel=True),
            ],
        )

    def test_runnable_steps_initial(self):
        plan = self._make_plan()
        runnable = plan.runnable_steps()
        assert len(runnable) == 1
        assert runnable[0].id == "s1"

    def test_runnable_after_step1_done(self):
        plan = self._make_plan()
        plan.steps[0].status = StepStatus.DONE
        runnable = plan.runnable_steps()
        assert len(runnable) == 2
        ids = {s.id for s in runnable}
        assert ids == {"s2", "s3"}

    def test_is_complete(self):
        plan = self._make_plan()
        assert not plan.is_complete()
        for s in plan.steps:
            s.status = StepStatus.DONE
        assert plan.is_complete()

    def test_has_failures(self):
        plan = self._make_plan()
        assert not plan.has_failures()
        plan.steps[1].status = StepStatus.FAILED
        assert plan.has_failures()


class TestMistakeRecord:
    def test_to_dict(self):
        m = MistakeRecord(
            step_id="s1",
            type=MistakeType.HALLUCINATION,
            description="Bad claim",
            correction="Removed",
            learned="Check sources",
        )
        d = m.to_dict()
        assert d["type"] == "hallucination"
        assert d["step_id"] == "s1"


class TestAgentOutput:
    def test_defaults(self):
        out = AgentOutput()
        assert out.result is None
        assert out.quality == Quality.OK
        assert out.confidence == 1.0
        assert out.mistakes == []
