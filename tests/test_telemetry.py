import io
import json
from pathlib import Path

import pytest

from contextengine.telemetry import FileSink, StdoutSink, TraceRecorder
from contextengine.types import AssembleResult, AssembleStats


def _result() -> AssembleResult:
    stats = AssembleStats(
        tokens_system=10,
        tokens_memory=0,
        tokens_tools=100,
        tokens_history=50,
        tokens_total=160,
        tools_loaded=("linear.create",),
        tools_dropped=("stripe.refund",),
        mcps_represented=("linear",),
        elapsed_ms=42.0,
    )
    return AssembleResult(system="", tools=[], messages=[], stats=stats)


async def test_recorder_emits_to_all_sinks(tmp_path: Path) -> None:
    buf = io.StringIO()
    stdout = StdoutSink(stream=buf)
    file_sink = FileSink(tmp_path / "traces.jsonl")
    recorder = TraceRecorder(sinks=[stdout, file_sink])
    recorder.start()
    recorder.span("phase_a", 10.0, ok=True)

    record = await recorder.emit(
        trace_id="abc",
        model="claude-sonnet-4-5",
        router_model="claude-haiku-4-5",
        entity_id="c1",
        role="sales",
        message="refund last order",
        budget=80_000,
        result=_result(),
    )
    assert record.trace_id == "abc"
    assert record.gen_ai_system == "anthropic"
    assert record.tokens_total == 160
    assert record.events[0].name == "phase_a"

    assert "contextengine" in buf.getvalue()

    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["trace_id"] == "abc"
    assert loaded["tokens_total"] == 160
    assert loaded["tools_loaded"] == ["linear.create"]
    assert loaded["events"][0]["name"] == "phase_a"


def test_add_sink_after_construction() -> None:
    recorder = TraceRecorder()
    buf = io.StringIO()
    recorder.add_sink(StdoutSink(stream=buf))
    assert len(recorder._sinks) == 1  # type: ignore[attr-defined]
