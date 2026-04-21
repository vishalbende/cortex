import json
from pathlib import Path

import pytest

from contextengine.dashboard import render_html, render_text, summarize


def _write_traces(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _rec(
    *,
    elapsed=50.0,
    tokens_total=1000,
    tokens_tools=400,
    tokens_memory=100,
    tokens_history=200,
    loaded=("a.x",),
    dropped=(),
    mcps=("a",),
    role="",
):
    return {
        "trace_id": "t",
        "elapsed_ms": elapsed,
        "tokens_total": tokens_total,
        "tokens_tools": tokens_tools,
        "tokens_memory": tokens_memory,
        "tokens_history": tokens_history,
        "tools_loaded": list(loaded),
        "tools_dropped": list(dropped),
        "mcps_represented": list(mcps),
        "role": role,
    }


def test_summary_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    s = summarize(path)
    assert s.total_calls == 0
    assert s.avg_elapsed_ms == 0.0
    assert s.waste_ratio == 0.0


def test_summary_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write_traces(
        path,
        [
            _rec(elapsed=50, loaded=("a.x", "a.y"), dropped=("a.z",), role="sales"),
            _rec(elapsed=150, loaded=("a.x",), dropped=("a.y", "a.z"), role="sales"),
            _rec(elapsed=100, loaded=("b.k",), mcps=("b",), role="support"),
        ],
    )
    s = summarize(path)
    assert s.total_calls == 3
    assert abs(s.avg_elapsed_ms - 100.0) < 0.01
    assert s.by_role == {"sales": 2, "support": 1}
    assert s.tools_loaded_counts["a.x"] == 2
    assert s.tools_dropped_counts["a.z"] == 2
    assert sorted(s.mcps_active) == ["a", "b"]
    # total loaded = 4, total dropped = 3 → waste = 3/7
    assert abs(s.waste_ratio - (3 / 7)) < 0.01


def test_render_text_mentions_counts(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write_traces(path, [_rec()])
    out = render_text(summarize(path))
    assert "total assemble() calls : 1" in out
    assert "tokens total" in out


def test_render_html_contains_metrics(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write_traces(path, [_rec(loaded=("a.x",)), _rec(loaded=("b.y",), mcps=("b",))])
    html_out = render_html(summarize(path))
    assert "contextengine telemetry" in html_out
    assert "assemble calls" in html_out
    assert "a.x" in html_out


def test_summary_ignores_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text("{\"tokens_total\":100,\"elapsed_ms\":10}\nnot json\n")
    s = summarize(path)
    assert s.total_calls == 1
