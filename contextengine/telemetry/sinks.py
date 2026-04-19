"""Sink interface and concrete implementations: stdout, JSONL file."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO

if TYPE_CHECKING:
    from contextengine.telemetry.recorder import TraceRecord


class Sink(Protocol):
    async def write(self, record: "TraceRecord") -> None: ...


class StdoutSink:
    """Human-readable one-liner per assemble call."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    async def write(self, record: "TraceRecord") -> None:
        line = (
            f"[contextengine] assemble "
            f"model={record.model} mcps={list(record.mcps_represented)} "
            f"tools={len(record.tools_loaded)} dropped={len(record.tools_dropped)} "
            f"tokens={record.tokens_total}/{record.tokens_budget} "
            f"elapsed={record.elapsed_ms:.0f}ms"
        )
        print(line, file=self._stream)


class FileSink:
    """JSONL file sink — one record per line. Thread-safe within a single process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, record: "TraceRecord") -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")
