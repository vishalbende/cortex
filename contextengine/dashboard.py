"""Dashboard: consume JSONL telemetry traces → text summary or static HTML.

Zero-dep (stdlib only). Consumes files produced by
`contextengine.telemetry.FileSink`.
"""
from __future__ import annotations

import html
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Summary:
    total_calls: int
    avg_elapsed_ms: float
    p50_elapsed_ms: float
    p95_elapsed_ms: float
    avg_tokens_total: float
    avg_tokens_tools: float
    avg_tokens_memory: float
    avg_tokens_history: float
    tools_loaded_counts: dict[str, int]
    tools_dropped_counts: dict[str, int]
    waste_ratio: float  # loaded but never referenced — approximated as dropped/loaded
    mcps_active: list[str]
    by_role: dict[str, int]


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return float(ordered[k])


def summarize(path: str | Path) -> Summary:
    records = _records(Path(path))
    if not records:
        return Summary(
            total_calls=0,
            avg_elapsed_ms=0.0,
            p50_elapsed_ms=0.0,
            p95_elapsed_ms=0.0,
            avg_tokens_total=0.0,
            avg_tokens_tools=0.0,
            avg_tokens_memory=0.0,
            avg_tokens_history=0.0,
            tools_loaded_counts={},
            tools_dropped_counts={},
            waste_ratio=0.0,
            mcps_active=[],
            by_role={},
        )

    elapsed = [float(r.get("elapsed_ms", 0.0)) for r in records]
    total = [float(r.get("tokens_total", 0)) for r in records]
    tools_tok = [float(r.get("tokens_tools", 0)) for r in records]
    mem_tok = [float(r.get("tokens_memory", 0)) for r in records]
    hist_tok = [float(r.get("tokens_history", 0)) for r in records]

    loaded: dict[str, int] = {}
    dropped: dict[str, int] = {}
    mcps: set[str] = set()
    by_role: dict[str, int] = {}

    total_loaded = 0
    total_dropped = 0
    for r in records:
        for name in r.get("tools_loaded", []):
            loaded[name] = loaded.get(name, 0) + 1
            total_loaded += 1
        for name in r.get("tools_dropped", []):
            dropped[name] = dropped.get(name, 0) + 1
            total_dropped += 1
        for m in r.get("mcps_represented", []):
            mcps.add(m)
        role = r.get("role") or "(none)"
        by_role[role] = by_role.get(role, 0) + 1

    waste = total_dropped / max(1, total_loaded + total_dropped)

    return Summary(
        total_calls=len(records),
        avg_elapsed_ms=statistics.fmean(elapsed),
        p50_elapsed_ms=_pct(elapsed, 0.5),
        p95_elapsed_ms=_pct(elapsed, 0.95),
        avg_tokens_total=statistics.fmean(total),
        avg_tokens_tools=statistics.fmean(tools_tok),
        avg_tokens_memory=statistics.fmean(mem_tok),
        avg_tokens_history=statistics.fmean(hist_tok),
        tools_loaded_counts=dict(sorted(loaded.items(), key=lambda kv: -kv[1])),
        tools_dropped_counts=dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
        waste_ratio=waste,
        mcps_active=sorted(mcps),
        by_role=by_role,
    )


def render_text(summary: Summary) -> str:
    lines = [
        "contextengine telemetry summary",
        "================================",
        f"total assemble() calls : {summary.total_calls}",
        f"avg elapsed            : {summary.avg_elapsed_ms:.1f}ms  (p50 {summary.p50_elapsed_ms:.1f} / p95 {summary.p95_elapsed_ms:.1f})",
        f"avg tokens total       : {summary.avg_tokens_total:.0f}",
        f"  └ tools              : {summary.avg_tokens_tools:.0f}",
        f"  └ memory             : {summary.avg_tokens_memory:.0f}",
        f"  └ history            : {summary.avg_tokens_history:.0f}",
        f"waste ratio (dropped)  : {summary.waste_ratio:.1%}",
        f"mcps active            : {', '.join(summary.mcps_active) or '(none)'}",
        "",
        "calls by role:",
    ]
    for role, n in sorted(summary.by_role.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {role}: {n}")
    lines.append("")
    lines.append("top tools loaded:")
    for name, n in list(summary.tools_loaded_counts.items())[:10]:
        lines.append(f"  {n:>4}  {name}")
    if summary.tools_dropped_counts:
        lines.append("")
        lines.append("top tools dropped (waste signal):")
        for name, n in list(summary.tools_dropped_counts.items())[:10]:
            lines.append(f"  {n:>4}  {name}")
    return "\n".join(lines)


_HTML_SHELL = """<!doctype html>
<html><head><meta charset="utf-8"><title>contextengine dashboard</title>
<style>
  body {{ font: 14px -apple-system, ui-sans-serif, sans-serif; padding: 2rem; max-width: 920px; margin: auto; color: #1a1a1a; }}
  h1 {{ margin-top: 0; font-size: 20px; letter-spacing: .2px; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin: 2rem 0 .5rem; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .metric {{ padding: 12px 14px; border: 1px solid #e5e5e5; border-radius: 6px; }}
  .metric .v {{ font-size: 22px; font-weight: 600; }}
  .metric .l {{ font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #888; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; }}
  th {{ color: #888; font-weight: 500; text-transform: uppercase; font-size: 10px; letter-spacing: .5px; }}
  .bar {{ display: inline-block; height: 8px; background: #4a7dff; border-radius: 2px; }}
  .bar.drop {{ background: #e07070; }}
  code {{ background: #f6f6f6; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
</style></head><body>
<h1>contextengine telemetry</h1>
<div class="metrics">
  <div class="metric"><div class="v">{total}</div><div class="l">assemble calls</div></div>
  <div class="metric"><div class="v">{p50:.0f}<span class="l">/{p95:.0f} ms</span></div><div class="l">p50 / p95 elapsed</div></div>
  <div class="metric"><div class="v">{tokens:.0f}</div><div class="l">avg tokens/call</div></div>
  <div class="metric"><div class="v">{waste:.1%}</div><div class="l">tool waste ratio</div></div>
</div>
<h2>token split</h2>
<table>
<tr><th>bucket</th><th>avg</th></tr>
<tr><td>tools</td><td>{tools:.0f}</td></tr>
<tr><td>memory</td><td>{memory:.0f}</td></tr>
<tr><td>history</td><td>{history:.0f}</td></tr>
</table>
<h2>mcps active</h2>
<p>{mcps}</p>
<h2>calls by role</h2>
<table><tr><th>role</th><th>calls</th></tr>
{role_rows}
</table>
<h2>top tools loaded</h2>
<table><tr><th>tool</th><th>count</th><th></th></tr>
{loaded_rows}
</table>
<h2>top tools dropped (waste)</h2>
<table><tr><th>tool</th><th>count</th><th></th></tr>
{dropped_rows}
</table>
</body></html>"""


def _bar(count: int, max_count: int, drop: bool = False) -> str:
    pct = (count / max_count) if max_count else 0
    cls = "bar drop" if drop else "bar"
    return f'<span class="{cls}" style="width:{pct * 120:.0f}px"></span>'


def _rows(items: dict[str, int], drop: bool = False) -> str:
    if not items:
        return "<tr><td colspan=3>(none)</td></tr>"
    max_count = max(items.values())
    rows = []
    for name, n in list(items.items())[:20]:
        rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td>"
            f"<td>{n}</td><td>{_bar(n, max_count, drop)}</td></tr>"
        )
    return "\n".join(rows)


def _role_rows(by_role: dict[str, int]) -> str:
    if not by_role:
        return "<tr><td colspan=2>(none)</td></tr>"
    return "\n".join(
        f"<tr><td>{html.escape(r)}</td><td>{n}</td></tr>"
        for r, n in sorted(by_role.items(), key=lambda kv: -kv[1])
    )


def render_html(summary: Summary) -> str:
    return _HTML_SHELL.format(
        total=summary.total_calls,
        p50=summary.p50_elapsed_ms,
        p95=summary.p95_elapsed_ms,
        tokens=summary.avg_tokens_total,
        waste=summary.waste_ratio,
        tools=summary.avg_tokens_tools,
        memory=summary.avg_tokens_memory,
        history=summary.avg_tokens_history,
        mcps=", ".join(f"<code>{html.escape(m)}</code>" for m in summary.mcps_active) or "(none)",
        role_rows=_role_rows(summary.by_role),
        loaded_rows=_rows(summary.tools_loaded_counts),
        dropped_rows=_rows(summary.tools_dropped_counts, drop=True),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="contextengine-dashboard")
    parser.add_argument("traces", help="path to JSONL traces from FileSink")
    parser.add_argument(
        "--format", choices=["text", "html"], default="text", help="output format"
    )
    parser.add_argument(
        "--output", "-o", help="write to file instead of stdout (required for html)"
    )
    args = parser.parse_args(argv)

    summary = summarize(args.traces)
    body = render_html(summary) if args.format == "html" else render_text(summary)
    if args.output:
        Path(args.output).write_text(body)
        print(f"wrote {args.format} → {args.output}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
