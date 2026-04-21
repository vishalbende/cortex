from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from contextengine.dashboard_server import DashboardServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_traces(path: Path, n: int = 3) -> None:
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "elapsed_ms": 50.0 + i,
                    "tokens_total": 1000,
                    "tokens_tools": 400,
                    "tokens_memory": 100,
                    "tokens_history": 200,
                    "tools_loaded": ["a.x"],
                    "tools_dropped": [],
                    "mcps_represented": ["a"],
                    "role": "sales",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _get(url: str, token: str | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


@pytest.fixture
def server(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    _write_traces(path)
    srv = DashboardServer(traces_path=path, host="127.0.0.1", port=_free_port())
    srv.start()
    yield srv
    srv.stop()


def test_health_endpoint(server: DashboardServer) -> None:
    status, body = _get(server.url("/health"))
    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_summary_endpoint(server: DashboardServer) -> None:
    status, body = _get(server.url("/summary"))
    assert status == 200
    payload = json.loads(body)
    assert payload["total_calls"] == 3
    assert payload["by_role"] == {"sales": 3}


def test_root_serves_html_with_refresh(server: DashboardServer) -> None:
    status, body = _get(server.url("/"))
    assert status == 200
    assert "contextengine telemetry" in body
    assert "window.location.reload" in body


def test_auth_required_when_token_set(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _write_traces(path)
    srv = DashboardServer(
        traces_path=path, host="127.0.0.1", port=_free_port(), auth_token="secret"
    )
    srv.start()
    try:
        status, _ = _get(srv.url("/"))
        assert status == 401

        status, body = _get(srv.url("/"), token="secret")
        assert status == 200
        assert "contextengine" in body

        # /health never requires auth — it's for readiness probes
        status, _ = _get(srv.url("/health"))
        assert status == 200
    finally:
        srv.stop()


def test_unknown_path_404s(server: DashboardServer) -> None:
    status, _ = _get(server.url("/nowhere"))
    assert status == 404
