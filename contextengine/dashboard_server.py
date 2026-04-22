"""Live dashboard server: serves an auto-refreshing HTML dashboard from a JSONL file.

Zero deps — uses stdlib `http.server` + `threading`. Optional token
auth via `CONTEXTENGINE_DASHBOARD_TOKEN`. Not hardened for public
internet exposure — intended for local / tunneled use.

Routes:
  GET /         → the dashboard HTML (auto-refreshes every 5s).
  GET /summary  → JSON summary (for custom frontends).
  GET /health   → {"ok": true}
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from contextengine.dashboard import render_html, summarize

_AUTOREFRESH_MS = 5000


def _auth_ok(header: str | None, expected: str | None) -> bool:
    if expected is None or expected == "":
        return True
    if not header:
        return False
    if header.startswith("Bearer "):
        return header[len("Bearer "):].strip() == expected
    return header.strip() == expected


def _html_with_refresh(body: str, refresh_ms: int = _AUTOREFRESH_MS) -> str:
    snippet = (
        f"<script>setTimeout(()=>window.location.reload(), {refresh_ms});</script>"
        f"<p style='color:#888;font-size:11px;margin-top:2rem'>auto-refreshes every "
        f"{refresh_ms / 1000:.0f}s</p>"
    )
    closing = "</body></html>"
    if closing in body:
        return body.replace(closing, f"{snippet}{closing}")
    return f"{body}\n{snippet}"


def make_handler(
    *,
    traces_path: Path,
    auth_token: str | None,
):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _auth(self) -> bool:
            ok = _auth_ok(self.headers.get("Authorization"), auth_token)
            if not ok:
                self.send_response(401)
                self.send_header("WWW-Authenticate", "Bearer")
                self.end_headers()
                self.wfile.write(b"unauthorized")
            return ok

        def _send(self, status: int, body: str, ctype: str = "text/html") -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, json.dumps({"ok": True}), "application/json")
                return
            if not self._auth():
                return
            summary = summarize(traces_path)
            if self.path == "/summary":
                self._send(200, json.dumps(asdict(summary)), "application/json")
                return
            if self.path in {"/", "/index.html"}:
                self._send(200, _html_with_refresh(render_html(summary)))
                return
            self._send(404, "not found", "text/plain")

    return _Handler


class DashboardServer:
    """Thin wrapper for programmatic start/stop — used in tests."""

    def __init__(
        self,
        *,
        traces_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 8765,
        auth_token: str | None = None,
    ) -> None:
        self.traces_path = Path(traces_path)
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = make_handler(
            traces_path=self.traces_path, auth_token=self.auth_token
        )
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def url(self, path: str = "/") -> str:
        return f"http://{self.host}:{self.port}{path}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="contextengine-dashboard-serve")
    parser.add_argument("traces", help="path to JSONL traces")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("CONTEXTENGINE_DASHBOARD_TOKEN"),
        help="Bearer token required for / and /summary (env: CONTEXTENGINE_DASHBOARD_TOKEN)",
    )
    args = parser.parse_args(argv)

    srv = DashboardServer(
        traces_path=args.traces,
        host=args.host,
        port=args.port,
        auth_token=args.auth_token,
    )
    srv.start()
    print(f"contextengine dashboard → {srv.url()}")
    print("  /         live HTML (auto-refreshes)")
    print("  /summary  JSON")
    print("  /health   status")
    if args.auth_token:
        print("  (auth: Bearer token required)")
    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
