import pytest

from contextengine.cli import _parse_mcp_spec
from contextengine.types import MCPServer


def test_parse_stdio_spec() -> None:
    s = _parse_mcp_spec("fs=npx -y @modelcontextprotocol/server-filesystem /tmp")
    assert isinstance(s, MCPServer)
    assert s.name == "fs"
    assert s.command == ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert s.url is None


def test_parse_sse_spec() -> None:
    s = _parse_mcp_spec("stripe=https://mcp.stripe.com/sse")
    assert s.name == "stripe"
    assert s.url == "https://mcp.stripe.com/sse"
    assert s.command is None


def test_parse_http_spec() -> None:
    s = _parse_mcp_spec("x=http://localhost:8000")
    assert s.url == "http://localhost:8000"


def test_parse_quoted_args() -> None:
    s = _parse_mcp_spec('fs=python -m server "a b"')
    assert s.command == ["python", "-m", "server", "a b"]


def test_parse_malformed_raises() -> None:
    with pytest.raises(SystemExit):
        _parse_mcp_spec("no-equals-sign")
