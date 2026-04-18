import pytest

from contextengine.types import MCPCatalog, MCPServer, Tool, ToolCategory


def test_mcp_server_requires_command_or_url() -> None:
    with pytest.raises(ValueError, match="requires either"):
        MCPServer(name="bad")


def test_mcp_server_rejects_both() -> None:
    with pytest.raises(ValueError, match="only one"):
        MCPServer(name="bad", command=["x"], url="http://y")


def test_mcp_server_stdio() -> None:
    s = MCPServer(name="linear", command=["npx", "@linear/mcp"])
    assert s.name == "linear"
    assert s.url is None
    assert s.env == {}


def test_mcp_server_http() -> None:
    s = MCPServer(name="stripe", url="https://mcp.stripe.com/sse")
    assert s.command is None


def test_tool_defaults() -> None:
    t = Tool(name="x.y", mcp="x", description="d", input_schema={})
    assert t.category == ""
    assert t.token_count == 0


def test_mcp_catalog_tools_flat() -> None:
    t1 = Tool(name="linear.create_issue", mcp="linear", description="", input_schema={})
    t2 = Tool(name="linear.list_issues", mcp="linear", description="", input_schema={})
    cat = ToolCategory(name="issues", summary="", tools=(t1, t2))
    mc = MCPCatalog(name="linear", summary="", categories=(cat,))
    assert {t.name for t in mc.tools_flat} == {"linear.create_issue", "linear.list_issues"}
