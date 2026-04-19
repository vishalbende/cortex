import pytest

from contextengine import ContextEngine, MCPServer


def test_engine_requires_mcps() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ContextEngine(mcps=[], model="claude-sonnet-4-5")


def test_engine_rejects_duplicate_mcp_names() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ContextEngine(
            mcps=[
                MCPServer(name="a", command=["x"]),
                MCPServer(name="a", command=["y"]),
            ],
            model="claude-sonnet-4-5",
        )


def test_engine_defaults() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["npx", "@linear/mcp"])],
        model="claude-sonnet-4-5",
    )
    assert e.model == "claude-sonnet-4-5"
    assert e.router_model == "claude-haiku-4-5"
    assert e.budget.total == 80_000
    assert str(e.cache_dir) == ".contextengine"


async def test_assemble_before_start_raises() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["npx", "@linear/mcp"])],
        model="claude-sonnet-4-5",
    )
    with pytest.raises(RuntimeError, match="start"):
        await e.assemble(message="hi")


class _FakeConnector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return {"ok": True, "name": name}

    async def close(self) -> None:
        return None


async def test_execute_routes_to_owning_mcp() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
    )
    fake = _FakeConnector()
    e._pool._connectors["linear"] = fake  # type: ignore[assignment]

    result = await e.execute({"name": "linear.create_issue", "input": {"title": "t"}})
    assert result == {"ok": True, "name": "create_issue"}
    assert fake.calls == [("create_issue", {"title": "t"})]


async def test_execute_rejects_unnamespaced_name() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
    )
    with pytest.raises(ValueError, match="namespaced"):
        await e.execute({"name": "create_issue", "input": {}})


async def test_execute_rejects_unknown_mcp() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
    )
    with pytest.raises(KeyError, match="Unknown MCP"):
        await e.execute({"name": "github.create_pr", "input": {}})


async def test_execute_accepts_object_form() -> None:
    e = ContextEngine(
        mcps=[MCPServer(name="linear", command=["x"])],
        model="claude-sonnet-4-5",
    )
    fake = _FakeConnector()
    e._pool._connectors["linear"] = fake  # type: ignore[assignment]

    class _Block:
        name = "linear.list_issues"
        input = {"limit": 5}

    result = await e.execute(_Block())
    assert fake.calls == [("list_issues", {"limit": 5})]
    assert result["ok"] is True
