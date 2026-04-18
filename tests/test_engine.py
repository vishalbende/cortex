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
