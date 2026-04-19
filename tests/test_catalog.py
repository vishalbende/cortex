import json
from pathlib import Path

import pytest

from contextengine.catalog import (
    build_catalog,
    compute_version_hash,
    load_catalog,
    save_catalog,
)
from contextengine.types import Catalog, MCPCatalog, Tool, ToolCategory
from tests.fakes import FakeAnthropicClient


def test_version_hash_is_order_independent() -> None:
    h1 = compute_version_hash({"linear": ["a", "b"], "github": ["c"]})
    h2 = compute_version_hash({"github": ["c"], "linear": ["b", "a"]})
    assert h1 == h2


def test_version_hash_changes_on_new_tool() -> None:
    h1 = compute_version_hash({"linear": ["a", "b"]})
    h2 = compute_version_hash({"linear": ["a", "b", "c"]})
    assert h1 != h2


def test_version_hash_length() -> None:
    assert len(compute_version_hash({"x": ["y"]})) == 16


def _sample_catalog() -> Catalog:
    t = Tool(
        name="linear.create_issue",
        mcp="linear",
        description="Create issue",
        input_schema={"type": "object"},
        category="issues",
        token_count=12,
    )
    return Catalog(
        mcps=(
            MCPCatalog(
                name="linear",
                summary="Linear.",
                categories=(
                    ToolCategory(name="issues", summary="", tools=(t,)),
                ),
            ),
        ),
        version_hash="abc",
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    save_catalog(catalog, tmp_path)
    loaded = load_catalog("abc", tmp_path)
    assert loaded == catalog


def test_load_miss_returns_none(tmp_path: Path) -> None:
    assert load_catalog("nope", tmp_path) is None


async def test_build_catalog_calls_llm_per_mcp(tmp_path: Path) -> None:
    tools = {
        "linear": [
            Tool(name="linear.create_issue", mcp="linear", description="new issue", input_schema={}),
            Tool(name="linear.list_issues", mcp="linear", description="list issues", input_schema={}),
        ]
    }
    client = FakeAnthropicClient(
        responses=[
            json.dumps(
                {
                    "summary": "Linear issue tracker.",
                    "categories": [
                        {
                            "name": "issues",
                            "summary": "CRUD issues",
                            "tool_names": ["create_issue", "list_issues"],
                        }
                    ],
                }
            )
        ]
    )
    catalog = await build_catalog(
        tools_by_mcp=tools,
        router_model="haiku",
        anthropic_client=client,
        cache_dir=tmp_path,
    )
    assert len(catalog.mcps) == 1
    assert catalog.mcps[0].summary == "Linear issue tracker."
    assert [c.name for c in catalog.mcps[0].categories] == ["issues"]
    assert {t.name for t in catalog.mcps[0].tools_flat} == {
        "linear.create_issue",
        "linear.list_issues",
    }
    assert all(t.category == "issues" for t in catalog.mcps[0].tools_flat)

    cache_file = tmp_path / "catalogs" / f"{catalog.version_hash}.json"
    assert cache_file.exists()


async def test_build_catalog_uses_cache(tmp_path: Path) -> None:
    tools = {"x": [Tool(name="x.a", mcp="x", description="d", input_schema={})]}
    client = FakeAnthropicClient(
        responses=[
            json.dumps(
                {"summary": "X.", "categories": [{"name": "g", "summary": "", "tool_names": ["a"]}]}
            )
        ]
    )
    c1 = await build_catalog(
        tools_by_mcp=tools, router_model="haiku", anthropic_client=client, cache_dir=tmp_path
    )
    assert len(client.messages.calls) == 1

    c2 = await build_catalog(
        tools_by_mcp=tools,
        router_model="haiku",
        anthropic_client=FakeAnthropicClient(responses=[]),
        cache_dir=tmp_path,
    )
    assert c2 == c1


async def test_build_catalog_catches_leftover_tools(tmp_path: Path) -> None:
    tools = {
        "linear": [
            Tool(name="linear.a", mcp="linear", description="d", input_schema={}),
            Tool(name="linear.b", mcp="linear", description="d", input_schema={}),
        ]
    }
    client = FakeAnthropicClient(
        responses=[
            json.dumps(
                {
                    "summary": "Linear.",
                    "categories": [{"name": "g", "summary": "", "tool_names": ["a"]}],
                }
            )
        ]
    )
    catalog = await build_catalog(
        tools_by_mcp=tools, router_model="haiku", anthropic_client=client, cache_dir=tmp_path
    )
    cat_names = [c.name for c in catalog.mcps[0].categories]
    assert "other" in cat_names
    all_tool_names = {t.name for t in catalog.mcps[0].tools_flat}
    assert all_tool_names == {"linear.a", "linear.b"}
