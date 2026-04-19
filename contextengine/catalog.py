"""Catalog builder and persistence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from contextengine._json import extract_json
from contextengine.types import Catalog, MCPCatalog, Tool, ToolCategory

_CATALOG_SUBDIR = "catalogs"


def compute_version_hash(mcp_tools: dict[str, list[str]]) -> str:
    """Stable hash of the MCP-to-tool-name mapping for cache invalidation."""
    payload = json.dumps({k: sorted(v) for k, v in mcp_tools.items()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def build_catalog(
    *,
    tools_by_mcp: dict[str, list[Tool]],
    router_model: str,
    anthropic_client: Any = None,
    cache_dir: Path | None = None,
) -> Catalog:
    """Build (or load from cache) the hierarchical MCP → category → tools catalog."""
    mcp_tool_names = {k: [t.name for t in v] for k, v in tools_by_mcp.items()}
    version_hash = compute_version_hash(mcp_tool_names)

    if cache_dir is not None:
        cached = load_catalog(version_hash, cache_dir)
        if cached is not None:
            return cached

    client = anthropic_client
    if client is None:
        import anthropic

        client = anthropic.AsyncAnthropic()

    mcps: list[MCPCatalog] = []
    for mcp_name, tools in tools_by_mcp.items():
        if not tools:
            mcps.append(MCPCatalog(name=mcp_name, summary="(no tools)", categories=()))
            continue
        categories, summary = await _categorize_mcp(
            mcp_name=mcp_name,
            tools=tools,
            client=client,
            model=router_model,
        )
        mcps.append(
            MCPCatalog(
                name=mcp_name,
                summary=summary,
                categories=tuple(categories),
            )
        )

    catalog = Catalog(mcps=tuple(mcps), version_hash=version_hash)
    if cache_dir is not None:
        save_catalog(catalog, cache_dir)
    return catalog


async def _categorize_mcp(
    *,
    mcp_name: str,
    tools: list[Tool],
    client: Any,
    model: str,
) -> tuple[list[ToolCategory], str]:
    tool_lines = "\n".join(
        f"- {t.name.split('.', 1)[-1]}: {t.description[:200]}" for t in tools
    )
    user_prompt = (
        f"Tools exposed by the {mcp_name!r} MCP server:\n\n"
        f"{tool_lines}\n\n"
        f"Return JSON with two keys:\n"
        f"  - summary: one sentence (≤20 words) describing this MCP's domain\n"
        f"  - categories: array of {{name, summary, tool_names}}. Group every "
        f"tool into exactly one of 2-5 categories. 'name' is a short "
        f"snake_case id. 'summary' is ≤15 words. 'tool_names' references the "
        f"tool names shown above.\n\n"
        f"Return ONLY the JSON object, no preamble."
    )

    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text
    data = extract_json(text)

    summary: str = data.get("summary", "")
    raw_categories = data.get("categories", [])

    tools_by_local = {t.name.split(".", 1)[-1]: t for t in tools}
    categories: list[ToolCategory] = []
    assigned: set[str] = set()
    for c in raw_categories:
        cat_name = c.get("name", "")
        cat_summary = c.get("summary", "")
        cat_tools: list[Tool] = []
        for local in c.get("tool_names", []):
            t = tools_by_local.get(local)
            if t is None or t.name in assigned:
                continue
            assigned.add(t.name)
            cat_tools.append(replace(t, category=cat_name))
        if cat_tools:
            categories.append(
                ToolCategory(name=cat_name, summary=cat_summary, tools=tuple(cat_tools))
            )

    leftover = [t for t in tools if t.name not in assigned]
    if leftover:
        categories.append(
            ToolCategory(
                name="other",
                summary="Tools not classified by the router.",
                tools=tuple(replace(t, category="other") for t in leftover),
            )
        )

    return categories, summary


def save_catalog(catalog: Catalog, cache_dir: Path) -> Path:
    """Persist catalog JSON at `cache_dir/catalogs/{version_hash}.json`."""
    target = Path(cache_dir) / _CATALOG_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{catalog.version_hash}.json"
    path.write_text(json.dumps(_catalog_to_dict(catalog), indent=2))
    return path


def load_catalog(version_hash: str, cache_dir: Path) -> Catalog | None:
    path = Path(cache_dir) / _CATALOG_SUBDIR / f"{version_hash}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return _catalog_from_dict(data)


def _catalog_to_dict(catalog: Catalog) -> dict[str, Any]:
    return {
        "version_hash": catalog.version_hash,
        "mcps": [
            {
                "name": m.name,
                "summary": m.summary,
                "categories": [
                    {
                        "name": c.name,
                        "summary": c.summary,
                        "tools": [
                            {
                                "name": t.name,
                                "mcp": t.mcp,
                                "description": t.description,
                                "input_schema": t.input_schema,
                                "category": t.category,
                                "token_count": t.token_count,
                            }
                            for t in c.tools
                        ],
                    }
                    for c in m.categories
                ],
            }
            for m in catalog.mcps
        ],
    }


def _catalog_from_dict(data: dict[str, Any]) -> Catalog:
    return Catalog(
        version_hash=data["version_hash"],
        mcps=tuple(
            MCPCatalog(
                name=m["name"],
                summary=m["summary"],
                categories=tuple(
                    ToolCategory(
                        name=c["name"],
                        summary=c["summary"],
                        tools=tuple(
                            Tool(
                                name=t["name"],
                                mcp=t["mcp"],
                                description=t["description"],
                                input_schema=t["input_schema"],
                                category=t.get("category", ""),
                                token_count=t.get("token_count", 0),
                            )
                            for t in c["tools"]
                        ),
                    )
                    for c in m["categories"]
                ),
            )
            for m in data["mcps"]
        ),
    )
