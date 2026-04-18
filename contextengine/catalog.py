"""Catalog builder and persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from contextengine.types import Catalog, Tool


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
    """Build (or load from cache) the hierarchical MCP → category → tools catalog.

    v1 is stubbed — the LLM-driven categorization + summarization pass
    lands next. The eventual flow:

    1. Compute version_hash from tools_by_mcp.
    2. If `cache_dir/catalogs/{hash}.json` exists, load and return.
    3. For each MCP, one Haiku call: group tools into 2-5 categories,
       write per-category summary, write per-MCP summary.
    4. Persist to disk, return Catalog.
    """
    del tools_by_mcp, router_model, anthropic_client, cache_dir
    raise NotImplementedError("build_catalog is stubbed — implement in next pass")


def save_catalog(catalog: Catalog, cache_dir: Path) -> Path:
    """Persist catalog JSON at `cache_dir/catalogs/{version_hash}.json`. Stub."""
    del catalog, cache_dir
    raise NotImplementedError


def load_catalog(version_hash: str, cache_dir: Path) -> Catalog | None:
    """Load a previously-saved catalog by hash, or None on miss. Stub."""
    del version_hash, cache_dir
    raise NotImplementedError
