"""
RAG Bridge — integrates VectifyAI/PageIndex for document-level retrieval.

PageIndex is a vectorless, reasoning-based RAG system that replaces
traditional vector databases with LLM-driven hierarchical tree search.
Instead of embeddings, it builds a table-of-contents tree from documents
and uses LLM reasoning to navigate to the right pages.

This bridge wraps PageIndex so Cortex can:
  1. Index PDF/text documents into the PageIndex tree structure
  2. Query them using natural-language intents
  3. Inject results as type="rag" PageIndex entries into the PageStore

Requires: pip install pageindex (or clone from github.com/VectifyAI/PageIndex)
Falls back to a simple keyword search if PageIndex is not installed.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cortex.models import PageIndex, PageType

logger = logging.getLogger(__name__)

# Try to import VectifyAI PageIndex components
_PAGEINDEX_AVAILABLE = False
try:
    from pageindex.core import toc_detector, toc_extractor, tree_parser, page_index
    _PAGEINDEX_AVAILABLE = True
except ImportError:
    logger.info(
        "VectifyAI PageIndex not installed. "
        "Install via: pip install pageindex  or  clone github.com/VectifyAI/PageIndex. "
        "Falling back to keyword-based retrieval."
    )


class PageIndexRAG:
    """
    Bridge between Cortex context and VectifyAI PageIndex.

    If PageIndex is installed, uses its reasoning-based tree search.
    Otherwise, falls back to simple keyword matching against stored pages.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", data_dir: str | None = None) -> None:
        self.model = model
        self.data_dir = Path(data_dir) if data_dir else Path(tempfile.mkdtemp(prefix="cortex_rag_"))
        self._indexed_docs: dict[str, dict[str, Any]] = {}  # doc_id -> tree metadata
        self._doc_pages: dict[str, list[PageIndex]] = {}     # doc_id -> page entries
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Indexing ─────────────────────────────────────────────────────

    async def index_document(self, doc_path: str, doc_id: str | None = None) -> str:
        """
        Index a document (PDF or text) into the PageIndex tree.

        Returns the doc_id assigned to this document.
        """
        path = Path(doc_path)
        doc_id = doc_id or path.stem

        if _PAGEINDEX_AVAILABLE and path.suffix.lower() == ".pdf":
            return await self._index_with_pageindex(path, doc_id)
        else:
            return await self._index_simple(path, doc_id)

    async def _index_with_pageindex(self, path: Path, doc_id: str) -> str:
        """Use VectifyAI PageIndex to build a reasoning tree from a PDF."""
        try:
            # PageIndex CLI: python3 run_pageindex.py --pdf_path <path>
            # We call the Python API directly if available
            tree_data = page_index(
                pdf_path=str(path),
                model=self.model,
                output_dir=str(self.data_dir / doc_id),
            )
            self._indexed_docs[doc_id] = {
                "path": str(path),
                "tree": tree_data,
                "method": "pageindex",
            }
            logger.info("Indexed '%s' with PageIndex tree search", doc_id)
        except Exception as e:
            logger.warning("PageIndex indexing failed for '%s': %s. Falling back.", doc_id, e)
            return await self._index_simple(path, doc_id)
        return doc_id

    async def _index_simple(self, path: Path, doc_id: str) -> str:
        """Fallback: read the file, chunk by paragraphs, store as pages."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        pages: list[PageIndex] = []
        for i, para in enumerate(paragraphs):
            page = PageIndex(
                id=f"page:rag:{doc_id}:{i}",
                type=PageType.RAG,
                summary=para[:80],
                token_count=len(para) // 4,
                tags=[f"doc:{doc_id}"],
                content=para,
            )
            pages.append(page)

        self._doc_pages[doc_id] = pages
        self._indexed_docs[doc_id] = {
            "path": str(path),
            "method": "simple",
            "num_chunks": len(pages),
        }
        logger.info("Indexed '%s' with simple chunking (%d chunks)", doc_id, len(pages))
        return doc_id

    # ── Querying ─────────────────────────────────────────────────────

    async def query(self, question: str, doc_id: str | None = None, top_k: int = 5) -> list[PageIndex]:
        """
        Query indexed documents. Returns matching PageIndex entries.

        If PageIndex is available, uses reasoning-based tree search.
        Otherwise falls back to keyword overlap scoring.
        """
        if _PAGEINDEX_AVAILABLE and doc_id and doc_id in self._indexed_docs:
            meta = self._indexed_docs[doc_id]
            if meta.get("method") == "pageindex":
                return await self._query_pageindex(question, doc_id, top_k)

        return self._query_simple(question, doc_id, top_k)

    async def _query_pageindex(self, question: str, doc_id: str, top_k: int) -> list[PageIndex]:
        """Use PageIndex tree search to find relevant pages."""
        try:
            tree_data = self._indexed_docs[doc_id]["tree"]
            # PageIndex returns page references with content
            results = tree_parser.search(tree_data, question, model=self.model, top_k=top_k)
            pages = []
            for r in results:
                page = PageIndex(
                    id=f"page:rag:{doc_id}:{r.get('page', 0)}",
                    type=PageType.RAG,
                    summary=r.get("summary", "")[:80],
                    token_count=len(r.get("content", "")) // 4,
                    tags=[f"doc:{doc_id}", f"page:{r.get('page', 0)}"],
                    content=r.get("content", ""),
                )
                pages.append(page)
            return pages
        except Exception as e:
            logger.warning("PageIndex query failed: %s. Falling back.", e)
            return self._query_simple(question, doc_id, top_k)

    def _query_simple(self, question: str, doc_id: str | None, top_k: int) -> list[PageIndex]:
        """Fallback: keyword overlap scoring."""
        query_words = set(question.lower().split())
        candidates: list[PageIndex] = []

        if doc_id and doc_id in self._doc_pages:
            candidates = self._doc_pages[doc_id]
        else:
            for pages in self._doc_pages.values():
                candidates.extend(pages)

        scored = []
        for page in candidates:
            page_words = set(page.content.lower().split())
            overlap = len(query_words & page_words)
            if overlap > 0:
                scored.append((overlap, page))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [page for _, page in scored[:top_k]]

    # ── Metadata ─────────────────────────────────────────────────────

    def list_indexed(self) -> dict[str, dict[str, Any]]:
        return dict(self._indexed_docs)

    def is_available(self) -> bool:
        return _PAGEINDEX_AVAILABLE
