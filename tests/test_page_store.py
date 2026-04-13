"""Tests for the PageStore context manager."""

import pytest
from cortex.context.page_store import PageStore
from cortex.models import PageIndex, PageType


class TestPageStore:
    def test_add_and_get(self):
        store = PageStore()
        page = PageIndex(id="page:1", type=PageType.RAG, content="Hello", token_count=10)
        store.add(page)
        assert store.get("page:1") is page
        assert len(store) == 1

    def test_remove(self):
        store = PageStore()
        page = PageIndex(id="page:1", content="Hello", token_count=10)
        store.add(page)
        store.remove("page:1")
        assert store.get("page:1") is None
        assert len(store) == 0

    def test_select_priority_order(self):
        store = PageStore()
        store.add(PageIndex(id="p1", type=PageType.CONVERSATION, token_count=10))
        store.add(PageIndex(id="p2", type=PageType.PLAN, token_count=10))
        store.add(PageIndex(id="p3", type=PageType.MISTAKE, token_count=10))
        store.add(PageIndex(id="p4", type=PageType.RAG, tags=["test"], token_count=10))

        selected = store.select_for_turn(intent_tags=["test"])
        types = [p.type for p in selected]
        # Plan should come first, then RAG (matching tags), then Mistake, then Conversation
        assert types[0] == PageType.PLAN
        assert types[1] == PageType.RAG

    def test_eviction_spares_mistakes(self):
        store = PageStore(max_tokens=50)
        store.add(PageIndex(id="p1", type=PageType.CONVERSATION, token_count=30, content="a" * 120))
        store.add(PageIndex(id="p2", type=PageType.MISTAKE, token_count=30, content="b" * 120))
        # Adding a third should trigger eviction of conversation, NOT mistake
        store.add(PageIndex(id="p3", type=PageType.RAG, token_count=30, content="c" * 120))
        assert store.get("p2") is not None  # mistake survives
        assert store.get("p1") is None      # conversation evicted

    def test_total_tokens(self):
        store = PageStore()
        store.add(PageIndex(id="p1", token_count=100))
        store.add(PageIndex(id="p2", token_count=200))
        assert store.total_tokens == 300
