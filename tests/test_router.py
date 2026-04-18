import pytest

from contextengine.router import Router
from contextengine.types import Catalog


def test_router_constructs() -> None:
    catalog = Catalog(mcps=(), version_hash="abc")
    r = Router(catalog=catalog, router_model="claude-haiku-4-5")
    assert r.router_model == "claude-haiku-4-5"
    assert r.catalog.version_hash == "abc"


async def test_router_select_stubbed() -> None:
    catalog = Catalog(mcps=(), version_hash="abc")
    r = Router(catalog=catalog, router_model="claude-haiku-4-5")
    with pytest.raises(NotImplementedError):
        await r.select(message="hello")
