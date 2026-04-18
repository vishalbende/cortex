from contextengine.catalog import compute_version_hash


def test_version_hash_is_order_independent() -> None:
    h1 = compute_version_hash({"linear": ["a", "b"], "github": ["c"]})
    h2 = compute_version_hash({"github": ["c"], "linear": ["b", "a"]})
    assert h1 == h2


def test_version_hash_changes_on_new_tool() -> None:
    h1 = compute_version_hash({"linear": ["a", "b"]})
    h2 = compute_version_hash({"linear": ["a", "b", "c"]})
    assert h1 != h2


def test_version_hash_length() -> None:
    h = compute_version_hash({"x": ["y"]})
    assert len(h) == 16
