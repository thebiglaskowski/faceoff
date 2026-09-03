"""Unit tests for utils/lru_cache.py (no GPU)."""


def _load():
    from utils.lru_cache import LRUModelCache

    return LRUModelCache


def test_explicit_max_size_honored():
    LRUModelCache = _load()
    cache = LRUModelCache("t", max_size=2)
    assert cache.max_size == 2
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert len(cache) == 2


def test_nonpositive_size_floored_to_one():
    LRUModelCache = _load()
    assert LRUModelCache("t", max_size=0).max_size == 1
    assert LRUModelCache("t", max_size=-3).max_size == 1


def test_eviction_calls_cleanup_and_keeps_newest():
    LRUModelCache = _load()
    cleaned = []
    cache = LRUModelCache("t", max_size=1, cleanup_fn=cleaned.append)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cleaned == [1]
    assert cache.get("b") == 2


def test_get_refreshes_recency():
    LRUModelCache = _load()
    cache = LRUModelCache("t", max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1


def test_remove_and_clear():
    LRUModelCache = _load()
    cache = LRUModelCache("t", max_size=5)
    cache.put("a", 1)
    assert cache.remove("a") is True
    assert cache.remove("missing") is False
    cache.put("b", 2)
    assert cache.clear() == 1
    assert len(cache) == 0
