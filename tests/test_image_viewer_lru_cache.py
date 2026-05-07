"""Tests for ``ImageViewer``'s internal frame cache (LRU)."""

from __future__ import annotations

import numpy as np

from ui.image_viewer import _LRUCache


def test_lru_cache_get_moves_key_to_most_recent() -> None:
    c = _LRUCache(capacity=2)
    a = np.array([1])
    b = np.array([2])
    c.set(0, a)
    c.set(1, b)
    assert c.get(0) is a
    # Key 0 should now be MRU; inserting 2 evicts key 1 (was LRU)
    c.set(2, np.array([3]))
    assert c.get(1) is None
    assert c.get(0) is not None
    assert c.get(2) is not None


def test_lru_cache_set_replaces_existing_key_without_eviction() -> None:
    c = _LRUCache(capacity=2)
    c.set(0, np.array([0]))
    c.set(1, np.array([1]))
    c.set(0, np.array([10]))
    assert c.get(0)[0] == 10
    assert c.get(1) is not None


def test_lru_cache_evicts_oldest_when_at_capacity() -> None:
    c = _LRUCache(capacity=2)
    c.set(0, np.zeros(1))
    c.set(1, np.ones(1))
    c.set(2, np.full(1, 2.0))
    assert c.get(0) is None
    assert np.allclose(c.get(1), [1.0])
    assert np.allclose(c.get(2), [2.0])
