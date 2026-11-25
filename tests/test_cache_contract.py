"""Contract tests: every CacheStore must pass these, whatever it stores its data in.
The store fixture runs each test once per store class (MemoryStore now; Task 8 adds RedisStore)."""

import numpy as np
import pytest

from gateway.cache import EMBED_DIM, CacheEntry, MemoryStore

STORE_FACTORIES = [MemoryStore]


@pytest.fixture(params=STORE_FACTORIES)
def store(request, clock):
    return request.param(clock=clock)


def unit_vector(index):
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def make_entry(id="e1", ctx="c", vector=None):
    return CacheEntry(
        id=id,
        ctx=ctx,
        prompt="How do I reset my password?",
        response={"id": id, "choices": []},
        vector=unit_vector(0) if vector is None else vector,
        upstream_ms=100.0,
        completion_tokens=5,
        created_at=1000.0,
    )


def test_put_then_get_exact(store):
    entry = make_entry()
    store.put(entry, "k", 60)
    found = store.get_exact("k")
    assert found.id == entry.id
    assert found.response == entry.response


def test_missing_exact_is_none(store):
    assert store.get_exact("nope") is None


def test_expired_entry_is_miss(store, clock):
    entry = make_entry()
    store.put(entry, "k", 10)
    clock.advance(11)
    assert store.get_exact("k") is None
    assert store.nearest(entry.ctx, entry.vector) is None
    assert store.count() == 0


def test_entry_alive_before_ttl(store, clock):
    store.put(make_entry(), "k", 10)
    clock.advance(9)
    assert store.get_exact("k") is not None


def test_nearest_picks_most_similar_in_same_ctx(store):
    store.put(make_entry(id="e1", vector=unit_vector(0)), "k1", 60)
    store.put(make_entry(id="e2", vector=unit_vector(1)), "k2", 60)
    query = np.zeros(EMBED_DIM, dtype=np.float32)
    query[0], query[1] = 0.9, 0.1
    query /= np.linalg.norm(query)
    entry, similarity = store.nearest("c", query)
    assert entry.id == "e1"
    assert similarity == pytest.approx(float(query[0]))


def test_nearest_ignores_other_contexts(store):
    store.put(make_entry(ctx="c1"), "k", 60)
    assert store.nearest("c2", unit_vector(0)) is None


def test_clear_and_count(store):
    store.put(make_entry(id="e1"), "k1", 60)
    store.put(make_entry(id="e2"), "k2", 60)
    assert store.count() == 2
    store.clear()
    assert store.count() == 0
    assert store.get_exact("k1") is None
