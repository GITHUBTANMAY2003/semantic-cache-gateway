"""Contract tests: every CacheStore must pass these, whatever it stores its data in.
The fixture runs each test once per store: MemoryStore and RedisStore."""

import time

import numpy as np
import pytest
import redis
from conftest import FakeClock

from gateway.cache import EMBED_DIM, CacheEntry, MemoryStore
from gateway.redis_store import RedisStore


def memory_store(settings):
    clock = FakeClock()
    return MemoryStore(clock=clock), clock.advance


def redis_store(settings):
    try:
        redis.Redis.from_url(settings.redis_url).ping()
    except redis.ConnectionError:
        pytest.skip("Redis not running: docker compose up -d redis")
    # Own index and key prefix, so tests never touch the gateway's real data.
    store = RedisStore(settings.redis_url, index="test_cache_idx", prefix="test:")
    store.clear()
    # Redis expires keys on its own clock, so the only way to move time forward is to wait.
    return store, time.sleep


STORE_FACTORIES = [memory_store, redis_store]


@pytest.fixture(params=STORE_FACTORIES, ids=["memory", "redis"])
def store_and_advance(request, settings):
    """(store, advance): advance(seconds) moves the store's time forward."""
    return request.param(settings)


@pytest.fixture
def store(store_and_advance):
    # Most tests do not need to move time.
    return store_and_advance[0]


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


@pytest.mark.slow  # the Redis variant sleeps ~2.5 s
def test_entry_lives_until_ttl_then_misses(store_and_advance):
    store, advance = store_and_advance
    entry = make_entry()
    store.put(entry, "k", 2)
    advance(1)
    assert store.get_exact("k") is not None
    advance(1.5)
    assert store.get_exact("k") is None
    assert store.nearest(entry.ctx, entry.vector) is None
    assert store.count() == 0


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


def test_redis_data_outlives_the_python_object(settings):
    store, _advance = redis_store(settings)
    store.put(make_entry(), "k", 60)
    # A new object with the same url, index and prefix: the data lives in Redis, not in Python.
    again = RedisStore(settings.redis_url, index="test_cache_idx", prefix="test:")
    assert again.get_exact("k").id == "e1"
