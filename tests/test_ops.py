"""Tests for running the gateway: fail-open when the cache breaks, GET /stats and DELETE /admin/cache."""

import httpx
import pytest
import redis
from conftest import BASE, FakeEmbedder, vector_at

from gateway.cache import MemoryStore
from gateway.main import create_app
from gateway.redis_store import RedisStore

Q = "How do I reset my password?"
P99 = "password reset steps"

REQUEST = {"model": "qwen2.5:1.5b", "temperature": 0, "messages": [{"role": "user", "content": Q}]}

RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "qwen2.5:1.5b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Click 'Forgot password'."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class BrokenStore:
    """Redis is down: every call fails."""

    def get_exact(self, key):
        raise redis.ConnectionError("down")

    def nearest(self, ctx, vector):
        raise redis.ConnectionError("down")

    def put(self, entry, exact_key, ttl_seconds):
        raise redis.ConnectionError("down")

    def clear(self):
        raise redis.ConnectionError("down")

    def count(self):
        raise redis.ConnectionError("down")


class PutFailsStore:
    """Reads work (and find nothing), but storing fails."""

    def get_exact(self, key):
        return None

    def nearest(self, ctx, vector):
        return None

    def put(self, entry, exact_key, ttl_seconds):
        raise redis.ConnectionError("down")

    def count(self):
        return 0


class BrokenEmbedder:
    def embed(self, text):
        raise RuntimeError("embedder broke")


# A test can replace `store` or `embedder` with @pytest.mark.parametrize("store", [...]):
# pytest then uses the given value instead of calling the fixture below.
@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def embedder():
    return FakeEmbedder({Q: BASE, P99: vector_at(0.99)})


@pytest.fixture
async def client(settings, store, embedder):
    async with httpx.AsyncClient() as upstream_client:
        app = create_app(settings, http_client=upstream_client, store=store, embedder=embedder)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client


# --- fail-open ---


@pytest.mark.parametrize("store", [BrokenStore()])
async def test_store_error_fails_open(client, upstream):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 200
    assert r.headers["x-cache"] == "BYPASS"
    assert r.json() == RESPONSE
    assert route.call_count == 1


@pytest.mark.parametrize("embedder", [BrokenEmbedder()])
async def test_embedder_error_fails_open(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 200
    assert r.headers["x-cache"] == "BYPASS"


@pytest.mark.parametrize("store", [PutFailsStore()])
async def test_put_failure_still_returns_answer(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 200
    assert r.headers["x-cache"] == "MISS"
    assert r.json() == RESPONSE


@pytest.mark.parametrize("store", [None])
async def test_no_store_bypasses(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 200
    assert r.headers["x-cache"] == "BYPASS"


async def test_startup_without_redis_does_not_crash(settings, embedder, upstream):
    settings.cache_backend = "redis"
    settings.redis_url = "redis://localhost:1"  # nothing listens there
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    async with httpx.AsyncClient() as upstream_client:
        app = create_app(settings, http_client=upstream_client, embedder=embedder)
        # ASGITransport does not run the lifespan, so run it by hand.
        async with app.router.lifespan_context(app):
            assert app.state.store is None
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
                r = await test_client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 200
    assert r.headers["x-cache"] == "BYPASS"


def test_redis_store_unreachable_raises_quickly():
    with pytest.raises(redis.ConnectionError):
        RedisStore("redis://localhost:1")


# --- /stats ---


async def test_stats_after_miss_then_exact_hit(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=REQUEST)
    await client.post("/v1/chat/completions", json=REQUEST)
    s = (await client.get("/stats")).json()
    assert s["counts"]["MISS"] == 1
    assert s["counts"]["HIT-EXACT"] == 1
    assert s["hit_rate"] == 0.5
    assert s["tokens_saved"] == RESPONSE["usage"]["completion_tokens"]
    assert s["entries"] == 1
    assert s["gpu_seconds_saved"] >= 0


async def test_stats_after_semantic_hit(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=REQUEST)
    paraphrase = {**REQUEST, "messages": [{"role": "user", "content": P99}]}
    r = await client.post("/v1/chat/completions", json=paraphrase)
    assert r.headers["x-cache"] == "HIT-SEMANTIC"
    s = (await client.get("/stats")).json()
    assert s["counts"]["HIT-SEMANTIC"] == 1
    assert s["tokens_saved"] == RESPONSE["usage"]["completion_tokens"]


async def test_bypass_counted_but_not_in_hit_rate(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json={**REQUEST, "temperature": 0.9})
    await client.post("/v1/chat/completions", json=REQUEST)
    s = (await client.get("/stats")).json()
    assert s["counts"]["BYPASS"] == 1
    assert s["counts"]["MISS"] == 1
    assert s["hit_rate"] == 0.0


@pytest.mark.parametrize("store", [BrokenStore()])
async def test_stats_with_broken_store(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    assert r.json()["entries"] is None


# --- DELETE /admin/cache ---


async def test_admin_clear_disabled_without_token(client):
    r = await client.delete("/admin/cache")
    assert r.status_code == 403


async def test_admin_clear_rejects_missing_or_wrong_token(client, settings):
    settings.admin_token = "secret"  # the app holds this same settings object
    r1 = await client.delete("/admin/cache")
    r2 = await client.delete("/admin/cache", headers={"Authorization": "Bearer wrong"})
    assert r1.status_code == 401
    assert r2.status_code == 401


async def test_admin_clear_with_right_token(client, settings, store, upstream):
    settings.admin_token = "secret"
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=REQUEST)
    assert store.count() == 1
    r = await client.delete("/admin/cache", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    assert store.count() == 0
    r2 = await client.post("/v1/chat/completions", json=REQUEST)
    assert r2.headers["x-cache"] == "MISS"
