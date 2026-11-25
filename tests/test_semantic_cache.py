"""App-level tests for the semantic cache: a paraphrase in the same context is served from
the cache only when its similarity reaches tau_high. FakeEmbedder sets each similarity exactly."""

import httpx
import numpy as np
import pytest
from conftest import BASE, FakeEmbedder, vector_at

from gateway.cache import MemoryStore
from gateway.keys import context_key, exact_key
from gateway.main import create_app

Q = "How do I reset my password?"
P99 = "password reset steps"
P98 = "steps to reset password"
P95 = "how do I reset my username"

RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "qwen2.5:1.5b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Click 'Forgot password'."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def req(text, system=None, temperature=0):
    messages = [{"role": "system", "content": system}] if system is not None else []
    messages.append({"role": "user", "content": text})
    return {"model": "qwen2.5:1.5b", "temperature": temperature, "messages": messages}


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def embedder():
    return FakeEmbedder({Q: BASE, P99: vector_at(0.99), P98: vector_at(0.98), P95: vector_at(0.95)})


@pytest.fixture
async def client(settings, store, embedder):
    # Same as conftest's client, but with a store and embedder the tests can look inside.
    async with httpx.AsyncClient() as upstream_client:
        app = create_app(settings, http_client=upstream_client, store=store, embedder=embedder)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client


async def test_paraphrase_at_099_is_semantic_hit(client, upstream):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r1 = await client.post("/v1/chat/completions", json=req(Q))
    r2 = await client.post("/v1/chat/completions", json=req(P99))
    assert r1.headers["x-cache"] == "MISS"
    assert r2.headers["x-cache"] == "HIT-SEMANTIC"
    assert r2.headers["x-cache-similarity"] == "0.9900"
    assert r2.json() == r1.json()
    assert route.call_count == 1


async def test_exactly_tau_high_is_hit(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=req(Q))
    r2 = await client.post("/v1/chat/completions", json=req(P98))
    assert r2.headers["x-cache"] == "HIT-SEMANTIC"


async def test_below_tau_high_is_miss(client, upstream, store):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=req(Q))
    r2 = await client.post("/v1/chat/completions", json=req(P95))
    assert r2.headers["x-cache"] == "MISS"
    assert route.call_count == 2
    assert store.count() == 2


async def test_tau_high_is_configurable(client, upstream, settings):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    settings.tau_high = 0.9  # the app holds this same settings object
    await client.post("/v1/chat/completions", json=req(Q))
    r2 = await client.post("/v1/chat/completions", json=req(P95))
    assert r2.headers["x-cache"] == "HIT-SEMANTIC"


async def test_same_text_different_system_prompt_is_miss(client, upstream):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r1 = await client.post("/v1/chat/completions", json=req(Q, system="A"))
    r2 = await client.post("/v1/chat/completions", json=req(Q, system="B"))
    assert r1.headers["x-cache"] == "MISS"
    assert r2.headers["x-cache"] == "MISS"
    assert route.call_count == 2


async def test_miss_stores_real_vector(client, upstream, store):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=req(Q))
    entry = store.get_exact(exact_key(context_key(req(Q)), Q))
    assert np.array_equal(entry.vector, BASE)


async def test_exact_hit_does_not_embed(client, upstream, embedder):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=req(Q))
    r2 = await client.post("/v1/chat/completions", json=req(Q))
    assert r2.headers["x-cache"] == "HIT-EXACT"
    assert embedder.calls == [Q]


async def test_bypassed_request_is_never_embedded(client, upstream, embedder):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r = await client.post("/v1/chat/completions", json=req(Q, temperature=0.9))
    assert r.headers["x-cache"] == "BYPASS"
    assert embedder.calls == []


async def test_semantic_hit_is_not_stored_again(client, upstream, store):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=req(Q))
    r2 = await client.post("/v1/chat/completions", json=req(P99))
    assert r2.headers["x-cache"] == "HIT-SEMANTIC"
    assert store.count() == 1
