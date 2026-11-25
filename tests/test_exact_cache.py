"""App-level tests for the exact cache: what is stored, what is served, and what never is."""

import httpx
import pytest
from conftest import HashEmbedder

from gateway.cache import MemoryStore
from gateway.keys import context_key, exact_key
from gateway.main import create_app

REQUEST = {
    "model": "qwen2.5:1.5b",
    "temperature": 0,
    "messages": [{"role": "user", "content": "How do I reset my password?"}],
}

RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "qwen2.5:1.5b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Click 'Forgot password'."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def with_text(text, system=None):
    messages = [{"role": "system", "content": system}] if system is not None else []
    messages.append({"role": "user", "content": text})
    return {**REQUEST, "messages": messages}


@pytest.fixture
def store(clock):
    return MemoryStore(clock=clock)


@pytest.fixture
async def client(settings, store):
    # Same as conftest's client, but with a store the tests can look inside.
    async with httpx.AsyncClient() as upstream_client:
        app = create_app(settings, http_client=upstream_client, store=store, embedder=HashEmbedder())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client


async def test_second_identical_request_is_exact_hit(client, upstream):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r1 = await client.post("/v1/chat/completions", json=REQUEST)
    r2 = await client.post("/v1/chat/completions", json=REQUEST)
    assert r1.headers["x-cache"] == "MISS"
    assert r2.headers["x-cache"] == "HIT-EXACT"
    assert r2.json() == r1.json()
    assert route.call_count == 1


async def test_normalized_variant_is_exact_hit(client, upstream):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=with_text("How do I reset my password?"))
    r2 = await client.post("/v1/chat/completions", json=with_text("how do I RESET my password"))
    assert r2.headers["x-cache"] == "HIT-EXACT"
    assert route.call_count == 1


async def test_different_system_prompt_never_shares(client, upstream):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r1 = await client.post("/v1/chat/completions", json=with_text("How do I reset my password?", system="A"))
    r2 = await client.post("/v1/chat/completions", json=with_text("How do I reset my password?", system="B"))
    assert r1.headers["x-cache"] == "MISS"
    assert r2.headers["x-cache"] == "MISS"
    assert route.call_count == 2


async def test_not_stored_when_finish_reason_length(client, upstream, store):
    cut_off = {**RESPONSE, "choices": [{**RESPONSE["choices"][0], "finish_reason": "length"}]}
    route = upstream.post("/chat/completions").respond(200, json=cut_off)
    r1 = await client.post("/v1/chat/completions", json=REQUEST)
    r2 = await client.post("/v1/chat/completions", json=REQUEST)
    assert r1.headers["x-cache"] == "MISS"
    assert r2.headers["x-cache"] == "MISS"
    assert route.call_count == 2
    assert store.count() == 0


async def test_not_stored_on_upstream_error(client, upstream, store):
    route = upstream.post("/chat/completions").respond(429, json={"error": "rate"})
    r1 = await client.post("/v1/chat/completions", json=REQUEST)
    r2 = await client.post("/v1/chat/completions", json=REQUEST)
    assert r1.headers["x-cache"] == "MISS"
    assert r2.headers["x-cache"] == "MISS"
    assert route.call_count == 2
    assert store.count() == 0


async def test_bypassed_requests_never_stored_or_served(client, upstream, store):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    body = {**REQUEST, "temperature": 0.9}
    r1 = await client.post("/v1/chat/completions", json=body)
    r2 = await client.post("/v1/chat/completions", json=body)
    assert r1.headers["x-cache"] == "BYPASS"
    assert r2.headers["x-cache"] == "BYPASS"
    assert route.call_count == 2
    assert store.count() == 0


async def test_stored_entry_has_right_fields(client, upstream, store):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=REQUEST)
    assert store.count() == 1
    entry = store.get_exact(exact_key(context_key(REQUEST), "How do I reset my password?"))
    assert entry.prompt == "How do I reset my password?"
    assert entry.response == RESPONSE
    assert entry.completion_tokens == 5
    assert entry.ctx == context_key(REQUEST)


async def test_expired_entry_is_miss_again(client, upstream, settings, clock):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    settings.cache_ttl_seconds = 10  # the app holds this same settings object
    await client.post("/v1/chat/completions", json=REQUEST)
    clock.advance(11)
    r2 = await client.post("/v1/chat/completions", json=REQUEST)
    assert r2.headers["x-cache"] == "MISS"
    assert route.call_count == 2
