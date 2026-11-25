"""Tests for the pass-through proxy: what goes in and what comes out must not change."""

import json

import httpx

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

SSE_BYTES = b'data: {"x":1}\n\ndata: [DONE]\n\n'


async def test_non_stream_passthrough(client, upstream):
    upstream.post("/chat/completions").respond(200, json=RESPONSE)
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 200
    assert r.json() == RESPONSE


async def test_upstream_error_status_passed_through(client, upstream):
    upstream.post("/chat/completions").respond(429, json={"error": "rate"})
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 429
    assert r.json() == {"error": "rate"}


async def test_upstream_unreachable_gives_502(client, upstream):
    upstream.post("/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_unreachable"


async def test_upstream_timeout_gives_504(client, upstream):
    upstream.post("/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    r = await client.post("/v1/chat/completions", json=REQUEST)
    assert r.status_code == 504
    assert r.json()["error"]["type"] == "upstream_timeout"


async def test_stream_passthrough(client, upstream):
    upstream.post("/chat/completions").respond(200, content=SSE_BYTES)
    r = await client.post("/v1/chat/completions", json={**REQUEST, "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.content == SSE_BYTES


async def test_models_passthrough(client, upstream):
    models = {"object": "list", "data": [{"id": "qwen2.5:1.5b", "object": "model"}]}
    upstream.get("/models").respond(200, json=models)
    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == models


async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


async def test_request_body_forwarded_unchanged(client, upstream):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    # Includes a field the gateway knows nothing about: it must still arrive.
    body = {**REQUEST, "max_tokens": 50, "some_unknown_field": {"a": [1, 2]}}
    await client.post("/v1/chat/completions", json=body)
    assert json.loads(route.calls.last.request.content) == body


# --- Extra tests beyond the Task 1 list ---


async def test_stream_upstream_error_status_passed_through(client, upstream):
    upstream.post("/chat/completions").respond(429, json={"error": "rate"})
    r = await client.post("/v1/chat/completions", json={**REQUEST, "stream": True})
    assert r.status_code == 429
    assert r.json() == {"error": "rate"}
    assert r.headers["content-type"] == "application/json"


async def test_stream_upstream_unreachable_gives_502(client, upstream):
    upstream.post("/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    r = await client.post("/v1/chat/completions", json={**REQUEST, "stream": True})
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_unreachable"


async def test_authorization_sent_only_when_key_set(client, upstream, settings):
    route = upstream.post("/chat/completions").respond(200, json=RESPONSE)
    await client.post("/v1/chat/completions", json=REQUEST)
    assert "authorization" not in route.calls.last.request.headers

    settings.upstream_api_key = "test-key"  # the app holds this same settings object
    await client.post("/v1/chat/completions", json=REQUEST)
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
