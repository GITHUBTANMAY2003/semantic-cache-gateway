"""All calls to the real LLM server (Ollama now, vLLM later).
Kept in one module so the routes do not deal with upstream URLs or auth headers."""

import time
from collections.abc import AsyncIterator

import httpx

from gateway.config import Settings


class UpstreamBadResponse(Exception):
    """The upstream replied, but the body was not valid JSON."""


def _json(response: httpx.Response) -> dict:
    # A proxy in front of the LLM can answer with an HTML error page instead of JSON.
    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamBadResponse() from exc


def _headers(settings: Settings) -> dict:
    # Ollama needs no key. An empty "Bearer " header would be invalid, so send nothing.
    if settings.upstream_api_key:
        return {"Authorization": f"Bearer {settings.upstream_api_key}"}
    return {}


async def forward_chat(client: httpx.AsyncClient, settings: Settings, body: dict) -> tuple[int, dict, float]:
    """Send a non-streaming chat request. Returns (status_code, response_json, elapsed_ms)."""
    start = time.perf_counter()
    response = await client.post(
        f"{settings.upstream_base_url}/chat/completions", json=body, headers=_headers(settings)
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response.status_code, _json(response), elapsed_ms


async def forward_models(client: httpx.AsyncClient, settings: Settings) -> tuple[int, dict]:
    response = await client.get(f"{settings.upstream_base_url}/models", headers=_headers(settings))
    return response.status_code, _json(response)


async def open_stream(client: httpx.AsyncClient, settings: Settings, body: dict) -> httpx.Response:
    """Send a streaming chat request and return once the status and headers have arrived.
    The body is not read yet and the connection stays open: the caller must pass the
    response to relay_stream, which closes it."""
    request = client.build_request(
        "POST", f"{settings.upstream_base_url}/chat/completions", json=body, headers=_headers(settings)
    )
    # stream=True is what stops httpx from downloading the whole body before returning.
    return await client.send(request, stream=True)


async def relay_stream(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield the upstream body piece by piece, as it arrives."""
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        # Runs when the body ends or reading it fails: frees the connection.
        await response.aclose()
