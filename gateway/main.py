"""The FastAPI app: OpenAI-compatible routes in front of the upstream LLM.
Apps use the gateway by changing only their base_url. A chat request is checked in this order:
skip rules (bypass), exact cache, semantic cache (nearest stored prompt in the same context),
and only then the upstream, whose complete answers are stored for later requests."""

import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.cache import CacheEntry, CacheStore, MemoryStore
from gateway.config import Settings
from gateway.keys import context_key, exact_key, last_user_text
from gateway.rules import bypass_reason
from gateway.upstream import UpstreamBadResponse, forward_chat, forward_models, open_stream, relay_stream


def error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type}})


def create_app(
    settings: Settings | None = None,
    http_client: httpx.AsyncClient | None = None,
    store: CacheStore | None = None,
    embedder=None,
) -> FastAPI:
    settings = settings or Settings()

    # Code before "yield" runs at server startup, code after it at shutdown.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.embedder is None:
            # Imported here so tests, which always pass an embedder, never import torch.
            from gateway.embedder import Embedder

            app.state.embedder = Embedder(settings.embed_model)  # ~2 s, once at startup
        if http_client is not None:
            # Passed in by the caller (tests), so the caller also closes it.
            yield
            return
        # One client for the whole app: it keeps upstream connections open and reuses them.
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            app.state.http_client = client
            yield

    app = FastAPI(lifespan=lifespan)
    app.state.http_client = http_client
    app.state.store = store if store is not None else MemoryStore()
    app.state.embedder = embedder

    # These handlers cover every route. Timeouts are a subclass of RequestError;
    # FastAPI picks the most specific handler, so they get 504 and the rest get 502.
    @app.exception_handler(httpx.TimeoutException)
    async def upstream_timeout(request: Request, exc: httpx.TimeoutException):
        return error_response(504, "upstream_timeout", "The upstream LLM server timed out.")

    @app.exception_handler(httpx.RequestError)
    async def upstream_unreachable(request: Request, exc: httpx.RequestError):
        return error_response(502, "upstream_unreachable", "Could not get a response from the upstream LLM server.")

    @app.exception_handler(UpstreamBadResponse)
    async def upstream_bad_response(request: Request, exc: UpstreamBadResponse):
        return error_response(502, "upstream_bad_response", "The upstream LLM server sent a reply that is not valid JSON.")

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict, request: Request):
        client = app.state.http_client
        store = app.state.store
        embedder = app.state.embedder
        reason = bypass_reason(body, request.headers, settings.max_cache_temperature)

        # A stream request always has a bypass reason ("stream"), so it is never cached.
        if body.get("stream"):
            # Opened before replying, so we know the upstream status before sending ours.
            upstream_response = await open_stream(client, settings, body)
            if upstream_response.status_code == 200:
                media_type = "text/event-stream"
            else:
                media_type = upstream_response.headers.get("content-type")
            return StreamingResponse(
                relay_stream(upstream_response),
                status_code=upstream_response.status_code,
                media_type=media_type,
                headers={"X-Cache": "BYPASS"},
            )

        if reason is not None:
            status, data, _elapsed_ms = await forward_chat(client, settings, body)
            return JSONResponse(status_code=status, content=data, headers={"X-Cache": "BYPASS"})

        # No bypass reason means last_user_text is not None, so context_key cannot raise.
        text = last_user_text(body)
        ctx = context_key(body)
        key = exact_key(ctx, text)

        entry = await asyncio.to_thread(store.get_exact, key)
        if entry is not None:
            return JSONResponse(content=entry.response, headers={"X-Cache": "HIT-EXACT"})

        # Embedding is CPU work (~10-30 ms), so it runs off the event loop like the store calls.
        vector = await asyncio.to_thread(embedder.embed, text)
        # nearest only searches entries with the same ctx, so it never crosses models or system prompts.
        found = await asyncio.to_thread(store.nearest, ctx, vector)
        if found is not None:
            entry, similarity = found
            if similarity >= settings.tau_high:
                headers = {"X-Cache": "HIT-SEMANTIC", "X-Cache-Similarity": f"{similarity:.4f}"}
                return JSONResponse(content=entry.response, headers=headers)

        status, data, elapsed_ms = await forward_chat(client, settings, body)

        # Store only complete answers. Read the body defensively: a strange upstream reply must not crash us.
        choices = data.get("choices") if isinstance(data, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        if status == 200 and isinstance(first, dict) and first.get("finish_reason") == "stop":
            usage = data.get("usage")
            entry = CacheEntry(
                id=uuid4().hex,
                ctx=ctx,
                prompt=text,
                response=data,
                vector=vector,
                upstream_ms=elapsed_ms,
                completion_tokens=usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0,
                created_at=time.time(),
            )
            await asyncio.to_thread(store.put, entry, key, settings.cache_ttl_seconds)

        # A miss whether or not it was stored.
        return JSONResponse(status_code=status, content=data, headers={"X-Cache": "MISS"})

    @app.get("/v1/models")
    async def models():
        status, data = await forward_models(app.state.http_client, settings)
        return JSONResponse(status_code=status, content=data)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


app = create_app()
