"""The FastAPI app: OpenAI-compatible routes in front of the upstream LLM.
Apps use the gateway by changing only their base_url. A chat request is checked in this order:
skip rules (bypass), exact cache, semantic cache (nearest stored prompt in the same context),
and only then the upstream, whose complete answers are stored for later requests.
The cache is optional: if the store or embedder fails, the request is forwarded (X-Cache: BYPASS)."""

import asyncio
import logging
import secrets
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.cache import CacheEntry, CacheStore, MemoryStore
from gateway.config import Settings
from gateway.keys import context_key, exact_key, last_user_text
from gateway.redis_store import RedisStore
from gateway.rules import bypass_reason
from gateway.stats import Stats
from gateway.upstream import UpstreamBadResponse, forward_chat, forward_models, open_stream, relay_stream


logger = logging.getLogger("gateway")


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
        if app.state.store is None:
            # In the lifespan so tests, which pass their own store, never connect to Redis.
            if settings.cache_backend == "redis":
                try:
                    app.state.store = RedisStore(settings.redis_url)
                except Exception:
                    # The store stays None, so requests bypass the cache.
                    # Limitation: the gateway must be restarted after Redis comes back.
                    logger.exception("Redis unavailable; caching disabled until restart")
            else:
                app.state.store = MemoryStore()
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
    app.state.store = store
    app.state.embedder = embedder
    app.state.stats = Stats()

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
        stats = app.state.stats
        reason = bypass_reason(body, request.headers, settings.max_cache_temperature)

        # A stream request always has a bypass reason ("stream"), so it is never cached.
        if body.get("stream"):
            # Opened before replying, so we know the upstream status before sending ours.
            upstream_response = await open_stream(client, settings, body)
            if upstream_response.status_code == 200:
                media_type = "text/event-stream"
            else:
                media_type = upstream_response.headers.get("content-type")
            stats.record("BYPASS")
            return StreamingResponse(
                relay_stream(upstream_response),
                status_code=upstream_response.status_code,
                media_type=media_type,
                headers={"X-Cache": "BYPASS"},
            )

        if reason is None and store is None:
            reason = "no-store"  # Redis was down at startup

        if reason is None:
            # Broad on purpose: any failure in the optional cache must degrade to "no cache", never to an error.
            try:
                # No bypass reason means last_user_text is not None, so context_key cannot raise.
                text = last_user_text(body)
                ctx = context_key(body)
                key = exact_key(ctx, text)

                entry = await asyncio.to_thread(store.get_exact, key)
                if entry is not None:
                    stats.record("HIT-EXACT", entry.completion_tokens, entry.upstream_ms)
                    return JSONResponse(content=entry.response, headers={"X-Cache": "HIT-EXACT"})

                # Embedding is CPU work (~10-30 ms), so it runs off the event loop like the store calls.
                vector = await asyncio.to_thread(embedder.embed, text)
                # nearest only searches entries with the same ctx, so it never crosses models or system prompts.
                found = await asyncio.to_thread(store.nearest, ctx, vector)
                if found is not None:
                    entry, similarity = found
                    if similarity >= settings.tau_high:
                        stats.record("HIT-SEMANTIC", entry.completion_tokens, entry.upstream_ms)
                        headers = {"X-Cache": "HIT-SEMANTIC", "X-Cache-Similarity": f"{similarity:.4f}"}
                        return JSONResponse(content=entry.response, headers=headers)
            except Exception:
                logger.exception("cache lookup failed; forwarding to upstream")
                reason = "cache-error"

        # The upstream calls stay outside the try, so their errors keep the 502/504 handlers.
        if reason is not None:
            status, data, _elapsed_ms = await forward_chat(client, settings, body)
            stats.record("BYPASS")
            return JSONResponse(status_code=status, content=data, headers={"X-Cache": "BYPASS"})

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
            try:
                await asyncio.to_thread(store.put, entry, key, settings.cache_ttl_seconds)
            except Exception:
                # The caller still gets the answer; only the caching of it failed.
                logger.exception("storing the answer failed; returning it uncached")

        # A miss whether or not it was stored.
        stats.record("MISS")
        return JSONResponse(status_code=status, content=data, headers={"X-Cache": "MISS"})

    @app.get("/v1/models")
    async def models():
        status, data = await forward_models(app.state.http_client, settings)
        return JSONResponse(status_code=status, content=data)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/stats")
    async def get_stats():
        entry_count = None  # None tells the caller the store could not be asked
        if app.state.store is not None:
            try:
                entry_count = await asyncio.to_thread(app.state.store.count)
            except Exception:
                logger.exception("could not count cache entries")
        return app.state.stats.snapshot(entry_count)

    @app.delete("/admin/cache")
    async def clear_cache(request: Request):
        if not settings.admin_token:
            return error_response(403, "admin_disabled", "The admin endpoint is disabled: set ADMIN_TOKEN to enable it.")
        given = request.headers.get("authorization", "")
        # Constant-time comparison so the token can't be guessed from response timing.
        # Compared as bytes: compare_digest raises on non-ASCII str, and a header can contain anything.
        if not secrets.compare_digest(given.encode(), f"Bearer {settings.admin_token}".encode()):
            return error_response(401, "unauthorized", "Missing or wrong admin token.")
        if app.state.store is None:
            return error_response(503, "cache_unavailable", "No cache store is available (Redis was down at startup).")
        await asyncio.to_thread(app.state.store.clear)
        return {"cleared": True}

    return app


app = create_app()
