"""The FastAPI app: OpenAI-compatible routes that pass requests through to the upstream LLM.
Apps use the gateway by changing only their base_url; caching will be added here later."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import Settings
from gateway.rules import bypass_reason
from gateway.upstream import UpstreamBadResponse, forward_chat, forward_models, open_stream, relay_stream


def error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type}})


def create_app(settings: Settings | None = None, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or Settings()

    # Code before "yield" runs at server startup, code after it at shutdown.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
        reason = bypass_reason(body, request.headers, settings.max_cache_temperature)
        headers = {"X-Cache": "BYPASS"} if reason is not None else {}

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
                headers=headers,
            )

        status, data, _elapsed_ms = await forward_chat(client, settings, body)
        return JSONResponse(status_code=status, content=data, headers=headers)

    @app.get("/v1/models")
    async def models():
        status, data = await forward_models(app.state.http_client, settings)
        return JSONResponse(status_code=status, content=data)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


app = create_app()
