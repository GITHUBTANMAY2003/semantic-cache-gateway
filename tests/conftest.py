"""Shared test fixtures: a fake upstream (respx) and a client that calls our app in memory.
Tests need neither Ollama nor a running server."""

import httpx
import pytest
import respx

from gateway.config import Settings
from gateway.main import create_app

# A made-up host, so a missing mock can never reach a real LLM server.
UPSTREAM_URL = "http://upstream.test/v1"


@pytest.fixture
def settings():
    # _env_file=None and explicit values: your local .env cannot change test results.
    return Settings(_env_file=None, upstream_base_url=UPSTREAM_URL, upstream_api_key="")


@pytest.fixture
def upstream():
    # Intercepts the gateway's outgoing calls. Each test says what the upstream replies.
    with respx.mock(base_url=UPSTREAM_URL) as mock:
        yield mock


@pytest.fixture
async def client(settings):
    async with httpx.AsyncClient() as upstream_client:
        app = create_app(settings, http_client=upstream_client)
        # ASGITransport hands requests straight to the app, without opening a network port.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client
