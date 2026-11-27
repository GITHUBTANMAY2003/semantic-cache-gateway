"""Shared test fixtures and helpers: a fake upstream (respx), a client that calls our app in
memory, a fake clock and fake embedders. Tests need neither Ollama, a server nor the real model."""

import hashlib
import math

import httpx
import numpy as np
import pytest
import respx

from gateway.cache import EMBED_DIM, MemoryStore
from gateway.config import Settings
from gateway.main import create_app

# A made-up host, so a missing mock can never reach a real LLM server.
UPSTREAM_URL = "http://upstream.test/v1"

BASE = np.zeros(EMBED_DIM, dtype=np.float32)
BASE[0] = 1.0


def vector_at(similarity: float) -> np.ndarray:
    """A unit vector whose dot product with BASE is exactly `similarity`,
    so tests can control similarity precisely."""
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[0] = similarity
    vector[1] = math.sqrt(1 - similarity**2)
    return vector


class FakeEmbedder:
    """Has the same embed method as Embedder, so tests never load the real model."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors
        self.calls: list[str] = []  # every text passed to embed, so tests can check when embedding happens

    def embed(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if text not in self.vectors:
            raise KeyError(f"FakeEmbedder has no vector for {text!r}")
        return self.vectors[text]


class HashEmbedder:
    """Same text gives the same vector; different texts give nearly unrelated vectors
    (random 384-d unit vectors have dot products near 0). So general tests never get
    accidental semantic hits and never load the real model."""

    def embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBED_DIM)
        return (v / np.linalg.norm(v)).astype(np.float32)


class FakeClock:
    """A clock tests can move forward, so TTL tests never sleep."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


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
        # Store and embedder passed in: the lifespan, which would create the real ones, does not run here.
        app = create_app(settings, http_client=upstream_client, store=MemoryStore(), embedder=HashEmbedder())
        # ASGITransport hands requests straight to the app, without opening a network port.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client
