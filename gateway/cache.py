"""Cache storage. A CacheStore keeps answered requests so later ones can reuse them:
found by exact key, or by the nearest vector within the same context key.
Methods are sync because the real store (Redis, Task 8) uses the sync redis client;
main.py calls them through asyncio.to_thread so a slow store never blocks the event loop."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

EMBED_DIM = 384  # vector size of BAAI/bge-small-en-v1.5


@dataclass
class CacheEntry:
    id: str
    ctx: str
    prompt: str
    response: dict
    vector: np.ndarray
    # Kept so each later hit can report the time and tokens it saved.
    upstream_ms: float
    completion_tokens: int
    created_at: float


class CacheStore(Protocol):
    def get_exact(self, key: str) -> CacheEntry | None:
        """The live entry stored under this exact key, or None."""

    def nearest(self, ctx: str, vector: np.ndarray) -> tuple[CacheEntry, float] | None:
        """The most similar live entry with the same ctx and its cosine similarity, or None.
        Vectors are L2-normalized, so cosine similarity is the dot product."""

    def put(self, entry: CacheEntry, exact_key: str, ttl_seconds: int) -> None:
        """Store the entry under the exact key; it expires after ttl_seconds."""

    def clear(self) -> None:
        """Remove every entry."""

    def count(self) -> int:
        """Number of live (not expired) entries."""


class MemoryStore:
    """A CacheStore kept in Python dicts, for tests and local runs.
    It does not inherit from CacheStore: a Protocol is structural, so having the same methods is enough."""

    def __init__(self, clock: Callable[[], float] = time.time):
        # Injectable so tests can move time forward without sleeping.
        self._clock = clock
        self._entries: dict[str, tuple[CacheEntry, float]] = {}  # entry id -> (entry, expires_at)
        self._exact: dict[str, str] = {}  # exact key -> entry id

    def get_exact(self, key: str) -> CacheEntry | None:
        entry_id = self._exact.get(key)
        if entry_id is None or entry_id not in self._entries:
            return None
        entry, expires_at = self._entries[entry_id]
        if self._clock() >= expires_at:
            del self._entries[entry_id]
            del self._exact[key]
            return None
        return entry

    def nearest(self, ctx: str, vector: np.ndarray) -> tuple[CacheEntry, float] | None:
        # Brute force over every entry is fine for tests; Redis does real vector search later.
        now = self._clock()
        best = None
        for entry, expires_at in self._entries.values():
            if entry.ctx != ctx or now >= expires_at:
                continue
            score = float(np.dot(entry.vector, vector))
            if best is None or score > best[1]:
                best = (entry, score)
        return best

    def put(self, entry: CacheEntry, exact_key: str, ttl_seconds: int) -> None:
        expires_at = self._clock() + ttl_seconds
        self._entries[entry.id] = (entry, expires_at)
        self._exact[exact_key] = entry.id

    def clear(self) -> None:
        self._entries.clear()
        self._exact.clear()

    def count(self) -> int:
        now = self._clock()
        return sum(1 for _entry, expires_at in self._entries.values() if now < expires_at)
