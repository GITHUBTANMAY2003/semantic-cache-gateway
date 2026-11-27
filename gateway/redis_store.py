"""A CacheStore kept in Redis Stack, so the cache survives restarts and can be shared.
Each answer is a hash ("entry:<id>") holding the response, its prompt vector and the numbers
a hit reports. A string ("exact:<key>") points from the exact key to the entry id. Both get
the same TTL, so they expire together. A vector index over the entry hashes finds the nearest
prompt, filtered by ctx so a search never crosses models, system prompts or conversations."""

import json

import numpy as np
import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from gateway.cache import EMBED_DIM, CacheEntry


class RedisStore:
    def __init__(
        self, url: str, dim: int = EMBED_DIM, index: str = "cache_idx", prefix: str = "", timeout: float = 1.0
    ):
        # Bytes responses (no decode_responses), because the vector is binary; text fields are decoded by hand.
        # Short timeouts: a hanging Redis must fail fast, or fail-open never gets a chance to run.
        self._r = redis.Redis.from_url(url, socket_timeout=timeout, socket_connect_timeout=timeout)
        self._index = index
        # The prefix lets tests use their own keys and index without touching real data.
        self._prefix = prefix
        try:
            self._r.ft(index).info()
        except redis.ResponseError:
            # The index does not exist yet ("Unknown index name"): create it once.
            self._r.ft(index).create_index(
                [
                    TagField("ctx"),
                    VectorField("vec", "HNSW", {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"}),
                ],
                definition=IndexDefinition(prefix=[f"{prefix}entry:"], index_type=IndexType.HASH),
            )

    def get_exact(self, key: str) -> CacheEntry | None:
        entry_id = self._r.get(f"{self._prefix}exact:{key}")
        if entry_id is None:
            return None
        return self._load(f"{self._prefix}entry:{entry_id.decode()}")

    def nearest(self, ctx: str, vector: np.ndarray) -> tuple[CacheEntry, float] | None:
        # ctx is a sha256 hex string, so it needs no escaping inside the TAG filter.
        query = (
            Query(f"(@ctx:{{{ctx}}})=>[KNN 1 @vec $v AS dist]")
            .sort_by("dist")
            .return_fields("dist")
            .dialect(2)
        )
        result = self._r.ft(self._index).search(query, query_params={"v": vector.astype(np.float32).tobytes()})
        if not result.docs:
            return None
        doc = result.docs[0]
        entry = self._load(doc.id)
        if entry is None:  # expired between the search and the load
            return None
        # Redis returns cosine DISTANCE (1 - similarity).
        return entry, 1 - float(doc.dist)

    def put(self, entry: CacheEntry, exact_key: str, ttl_seconds: int) -> None:
        entry_key = f"{self._prefix}entry:{entry.id}"
        # A pipeline sends all writes together in one round trip.
        pipe = self._r.pipeline()
        pipe.hset(
            entry_key,
            mapping={
                "ctx": entry.ctx,
                "prompt": entry.prompt,
                "response": json.dumps(entry.response),
                "vec": entry.vector.astype(np.float32).tobytes(),
                "upstream_ms": entry.upstream_ms,
                "completion_tokens": entry.completion_tokens,
                "created_at": entry.created_at,
            },
        )
        pipe.expire(entry_key, ttl_seconds)
        pipe.set(f"{self._prefix}exact:{exact_key}", entry.id, ex=ttl_seconds)
        pipe.execute()

    def clear(self) -> None:
        # Deletes the data but keeps the index, which stays valid for new entries.
        for pattern in (f"{self._prefix}entry:*", f"{self._prefix}exact:*"):
            for key in self._r.scan_iter(match=pattern):
                self._r.delete(key)

    def count(self) -> int:
        # Redis has already removed expired keys. Scanning every key is fine at our scale.
        return sum(1 for _key in self._r.scan_iter(match=f"{self._prefix}entry:*"))

    def _load(self, entry_key: str) -> CacheEntry | None:
        fields = self._r.hgetall(entry_key)
        if not fields:  # expired or never stored
            return None
        return CacheEntry(
            id=entry_key.removeprefix(f"{self._prefix}entry:"),
            ctx=fields[b"ctx"].decode(),
            prompt=fields[b"prompt"].decode(),
            response=json.loads(fields[b"response"]),
            vector=np.frombuffer(fields[b"vec"], dtype=np.float32),
            upstream_ms=float(fields[b"upstream_ms"]),
            completion_tokens=int(fields[b"completion_tokens"]),
            created_at=float(fields[b"created_at"]),
        )
