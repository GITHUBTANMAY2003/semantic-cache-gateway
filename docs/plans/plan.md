# Semantic AI Gateway & Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An OpenAI-compatible FastAPI gateway that answers repeated-meaning prompts from a Redis vector cache instead of the GPU, plus benchmarks that prove hit rate, savings and false-hit rate on vLLM.

**Architecture:** Request → skip rules → context key (request minus last user message) → exact cache → embed + nearest-neighbour search filtered by context key → threshold / grey-zone verifier → else forward to upstream LLM and store. Storage behind a `CacheStore` protocol with `MemoryStore` (tests) and `RedisStore` (real).

**Tech Stack:** Python 3.11, FastAPI, uvicorn, httpx, sentence-transformers, numpy, redis-py + Redis Stack, pydantic-settings, pytest, pytest-asyncio, respx, matplotlib, docker-compose.

**Spec:** `docs/superpowers/specs/2026-11-25-semantic-cache-gateway-design.md`

## Global Constraints

- **Explainability first:** no library or abstraction without a one-line reason in the file's docstring; one responsibility per module (spec §0, §5.9 layout).
- Every task ends with an **explain-back checkpoint**: Tanmay explains the new code in his own words and answers 2–3 interview-style questions before the next task starts.
- Python ≥ 3.11. Ollama upstream for dev: `http://localhost:11434/v1`, model `qwen2.5:1.5b`. Benchmark: vLLM `Qwen/Qwen2.5-1.5B-Instruct --dtype half` on T4.
- Embed model `BAAI/bge-small-en-v1.5` (384-d, L2-normalized). Verifier `cross-encoder/quora-distilroberta-base`.
- Defaults: `TAU_HIGH=0.98`, `TAU_LOW=0.85`, `P_VERIFY=0.8`, `MAX_CACHE_TEMPERATURE=0.3`, `CACHE_TTL_SECONDS=86400`.
- `X-Cache` values exactly: `BYPASS`, `HIT-EXACT`, `HIT-SEMANTIC`, `HIT-VERIFIED`, `MISS`. Semantic/verified hits also send `X-Cache-Similarity` (4 decimals).
- Store only when upstream status is 200 and `choices[0].finish_reason == "stop"`.
- Cached responses are returned exactly as stored (the `X-Cache` header is how callers tell).
- The cache must never take the gateway down: any store/embedder error → forward to upstream, `X-Cache: BYPASS`.
- Stretch items (Prometheus/Grafana, streaming cache, concurrent bench, per-user scope) are **not** in this plan.

## Review Focus

1. **Upstream down / timeout / non-200** → client gets upstream's status and body (or 502 with a JSON error on connection failure); nothing stored. Test in Task 1.
2. **Redis unreachable or erroring** mid-request → request still answered via upstream with `X-Cache: BYPASS`. Test in Task 8.
3. **No usable last user message** (only system message, last message is assistant, empty/whitespace text, or `content` is a list of parts) → bypass, no crash. Test in Task 3.
4. **Exact key present but its entry already expired** → treated as a miss, not a crash or stale hit. Test in Task 4.
5. **Same question, different system prompt or model** → never a hit. Test in Task 2 and Task 6.

---

### Task 0: Phase 0 — learning notebook (no product code)

**Files:**
- Create: `notebooks/00_basics.ipynb`

- [ ] **Step 1:** Load `BAAI/bge-small-en-v1.5`, embed "How do I reset my password?", "Password reset steps?", "How do I reset my username?", "What's the weather?"; print the 4×4 cosine matrix (normalized dot products).
- [ ] **Step 2:** Call Ollama's `/v1/chat/completions` with `httpx` and print the raw JSON; identify `choices[0].message.content`, `finish_reason`, `usage`.
- [ ] **Step 3 (checkpoint):** Tanmay writes 3 lines in the notebook: what an embedding is, why paraphrases score ~0.85–0.95 not 0.98, what `finish_reason` means.

---

### Task 1: Project skeleton + pass-through proxy

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.gitignore`, `gateway/__init__.py`, `gateway/config.py`, `gateway/upstream.py`, `gateway/main.py`, `tests/conftest.py`, `tests/test_proxy.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings) with fields `upstream_base_url: str`, `upstream_api_key: str = ""`, `redis_url: str = "redis://localhost:6379"`, `embed_model`, `tau_high: float`, `tau_low: float`, `verifier_enabled: bool = True`, `p_verify: float`, `max_cache_temperature: float`, `cache_ttl_seconds: int`, `admin_token: str = ""`; `get_settings() -> Settings`.
- Produces: `async def forward_chat(client: httpx.AsyncClient, body: dict) -> tuple[int, dict, float]` (status, json, elapsed_ms); `async def forward_stream(client, body) -> AsyncIterator[bytes]`; `async def forward_models(client) -> tuple[int, dict]`.
- Produces: `create_app(settings: Settings, store=None, embedder=None, verifier=None) -> FastAPI` (later tasks fill the optional deps).

- [ ] **Step 1: `git init`, write pyproject with deps listed in Tech Stack (dev group: pytest, pytest-asyncio, respx).**
- [ ] **Step 2: Write failing tests** (respx mocks `UPSTREAM/chat/completions`; client = `httpx.AsyncClient(transport=ASGITransport(app))`):

```python
async def test_non_stream_passthrough(client, upstream):
    upstream.post("/chat/completions").respond(200, json=OK_RESPONSE)
    r = await client.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 200 and r.json() == OK_RESPONSE

async def test_upstream_error_status_passed_through(client, upstream):
    upstream.post("/chat/completions").respond(429, json={"error": "rate"})
    r = await client.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 429 and r.json() == {"error": "rate"}

async def test_upstream_unreachable_gives_502(client, upstream):
    upstream.post("/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    r = await client.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 502 and "error" in r.json()

async def test_stream_passthrough(client, upstream):
    upstream.post("/chat/completions").respond(200, content=SSE_BYTES)
    r = await client.post("/v1/chat/completions", json={**REQ, "stream": True})
    assert r.content == SSE_BYTES

async def test_models_passthrough(client, upstream): ...
async def test_healthz(client): assert (await client.get("/healthz")).json() == {"ok": True}
```

- [ ] **Step 3: Run** `pytest tests/test_proxy.py -v` → FAIL (import errors).
- [ ] **Step 4: Implement** config, upstream, `main.py` routes. `stream: true` returns `StreamingResponse(forward_stream(...), media_type="text/event-stream")`. One shared `httpx.AsyncClient` created in the app lifespan, timeout 120 s.
- [ ] **Step 5: Run** tests → PASS. Manual check: `uvicorn gateway.main:app` then the OpenAI Python SDK with `base_url="http://localhost:8000/v1"` against Ollama returns an answer.
- [ ] **Step 6: Commit** `feat: pass-through OpenAI-compatible proxy`.
- [ ] **Step 7 (checkpoint):** Tanmay explains: why httpx async, what the lifespan client is for, what happens on 429 and on connection error.

---

### Task 2: Keys (the one matching rule)

**Files:**
- Create: `gateway/keys.py`, `tests/test_keys.py`

**Interfaces:**
- Produces: `last_user_text(body: dict) -> str | None` (text of final message if role is `user` and content is a non-empty string after strip; else None); `normalize(text: str) -> str` (lowercase, collapse whitespace, strip trailing `?.!`); `context_key(body: dict) -> str` (sha256 hex of `json.dumps(body_without_last_user_content, sort_keys=True)`, where `stream` is also removed); `exact_key(ctx: str, text: str) -> str` (sha256 of `ctx + "\n" + normalize(text)`).

- [ ] **Step 1: Failing tests:**

```python
def test_context_key_ignores_last_user_wording():
    assert context_key(req("reset password?")) == context_key(req("password reset steps"))

def test_context_key_differs_on_system_prompt():
    assert context_key(req("x", system="A")) != context_key(req("x", system="B"))

def test_context_key_differs_on_model_and_temperature():
    assert context_key(req("x", model="a")) != context_key(req("x", model="b"))
    assert context_key(req("x", temperature=0)) != context_key(req("x", temperature=0.2))

def test_context_key_differs_on_earlier_turns(): ...
def test_exact_key_normalizes(): assert exact_key("c", "Reset  Password?") == exact_key("c", "reset password")
def test_last_user_text_none_cases():
    assert last_user_text({"messages": [{"role": "system", "content": "s"}]}) is None
    assert last_user_text({"messages": [{"role": "user", "content": "   "}]}) is None
    assert last_user_text({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}) is None
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (deep-copy body; never mutate input). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: context and exact cache keys`.
- [ ] **Step 6 (checkpoint):** Why hashing "everything except the last message" is safer than picking fields.

---

### Task 3: Skip rules

**Files:**
- Create: `gateway/rules.py`, `gateway/bypass_patterns.py`, `tests/test_rules.py`

**Interfaces:**
- Produces: `bypass_reason(body: dict, headers: Mapping[str, str], max_temperature: float) -> str | None` returning one of `"stream"`, `"temperature"`, `"n"`, `"tools"`, `"no-cache-header"`, `"no-user-text"`, `"time-sensitive"`, `"identifier"`, or None. Missing `temperature` counts as 1.0 (OpenAI default) → bypass.
- `bypass_patterns.py` holds two lists of compiled regexes: `TIME_SENSITIVE`, `IDENTIFIERS`.

- [ ] **Step 1: Failing tests** (parametrized):

```python
@pytest.mark.parametrize("text", ["How do I cancel my order?", "my account settings help", "reset my password"])
def test_generic_personal_wording_is_cacheable(text): assert bypass_reason(req(text), {}, 0.3) is None

@pytest.mark.parametrize("text", ["cancel order #48213", "refund for ORD-99812", "email me at a@b.com",
                                  "call +91 98765 43210", "where is {{Order Number}}"])
def test_identifiers_bypass(text): assert bypass_reason(req(text), {}, 0.3) == "identifier"

@pytest.mark.parametrize("text", ["what's the latest version", "news today", "price on 2026-12-01"])
def test_time_sensitive_bypass(text): ...

def test_temperature_missing_bypasses(): ...
def test_temperature_above_max_bypasses(): ...
def test_stream_n_tools_header_bypass(): ...
def test_no_user_text_bypass(): ...   # system-only, assistant-last, list content
```

- [ ] **Step 2–4:** FAIL → implement (check order: stream, header, temperature, n, tools, no-user-text, time, identifier) → PASS.
- [ ] **Step 5: Wire into `main.py`:** if `bypass_reason` is not None → forward, header `X-Cache: BYPASS`. Add a test in `tests/test_proxy.py`: `test_bypass_header_set`.
- [ ] **Step 6: Commit** `feat: skip rules`. **Step 7 (checkpoint):** why "my order" is cacheable but "order #48213" is not.

---

### Task 4: CacheStore protocol + MemoryStore + exact cache wired in

**Files:**
- Create: `gateway/cache.py`, `tests/test_cache_contract.py`, `tests/test_exact_cache.py`
- Modify: `gateway/main.py`

**Interfaces:**
- Produces:
  - `@dataclass CacheEntry: id: str, ctx: str, prompt: str, response: dict, vector: np.ndarray, upstream_ms: float, completion_tokens: int, created_at: float`
  - `class CacheStore(Protocol)`: `get_exact(key: str) -> CacheEntry | None`; `nearest(ctx: str, vector: np.ndarray) -> tuple[CacheEntry, float] | None` (similarity in [−1, 1]); `put(entry: CacheEntry, exact_key: str, ttl_seconds: int) -> None`; `clear() -> None`; `count() -> int`.
  - `class MemoryStore(CacheStore)` with injectable `clock: Callable[[], float] = time.time` for TTL tests.
- Methods are sync; `main.py` calls them via `asyncio.to_thread` (one reason line in docstring).

- [ ] **Step 1: Failing contract tests**, written as a fixture parametrized over store classes (only `MemoryStore` now; Task 8 adds `RedisStore`):

```python
def test_put_then_get_exact(store): store.put(E, "k", 60); assert store.get_exact("k").id == E.id
def test_missing_exact_is_none(store): assert store.get_exact("nope") is None
def test_expired_entry_is_miss(store_with_clock):   # Review Focus 4
    s, clock = store_with_clock; s.put(E, "k", 10); clock.advance(11)
    assert s.get_exact("k") is None and s.nearest(E.ctx, E.vector) is None
def test_clear_and_count(store): ...
```

- [ ] **Step 2: Failing app tests** (`test_exact_cache.py`, MemoryStore injected, upstream mocked and call-counted):

```python
async def test_second_identical_request_is_exact_hit(...):
    r1 = await post(REQ); r2 = await post(REQ)
    assert r1.headers["x-cache"] == "MISS" and r2.headers["x-cache"] == "HIT-EXACT"
    assert r2.json() == r1.json() and upstream_calls == 1

async def test_not_stored_when_finish_reason_length(...): ...   # second call is MISS
async def test_not_stored_on_upstream_error(...): ...
```

- [ ] **Step 3–4:** FAIL → implement MemoryStore + exact path in `main.py` (on MISS store with a zero vector placeholder until Task 6 supplies embeddings) → PASS.
- [ ] **Step 5: Commit** `feat: exact-match cache with MemoryStore`. **Step 6 (checkpoint):** what a Protocol is and why tests use MemoryStore.

---

### Task 5: Embedder

**Files:**
- Create: `gateway/embedder.py`, `tests/test_embedder.py`

**Interfaces:**
- Produces: `class Embedder: __init__(model_name: str)`, `embed(text: str) -> np.ndarray` (float32, shape (384,), unit norm). Tests elsewhere use `FakeEmbedder` in `tests/conftest.py` mapping known strings to fixed unit vectors.

- [ ] **Step 1: Failing tests** (mark `@pytest.mark.slow`, downloads model):

```python
def test_shape_and_unit_norm(emb): v = emb.embed("hi"); assert v.shape == (384,) and abs(np.linalg.norm(v) - 1) < 1e-5
def test_paraphrase_closer_than_unrelated(emb):
    a, b, c = (emb.embed(t) for t in ["How do I reset my password?", "Password reset steps?", "What's the weather?"])
    assert a @ b > a @ c
```

- [ ] **Step 2–4:** FAIL → implement with `SentenceTransformer(...).encode(text, normalize_embeddings=True)` → PASS.
- [ ] **Step 5: Commit** `feat: embedder`. **Step 6 (checkpoint):** why normalize makes cosine a dot product.

---

### Task 6: Semantic cache path

**Files:**
- Modify: `gateway/cache.py` (MemoryStore.nearest = brute-force dot product over unexpired entries with matching ctx), `gateway/main.py`
- Create: `tests/test_semantic_cache.py`

**Interfaces:**
- Consumes: `Embedder.embed`, `CacheStore.nearest/put`, `context_key`, `exact_key`, `Settings.tau_high`.
- Produces in `main.py`: `async def lookup(body) -> tuple[str, CacheEntry | None, float | None]` returning (`x_cache_value`, entry, similarity). Verifier hook added in Task 9.

- [ ] **Step 1: Failing tests** (FakeEmbedder with controlled similarities):

```python
async def test_paraphrase_above_tau_high_is_semantic_hit(...):   # sim 0.99
    assert r2.headers["x-cache"] == "HIT-SEMANTIC" and r2.headers["x-cache-similarity"] == "0.9900"
async def test_below_tau_high_is_miss_when_verifier_disabled(...):   # sim 0.95
async def test_same_text_different_system_prompt_is_miss(...):     # Review Focus 5
async def test_miss_stores_real_vector(...): ...
```

- [ ] **Step 2–4:** FAIL → implement → PASS (also run Tasks 1–4 tests).
- [ ] **Step 5: Manual:** with Ollama, ask "How do I reset my password?" then "how can i reset my password" and see headers with `curl -i`.
- [ ] **Step 6: Commit** `feat: semantic cache`. **Step 7 (checkpoint):** walk the request flow diagram (spec §5.1) against the code.

---

### Task 7: RedisStore + docker-compose

**Files:**
- Create: `gateway/redis_store.py`, `docker-compose.yml` (services: `redis` = `redis/redis-stack-server`, `gateway`), `Dockerfile`
- Modify: `tests/test_cache_contract.py` (add `RedisStore` to the parametrized fixture; skip if `REDIS_URL` not reachable), `gateway/main.py` (build `RedisStore` from settings by default)

**Interfaces:**
- Produces: `class RedisStore(CacheStore): __init__(url: str, dim: int = 384, index: str = "cache_idx")`; creates on init (if absent) an index `ON HASH PREFIX entry:` with `ctx TAG`, `vec VECTOR HNSW TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE`. `nearest` runs KNN 1 with filter `@ctx:{<ctx>}` and returns similarity = `1 - distance`. `put` writes `entry:{id}` hash + `exact:{key}` string → id, both with `EXPIRE ttl`. Response stored as JSON string, vector as `float32.tobytes()`.

- [ ] **Step 1:** `docker compose up -d redis`; run contract tests → RedisStore FAIL.
- [ ] **Step 2: Implement** → all contract tests PASS for both stores (TTL test for Redis uses `ttl=1` + `time.sleep(1.5)`, marked slow).
- [ ] **Step 3: Commit** `feat: Redis Stack store`. **Step 4 (checkpoint):** what HNSW is in one sentence, why TAG filter, why ctx contains only hex (no escaping issues).

---

### Task 8: Fail-open, /stats, admin clear

**Files:**
- Create: `gateway/stats.py`, `tests/test_ops.py`
- Modify: `gateway/main.py`

**Interfaces:**
- Produces: `class Stats: record(result: str, tokens_saved: int = 0, upstream_ms_saved: float = 0.0)`, `snapshot(entry_count: int) -> dict` with keys `counts` (per X-Cache value), `hit_rate` (hits / non-bypass requests), `tokens_saved`, `gpu_seconds_saved`, `entries`. In-process counters (reason line: single process; Prometheus is stretch).
- Endpoints: `GET /stats`; `DELETE /admin/cache` requires header `Authorization: Bearer <ADMIN_TOKEN>`, 401 otherwise, 403 if `admin_token` is empty (disabled).

- [ ] **Step 1: Failing tests:**

```python
async def test_store_error_fails_open(...):   # Review Focus 2: store.get_exact raises ConnectionError
    r = await post(REQ); assert r.status_code == 200 and r.headers["x-cache"] == "BYPASS"
async def test_embedder_error_fails_open(...): ...
async def test_stats_counts_and_savings(...):  # MISS then HIT-EXACT → tokens_saved == completion_tokens of entry
async def test_admin_clear_requires_token(...): ...
async def test_admin_clear_empties_store(...): ...
```

- [ ] **Step 2–4:** FAIL → implement (wrap lookup+store in try/except, log the error) → PASS.
- [ ] **Step 5: Commit** `feat: fail-open, stats, admin clear`. **Step 6 (checkpoint):** why "the cache must never take the gateway down".

---

### Task 9: Grey-zone verifier

**Files:**
- Create: `gateway/verifier.py`, `tests/test_verifier.py`
- Modify: `gateway/main.py` (`lookup`)

**Interfaces:**
- Produces: `class Verifier: __init__(model_name: str)`, `score(a: str, b: str) -> float` in [0, 1] (`CrossEncoder.predict`, sigmoid applied if the model returns logits — check once in the notebook and note in docstring). Tests use `FakeVerifier(score=...)`.
- Rule: if `tau_low <= sim < tau_high` and `verifier_enabled` and `score(new_text, entry.prompt) >= p_verify` → `HIT-VERIFIED`.

- [ ] **Step 1: Failing tests:**

```python
async def test_grey_zone_verified_hit(...):    # sim 0.92, verifier 0.9 → HIT-VERIFIED
async def test_grey_zone_rejected(...):        # sim 0.92, verifier 0.3 → MISS
async def test_below_tau_low_skips_verifier(...):  # sim 0.80 → MISS, verifier never called
def test_real_verifier_orders_pairs(...):      # slow: score(dup pair) > score("reset password","reset username")
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat: grey-zone verifier`. **Step 6 (checkpoint):** bi-encoder vs cross-encoder, and why the cross-encoder only runs in the grey zone.

---

### Task 10: Threshold study

**Files:**
- Create: `bench/datasets/hard_negatives.csv` (columns `a,b,label`; ~150 rows hand-written, label 0 for different meaning, plus some label-1 paraphrases), `bench/threshold_eval.py`, `bench/results/threshold_*.csv|png`

**Interfaces:**
- Consumes: `Embedder`, `Verifier`.
- `threshold_eval.py --qqp-n 5000 --seed 42` loads a QQP sample (HF `glue/qqp` validation), adds hard negatives, computes similarities once, then for τ in 0.70…0.99 (step 0.01) writes precision, recall, false-hit rate for (a) cosine ≥ τ and (b) cosine ≥ τ_high OR (τ ≤ cosine < τ_high AND verifier ≥ p) for p in {0.5, 0.7, 0.8, 0.9}. Plots precision–recall and false-hit-rate vs τ.

- [ ] **Step 1:** Unit test `tests/test_threshold_math.py`: `metrics(sims, labels, tau)` on a 6-item toy list returns hand-computed precision/recall/false-hit rate.
- [ ] **Step 2:** Implement `metrics` → PASS; implement script; run on Mac.
- [ ] **Step 3:** Choose `TAU_LOW` and `P_VERIFY` (rule: highest recall with false-hit rate ≤ 1% on the combined set); write the choice and the 0.98 numbers into `bench/results/threshold_summary.md`; update `.env.example`.
- [ ] **Step 4: Commit** `bench: threshold study`. **Step 5 (checkpoint):** Tanmay explains the plot as he would to the founder.

---

### Task 11: Workload replay + Kaggle vLLM benchmark

**Files:**
- Create: `bench/prepare_bitext.py`, `bench/replay.py`, `notebooks/kaggle_vllm_benchmark.ipynb`, `bench/results/replay_*.csv|png`, `bench/results/hand_check_sample.csv`

**Interfaces:**
- `prepare_bitext.py --n 2500 --seed 42` → `bench/datasets/stream.jsonl` rows `{text, intent}`; fills each `{{...}}` placeholder with a random realistic value (order numbers, names, dates).
- `replay.py --base-url URL --mode direct|gateway --out file.csv` sends requests **sequentially** (temperature 0, `max_tokens` 256), records per request: `text, intent, x_cache, similarity, latency_ms, completion_tokens, matched_prompt` (from response header `X-Cache-Matched-Prompt`, URL-encoded, which the gateway adds only when setting `debug_headers: bool = False` is true; add the field to `Settings` and a test in `tests/test_semantic_cache.py`).
- Summary function `summarize(csv) -> dict`: hit rate by type, bypass rate, p50/p95 latency by type, tokens and GPU-seconds avoided, intent-mismatch false-hit rate; plus rolling hit rate plot.

- [ ] **Step 1:** Unit test `summarize` on a 10-row fixture CSV with hand-computed numbers → implement → PASS.
- [ ] **Step 2: Kaggle smoke test (do this early, week 1):** notebook starts vLLM, Redis Stack server binary, uvicorn gateway; one request works through the gateway. If Redis fails, set store to MemoryStore and note it.
- [ ] **Step 3:** Run three modes on T4: direct, gateway pure-0.98 (`VERIFIER_ENABLED=false`), gateway tuned. Overhead: run `replay.py` once with an empty cache and fresh unique prompts, comparing direct vs gateway miss latency.
- [ ] **Step 4:** Sample 200 hits into `hand_check_sample.csv`; Tanmay labels `wrong_for_user` 0/1. Final false-hit rate = max(intent-mismatch rate, hand-check rate).
- [ ] **Step 5: Commit** results + notebook. **Step 6 (checkpoint):** explain why sequential replay is needed for honest GPU-seconds.

---

### Task 12: README, diagram, hiring package

**Files:**
- Create/modify: `README.md`, `docs/architecture.md` (Mermaid diagram of spec §5.1)

- [ ] **Step 1:** README sections in order: one-line pitch; the problem; results table (only numbers from `bench/results/`); threshold plot; architecture diagram; how it works (matching rule, skip rules, thresholds, verifier); run in one command (`docker compose up` + Ollama); benchmark reproduction; limitations; future work (stretch list + Qdrant split).
- [ ] **Step 2:** Verify every number in README against result files (grep each).
- [ ] **Step 3:** Fresh-clone test: clone into a temp dir, follow README exactly, confirm a HIT-SEMANTIC.
- [ ] **Step 4: Commit** `docs: README and architecture`.
- [ ] **Step 5 (final checkpoint):** mock interview — Tanmay walks any file picked at random, plus the 8 interview answers in spec §8. Then draft the LinkedIn post and message to the founder.
