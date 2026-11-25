# Semantic AI Gateway & Cache — Design

Date: 2026-11-25 · Owner: Tanmay Kumar · Status: **approved** (trimmed version)

## 0. Guiding principle: explainable end to end

Tanmay must be able to open any file in an interview and explain every line. So:
no abstraction or library without a one-sentence reason; one idea per module; built
phase by phase, and each phase ends with Tanmay explaining the new code back before
moving on. Claude Code writes code; Tanmay owns understanding it.

## 1. Intent and why

**Problem.** LLM inference is expensive and slow: every request uses GPU time, and
answers take seconds. Real traffic repeats the same *meaning* in different *words*
("How do I reset my password?" / "Password reset steps?"). An exact-match cache misses
these, so the GPU regenerates the same answer.

**Solution.** An OpenAI-compatible gateway between apps and the inference server. It
embeds each prompt, looks for a previously answered prompt with the same meaning (in
an otherwise identical request), and if one is close enough it returns the stored
answer without touching the GPU. Misses go to the LLM and the answer is stored. Apps
adopt it by changing only `base_url`.

**Why.** Ravi A Samy (co-founder, stealth AI startup) listed 12 inference-infra problems
and said he'll hire in Jan 2027. This is #9: *embed prompts, check for >98% vector
similarity, return cached outputs without hitting the GPU.* His list is about GPU
economics, so the project must read as **inference engineering that removes GPU work,
proven with measurements**.

**Success** = public repo + write-up that (a) runs with one command, (b) reports real
numbers measured against vLLM on a GPU, (c) engages with his 98% figure using data,
(d) Tanmay can explain line by line. Target: sent to the founder by ~20 Dec 2026.

## 2. Goals, non-goals, anti-goals

**Goals (v1)**
- `POST /v1/chat/completions` (non-streaming cached; streaming passed through) and `GET /v1/models`.
- Exact-match cache, then semantic cache, then grey-zone verifier.
- Safety rules: reuse an answer only where it is valid.
- Threshold chosen from a measured precision/recall curve.
- `GET /stats` JSON; benchmark on vLLM/T4 with plots; README with diagram and results.

**Stretch (only after all v1 phases are done)**
1. Prometheus `/metrics` + Grafana dashboard.
2. Caching streamed responses (replay stored answer as SSE chunks).
3. Concurrent-load benchmark (throughput / wall-clock with vs without gateway).
4. Per-user cache scope, finer admin invalidation.

**Non-goals**: multi-model routing, auth/rate limiting, UI, training models, Kubernetes,
caching tool/function calls.

**Anti-goals**
- **Silently serving a wrong answer.** Precision over hit rate, always.
- **Unmeasured numbers.** Every README/resume figure comes from a committed run.
- **Slowing down misses** noticeably. Overhead (embed + search) is measured and reported; target p95 < 25 ms on CPU.
- **Code Tanmay can't explain.**

## 3. Constraints

- Dev: Mac + Ollama (`qwen2.5:1.5b`). Benchmark: vLLM on free Kaggle/Colab **T4**
  (`Qwen/Qwen2.5-1.5B-Instruct`, `--dtype half`; T4 has no bf16). Only the upstream URL changes.
- Kaggle has no Docker: the benchmark notebook runs Redis Stack server binary, vLLM and
  the gateway as processes in one notebook. Week-1 smoke test; if Redis Stack won't run
  there, use the in-memory store for that run and say so in the write-up.
- Free tools, Python 3.11+, ~4 weeks part-time. Embedding on CPU.

## 4. Approaches (decided)

| Option | Trade-off | Decision |
|---|---|---|
| **A. Redis Stack** — vector search + storage + TTL in one service | Learn RediSearch query syntax | **Chosen** |
| B. FAISS in-process + SQLite | Lost on restart, not shareable, hand-rolled TTL; reads as a toy | Rejected |
| C. Qdrant (vectors) + Redis (answers) | Two stores to keep in sync | **Future scale path** |

Storage sits behind a small `CacheStore` interface (Redis implementation + in-memory
implementation for tests), so C is a new class, not a rewrite.

## 5. Design

### 5.1 Request flow

```
request ─► [1] skip rules? ── yes ─► forward to LLM, don't store        X-Cache: BYPASS
             │ no
             ▼
           [2] context_key = sha256(request with last user message removed)
           [3] exact_key   = sha256(context_key + normalized last user message)
             ├─ found ─► return stored answer                          X-Cache: HIT-EXACT
             ▼
           [4] embed last user message (bge-small, normalized)
           [5] nearest stored prompt with same context_key → similarity s
             ├─ s ≥ τ_high ─► return                                   X-Cache: HIT-SEMANTIC
             ├─ τ_low ≤ s < τ_high and verifier says "same" ─► return    X-Cache: HIT-VERIFIED
             ▼
           [6] forward to LLM ─► return                                X-Cache: MISS
               if 200 and finish_reason == "stop": store (exact key, vector, answer) with TTL
```

Semantic responses also carry `X-Cache-Similarity`.

### 5.2 The one matching rule (context key)

Two requests may share an answer **only if everything except the wording of the last
user message is identical**: same model, temperature, system prompt, earlier turns,
max_tokens, etc. Implementation: copy the request, remove the last user message text,
serialize with sorted keys, sha256 it. Vector search is filtered by this key (Redis TAG
field), so it can never cross models, system prompts or conversations.

### 5.3 Skip rules (bypass lookup and store)

- `stream: true` (passed through unchanged in v1).
- `temperature > 0.3` (configurable): caller wants variety.
- `n > 1`, or `tools`/`functions` present.
- Header `Cache-Control: no-cache`.
- Time-sensitive words: "today", "now", "latest", "current", "this week", explicit dates.
- **Specific identifiers** *(decided with Tanmay)*: order/ticket/booking numbers, long
  digit or alphanumeric IDs, emails, phone numbers, unfilled `{{...}}` placeholders.
  Generic wording like "how do I cancel my order?" is **not** skipped: same answer for
  everyone. A cached answer about order #48213 would be wrong for #55190, so prompts with
  specific facts are skipped.

Patterns live in one config file. Store only on HTTP 200 with `finish_reason == "stop"`.

### 5.4 Similarity and thresholds

Embeddings are L2-normalized, so cosine similarity is just a dot product:

$$\text{sim}(A,B)=\frac{A\cdot B}{\lVert A\rVert\,\lVert B\rVert}=A\cdot B\qquad(\lVert A\rVert=\lVert B\rVert=1)$$

Redis returns cosine distance $d=1-\text{sim}$; the gateway converts.

- $\tau_{high}=0.98$ by default (the founder's number) → direct hit.
- $\tau_{low}=0.85$ to start, replaced by the tuned value from §6.1 → grey zone below $\tau_{high}$.
- Verifier: `cross-encoder/quora-distilroberta-base` scores (new prompt, cached prompt);
  hit if score ≥ $p_{verify}$ (default 0.8, tuned in §6.1). Can be turned off ("pure 0.98" mode).

### 5.5 Storage (Redis Stack)

- `exact:{sha256}` → entry id, with TTL.
- Hash `entry:{id}`: `ctx` (TAG), `vec` (FLOAT32 384-d, HNSW, COSINE), `prompt`,
  `response` (OpenAI JSON), `created_at`, `upstream_ms`, `completion_tokens`. TTL default 24 h.
- `upstream_ms` and `completion_tokens` of the original generation let each later hit
  count GPU time and tokens saved.

### 5.6 Endpoints

`POST /v1/chat/completions`, `GET /v1/models` (passthrough), `GET /healthz`,
`GET /stats` (JSON: counts per X-Cache result, hit rate, tokens saved, GPU-seconds saved,
entry count), `DELETE /admin/cache` (clears all; protected by `ADMIN_TOKEN`).

### 5.7 Config (`.env`)

`UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, `REDIS_URL`, `EMBED_MODEL`, `TAU_HIGH`,
`TAU_LOW`, `VERIFIER_ENABLED`, `P_VERIFY`, `MAX_CACHE_TEMPERATURE`, `CACHE_TTL_SECONDS`,
`ADMIN_TOKEN`.

### 5.8 Stack

Python 3.11, FastAPI + uvicorn, httpx, sentence-transformers (`BAAI/bge-small-en-v1.5`,
cross-encoder), redis-py + Redis Stack, pydantic-settings, pytest (fake upstream +
in-memory store), matplotlib for plots, docker-compose with two services (redis-stack,
gateway). Ollama runs natively on the Mac. Embedding runs in a thread so the server
isn't blocked.

### 5.9 Repo layout

```
semantic-cache-gateway/
  gateway/   main.py  config.py  rules.py  keys.py  embedder.py  verifier.py
             cache.py (CacheStore + RedisStore + MemoryStore)  upstream.py  stats.py
  tests/
  bench/     threshold_eval.py  replay.py  datasets/  results/
  notebooks/ kaggle_vllm_benchmark.ipynb
  docker-compose.yml  .env.example  README.md
```

## 6. Proving it works

### 6.1 Threshold study — "what does 98% actually buy you?"
- Data: Quora Question Pairs sample (~5k labelled pairs) + ~150 hand-written hard
  negatives (similar text, different meaning: "reset password" vs "reset username",
  "cancel order" vs "track order", negations, swapped numbers).
- For thresholds 0.70→0.99: precision, recall, false-hit rate, for cosine alone and
  cosine + verifier. Plot both. Pick $\tau_{low}$ and $p_{verify}$.

### 6.2 Workload replay on vLLM (Kaggle T4)
- Data: Bitext customer-support dataset (≈27k queries with intent labels). Fill
  `{{Placeholder}}` slots with random values so identifier rules are tested. Sample
  ~2–3k queries, random order. Cache starts empty (cold start); plot rolling hit rate.
- Runs, **one request at a time**: (a) direct to vLLM, (b) gateway pure-0.98,
  (c) gateway tuned + verifier.
- Report: hit rate by type, bypass rate, p50/p95 latency hit vs miss, tokens and
  GPU-seconds generated vs avoided, $ saved computed in README from a stated T4 price.
- **False-hit rate, two ways** *(decided with Tanmay)*: automatic intent-label mismatch,
  and a hand-checked sample of ~200 hits (labels committed as CSV). Quote the stricter.

### 6.3 Overhead
p50/p95 of embed + lookup (+ verifier) on misses, on Mac and Kaggle CPU.

All scripts, raw results and plots committed under `bench/results/`.

## 7. Phases

| # | Phase | Est. |
|---|---|---|
| 0 | Learn: embeddings, cosine, HNSW, OpenAI format (tiny notebook) | 1 day |
| 1 | Pass-through proxy + tests | 1 day |
| 2 | Keys + exact cache (MemoryStore, then Redis) | 1–2 days |
| 3 | Semantic cache | 2 days |
| 4 | Skip rules, TTL, `/stats`, admin clear | 1–2 days |
| 5 | Grey-zone verifier | 1 day |
| 6 | Threshold study | 2 days |
| 7 | Kaggle vLLM benchmark | 2 days |
| 8 | README, diagram, results, post, message to founder | 2 days |
| — | Stretch (§2), only if time remains | — |

Each phase ends with: tests pass, a commit, and Tanmay explains the new code back.

## 8. Hiring package

- README: problem → results table → threshold curve → architecture diagram → run in one command.
- Resume bullet (fill only with measured numbers): "Built an OpenAI-compatible semantic
  cache gateway (FastAPI, Redis vector search, bge-small) in front of vLLM; on a
  customer-support workload it served X% of requests from cache at Y ms p95 vs Z s for
  generation, avoiding N GPU-seconds, at a measured F% false-hit rate."
- Interview answers to prepare: why exact caching fails; how the threshold was chosen;
  why 0.98 is safe but low-yield; temperature/personal/time-sensitive prompts; stale
  answers (TTL + clear); overhead on misses; scaling (stateless gateway replicas, Redis
  cluster → Qdrant + Redis); how this complements vLLM prefix caching (that saves
  prefill; this saves the whole generation).

## 9. Left to the builder

Exact regexes, library versions, dataset sample sizes and seeds, and small internal
helpers within the files listed in §5.9.
