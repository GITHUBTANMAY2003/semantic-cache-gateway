# Semantic AI Gateway & Cache: handoff (end of Task 1, 2026-11-25)

Paste this into a new chat to continue. The full design and plan are in the project folder:
`~/semantic-cache-gateway/docs/specs/design.md` and `docs/plans/plan.md`.

## Who / why
- I'm Tanmay (ECE, BIT Mesra), moving toward AI engineering roles.
- Ravi A Samy (co-founder, stealth AI startup) posted 12 inference-infra problems on LinkedIn
  and said he will hire in Jan 2027. I picked **#9: Semantic AI Gateway & Cache**: "embed
  incoming prompts, check for >98% vector similarity, return cached outputs without hitting
  the GPU. The cheapest and fastest token is the one you never generate."
- Target: ship the repo + write-up and message him by ~20 Dec 2026.

## How we work (important)
- I build with **Claude Code (desktop app)** on my Mac. The chat assistant writes one
  **prompt per task** for me to paste into Claude Code; before each task it tells me any
  setup I must do myself, step by step.
- **No vibe code:** I must be able to explain every line in an interview. Minimal,
  plain code; docstrings on every module; no extra abstractions.
- Claude Code must **not run git** (I want to be the only contributor; I'll commit myself).
- Test-first in every task. After each task: a walkthrough, then 3 interview questions.
  For each, I may ask for **hints first**, then the **model answer**.
- Explain everything so I can say it to an interviewer: plain words, the why, an
  interview-ready answer. Maths in rendered LaTeX.

## Decisions (approved)
- Stack: Python 3.12, FastAPI, httpx, sentence-transformers (**bge-small-en-v1.5**, 384-d,
  CPU), **Redis Stack** for vectors + answers + TTL (behind a `CacheStore` interface;
  future scale path: Qdrant for vectors + Redis for answers), cross-encoder verifier
  (`cross-encoder/quora-distilroberta-base`), pytest + respx.
- Hardware "option B": build on Mac with **Ollama `qwen2.5:1.5b`**; final benchmark with
  **vLLM on a free Kaggle/Colab T4** (`Qwen/Qwen2.5-1.5B-Instruct --dtype half`).
- Matching rule: two requests share an answer only if **everything except the wording of
  the last user message is identical** (context key = hash of the request minus that text).
- Thresholds: τ_high = **0.98** (founder's number) → direct hit; grey zone 0.85–0.98 →
  verifier decides; below → miss. τ_low and p_verify tuned from data in Task 10.
- Skip rules: stream, temperature > 0.3 (missing = 1.0), n > 1, tools, no-cache header,
  no usable user text, time-sensitive words, **specific identifiers** (order/ticket IDs,
  emails, phones, `{{placeholders}}`). Generic "how do I cancel my order?" IS cacheable.
- Store only on HTTP 200 + `finish_reason == "stop"`; TTL 24 h; fail open if Redis or the
  embedder breaks (forward to LLM, `X-Cache: BYPASS`).
- `X-Cache` header values: BYPASS, HIT-EXACT, HIT-SEMANTIC, HIT-VERIFIED, MISS.
- Trimmed scope: **no Prometheus/Grafana, no stream caching, no concurrent benchmark,
  no per-user scope** until all phases are done (stretch only). `/stats` JSON instead.
- Benchmark: Quora pairs + ~150 hand-written hard negatives for the threshold study;
  Bitext customer-support queries (placeholders filled with random values) replayed
  **sequentially**, cold cache, three modes (direct / pure 0.98 / tuned + verifier).
  False-hit rate = stricter of intent-mismatch and a hand-checked sample of ~200 hits.

## Environment (done)
- MacBook Air M1, 8 GB RAM, ~50 GB free. Close Chrome tabs while working.
- Ollama app 0.40.18 (old Homebrew copy removed), `qwen2.5:1.5b` pulled; API at
  `http://localhost:11434/v1` verified. No Ollama account needed.
- Python 3.12 via Homebrew (`python3.12`); system `python3` is 3.9 and must not be used.
- Project: `~/semantic-cache-gateway` with `.venv` (3.12.16), opened in Claude Code
  desktop and VS Code (notebook kernel: `.venv`).
- Docker/Redis not installed yet (needed at Task 7).

## Task 0: done (notebooks/00_basics.ipynb)
- Embedded 6 support questions → 6 × 384 normalized vectors; cosine = dot product.
- Key finding: real paraphrases scored **0.83–0.92**; "reset password" vs "reset
  **username**" scored **0.863**, higher than a real paraphrase (0.834). So 0.98 alone
  gives almost no hits, and no single threshold separates right from wrong → tuned
  threshold + verifier.
- Timing on M1: embedding ~10 ms vs LLM ~3,750 ms (~368×). The check adds <1% to a miss.
  Not the full hit speed-up (lookup not included).
- Ollama's `cached_tokens` = its own prefix/KV cache (saves prefill). Ours skips the
  whole generation (the 166 completion tokens).

## Task 1: done (pass-through proxy)
Files: `requirements.txt` (10 packages), `pytest.ini`, `.gitignore`, `.env.example`,
`gateway/__init__.py`, `gateway/config.py` (Settings: upstream_base_url, upstream_api_key,
upstream_timeout_seconds=120), `gateway/upstream.py` (`forward_chat` → (status, json,
elapsed_ms), `forward_models`, `open_stream` + `relay_stream`), `gateway/main.py`
(`create_app(settings, http_client)` factory, shared AsyncClient in lifespan, routes
`POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`; timeout → 504,
connection error → 502; upstream 4xx/5xx passed through), `tests/conftest.py`,
`tests/test_proxy.py` (**11 tests pass**). Live check against Ollama worked.

Understood and can explain: async/await; why streaming is split into open (get the status
first) + relay (stream the body, close in `finally`), because the status is sent first and
can't change; 429 is a reply (passes through) vs timeout is no reply (exception → 504);
the 120 s timeout is per wait, not total; one shared client = connection pooling;
`http_client` injection exists for tests (lifespan doesn't run under ASGITransport).

Known limits from Task 1: non-JSON upstream reply → `.json()` crashes → 500 (**fix first
in Task 2: return 502 with a test**); most headers not forwarded; mid-stream failure
can't change the status.

## Next: Task 2 (keys + exact cache)
Per plan.md Task 2: `gateway/keys.py` with `last_user_text`, `normalize`, `context_key`,
`exact_key` + tests. Start the Task 2 prompt with the small non-JSON → 502 fix (test first).
Then Task 3 skip rules, Task 4 CacheStore/MemoryStore + exact cache wired in, and so on.
Note: plan.md's "Spec:" line points to docs/superpowers/specs/...; the real path is
docs/specs/design.md.
