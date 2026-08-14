# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAGBook — a RAG (retrieval-augmented generation) service: FastAPI backend + Gradio UI for ingesting documents (text/PDF/CSV/image), searching them (vector / TF-IDF / hybrid), and answering questions via an LLM (Gemini or Ollama).

## Code comments

All code comments must always be written in English, regardless of the language used in the conversation.

## Commands

```bash
# Environment (Python 3.12, venv already in .venv)
source .venv/bin/activate
pip install -r requirements-dev.txt

# Dependencies are managed with pip-tools: edit requirements.in /
# requirements-dev.in, then regenerate the pinned .txt files:
pip-compile requirements.in
pip-compile requirements-dev.in

# Docker (no local deps required — see README "Run with Docker")
cp -n .env.example .env                                 # then set OLLAMA_BASE_URL=http://ollama:11434
docker compose up                                       # full stack (app + Ollama), host port 8001 (APP_PORT; container listens on 8000)
docker compose up app                                   # app only (Ollama on host or Gemini)
docker compose exec ollama ollama pull qwen3:4b         # pull model once; must match LLM_MODEL

# Run the server (reads .env; auto-reload when DEBUG=true)
python run.py
# or: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Tests (async tests use @pytest.mark.asyncio; no pytest config file)
pytest
pytest tests/unit/
pytest tests/integration/
pytest tests/unit/test_hybrid_search.py::test_name   # single test

# Architectural contract check (5 contracts in pyproject.toml)
lint-imports

# Postman E2E — isolated test server: port 8010, data under data/test/
# (never touches the dev server on 8000 or data/). See postman/README.md.
bash postman/run_test_server.sh
python postman/build_collection.py   # regenerate the collection from postman/src/

# Retrieval-quality evaluation (server must be running with documents ingested)
python -m evaluation.evaluate --collection-id ID [--api-url URL] [--top-k N] [--strategy vector|tfidf|hybrid]
python -m evaluation.compare evaluation/results/<before>.json evaluation/results/<after>.json
```

Configuration comes from `.env` (see `.env.example`). `LLM_PROVIDER` and `LLM_MODEL` are required — validated at startup via `settings.validate_llm()` (raises `ConfigurationError` if either is unset or unknown). The API serves at `http://HOST:PORT/api` (health check: `/api/health`); the Gradio UI is mounted on the same server at `/ui`. The embedding model loads in a background task at startup (first boot downloads it from Hugging Face via `src/infrastructure/embeddings/model_prefetch.py`), so the port opens immediately. The warm-up runs in two phases: `downloading` (byte-level prefetch with progress) → `loading` (model initialisation). `/api/health` reports four fields: `embedding_status` (`warming|ready|error`), `embedding_phase` (`downloading|loading|null`), `embedding_progress` (`{downloaded_bytes, total_bytes, percent}` or `null` — populated only during the download phase).

## Architecture

Clean/hexagonal architecture; dependencies point inward (`ui → api → application → domain`). `application` and `infrastructure` are mutually independent — neither imports from the other. Both rules are enforced by `lint-imports` (5 contracts in `pyproject.toml`). Detailed docs live in `docs/` (numbered 01–10, plus `100-best-practices-and-troubleshooting.md`).

- **`src/domain/`** — entities (`Document`, `Chunk`, `SearchResult`, `Collection`, `SearchQuery`), enums, `query_mode.py` (`parse_mode`, `QUERY_MODES` — the `/mode` prefix parser used by both application and LLM adapters), and abstract **ports** (`src/domain/ports/`): async ABCs for loader, chunker, embedding, vector store, TF-IDF, LLM, reranker, metadata store, search. The domain imports nothing from outer layers.
- **`src/application/`** — use cases (ingest, search, collection, document) that orchestrate only through ports.
- **`src/infrastructure/`** — concrete adapters, one subpackage per port: vector stores (FAISS/Chroma/Pinecone/Qdrant), LLMs (Gemini/Ollama), chunkers (recursive/semantic/CSV), loaders, lexical search behind `TfidfPort` (sklearn `SklearnTfidf` by default, `Bm25Lexical` with `LEXICAL_BACKEND=bm25`; BM25 indexes live in `data/bm25_indexes/` — switching backend with existing collections leaves the index empty, re-ingest required), cross-encoder reranker, SQLite metadata, search strategies.
- **`src/api/`** — FastAPI app and routers. **`src/api/dependencies.py` is the composition root**: `@lru_cache` factory functions pick adapters based on `settings` (e.g. `VECTOR_STORE`, `LLM_PROVIDER`, `CHUNKER`, `SEARCH_STRATEGY`, `RERANKER_ENABLED`, `FUSION`, `LEXICAL_BACKEND`) and expose `Annotated` type aliases for FastAPI DI. Adding a new adapter means implementing the port and wiring it here. It is therefore the **only** file under `src/api/` allowed to import from `src/infrastructure/` — every other file there must depend on ports and injected aliases. Consequence for the layered agent rules: `ai-feature-builder.config.json` maps both `layers.presentation` and `layers.composition` to `src/api`, so `ai-fb-check-arch` reports this file's infrastructure imports as a direction violation. That single finding is expected — do not "fix" it by rewriting the imports.
- **`src/ui/`** — Gradio Blocks UI (tabs in `src/ui/tabs/`). It does **not** call use cases directly; it goes through `ApiClient`, which calls the REST API over HTTP.
- **`src/api/embedding_state.py`** — thread-safe in-process state for the embedding warm-up; mutated by the prefetch/load thread, read by `/api/health`.
- **`src/config/settings.py`** — pydantic-settings singleton `settings`, loaded from `.env`.
- **Pinecone adapter** (`src/infrastructure/vector_stores/pinecone_store.py`): on startup, `_ensure_index` checks `has_index`; creates a serverless index (`ServerlessSpec(cloud, region)`, metric cosine) if absent, or raises `ValueError` on dimension mismatch. Configured via `PINECONE_CLOUD` (default `aws`) and `PINECONE_REGION` (default `us-east-1`). Manual smoke-test procedure: `docs/pinecone-smoke-test.md`.

### Key flows

- **Ingest** (`IngestUseCase`): dedup check (SHA-256 content hash against `metadata_store.find_document_by_content_hash`) → if duplicate in the same collection, skip pipeline, return existing document with `already_ingested: true`; otherwise load → chunk → embed → vector store add + TF-IDF fit → SQLite metadata. CSV documents always use `CsvChunker` regardless of the configured chunker. Both ingest and search routers call `check_model_compatibility(collection, model, dim)` — returns 409 if the collection's embedding model/dimension differs from current settings.
- **Search** (`SearchUseCase.execute`): over-fetch (4× `top_k` with reranker, 3× without) → optional cross-encoder rerank of *all* candidates → filter by `min_score` (on reranker sigmoid scores when the reranker is enabled — default `RERANK_MIN_SCORE`; on raw retrieval scores otherwise — default `MIN_SCORE`) → `diversify_results` (caps per-document results, cuts to `top_k`) → LLM generates a grounded answer from chunk contents, citing sources as `[n]` inline; declares explicitly when information is not in the documents. `execute_raw` is the same pipeline without the LLM step. The search API requires a valid `collection_id` (router returns 400/404 otherwise); there is no "all collections" search. Queries may start with a `/mode` prefix (see `src/domain/query_mode.py`) that controls answer style — it is stripped before retrieval and re-parsed by the LLM adapters. Eight per-request tuning parameters are carried in `SearchQuery` via two value objects (`RetrievalTuning` for retrieval knobs: `fusion`, `vector_weight`, `max_results_per_document`, `reranker_enabled`; `LlmOptions` for generation: `temperature`, `think`, `num_ctx`) — assembled in `_build_search_query` in the search router, which resolves each `None` field against the server default. The effective `min_score` default is reranker-aware: `RERANK_MIN_SCORE` when the *effective* reranker flag (per-request override wins over global setting) is `true`, otherwise `MIN_SCORE`. `get_reranker_port()` always constructs the cross-encoder adapter (model weights are lazy-loaded on first use); the on/off decision is a per-request `reranker_enabled` field in `RetrievalTuning`, resolved in `SearchUseCase`. `GET /api/config` exposes the effective defaults and valid ranges so the UI can populate its Advanced accordion without duplicating constants.
- **Streaming search** (`SearchUseCase.execute_stream`): same retrieval pipeline as `execute`, then calls `LLMPort.generate_stream` (async generator on the LLM port) instead of `generate`. Exposed as `POST /api/search/stream` (SSE); the event sequence is: one `sources` event (chunks list) → N `delta` events (answer tokens) → one `done` event. If the LLM raises mid-stream an `error` event is emitted and the stream closes. The Search tab in the UI consumes this endpoint via `ApiClient.search_stream` and renders the answer incrementally.
- **Hybrid search** merges vector and lexical scores via `FUSION=weighted|rrf`: `weighted` computes a max-norm linear combination weighted by `HYBRID_VECTOR_WEIGHT` (default 0.7); `rrf` applies Reciprocal Rank Fusion (k=60). The lexical branch is served by `LEXICAL_BACKEND=tfidf|bm25` (see `src/infrastructure/` above). `SearchDispatcher` routes per-query strategy overrides. **`min_score` semantics without reranker**: RRF raw scores are ~1/(60·n) and BM25 scores are unbounded — the default `MIN_SCORE=0.15` may filter out all results; lower it (e.g. 0.01) when `RERANKER_ENABLED=false`. With the reranker active (default), filtering runs on sigmoid scores and is unaffected.

### Testing conventions

- Unit tests mock ports with `AsyncMock`; shared entity fixtures live in `tests/conftest.py`.
- Integration tests (`tests/integration/`) test routers with mocked use cases; their `conftest.py` stubs modules missing from the environment via `sys.modules`.

### Retrieval tuning workflow

When changing retrieval parameters (min_score, weights, reranker, chunking), run the evaluation baseline before the change, re-run after, and compare — see `evaluation/README.md` for targets (recall 1.0, avg_score > 0.45, latency < 500ms).
