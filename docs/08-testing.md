# 8. Testing — Verifying Every Layer of RAGBook

In the previous chapters we covered all the major building blocks: document loading ([chapter 3](03-loading-documents.md)), embeddings and vector search ([chapter 4](04-embeddings-and-vector-store.md)), hybrid retrieval ([chapter 5](05-tfidf-and-hybrid-search.md)), LLM answer generation ([chapter 6](06-llm-and-generation.md)), and the REST API ([chapter 7](07-api-and-interface.md)). This chapter explains how the test suite is structured, what each group of tests covers, and the design decisions behind it.

## 8.1 Overview

RAGBook uses **pytest** with **pytest-asyncio** for all tests. No external services are needed — FAISS and SQLite run in-memory or in temporary directories, and heavy dependencies are mocked only where necessary.

```
tests/
├── conftest.py                  # Shared fixtures (entities, mocks)
├── unit/
│   ├── test_entities.py         # Domain entities and enums
│   ├── test_ingest_use_case.py  # IngestUseCase
│   ├── test_search_use_case.py  # SearchUseCase
│   ├── test_collection_use_case.py
│   ├── test_text_loader.py
│   ├── test_csv_loader.py
│   ├── test_pdf_loader.py
│   ├── test_recursive_chunker.py
│   ├── test_semantic_chunker.py
│   ├── test_sklearn_tfidf.py
│   ├── test_faiss_store.py
│   ├── test_vector_search.py
│   ├── test_tfidf_search.py
│   ├── test_hybrid_search.py
│   ├── test_cross_encoder_reranker.py
│   └── test_result_diversifier.py
└── integration/
    ├── conftest.py              # FastAPI test app fixture
    ├── test_ingest_router.py
    ├── test_search_router.py
    ├── test_collection_router.py
    └── test_health.py
```

Run the full suite with:

```bash
pytest tests/ -v
```

## 8.2 Shared Fixtures (`tests/conftest.py`)

All fixtures that multiple test modules need live in a single root-level `conftest.py`. This avoids duplication and keeps each test file focused on assertions.

The key fixtures are:

| Fixture | What it provides |
|---|---|
| `sample_document` | A `Document` entity with realistic fields |
| `sample_chunks` | Three `Chunk` objects with pre-set embeddings |
| `sample_collection` | A `Collection` entity |
| `sample_search_query` | A `SearchQuery` with a question and `top_k=5` |
| `sample_search_results` | Two `SearchResult` objects linking chunks to scores |
| `mock_loader` | `AsyncMock` of `DocumentLoaderPort` |
| `mock_chunker` | `AsyncMock` of `ChunkerPort` |
| `mock_embedding` | `AsyncMock` — `embed()` returns three float lists, `embed_query()` returns one |
| `mock_vector_store` | `AsyncMock` of `VectorStorePort` |
| `mock_tfidf` | `AsyncMock` of `TfidfPort` |
| `mock_llm` | `AsyncMock` — `generate()` returns a fixed answer string |
| `mock_metadata_store` | `AsyncMock` of `MetadataStorePort` |

The `mock_embedding` fixture is worth a closer look: it pre-configures the return values so that unit tests for `IngestUseCase` and `SearchUseCase` do not need to set them up themselves:

```python
mock.embed.return_value = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]]
mock.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
```

## 8.3 Unit Tests

### 8.3.1 Domain Entities (`test_entities.py`)

These tests verify that the core data structures behave correctly in isolation, with no infrastructure involved.

Covered scenarios:
- Creating a `Document` with all fields or only the required ones (auto-generated `id` and `created_at`).
- Creating a `Chunk` with and without an embedding.
- Creating `SearchResult` and `Collection`.
- `SearchQuery` defaults: `strategy` is `None` unless explicitly set.
- `DocumentType` and `SearchStrategy` enums carry the expected string values.

### 8.3.2 Use Cases

#### IngestUseCase (`test_ingest_use_case.py`)

The ingest pipeline has five steps: load, chunk, embed, store vectors, fit TF-IDF, save metadata. The tests verify each step is invoked correctly and that the orchestration logic is sound.

Key tests:

- **`test_execute_success`** — the happy path. All mocks are called once with the expected arguments, and the returned `(document, chunks)` tuple matches what the mocks returned.
- **`test_execute_with_collection_id`** — when a `collection_id` is passed, it is assigned to `document.collection_id` as a UUID.
- **`test_execute_sets_embeddings_on_chunks`** — verifies that `embed()` is called with the correct list of chunk texts. Note: after ingestion, embeddings are cleared from the returned chunks (a deliberate memory optimization — the vector store already holds them). The test asserts this post-ingestion state.
- **`test_execute_calls_save_on_vector_store`** — verifies that `vector_store.save()` is called once, persisting the index to disk.

#### SearchUseCase (`test_search_use_case.py`)

- **`test_execute_returns_answer_and_results`** — verifies the return type is `(str, list[SearchResult])`.
- **`test_execute_passes_context_to_llm`** — the chunk contents retrieved by the search are assembled into a context string and forwarded to the LLM.
- **`test_execute_no_results`** — when the search returns nothing, the LLM is still called (with empty context) and a string answer is returned.
- **`test_execute_raw_returns_results_without_llm`** — `execute_raw()` skips the LLM entirely and returns only the list of `SearchResult` objects.

#### CollectionUseCase (`test_collection_use_case.py`)

- **`test_create_collection`** — delegates to the metadata store and returns the created `Collection`.
- **`test_list_collections`** — returns whatever the metadata store provides.
- **`test_delete_collection_cleans_all_stores`** — deleting a collection must touch all three stores: vector store, TF-IDF index, and metadata store. The test verifies all three mocks are called.

### 8.3.3 Infrastructure: Loaders

All loader tests create real temporary files using pytest's `tmp_path` fixture — no mocking of file I/O.

#### TextLoader (`test_text_loader.py`)

- `.txt` and `.md` files are loaded correctly.
- Missing files raise `FileNotFoundError`.
- Metadata contains `filename` and `size`.

#### CsvLoader (`test_csv_loader.py`)

- A 3-row CSV is loaded and formatted as `"column: value"` lines.
- Metadata includes `num_rows` and `columns`.
- An empty CSV (header only) is handled gracefully.

#### PdfLoader (`test_pdf_loader.py`)

PDF loading is tested by mocking `PdfReader` directly (creating a real minimal PDF is fragile and format-dependent). The mock simulates two pages and verifies:

- `content` is the concatenation of both pages.
- Metadata contains `filename`, `num_pages`, and `page_range`.
- A non-existent file path raises `FileNotFoundError`.

### 8.3.4 Infrastructure: Chunker

#### RecursiveChunker (`test_recursive_chunker.py`)

Uses `langchain_text_splitters.RecursiveCharacterTextSplitter` under the hood. Tests use real text — no mocking.

- A document shorter than `chunk_size` produces exactly one chunk.
- A long document produces multiple chunks.
- Every chunk carries the `document_id` of the source document.
- Chunk `index` values are `0, 1, 2, ...` (sequential).
- When `chunk_size=50`, no chunk's content exceeds 50 characters.

#### SemanticChunker (`test_semantic_chunker.py`)

This is the **default** chunker (`CHUNKER=semantic`); it picks a splitting strategy per file type (see [chapter 3](03-loading-documents.md)). Tests use real text — no mocking.

- **`test_markdown_splits_on_headings`** — a `.md` document is split on heading boundaries.
- **`test_python_splits_on_function_boundaries`** — a `.py` file is split on function/class boundaries.
- **`test_js_file_uses_js_splitter`** — a `.js`/`.ts` file uses the JS language splitter.
- **`test_plain_text_falls_back_to_default`** — an unrecognised type falls back to the recursive splitter.
- **`test_chunk_metadata_contains_filename`** — chunk metadata carries the source `filename`.
- **`test_chunk_indexes_are_sequential`** — chunk `index` values are `0, 1, 2, ...`.

### 8.3.5 Infrastructure: TF-IDF (`test_sklearn_tfidf.py`)

`SklearnTfidf` is tested with real `scikit-learn` objects and a real `tmp_path` index directory.

- **`test_fit_and_search`** — after fitting three thematically distinct chunks, a query about Python programming returns the Python-related chunk as the top result.
- **`test_search_empty_collection`** — searching a collection that was never fitted returns an empty list immediately.
- **`test_fit_creates_index_files`** — fitting creates three pickle files in the collection subdirectory: `vectorizer.pkl`, `matrix.pkl`, and `chunks.pkl`.
- **`test_search_result_scores_normalized`** — all returned scores are in `[0.0, 1.0]`.

### 8.3.6 Infrastructure: FAISS Vector Store (`test_faiss_store.py`)

Tests use real FAISS indices with 8-dimensional unit-norm numpy vectors.

- **`test_add_and_search`** — after adding three chunks, searching with the first chunk's embedding returns it as the top result.
- **`test_search_empty_index`** — searching an uninitialised collection returns `[]`.
- **`test_save_and_load`** — the index is serialized to `tmp_path`, a new store instance loads it, and a search still returns the correct result.
- **`test_delete_collection`** — after deletion, searching that collection returns `[]`.
- **`test_multiple_collections`** — chunks in `colA` and `colB` are stored independently; searching `colA` never returns chunks from `colB`.

### 8.3.7 Infrastructure: Search Strategies

#### VectorSearch (`test_vector_search.py`)

- `embed_query()` is called, its result is passed to `vector_store.search()`.
- Every `SearchResult` in the response has `source="vector"`.

#### TfidfSearch (`test_tfidf_search.py`)

- `tfidf.search()` is called with the correct query and `collection_id`.
- Every result has `source="tfidf"`.

#### HybridSearch (`test_hybrid_search.py`)

This is the most behaviorally rich group of unit tests.

- **`test_hybrid_combines_results`** — results from vector search and TF-IDF search are merged into a single ranked list.
- **`test_hybrid_respects_weights`** — with `vector_weight=1.0`, the output is identical to vector-only search.
- **`test_hybrid_deduplicates_by_chunk_id`** — if the same chunk appears in both result sets, it appears exactly once in the output, with its score combining contributions from both sources.
- **`test_hybrid_fallback_single_source`** — if one source returns no results, the other's results are used as-is.
- **`test_normalize_scores`** — the `_normalize_scores` helper maps scores linearly to `[0, 1]`.
- **`test_normalize_scores_same_value`** — when all scores are equal, normalization returns `1.0` for all (no division by zero).
- **`test_normalize_scores_empty`** — an empty list returns an empty list.

### 8.3.8 Infrastructure: Cross-Encoder Reranker (`test_cross_encoder_reranker.py`)

`CrossEncoderReranker` re-scores candidate chunks against the query with a cross-encoder (the model is mocked so no weights are loaded).

- **`test_pairs_constructed_correctly`** — the reranker builds `(query, chunk_content)` pairs and hands them to the cross-encoder.
- **`test_results_reordered_by_cross_encoder_score`** — results are re-ordered by the cross-encoder score, not the original retrieval score.
- **`test_top_k_respected`** — only `top_k` results are returned after reranking.
- **`test_empty_input_returns_empty`** — an empty candidate list returns an empty list (the model is never called).

### 8.3.9 Application: Result Diversifier (`test_result_diversifier.py`)

`result_diversifier` caps the number of chunks per document and cuts to `top_k` (see [chapter 9](09-retrieval-optimization.md), OPT-4). These are synchronous tests.

- **`test_all_results_from_same_document`** — when every candidate comes from one document, only the per-document cap survives.
- **`test_results_from_different_documents`** — results spread across documents are preserved.
- **`test_duplicate_content_dropped`** — chunks with duplicate content are dropped.
- **`test_top_k_limit`** — the output is cut to `top_k`.
- **`test_empty_input`** — an empty list returns an empty list.

## 8.4 Integration Tests

Integration tests exercise the full FastAPI request/response cycle. All use cases are mocked at the dependency-injection level — no real infrastructure is started.

### 8.4.1 Test App Fixture

`tests/integration/conftest.py` provides a `test_app` fixture that:

1. Creates a minimal `FastAPI` application with the three API routers (ingest, search, collections).
2. Adds a `/api/health` endpoint.
3. Overrides every dependency (`get_search_use_case`, `get_collection_use_case`, etc.) with pre-configured mocks.
4. Clears all overrides via `yield` + `app.dependency_overrides.clear()` after each test.

Tests use `httpx.AsyncClient` with `ASGITransport` to send real HTTP requests to the in-process ASGI app without binding a port:

```python
async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
    response = await client.post("/api/ingest", files={"file": ("doc.txt", b"hello", "text/plain")})
```

### 8.4.2 Ingest Router (`test_ingest_router.py`)

- **`test_ingest_txt_file`** — `POST /api/ingest` with a `.txt` file returns `200` with `document_id`, `filename`, and `num_chunks`.
- **`test_ingest_unsupported_extension`** — `.xyz` extension returns `400` with `"Unsupported file type"` in the detail.
- **`test_ingest_with_collection_id`** — the `collection_id` form field is forwarded to `IngestUseCase.execute()`.

### 8.4.3 Search Router (`test_search_router.py`)

- **`test_search_returns_answer_and_sources`** — `POST /api/search` returns `200` with an `answer` string and a `sources` list.
- **`test_search_raw_returns_sources_only`** — `POST /api/search/raw` returns `200` with only `sources` (no `answer`).
- **`test_search_with_strategy`** — the `strategy` field in the request body is parsed and passed to the use case.

### 8.4.4 Collection Router (`test_collection_router.py`)

- `POST /api/collections` — returns `201` with `id` and `name`.
- `GET /api/collections` — returns `200` with a list.
- `DELETE /api/collections/{id}` — returns `200`.

### 8.4.5 Health (`test_health.py`)

- `GET /api/health` — returns `200` with `{"status": "ok"}`.

## 8.5 Design Decisions

### Why real implementations for unit tests?

Where the library is available in the environment (FAISS, scikit-learn, langchain text splitters), unit tests use the real implementations with small in-memory or `tmp_path`-backed stores. This catches real edge cases — for example, FAISS index format details, pickle compatibility for TF-IDF state, and actual chunking behavior — that mocks would silently hide.

Mocks are reserved for ports (interfaces) when testing use cases, and for heavy side effects (LLM calls, file uploads in routers).

### Why not stub all third-party libraries for integration tests?

The integration test conftest no longer injects `MagicMock` substitutes for installed packages like FAISS, scikit-learn, or `pypdf`. Doing so at module collection time caused those stubs to persist into unit tests (Python caches module objects in `sys.modules`), breaking tests that need the real implementations. Since all dependencies are installed in the virtual environment, there is no reason to stub them.

Only `langchain_community` — a package not present in the environment — is stubbed.

### Embeddings are cleared after ingestion

`IngestUseCase` deliberately sets `chunk.embedding = None` after calling `vector_store.add()`. The vector store already owns the embedding; keeping a second copy in the chunk would waste memory. Tests for the ingest use case explicitly verify this post-ingestion state rather than expecting embeddings to persist on the returned chunks.
