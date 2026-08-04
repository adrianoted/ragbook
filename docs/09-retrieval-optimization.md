# Retrieval Optimization

## Diagnosis

The system works end-to-end: documents are ingested, indexed, and queried. But retrieval quality showed measurable problems:

| Symptom | Observed | Expected |
|---------|----------|----------|
| Avg top-5 score | 0.11 - 0.14 | > 0.40 |
| Unique docs in top-5 | 1-2 out of 5 | 4-5 out of 5 |
| Chunk content | Mixed code+text, truncated mid-sentence | Semantically coherent blocks |

### How scores were computed (before optimization)

```
User query
    |
    +---> Embedding (all-MiniLM-L6-v2, 384 dim)
    |         |
    |         v
    |     Vector Search (cosine similarity) --> raw score [0, 1]
    |
    +---> TF-IDF (sklearn TfidfVectorizer)
    |         |
    |         v
    |     Cosine Similarity --> raw score [0, 1]
    |
    v
Hybrid Merge
    +-- Min-max normalization (independent per set)
    +-- Weight: 0.7 x vector_norm + 0.3 x tfidf_norm
    +-- Sort descending --> top_k results
```

**Normalization problem**: `_normalize_scores` used min-max on each batch independently. If all vector scores were 0.10-0.14, after normalization they became 0.0-1.0. This masked the fact that *no result was truly relevant* — a chunk with raw score 0.10 became 0.0, while 0.14 became 1.0, an enormous gap for a negligible real difference.

### Root causes

1. **Chunk size 500, overlap 50**: chunks too short, splitting semantic context mid-sentence. Embeddings captured only fragments of meaning.

2. **Lightweight embedding model** (`all-MiniLM-L6-v2`, 22M params, 384 dim): optimized for speed, not precision. Max context 256 tokens (~350 words) — longer chunks silently truncated. Weak on technical text, code, and non-English languages.

3. **No relevance filtering**: `search_use_case.py` passed all results to the LLM regardless of score. The LLM received near-random context and generated responses that *seemed* authoritative because they cited the document.

4. **No result diversity**: the same document, split into N chunks, could have M chunks weakly matching the query. Result: 4 out of 5 results from the same file, zero informational diversity.

---

## Optimization plan

All optimizations are implemented. Ordered by impact/complexity ratio, each was applied and measured independently.

```
Phase 1 - Quick wins (no new dependencies)                      DONE
+-- OPT-1: Chunk size + overlap                                 DONE
+-- OPT-2: Score threshold                                      DONE
+-- OPT-3: Score normalization fix                               DONE
+-- OPT-4: Result deduplication by document                      DONE
    |
    v  Checkpoint: avg score > 0.30
    |
Phase 2 - Model upgrade (requires re-ingestion)                  DONE
+-- OPT-5: Embedding model -> Qwen/Qwen3-Embedding-4B + MPS     DONE
+-- OPT-6: Semantic chunker for markdown/code                    DONE
    |
    v  Checkpoint: avg score > 0.45
    |
Phase 3 - Reranking + Evaluation (new components)                DONE
+-- OPT-7: Cross-encoder reranker                                DONE
+-- OPT-8: Evaluation framework                                  DONE
```

---

## Phase 1 - Quick wins

### OPT-1: Chunk size and overlap

**Impact**: high | **Complexity**: low | **Files**: `src/config/settings.py`

**Problem**: with `chunk_size=500` and `chunk_overlap=50`, an 800-character paragraph was split into 2 chunks, each with half the meaning. The 50-character overlap (10%) was insufficient for semantic continuity.

**Change**: `chunk_size=1000`, `chunk_overlap=200`.

- **1000 characters**: ~150-200 words, enough to contain a full paragraph. Fits within the context window of Qwen3-Embedding-4B (up to 32K tokens).
- **200 overlap (20%)**: border concepts appear in both adjacent chunks.

> Changing chunk size requires document re-ingestion. Existing chunks in the vector store retain the old dimensions.

### OPT-2: Score threshold

**Impact**: medium-high | **Complexity**: low | **Files**: `src/config/settings.py`, `src/domain/entities.py`, `src/application/search_use_case.py`, `src/api/routers/search_router.py`

**Problem**: the use case passed all top-k results to the LLM without filtering. If the best result had score 0.14, the LLM received near-random context and generated responses based on irrelevant fragments.

**Change**: a relevance threshold in settings and `SearchQuery`, applied at the use case layer before passing context to the LLM. Also configurable per-request via the API (`min_score` in `SearchRequest`).

If no results survive the threshold, the system returns "No sufficiently relevant results found." instead of hallucinating from bad context.

> **Evolution (2026-07)**: retrieval scores proved to be a poor base for an *absolute*
> threshold — hybrid scores are max-normalized per batch (see OPT-3), so the top result
> scores ~1.0 even for a completely off-topic query. The threshold now applies to the
> cross-encoder sigmoid scores **after** reranking (`RERANK_MIN_SCORE=0.3`, calibrated by
> comparing golden queries vs off-topic probes: relevant chunks score 0.9–1.0, garbage
> 0.0–0.1). `MIN_SCORE=0.15` remains as the fallback filter when the reranker is disabled.

### OPT-3: Score normalization fix

**Impact**: medium | **Complexity**: low | **Files**: `src/infrastructure/search/hybrid_search.py`

**Problem**: `_normalize_scores` used min-max normalization, always mapping the batch to [0, 1] and destroying absolute relevance information.

```python
# Before: min-max (broken)
scores [0.10, 0.12, 0.14] -> [0.0, 0.5, 1.0]

# After: max-score normalization
scores [0.10, 0.12, 0.14] -> [0.71, 0.86, 1.0]
```

**Change**: replaced with `score / max_score`. This preserves proportions — a low raw score stays low after normalization. Works in combination with OPT-2: normalized scores are more meaningful, so the threshold filters correctly.

### OPT-4: Result deduplication by document

**Impact**: medium | **Complexity**: low | **Files**: `src/config/settings.py`, `src/application/result_diversifier.py`, `src/application/search_use_case.py`

**Problem**: when multiple chunks from the same document matched weakly, they monopolized the top-k results (e.g., 4 out of 5 from the same file). No informational diversity — the user never discovered relevant content from other documents.

**Change**: post-retrieval diversification with `max_results_per_document=2`. The **use case** over-fetches (`overfetch_factor = 4 if reranker else 3` — so **4x** with the reranker enabled, which is the default, **3x** otherwise), then the diversifier limits chunks per document and selects the top_k most diverse results. This over-fetch happens at the use case layer, not the search layer.

---

## Phase 2 - Model upgrade

### OPT-5: Embedding model -> Qwen/Qwen3-Embedding-4B + MPS

**Impact**: high | **Complexity**: medium | **Files**: `src/config/settings.py`, `src/infrastructure/embeddings/sentence_transformer_embedding.py`

**Problem**: `all-MiniLM-L6-v2` (22M params, 384 dim) was designed for speed, not accuracy:
- Max context 256 tokens — longer chunks silently truncated
- Weak on technical text, code, and non-English languages
- 384 dimensions limit the ability to distinguish semantic nuances

**Model comparison** (considered during selection):

| Model | Params | Dim | Max tokens | Multilingual | Notes |
|-------|--------|-----|------------|--------------|-------|
| `all-MiniLM-L6-v2` (old) | 22M | 384 | 256 | No | Fast, low accuracy |
| `all-mpnet-base-v2` | 110M | 768 | 384 | No | 2x more accurate |
| `intfloat/multilingual-e5-large` | 560M | 1024 | 512 | Yes | Good for mixed IT/EN |
| `Qwen/Qwen3-Embedding-4B` (chosen) | 4B | 2560 | 32K | Yes (100+ langs) | #1 on MTEB multilingual |

**Change**: `embedding_model="Qwen/Qwen3-Embedding-4B"` with MPS device support for Apple Silicon.

- Device detection: CUDA -> MPS -> CPU (added `torch.backends.mps.is_available()`)
- 4B params, 2560-dim embeddings, #1 MTEB multilingual
- ~8GB RAM in fp16 (fits on M1 Pro 32GB)
- First run downloads ~8GB, subsequent runs use cache
- Requires `transformers>=4.51.0`, `sentence-transformers>=2.7.0`

**Infrastructure impact**:
- All existing vector indexes must be deleted and documents re-ingested (dimension changes from 384 to 2560)
- FAISS indexes recreated, Qdrant collections recreated with new `VectorParams`
- Ingestion ~5-10x slower due to model size. Single query latency adds ~100-200ms (acceptable)

### OPT-6: Semantic chunker for markdown and code

**Impact**: medium-high | **Complexity**: medium | **Files**: `src/config/settings.py`, `src/infrastructure/chunkers/semantic_chunker.py`, `src/api/dependencies.py`

**Problem**: `RecursiveCharacterTextSplitter` splits by character count, ignoring document structure. Markdown files get cut mid-heading, code files mid-function.

**Change**: `chunker="semantic"` in settings. The `SemanticChunker` respects document structure:

| Extension | Strategy | Separators |
|-----------|----------|------------|
| `.md`, `.markdown` | Heading-aware | `\n## `, `\n### `, `\n#### `, `\n\n`, `\n`, `" "` (space fallback) |
| `.py` | Language-aware | `Language.PYTHON` boundaries (function/class) |
| `.js`, `.ts` | Language-aware | `Language.JS` boundaries (function) |
| Other | Default | `RecursiveCharacterTextSplitter` standard |

Selected automatically in `src/api/dependencies.py` via `get_chunker()`. Configurable: `"recursive"` or `"semantic"`.

---

## Phase 3 - Reranking + Evaluation

### OPT-7: Cross-encoder reranker

**Impact**: high | **Complexity**: medium

**Problem**: embedding-based retrieval (bi-encoder) encodes query and document *separately*, then compares vectors. This is fast but imprecise — it does not capture interactions between the query and document words.

A cross-encoder takes query + document *together* and produces a direct relevance score. 10-100x slower, but much more accurate.

#### Architecture

Follows the project's port/adapter pattern:

| Layer | File | Role |
|-------|------|------|
| Domain | `src/domain/ports/reranker_port.py` | Abstract interface `RerankerPort` |
| Infrastructure | `src/infrastructure/rerankers/cross_encoder_reranker.py` | `CrossEncoderReranker` implementation |
| Config | `src/config/settings.py` | `reranker_enabled`, `reranker_model` |
| Application | `src/application/search_use_case.py` | Pipeline integration |
| DI | `src/api/dependencies.py` | `get_reranker_port()` + wiring |
| Test | `tests/unit/test_cross_encoder_reranker.py` | 4 unit tests |

#### Updated search pipeline

```
search(top_k x 4)  ->  rerank(ALL)  ->  score filter  ->  diversify  ->  LLM
      ^                    ^                 ^                ^
  Over-fetch         Cross-encoder     rerank_min_score   Max 2/doc,
  (if reranker       reorders all      >= 0.3 on sigmoid  cuts to top_k
   enabled: 4x,      candidates,       scores (OPT-2      (OPT-4)
   otherwise 3x)     no cut            evolution)
```

The pattern is **retrieve-then-rerank**: over-fetch more candidates than needed, then reorder with a more precise model. The reranker does **not** trim to `top_k`: the diversifier needs the full reordered list to backfill when a document exceeds its per-document cap, and the final cut to `top_k` happens there.

#### Model

- **Default**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, fast)
- Lazy-loading with `threading.Lock` (double-checked locking)
- Automatic device detection: CUDA -> MPS -> CPU
- Prediction in separate thread via `asyncio.to_thread()`

**Alternative models**:

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | Good | Recommended to start |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Medium | Better | Good compromise |
| `BAAI/bge-reranker-large` | Slow | Excellent | State of the art |

#### Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable/disable the reranker |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |

To disable the reranker, in `.env`:
```
RERANKER_ENABLED=false
```

---

### OPT-8: Evaluation framework

Framework to measure retrieval quality before and after optimizations. Full documentation: [docs/10-evaluation.md](10-evaluation.md).

---

## Expected impact summary

| Optimization | Expected avg score | Latency | Dependencies |
|-------------|-------------------|---------|--------------|
| Baseline (before) | 0.12 | ~100ms | - |
| + OPT-1 (chunk size) | 0.25 | ~100ms | Re-ingestion |
| + OPT-2 (threshold) | 0.25 (filtered) | ~100ms | - |
| + OPT-3 (normalization) | 0.30 | ~100ms | - |
| + OPT-4 (dedup) | 0.30 + diversity | ~100ms | - |
| + OPT-5 (embedding) | 0.45 | ~200ms | Re-ingestion |
| + OPT-6 (semantic chunker) | 0.45+ | ~200ms | Re-ingestion |
| + OPT-7 (reranker) | 0.55+ | ~350ms | `sentence-transformers` |

> Expected scores are estimates based on public benchmarks (MTEB, MS MARCO) and known RAG literature patterns. Actual values depend on document content and queries.

## Measured results (2026-07-07, full pipeline)

Corpus: 2 LangGraph guides, 150 chunks (chunk 1000/200, semantic chunker), 10 golden queries,
`evaluation.evaluate --strategy hybrid --collection-id <id>`. Scores are cross-encoder sigmoid
(post-rerank scale, not comparable with the retrieval-scale estimates above).

| Metric | Before post-rerank filter | After (`RERANK_MIN_SCORE=0.3`) |
|--------|--------------------------|-------------------------------|
| avg_score | 0.664 | **0.989** (garbage tails filtered out) |
| avg_recall | 1.000 | **1.000** |
| avg_keyword_hit_rate | 0.913 | 0.873 (fewer but cleaner chunks) |
| latency (warm query) | ~350ms | **~350ms** |
| Off-topic queries (5 probes) | 4-5 results each, all noise | **0 results → fallback message** |

### Tuning the threshold

`RERANK_MIN_SCORE=0.3` was calibrated on this corpus: cross-encoder sigmoid scores are
bimodal (relevant chunks 0.9–1.0, off-topic 0.0–0.1), so there is wide margin on both sides.
If behaviour drifts after changing corpus, embedding model, or reranker model:

- System **too strict** (answers "No sufficiently relevant results found." on legitimate
  queries) → lower `RERANK_MIN_SCORE` to `0.2`.
- System **too permissive** (answers built on irrelevant chunks) → raise it to `0.5`.
- To re-calibrate properly: run `python -m evaluation.evaluate` for the relevant-score
  distribution, probe a few off-topic queries via `/api/search/raw` for the garbage
  distribution, and pick a value in the gap between the two.
