# 4. Embeddings and Vector Stores — Turning Text into Numbers

In [chapter 3](03-loading-documents.md) we saw how documents get loaded and split into chunks. But chunks are still plain text, and computers are much better at comparing numbers than words. This chapter explains how RAGBook transforms text into numerical vectors (embeddings) and stores them in a way that makes similarity search fast and accurate.

## 4.1 What Is an Embedding?

Imagine a huge map where every sentence occupies a specific point. Sentences with similar meaning sit close together, while unrelated sentences are far apart. An **embedding** is the set of coordinates that places a piece of text on this map.

Technically, an embedding is a list of floating-point numbers (a vector) — around 2560 dimensions for the model RAGBook uses by default (`Qwen/Qwen3-Embedding-4B`). Each dimension captures some abstract aspect of meaning, learned during training. You cannot interpret a single dimension, but the overall position encodes semantic content remarkably well.

Why does this matter? Because once two chunks are represented as vectors, we can measure how "close" they are with a simple mathematical operation — **cosine similarity** — instead of trying to match keywords.

## 4.2 Sentence Transformers: Free, Local Embeddings

RAGBook uses the `sentence-transformers` library with the **Qwen3-Embedding-4B** model (`Qwen/Qwen3-Embedding-4B`, set via `EMBEDDING_MODEL`) as its default embedding engine. This model runs entirely on your machine — no API key, no network, no per-call cost.

### 4.2.1 The Model: Qwen3-Embedding-4B

This is a 4-billion-parameter embedding model from the Qwen3 family, trained to produce high-quality multilingual sentence embeddings. It outputs **2560-dimensional** vectors and handles long inputs (thousands of tokens). It favors retrieval quality over footprint: it is much heavier than small MiniLM-class models, so a GPU (CUDA or Apple Silicon MPS) is recommended for fast ingestion — though it still runs on CPU.

To trade quality for speed and memory, set `EMBEDDING_MODEL` to a lighter model such as `all-MiniLM-L6-v2` (384-dimensional vectors, 256-token inputs), which is small enough to run comfortably on a laptop CPU.

### 4.2.2 Lazy Loading and Device Detection

The model is loaded only when the first embedding is requested, not at startup. This keeps application boot fast. The implementation also picks the best available device — NVIDIA CUDA, then Apple Silicon (MPS), then CPU:

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
self._model = SentenceTransformer(self._model_name, device=device)
```

A `threading.Lock` ensures that concurrent requests don't accidentally load the model twice (double-check locking pattern).

### 4.2.3 Batch vs. Single Embedding

The `EmbeddingPort` interface defines three methods:

- `embed(texts)` — encodes a list of texts in one batch (efficient for ingestion)
- `embed_query(text)` — encodes a single query (delegates to `embed` internally)
- `dimension()` — returns the vector size the model produces (synchronous; used to size vector-store collections)

Batching matters: encoding 100 chunks at once is dramatically faster than encoding them one by one, because the model can leverage matrix parallelism.

## 4.3 FAISS: The Local Vector Warehouse

**FAISS** (Facebook AI Similarity Search) is a library designed for fast nearest-neighbor search in high-dimensional vector spaces. RAGBook uses it as its default vector store.

### 4.3.1 How Similarity Search Works

When you search for similar vectors, FAISS compares your query vector against every stored vector and returns the closest ones. RAGBook uses `IndexFlatIP` — a flat index with **inner product** distance. With normalized vectors, inner product equals cosine similarity.

The search flow:

1. Normalize the query vector (L2 normalization)
2. Compute inner product against all stored vectors
3. Return the top-K highest scores

### 4.3.2 Cosine Similarity via Inner Product

A crucial detail: FAISS's `IndexFlatIP` computes **inner product**, not cosine similarity directly. But if all vectors are L2-normalized (unit length), the inner product *equals* cosine similarity. That's why RAGBook normalizes every vector before insertion:

```python
arr = np.array(vectors, dtype=np.float32)
faiss.normalize_L2(arr)
```

This is done both at insertion time and at query time, ensuring consistent results.

### 4.3.3 Collections and Multi-Tenancy

RAGBook supports multiple **collections** — logical groups of documents. FAISS handles this by maintaining a separate index per collection, stored in memory as:

```python
self._indexes: dict[str, faiss.IndexFlatIP]   # one index per collection
self._mappings: dict[str, dict[str, Chunk]]    # chunk data by ID
self._id_lists: dict[str, list[str]]           # insertion order
```

The ID list preserves the insertion order so that FAISS's integer-based results (position 0, 1, 2...) can be mapped back to chunk UUIDs.

### 4.3.4 Thread Safety

FAISS is not thread-safe. RAGBook wraps all index operations in a `threading.Lock()` to prevent concurrent read/write corruption.

### 4.3.5 Persistence: Saving and Loading Indexes

FAISS indexes live in memory. To survive restarts, RAGBook serializes them to disk with the following structure:

```
data/faiss_indexes/{collection_id}/
    index.faiss      # binary FAISS index
    mappings.json    # chunk metadata (Pydantic → JSON)
    id_list.json     # ID ordering
```

On startup, `load()` scans this directory and reconstructs all indexes. The separation into three files keeps each concern independent: the raw vectors, the metadata, and the ordering.

## 4.4 ChromaDB as an Alternative

**ChromaDB** is an embedding database that handles both storage and search. It's a higher-level alternative to FAISS, trading some control for convenience.

Key differences from FAISS:

| Aspect | FAISS | ChromaDB |
|--------|-------|----------|
| Persistence | Manual (save/load) | Automatic (`PersistentClient`) |
| Metadata | Custom JSON files | Built-in metadata storage |
| Thread safety | Manual locking | Handled internally |
| Score | Cosine similarity directly | Distance → converted to score |

ChromaDB returns **distances** (lower is better), while RAGBook's interface expects **similarity scores** (higher is better). RAGBook configures each Chroma collection to use cosine space (`hnsw:space=cosine`), so the distance is a cosine distance (`1 - cosine similarity`). The conversion back to a score is therefore:

```python
score = 1.0 - distance
```

Cosine space keeps Chroma's scores on the same scale as FAISS (normalized inner product = cosine similarity), so `min_score` thresholds behave consistently across stores.

ChromaDB's `save()` and `load()` are essentially no-ops, since the `PersistentClient` writes changes to disk automatically.

## 4.5 Pinecone: Cloud-Managed Vector Search

For production deployments or large-scale datasets, RAGBook supports **Pinecone** — a fully managed cloud vector database.

### 4.5.1 Namespaces for Collections

Pinecone uses **namespaces** to logically separate collections within a single index. This maps cleanly to RAGBook's collection concept without creating multiple Pinecone indexes (which would require separate paid plans).

### 4.5.2 Batch Upsert and Metadata Limits

Pinecone has a per-request size limit, so RAGBook sends vectors in batches of 100. It also enforces a 40KB metadata limit per vector — content is truncated if necessary:

```python
encoded = content.encode("utf-8")
if len(encoded) <= max_bytes:
    return content
return encoded[:max_bytes].decode("utf-8", errors="ignore")
```

Only primitive types (str, int, float, bool) are stored in metadata — nested structures are filtered out.

### 4.5.3 No Local Persistence

Since Pinecone is cloud-managed, `save()` and `load()` are no-ops. Data persists on Pinecone's servers, surviving application restarts without any local state.

### 4.5.4 Error Handling

The Pinecone adapter catches `PineconeApiException` specifically, logging detailed errors for rate limiting, authentication failures, or network issues. Generic exceptions are caught separately to avoid silencing unexpected errors.

## 4.6 Qdrant: High-Performance Vector Search, Local or Cloud

**Qdrant** is an open-source vector database written in Rust, designed for high-performance similarity search. It can run locally via Docker or as a fully managed cloud service.

### 4.6.1 Local or Cloud — Same Configuration

Qdrant's adapter works identically in both environments. The only difference is the URL and API key:

| Environment | URL | API Key |
|-------------|-----|---------|
| Local (Docker) | `http://localhost:6333` | Not required |
| Qdrant Cloud | `https://<cluster-id>.aws.cloud.qdrant.io:6333` | Required |

To run Qdrant locally with Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

To use Qdrant Cloud:

1. Create an account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster (a free tier with 1GB is available)
3. Copy the cluster URL and generate an API key from the dashboard

Configure via `.env`:

```
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333        # or your cloud URL
QDRANT_API_KEY=                          # required for cloud, empty for local
```

### 4.6.2 Collections as First-Class Citizens

Unlike Pinecone (which uses namespaces within a single index), Qdrant creates a **separate collection** for each `collection_id`. Collections are auto-created on first use with cosine distance and a vector size matching the embedding model (2560 dimensions for the default `Qwen/Qwen3-Embedding-4B`).

```python
VectorParams(size=self._vector_size, distance=Distance.COSINE)
```

This means each collection is fully isolated — it can be deleted independently without affecting others.

### 4.6.3 Filtering by Document

Qdrant stores `document_id` in each point's payload, enabling efficient filtered operations. When deleting a document's chunks, the adapter uses a payload filter rather than fetching and listing IDs:

```python
self._client.delete(
    collection_name=collection_id,
    points_selector=Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    ),
)
```

### 4.6.4 No Manual Persistence

Like Pinecone, Qdrant handles persistence automatically — both locally (data is written to disk) and in the cloud. The `save()` and `load()` methods are no-ops.

### 4.6.5 Qdrant vs. Other Vector Stores

| Aspect | FAISS | ChromaDB | Pinecone | Qdrant |
|--------|-------|----------|----------|--------|
| Deployment | Local only | Local only | Cloud only | Local + Cloud |
| Persistence | Manual | Automatic | Managed | Automatic |
| Collection isolation | Separate indexes | Separate collections | Namespaces (shared index) | Separate collections |
| Filtering | Not built-in | Built-in | Built-in | Built-in (payload filters) |
| Language | C++ (Python bindings) | Python | SaaS | Rust |
| Free tier | Always free | Always free | Limited | Local free, Cloud 1GB free |

## 4.7 When Embeddings Are Not Enough

Embeddings capture **semantic meaning** — they understand that "car" and "automobile" are similar. But they struggle with:

- **Exact matches**: searching for a specific product code like "SKU-4872"
- **Proper nouns**: names, acronyms, or domain-specific jargon not well represented in the model's training data
- **Rare terms**: the model may map uncommon words to generic regions of vector space

This is where TF-IDF comes in — a complementary approach that excels at exact and keyword-based matching. The next chapter explores how RAGBook combines both strategies for the best of both worlds.

---

Next: [Chapter 5 — TF-IDF and Hybrid Search](05-tfidf-and-hybrid-search.md) | Previous: [Chapter 3 — Loading Documents](03-loading-documents.md)
