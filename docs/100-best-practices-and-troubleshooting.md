# 100 - Best Practices and Troubleshooting

## Best Practices

### 1. Do not block the event loop with CPU-bound operations

FastAPI/uvicorn use an asynchronous event loop: if an `async def` function executes blocking (CPU-bound) code, the entire server freezes until the operation completes. No other requests are served and active connections can die.

**Rule:** any heavy operation (encoding, numerical computations, synchronized I/O) must be executed in a separate thread.

```python
# WRONG - blocks the event loop
async def embed(self, texts: list[str]) -> list[list[float]]:
    model = self._get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [row.tolist() for row in embeddings]

# CORRECT - executes in a separate thread
async def embed(self, texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(self._encode_sync, texts)

def _encode_sync(self, texts: list[str]) -> list[list[float]]:
    model = self._get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [row.tolist() for row in embeddings]
```

**When to use `asyncio.to_thread()`:**
- Embedding encoding (sentence-transformers)
- FAISS/TF-IDF operations (fit, search)
- Reading/writing large files
- Any call to libraries that are not async-native

### 2. Process large data in batches

Loading thousands of chunks into memory at once (with their embeddings) can cause a silent OOM (Out Of Memory): the operating system terminates the process without any visible errors in the terminal.

**Rule:** when processing N elements where N can be large, work in batches and free memory after each batch.

```python
# WRONG - all chunks and embeddings in memory at once
chunks = await self._chunker.chunk(document)
texts = [chunk.content for chunk in chunks]
embeddings = await self._embedding.embed(texts)       # 12k embeddings in RAM
for chunk, emb in zip(chunks, embeddings):
    chunk.embedding = emb
await self._vector_store.add(chunks, coll_id)          # duplicated in FAISS

# CORRECT - batch processing with memory release
BATCH_SIZE = 500

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    texts = [chunk.content for chunk in batch]
    embeddings = await self._embedding.embed(texts)
    for chunk, emb in zip(batch, embeddings):
        chunk.embedding = emb

    await self._vector_store.add(batch, coll_id)

    # Release embeddings - the vector store has already copied them
    for chunk in batch:
        chunk.embedding = None

await self._vector_store.save()
```

**Why Python embeddings consume so much memory:**
- Each Python `float` takes ~28 bytes (not 4 like in numpy)
- 12,000 chunks × 384 dimensions × 28 bytes = ~130 MB just for embeddings (illustrative — the old `all-MiniLM-L6-v2` at 384 dim; the current default `Qwen/Qwen3-Embedding-4B` uses ~2560 dim, so the real figure is several times larger)
- Plus copies in FAISS mappings, plus the model in memory
- Total easily exceeds 500 MB, which can cause OOM

**Other tips:**
- Free `document.content` after chunking (the original text is no longer needed)
- The internal embedding phase can also use batching (see `BATCH_SIZE = 256` in `SentenceTransformerEmbedding`)

### 3. Configure uvicorn reload correctly

With `reload=True`, uvicorn monitors files on disk and restarts the server when something changes. If it also monitors the data directories (`data/`), any ingestion operation (which writes indexes, uploads, database entries) causes a reload that kills active connections.

**Rule:** limit the watcher to the source code directory only.

```python
# WRONG - uvicorn monitors everything, including data/
uvicorn.run("src.api.app:app", reload=True)

# CORRECT - uvicorn monitors only src/
uvicorn.run("src.api.app:app", reload=True, reload_dirs=["src"])
```

---

## Troubleshooting

### The server process terminates without visible errors

**Symptom:** `python run.py` terminates with `[Process completed]` without traceback or error messages.

**Cause:** OOM killer from the operating system. macOS and Linux silently terminate processes that consume too much memory.

**Diagnosis on macOS:**
```bash
# Search the system log
log show --predicate 'eventMessage contains "Killed"' --last 5m
```

**Solution:** reduce memory consumption with batch processing and explicit resource release after use.

### Ingestion of large files is very slow

**Symptom:** a file of a few MB takes many minutes to ingest.

**Causes:**
- Embedding on CPU is the main bottleneck (~12k chunks require several minutes even with a small model like the old `all-MiniLM-L6-v2` on CPU; the current default `Qwen/Qwen3-Embedding-4B` is far heavier — use GPU/MPS where possible)
- TF-IDF refit on many chunks is expensive

**Mitigations:**
- Logs in the terminal show batch-by-batch progress (`Batch 1/25 done`)
- If frequently used with large files, consider a lighter embedding model or GPU usage
- The ingestion batch size (`INGEST_BATCH_SIZE`) and embedding batch size (`BATCH_SIZE`) are configurable in their respective modules

### Upload fails with "timed out" in Docker

**Symptom:** in the Gradio Upload tab the document row comes back as `ERROR`, `0` chunks, and the text `timed out`. Nothing is added to the collection.

**Cause:** this is a *client-side* timeout, not a server error. `ApiClient.ingest` posts to `/api/ingest` with `timeout=TIMEOUT_INGEST` (120s, `src/ui/constants.py`); when embedding takes longer, `httpx` raises `ReadTimeout` and the UI renders the exception. The server does **not** abort — it keeps embedding after the client disconnects, so the document can appear in the collection minutes later.

On macOS and Windows the container has no GPU: `SentenceTransformer` falls back to `cpu` (`src/infrastructure/embeddings/sentence_transformer_embedding.py`) because neither `torch.cuda.is_available()` nor `torch.backends.mps.is_available()` holds inside the Linux VM, and the image ships the CPU-only PyTorch build. With the default `Qwen/Qwen3-Embedding-4B`, a few hundred chunks exceed 120s easily.

**Diagnosis:**
```bash
# Ingest started but never logged completion?
docker compose logs app | grep "Ingesting"

# Still burning CPU → embedding is running, not stuck
docker stats --no-stream

# Nothing committed yet
curl -s http://localhost:8001/api/collections/<collection-id>/documents
```

A pair of `Ingesting N chunks` lines exactly 120s apart is the signature: the client gave up on the first file and moved to the next.

**Note:** switching `LLM_PROVIDER` to `gemini` does **not** help here. Embeddings and reranking always run in-process, whatever the LLM provider; only answer generation moves off the machine.

**Solutions:**
- Lighter embedding model (`EMBEDDING_MODEL=intfloat/multilingual-e5-small`) — changes the vector dimension, so existing collections return 409 and must be re-ingested
- Raise `TIMEOUT_INGEST` in `src/ui/constants.py`
- Run the app natively (MPS/CUDA available) and keep only supporting services in Docker
- On Linux, enable the `deploy.resources` GPU block in `compose.yml`
