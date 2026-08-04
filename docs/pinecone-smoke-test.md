# Pinecone Smoke Test

Manual procedure to verify `PineconeVectorStore` with a real key.
Requires live credentials and access to the Pinecone console.

## Prerequisites

- Pinecone account (free tier is enough)
- API key from [app.pinecone.io](https://app.pinecone.io) → API Keys
- **Free tier**: check that the plan supports `aws`/`us-east-1` serverless indexes
  (free tier behavior is subject to change — check the console at run time)
- Python 3.12, venv activated: `source .venv/bin/activate`
- LLM configured (e.g. `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-2.0-flash`, valid `GOOGLE_API_KEY`)

## `.env` configuration

Add/edit these variables before the run. **Do not commit real keys.**

```env
VECTOR_STORE=pinecone
PINECONE_API_KEY=<your-key>
PINECONE_INDEX_NAME=ragbook          # default — change if you want a dedicated test index
# PINECONE_CLOUD=aws                   # default aws
# PINECONE_REGION=us-east-1            # default us-east-1
```

Leave the rest of `.env` unchanged (LLM, embedding model, etc.).

## Sequence

### Step 1 — Startup (ensure-index)

```bash
python run.py
```

**Expected outcome**: the server starts; the logs show a line similar to:

```
INFO pinecone_store: Pinecone index 'ragbook' not found — creating serverless index (dim=...)
```

If the index already exists with the correct dimension: no creation line, normal startup.

If the index exists with a different dimension: `ValueError` at startup with message
`"Pinecone index 'ragbook' has dimension X, embedding model produces Y — recreate the index or change EMBEDDING_MODEL"`.
In this case delete the index from the Pinecone console and restart.

### Step 2 — Create collection

```bash
curl -s -X POST http://localhost:8000/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "smoke-test", "description": "Pinecone smoke test"}' | python -m json.tool
```

Note the returned `collection_id` (`<COLLECTION_ID>`).

### Step 3 — Ingest sample document

Create a minimal text file:

```bash
echo "RAGBook is a RAG service that indexes documents and answers questions." > /tmp/smoke.txt
```

```bash
curl -s -X POST http://localhost:8000/api/ingest \
  -F "file=@/tmp/smoke.txt" \
  -F "collection_id=<COLLECTION_ID>" | python -m json.tool
```

**Expected outcome**: response with `document_id` and completed ingest status.

### Step 4 — Raw search

```bash
curl -s -X POST http://localhost:8000/api/search/raw \
  -H "Content-Type: application/json" \
  -d '{"query": "what is RAGBook?", "collection_id": "<COLLECTION_ID>", "top_k": 3}' | python -m json.tool
```

**Expected outcome**: `results` array with at least one element, `score > 0`.

### Step 5 — Search with LLM

```bash
curl -s -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is RAGBook?", "collection_id": "<COLLECTION_ID>", "top_k": 3}' | python -m json.tool
```

**Expected outcome**: LLM answer consistent with the ingested text, with a `[1]` citation.

### Step 6 — Check Pinecone console

Go to [app.pinecone.io](https://app.pinecone.io) → index `ragbook` → namespace `<COLLECTION_ID>`.
**Expected outcome**: vectors present (count > 0).

### Step 7 — Delete collection

```bash
curl -s -X DELETE http://localhost:8000/api/collections/<COLLECTION_ID> | python -m json.tool
```

**Expected outcome**: deletion confirmation response.

### Step 8 — Index cleanup (optional)

If you want to remove the Pinecone index after the test, do it from the console:
app.pinecone.io → index `ragbook` → Delete Index.

**Note**: the index can be reused for later runs — no need to recreate it.

---

## Run log

| Date | Outcome | Notes |
|------|-------|------|
| — | to run | first run: maintainer |
