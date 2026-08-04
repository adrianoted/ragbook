# Evaluation Framework

Framework to measure retrieval quality in an objective, repeatable way. It communicates with the API over HTTP (does not import application code), so it works independently of internal changes.

## File structure

```
evaluation/
  golden_queries.json   # Test queries with expected results
  evaluate.py           # Runs queries and computes metrics
  compare.py            # Side-by-side comparison between two runs
  results/              # Timestamped result files (auto-created)
```

---

## Prerequisites

1. **Server running** — the evaluation calls the `/api/search/raw` endpoint over HTTP
2. **Documents ingested into a collection** — you need the **ID of an already-ingested collection**; every command below requires `--collection-id`
3. **`httpx` dependency** — used for HTTP calls (already in the project)

---

## Golden Queries

The file `evaluation/golden_queries.json` contains the reference queries. Each entry has three fields:

```json
{
  "query": "How to add short-term memory to a LangGraph agent",
  "expected_documents": ["default-file-memory.md"],
  "expected_keywords": ["checkpointer", "InMemorySaver", "thread_id"]
}
```

| Field | Description |
|-------|-------------|
| `query` | Search query text |
| `expected_documents` | Filenames that should appear in results (substring match: `"doc.md"` also matches `"abc123_doc.md"`) |
| `expected_keywords` | Keywords that should be present in the returned chunk content |

### Writing good golden queries

- Use realistic queries, as a real user would phrase them
- Each query should have at least one expected document and 3-5 keywords
- Cover different cases: specific queries, broad queries, queries with technical terms
- Update golden queries when new documents are added to the system

---

## Metrics

| Metric | What it measures | Target | Direction |
|--------|------------------|--------|-----------|
| `avg_score` | Mean score of top-k results | > 0.45 | Higher is better |
| `recall` | Fraction of expected documents found in results | 1.0 | Higher is better |
| `keyword_hit_rate` | Fraction of expected keywords present in content | 1.0 | Higher is better |
| `unique_docs_ratio` | Unique documents / total results (diversity) | > 0.8 | Higher is better |
| `latency_ms` | Query response time in milliseconds | < 500ms | Lower is better |

---

## Running an evaluation

### 1. Start the server

```bash
# Standard server — port 8000
python run.py

# Or test server with isolated database — port 8010
bash postman/run_test_server.sh
```

> The test server listens on **8010**, not 8000, so it can never be confused with the
> development server. When evaluating against it, pass the port explicitly:
> `--api-url http://localhost:8010`.

### 2. Run the evaluation

```bash
python -m evaluation.evaluate --collection-id <collection-id>
```

#### Available options

| Flag | Default | Description |
|------|---------|-------------|
| `--collection-id` | **(required)** | Collection ID to search in — a search always targets one collection |
| `--api-url` | `http://localhost:8000` | API base URL |
| `--top-k` | `5` | Number of results per query |
| `--strategy` | Server default | Search strategy: `vector`, `tfidf`, `hybrid` |

#### Examples

```bash
# Evaluation with default parameters
python -m evaluation.evaluate --collection-id <collection-id>

# Specify strategy and number of results
python -m evaluation.evaluate --collection-id <collection-id> --strategy hybrid --top-k 10

# Point to a different server
python -m evaluation.evaluate --collection-id <collection-id> --api-url http://localhost:8000
```

### 3. Output

The script prints per-query results and aggregates to the terminal:

```
================================================================================
RETRIEVAL EVALUATION RESULTS
================================================================================

Query: How to add short-term memory to a LangGraph agent
  avg_score:        0.5432
  recall:           1.0000
  keyword_hit_rate: 0.8000
  unique_docs_ratio:0.6000
  latency_ms:       342.5
  num_results:      5
  files:            abc_default-file-memory.md, xyz_default-file-streaming.md

--------------------------------------------------------------------------------
AGGREGATES
--------------------------------------------------------------------------------
  avg_score: 0.4876
  avg_recall: 0.9000
  avg_keyword_hit_rate: 0.7500
  avg_unique_docs_ratio: 0.7200
  avg_latency_ms: 315.2
```

Results are automatically saved to `evaluation/results/<YYYYMMDD_HHMMSS>.json`.

---

## Comparing two runs

After running two evaluations (e.g. before and after an optimization), compare them:

```bash
python -m evaluation.compare evaluation/results/20260324_100000.json evaluation/results/20260324_110000.json
```

The output shows a table with:
- **Green**: improvements
- **Red**: regressions
- Numeric delta for each metric

---

## Typical workflow

```bash
# 1. Run the baseline before making changes
python -m evaluation.evaluate --collection-id <collection-id>
# -> saves evaluation/results/20260324_100000.json

# 2. Apply the optimization (e.g. change model, chunk size, etc.)
#    Re-ingest documents if needed

# 3. Run the evaluation after the change
python -m evaluation.evaluate --collection-id <collection-id>
# -> saves evaluation/results/20260324_110000.json

# 4. Compare results
python -m evaluation.compare \
  evaluation/results/20260324_100000.json \
  evaluation/results/20260324_110000.json
```

### Example: comparing search strategies

```bash
# Evaluate vector search only
python -m evaluation.evaluate --collection-id <collection-id> --strategy vector
# -> results/20260324_120000.json

# Evaluate hybrid search
python -m evaluation.evaluate --collection-id <collection-id> --strategy hybrid
# -> results/20260324_120100.json

# Compare
python -m evaluation.compare \
  evaluation/results/20260324_120000.json \
  evaluation/results/20260324_120100.json
```

---

## Adding new golden queries

1. Open `evaluation/golden_queries.json`
2. Add a new entry:
   ```json
   {
     "query": "Your test query",
     "expected_documents": ["expected-file.md"],
     "expected_keywords": ["keyword1", "keyword2", "keyword3"]
   }
   ```
3. Run the evaluation to verify the results are consistent

> The current golden queries cover two documents: `default-file-memory.md` (LangGraph memory) and `default-file-streaming.md` (LangGraph streaming). Add queries for each new document ingested into the system.

---

## Notes

- The script uses the `/api/search/raw` endpoint which returns raw results (without passing through the LLM)
- Document name matching is partial: `"doc.md"` also matches UUID-prefixed names like `"abc123_doc.md"`
- Saved JSON results contain the configuration used, per-query results, and aggregates — useful for historical tracking
- HTTP timeout is set to 30 seconds per query
