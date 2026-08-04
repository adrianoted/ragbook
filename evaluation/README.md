# Evaluation Framework — Quick Reference

Measures retrieval quality by running golden queries against the API and computing metrics.

## Setup

Make sure the server is running and documents are ingested:

```bash
# from project root
python run.py
```

## Commands

### Run evaluation

```bash
python -m evaluation.evaluate --collection-id <ID>
```

Options:

```
--api-url URL    API base URL (default: http://localhost:8000)
--top-k N        Number of results per query (default: 5)
--strategy STR   Search strategy: vector | tfidf | hybrid (default: server config)
```

Results are saved to `evaluation/results/<YYYYMMDD_HHMMSS>.json`.

### Compare two runs

```bash
python -m evaluation.compare evaluation/results/<before>.json evaluation/results/<after>.json
```

### Typical workflow

```bash
# 1. Baseline
python -m evaluation.evaluate --collection-id <ID>

# 2. Make changes, re-ingest if needed

# 3. Evaluate again
python -m evaluation.evaluate --collection-id <ID>

# 4. Compare
python -m evaluation.compare evaluation/results/<before>.json evaluation/results/<after>.json
```

### Compare strategies

```bash
python -m evaluation.evaluate --collection-id <ID> --strategy vector
python -m evaluation.evaluate --collection-id <ID> --strategy hybrid
python -m evaluation.compare evaluation/results/<vector>.json evaluation/results/<hybrid>.json
```

## Golden queries

Edit `golden_queries.json` to add test cases.

**Positive query** — expects documents in the results:

```json
{
  "query": "your search query",
  "expected_documents": ["expected-file.md"],
  "expected_keywords": ["keyword1", "keyword2"]
}
```

**Negative query** — expects zero results (query is off-topic, no relevant document in corpus):

```json
{
  "query": "unrelated topic query",
  "negative": true
}
```

Negative queries are excluded from all positive averages (`avg_recall`, `avg_mrr`,
`avg_score`, `avg_keyword_hit_rate`, `avg_unique_docs_ratio`). They
contribute only to `negative_pass_rate`.

## Metrics

### Retrieval quality (positive queries)

| Metric | Description | Target |
|--------|-------------|--------|
| `avg_score` | Mean similarity score of top results | > 0.45 |
| `avg_recall` | Fraction of expected docs found across queries | 1.0 |
| `avg_keyword_hit_rate` | Fraction of expected keywords present in retrieved chunks | 1.0 |
| `avg_unique_docs_ratio` | Diversity of source documents in results | > 0.8 |
| `avg_mrr` | Mean Reciprocal Rank — rewards finding the right doc early | > 0.8 *(proposed)* |
| `avg_latency_ms` | Mean query latency | < 500ms |

**MRR** (Mean Reciprocal Rank): for each query, `1/rank` of the first relevant result
(1-based). `0.0` if no relevant result is returned. Rewards retrieving the right document
near the top.

### Negative-query robustness

| Metric | Description | Target |
|--------|-------------|--------|
| `negative_pass_rate` | Fraction of negative queries that return zero results | 1.0 *(proposed)* |

`negative_pass` (per-query): `1.0` if the query returns zero results (the server's
`min_score` filter correctly rejects all candidates), `0.0` otherwise.

### Backward compatibility

When comparing a new run against a JSON produced before MRR/negative metrics were
introduced, missing keys fall back to `0.0`. Per-query diffs show `0 → x` for the new
metrics; negative queries added to the golden set do not appear in the per-query diff
(only queries present in both JSONs are compared).

*Proposed targets (`avg_mrr`, `negative_pass_rate`) are to be confirmed
after the first baseline run on the extended golden set.*

Full documentation: [docs/10-evaluation.md](../docs/10-evaluation.md)

---

## Generation evaluation

Measures **faithfulness**: how well the LLM answer is grounded in the retrieved
chunks, with no hallucinated claims. Uses a separate Gemini model as judge
(LLM-as-judge pattern).

### Prerequisites

- Server running and documents ingested (same as retrieval eval)
- `GOOGLE_API_KEY` set in the environment:
  ```bash
  export GOOGLE_API_KEY=<your-key>
  ```

### Command

```bash
python -m evaluation.evaluate_generation --collection-id <ID>
```

Options:

```
--api-url URL        API base URL (default: http://localhost:8000)
--top-k N            Number of results per query (default: 5)
--strategy STR       Search strategy: vector | tfidf | hybrid (default: server config)
--judge-model MODEL  Gemini model used as judge (default: gemini-2.5-flash)
--timeout SECONDS    HTTP timeout for /api/search (default: 300)
```

`/api/search` is not streamed here, so the response arrives only when generation
is complete: the timeout must cover the slowest answer end to end (raise it above
300s for slow local models, e.g. thinking models on Ollama).

Results are saved to `evaluation/results/gen_<YYYYMMDD_HHMMSS>.json` — separate
from retrieval JSONs (`evaluation/results/<YYYYMMDD_HHMMSS>.json`).

### Faithfulness metric (claim-based binary)

The judge reads the answer and the numbered source chunks, then extracts each
factual claim in the answer and labels it `supported: true/false` against the
chunks.

```
faithfulness = supported_claims / total_claims
```

**Why binary labels instead of a 1–5 scale?** Binary judgements have higher
inter-rater agreement and are less sensitive to prompt phrasing. A graded scale
introduces ambiguity ("is this a 3 or a 4?") that increases noise without
improving the signal for retrieval tuning.

**Edge cases:**

| Condition | faithfulness |
|-----------|-------------|
| Answer is the fallback string | `1.0` per query; excluded from `avg_faithfulness` |
| Judge returns zero claims | `null`; excluded from `avg_faithfulness`, counted in `no_claims_count` |
| Judge failure after 1 retry | `null`; excluded from `avg_faithfulness`, counted in `judge_error_count` |

**Why zero claims is not 1.0.** When retrieval returns off-topic chunks, the
grounded prompt answers "the provided documents do not contain information about
this" — an answer with no factual claim. Scoring it `1.0` would make
`avg_faithfulness` *rise* as retrieval degrades. Those queries are excluded
instead, and `faithfulness_n` reports how many queries the average is actually
computed on — always read the average together with `faithfulness_n`,
`no_claims_count` and `judge_error_count`, since the denominator varies between
runs.

### `negative_refusal` metric

For queries marked `"negative": true` in the golden set, the script skips the
judge and instead checks whether the answer is exactly the server's fallback
string (`"No sufficiently relevant results found."`).

```
negative_refusal = 1.0  ←  exact fallback match
negative_refusal = 0.0  ←  any other answer
```

The `"negative"` flag was introduced by the eval-metrics feature (B.6). Golden
set entries without the flag are treated as positive queries.

**Known limit:** the fallback string is emitted by the *retrieval* layer, when no
chunk survives `min_score` — the LLM is never called in that branch. So
`negative_refusal` measures empty retrieval, not the model's ability to refuse:
a correct refusal phrased in natural language scores `0.0`. It duplicates
`negative_pass` in the retrieval eval by construction.

### Judge bias

The judge is always a Gemini model (`gemini-2.5-flash` by default), **regardless
of which LLM provider the application is configured to use**. This keeps the
judge independent when the app runs with Ollama.

**When the app runs with a Gemini model**: the judge and the generator share the
same provider family. Residual self-evaluation bias may inflate `avg_faithfulness`
slightly — Gemini tends to rate its own outputs more charitably. Treat baselines
taken in this configuration with an extra margin of scepticism; compare relative
changes between runs rather than relying on absolute scores.

### Cost per run

Each query incurs up to **two API calls**:

1. `POST /api/search` — generation call (billed against the configured LLM)
2. Judge call — billed against `GOOGLE_API_KEY`, model `gemini-2.5-flash` by default

On any judge failure (malformed JSON, rate limit, network error) the judge is
retried once after a 5s pause, so the maximum is **2 judge calls per positive
query** (`max_judge_calls = 2N` where N is the number of positive queries).
Negative queries and fallback answers skip the judge entirely.

`gemini-2.5-flash` is the default judge to keep per-run cost low. For higher
accuracy at higher cost, pass `--judge-model gemini-2.5-pro`.

### Comparing runs

`python -m evaluation.compare` does **not** support `gen_*.json`: it reads
retrieval fields only (`recall`, `mrr`, `negative`), so generation runs render as
all-zero rows. Compare `gen_*.json` files manually for now.

### Baseline note

**Do not treat the first baseline as a target.** The prerequisite — feature A2
(grounded prompt) — is merged, so a baseline can be taken now. Read
`avg_faithfulness` together with `faithfulness_n`, `fallback_count`,
`no_claims_count` and `judge_error_count`: the average is computed only on
judged queries, so the denominator moves between runs and a higher average may
just mean fewer queries were judged.

The `avg_faithfulness` target will be set once that first baseline exists.
