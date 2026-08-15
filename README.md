# RAGBook

A RAG (retrieval-augmented generation) service: FastAPI backend + Gradio UI for
ingesting documents (text/PDF/CSV/image), searching them (vector / TF-IDF /
hybrid), and answering questions via an LLM (Gemini or Ollama).

> **Native or Docker?** On macOS and Windows, **run it natively** if you plan to actually use it. Containers there have no GPU access, and the embedding model — the heaviest part of the pipeline — then runs on CPU, making ingestion an order of magnitude slower. Docker is the right choice for a zero-setup demo, for reproducibility, and on Linux with an NVIDIA GPU, where it matches native speed. See [Models and Resources](#models-and-resources) and [Performance Limits in Docker](#performance-limits-in-docker).

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- pip
- (Optional) pyenv for Python version management
- An LLM backend: [Ollama](https://ollama.com) running locally, or a Google Gemini API key
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (required for image ingestion via OCR)

### Installation

#### 1. Clone the repository

#### 2. Set up Python environment

<details>
<summary>📦 Using pyenv (recommended for managing Python versions)</summary>

```bash

# If you have already a Python versions installed
pyenv install


# ELSE install a specific version for the project
# List all available Python versions
pyenv install --list

# Install Python 3.12
pyenv install 3.12.10

# Set local Python version for this project
pyenv local 3.12.10

# Verify Python version
python --version
```

</details>

Create and activate virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### 3. Install dependencies

<details>
<summary>🔧 Using pip-tools (recommended for dependency management)</summary>

**pip-tools** helps manage Python dependencies with pinned versions for reproducible builds.

##### Install pip-tools
```bash
pip install pip-tools
```

##### Working with dependencies

1. **Create source files:**
   ```bash
   # requirements.in - Production dependencies
   fastapi
   uvicorn[standard]
   pydantic
   google-generativeai

   # requirements-dev.in - Development dependencies
   -r requirements.in
   pytest
   pytest-asyncio
   httpx
   ```

2. **Compile pinned versions:**
   ```bash
   # Generate requirements.txt with exact versions
   pip-compile requirements.in

   # Generate requirements-dev.txt
   pip-compile requirements-dev.in
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   pip install -r requirements.txt
   ```

4. **Update dependencies:**
   ```bash
   # Update all packages to latest versions
   pip-compile --upgrade requirements.in
   pip-sync requirements.txt
   ```

</details>

<details>
<summary> Using standard installation:</summary>

**No pip-tools:**

```bash
pip install -r requirements.txt
```
</details>

#### 4. Configure environment


Copy `.env.example` to `.env` and adjust as needed. `LLM_PROVIDER` and `LLM_MODEL`
are **required** — the app exits on startup if they are unset.

```env
HOST=0.0.0.0
PORT=8000
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=info

# LLM: gemini | ollama  (required)
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:14b
# For Gemini instead, set LLM_PROVIDER=gemini and GOOGLE_API_KEY=<your-key>
```

See `.env.example` for the full list of options (embedding model, vector store,
chunking, retrieval thresholds, search strategy, provider keys).

#### 5. Run the service

```bash
# Option 1: Using run.py (recommended; auto-reload when DEBUG=true)
python run.py

# Option 2: Using uvicorn directly
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Tests
pytest
pytest tests/integration/

# Production
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

The service starts at `http://localhost:8000`:

> **Local demo — no authentication.** CORS is restricted to `localhost:8000` and `127.0.0.1:8000` by default; widen it via `CORS_ORIGINS` in `.env` if needed.

- REST API — `http://localhost:8000/api` (health check: `/api/health`)
- Gradio UI — `http://localhost:8000/ui`
- API docs (Swagger) — `http://localhost:8000/docs`

## Models and Resources

Three different models run at three different stages. Sizing the machine means looking at all three — not just the LLM, which is the one people usually think of.

| Phase | Model | Default | Size | Runs where |
|-------|-------|---------|------|-----------|
| **Ingest** — embed every chunk | Embedding | `Qwen/Qwen3-Embedding-4B` | ~8 GB on disk | **Always local**, in-process |
| **Search** — embed the query | Embedding (same instance) | ↑ | ↑ | **Always local**, in-process |
| **Search** — rerank candidates | Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | **Always local**, in-process |
| **Answer** — generate the response | LLM | `qwen3:14b` (Ollama) | ~9 GB (`qwen3:4b` ≈ 2.6 GB) | Ollama, **or a hosted API** |

Two consequences worth internalising:

- **Only the last row can be offloaded.** Setting `LLM_PROVIDER=gemini` removes the LLM from your machine entirely, but embedding and reranking stay local whatever you configure. If ingestion is slow, the LLM provider is irrelevant.
- **The embedding model is the expensive one, not the LLM.** `Qwen3-Embedding-4B` runs once per chunk at ingest time and once per query at search time. It loads on first use and stays resident.

**Hardware guidance:**

- **Apple Silicon (native)** — `SentenceTransformer` and `CrossEncoder` auto-select `mps`; Ollama uses Metal. Recommended setup on a Mac. Budget ~16 GB RAM with `qwen3:4b`, more with `qwen3:14b`.
- **NVIDIA GPU** — `cuda` is selected automatically, natively or in Docker (with `nvidia-container-toolkit`). Best case overall.
- **CPU only** — works, but expect minutes per document at ingest. Measured in a container on Apple Silicon: **~16 s to embed a single query** with `Qwen3-Embedding-4B`. Switch to a lighter model (`intfloat/multilingual-e5-small`, `sentence-transformers/all-MiniLM-L6-v2`) — the difference is one or two orders of magnitude. Note that changing the embedding model changes the vector dimension: existing collections then return **409** and must be re-ingested.
- **Low RAM** — the embedding model and the LLM are resident at the same time during a search. Pair a small embedding model with `qwen3:4b`, or move the LLM to Gemini.

Device selection is automatic (`cuda` → `mps` → `cpu`) in `src/infrastructure/embeddings/sentence_transformer_embedding.py` and `src/infrastructure/rerankers/cross_encoder_reranker.py`; there is no setting to override it.

## Run with Docker

No local Python, Tesseract, or Ollama required — Docker handles everything.

> ⚠️ **On macOS and Windows this trades speed for convenience.** The container cannot reach the GPU, so all three models run on CPU and ingestion becomes very slow — see [Performance Limits in Docker](#performance-limits-in-docker) before uploading anything sizeable. Use Docker here for a quick demo or a reproducible environment; for day-to-day use, run natively. On Linux with an NVIDIA GPU this caveat does not apply.

### Quick Start (full stack)

```bash
cp -n .env.example .env    # -n: never overwrite an existing .env
```

Then edit `.env` for Docker:
- `OLLAMA_BASE_URL=http://ollama:11434` — point the app at the compose `ollama` service
- `LLM_MODEL=qwen3:4b` — recommended on CPU (keep `qwen3:14b` on a powerful host)

```bash
docker compose up
```

Then pull the Ollama model once (persisted across restarts in the `ollama-models` volume):

```bash
docker compose exec ollama ollama pull qwen3:4b    # or qwen3:14b
```

The pulled model must match `LLM_MODEL` in `.env`, otherwise the first search fails with a "model not found" error from Ollama.

The API and UI come up right away on port **8001** (so Docker never clashes with a native `python run.py` on 8000; override with `APP_PORT` in `.env`):
- REST API — `http://localhost:8001/api`
- Gradio UI — `http://localhost:8001/ui`

**On first launch the embedding model (~8 GB) is downloaded in the background** — the server is reachable immediately, but ingest/search wait for the model. The Gradio UI shows a live progress banner during warm-up. Check readiness with:

```bash
curl http://localhost:8001/api/health
# → downloading: {"status":"ok","embedding_status":"warming","embedding_phase":"downloading","embedding_progress":{"downloaded_bytes":1234567890,"total_bytes":8000000000,"percent":15.4}}
# → loading:     {"status":"ok","embedding_status":"warming","embedding_phase":"loading","embedding_progress":null}
# → ready:       {"status":"ok","embedding_status":"ready","embedding_phase":null,"embedding_progress":null}
```

The download is cached in the `hf-cache` volume, so later launches are fast.

### LLM Scenarios

| Scenario | Command | `.env` settings |
|----------|---------|-----------------|
| **Everything in Docker** | `docker compose up` | `OLLAMA_BASE_URL=http://ollama:11434`; LLM runs in the container on CPU |
| **App only + Ollama on host** | `docker compose up app` | `OLLAMA_BASE_URL=http://host.docker.internal:11434`; Ollama already running locally (e.g. GPU/Metal) |
| **App only + Gemini** | `docker compose up app` | `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY=<your-key>` |

All three variants are documented as comments in `.env.example`.

The `docker compose up app` command starts only the app service — it does **not** start the `ollama` container.

### Resource Requirements

**First launch** downloads:
- Embedding model (Qwen3-Embedding-4B): ~8 GB — cached in the `hf-cache` volume, not re-downloaded on restart
- LLM model (qwen3:4b ≈ 2.6 GB, qwen3:14b ≈ 9 GB) — cached in the `ollama-models` volume

**Memory**: allocate at least **16 GB RAM** to Docker (Docker Desktop → Settings → Resources).

**GPU on macOS**: Metal is not supported inside containers — the LLM and embedding model run on CPU. For GPU-accelerated Ollama, run it natively on the host and use the `docker compose up app` scenario instead.

**GPU on Linux**: uncomment the `deploy.resources` block in `compose.yml` (requires `nvidia-container-toolkit` on the host).

### Data Persistence

`compose.yml` mounts `./data:/app/data` as a **bind mount** — a real directory on the host, not a Docker volume. Everything ingested lands there:

| Path | Contents |
|------|----------|
| `data/faiss_indexes/<collection>/` | vector index |
| `data/tfidf_indexes/<collection>/` | vectorizer + matrix (pickle) |
| `data/ragbook.db` | SQLite: collections, documents, chunks |
| `data/uploads/` | original uploaded files |

It survives `docker compose down`, `restart`, `up --build`, and image rebuilds — and also `docker compose down -v`, since `-v` removes named volumes and leaves bind mounts alone. On startup the lifespan hook calls `vector_store.load()`, which reads the indexes back from `data/`.

What `down -v` *does* destroy is the two named volumes: `hf-cache` (an ~8 GB re-download of the embedding model) and `ollama-models` (re-pull of the LLM). Prefer plain `docker compose down` unless you deliberately want a clean slate.

### Performance Limits in Docker

On macOS and Windows, containers have **no access to the host GPU** (Metal is not exposed to the Linux VM). The image ships the CPU-only PyTorch build, so every model runs on CPU — noticeably slower than a native run, where `SentenceTransformer` picks up `mps`/`cuda` automatically.

The most visible symptom is **slow ingest**: with the default `Qwen/Qwen3-Embedding-4B` (4B parameters), embedding a few hundred chunks on CPU takes several minutes. The Gradio UI submits files to `POST /api/ingest/async`, which returns a job ID immediately, then polls `GET /api/ingest/jobs/{id}` every 1.5 s — the upload table updates in place, cycling through pipeline phases (e.g. `embedding 12/42 (29%)`), so you see progress rather than a timeout error.

A common misconception: switching to a hosted LLM does **not** fix this. As the [Models and Resources](#models-and-resources) table shows, only answer generation can be offloaded — embedding and reranking are in-process regardless of `LLM_PROVIDER`. Gemini removes the slowest part of *search* and frees the RAM and CPU the `ollama` container would take, but ingest stays exactly as slow.

**An interrupted ingest loses all its work.** `IngestUseCase` writes nothing until every batch is embedded: `vector_store.save()`, the TF-IDF pickles, and the SQLite rows all run after the loop, and there is no save on shutdown. Stopping the container mid-ingest means starting that document over from scratch. Wait for the document to appear in the Collections tab before running `docker compose down`.

Ways to make Docker usable:

1. **Lighter embedding model** — the real fix for ingest. In `.env`:
   ```bash
   EMBEDDING_MODEL=intfloat/multilingual-e5-small
   ```
   ⚠️ Changing the model changes the vector dimension. Existing collections then fail with **409** from `check_model_compatibility` — create a new collection and re-ingest.
2. **Smaller LLM** — `LLM_MODEL=qwen3:4b`; `qwen3:14b` on CPU generates at a few tokens per second.
3. **Hosted LLM** — `LLM_PROVIDER=gemini` + `GOOGLE_API_KEY`, with `docker compose up app` (no `ollama` container at all).
4. **Hybrid setup** — run the app natively (`python run.py`, uses MPS) and keep only supporting services in Docker. Best option on Apple Silicon.

On Linux with `nvidia-container-toolkit` none of this applies: the container reaches the GPU and performance matches a native run.

## License

MIT — see [LICENSE](LICENSE).