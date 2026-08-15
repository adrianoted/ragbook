"""UI constants for the Gradio interface."""

from src.config.settings import settings

# ── App ──────────────────────────────────────────────────────
APP_TITLE = "RAGBook"
APP_HEADER = "# RAGBook\nUpload documents, search with AI, manage collections."
GRADIO_MOUNT_PATH = "/ui"

# ── Tab names ────────────────────────────────────────────────
TAB_UPLOAD = "Upload Documents"
TAB_SEARCH = "Search"
TAB_COLLECTIONS = "Collections"

# ── Supported file types ─────────────────────────────────────
SUPPORTED_FILE_TYPES = [".txt", ".md", ".pdf", ".csv", ".png", ".jpg", ".jpeg"]

# ── Search defaults ──────────────────────────────────────────
SEARCH_STRATEGIES = ["vector", "tfidf", "hybrid"]
DEFAULT_SEARCH_STRATEGY = "hybrid"
TOP_K_MIN = 1
TOP_K_MAX = 20
TOP_K_DEFAULT = 5
CHUNK_PREVIEW_MAX_CHARS = 300

# ── Tuning fallbacks ─────────────────────────────────────────
# Used ONLY when the server config is unreachable (`ApiClient.get_config()`
# returns `{}`, e.g. during UI build before the server listens). The real
# defaults, ranges and steps come from `GET /api/config` at page load and
# overwrite these. They mirror the server's values so the Advanced controls stay
# usable offline, but they are NOT a second source of truth — the server wins.
TUNING_LABELS = {
    "min_score": "Min score",
    "fusion": "Fusion",
    "hybrid_vector_weight": "Hybrid vector weight",
    "max_results_per_document": "Max results per document",
    "reranker_enabled": "Reranker enabled",
    "llm_temperature": "LLM temperature",
    "llm_think": "LLM think",
    "llm_num_ctx": "LLM context window",
}

FUSION_CHOICES = ["weighted", "rrf"]

# (min, max, step) for the five slider controls; the radio/checkbox controls
# (fusion, reranker_enabled, llm_think) have no numeric range.
TUNING_FALLBACK_RANGES = {
    "min_score": (0.0, 1.0, 0.05),
    "hybrid_vector_weight": (0.0, 1.0, 0.05),
    "max_results_per_document": (1, 10, 1),
    "llm_temperature": (0.0, 2.0, 0.1),
    "llm_num_ctx": (2048, 32768, 2048),
}

# Fallback default value for each of the eight controls.
TUNING_FALLBACK_DEFAULTS = {
    "min_score": 0.3,
    "fusion": "weighted",
    "hybrid_vector_weight": 0.7,
    "max_results_per_document": 2,
    "reranker_enabled": True,
    "llm_temperature": 0.3,
    "llm_think": False,
    "llm_num_ctx": 8192,
}

# Info notes shown under specific controls (decisions taken during spec).
RERANKER_INFO = (
    "If the reranker is disabled in the server configuration, the first search "
    "with this option enabled downloads the model (~90 MB) and may take a few "
    "minutes."
)
LLM_THINK_INFO = "Ollama only."
LLM_NUM_CTX_INFO = "Ollama only. The loaded model may reject high values."

# ── HTTP timeouts (seconds) ──────────────────────────────────
TIMEOUT_DEFAULT = 10
TIMEOUT_INGEST = 120
TIMEOUT_SEARCH = 120

# ── Ingest polling ───────────────────────────────────────────
# Cap prevents a Gradio queue slot from being held indefinitely on stuck jobs.
# With the elapsed-seconds timer on screen (no spinner to animate), cadence
# matters less, so this can run slower and halve the request volume.
# The cadence is configurable via INGEST_POLL_INTERVAL in .env (default 5s).
INGEST_POLL_INTERVAL = settings.ingest_poll_interval
INGEST_MAX_WAIT = 3600

# ── Warm-up banner ───────────────────────────────────────
HEALTH_POLL_SECONDS = 10
BANNER_DOWNLOADING = "⏳ Downloading model… {percent}% ({done} / {total} GB)"
BANNER_LOADING = "⏳ Loading model into memory…"
BANNER_ERROR = "❌ Model loading failed — check server logs."
BANNER_VARIANT_INFO = "warmup-banner-info"
BANNER_VARIANT_ERROR = "warmup-banner-error"

# ── Upload status ────────────────────────────────────────────
UPLOAD_STATUS_QUEUED = "⏳ queued"
UPLOAD_STATUS_DONE = "✓"
UPLOAD_STATUS_DUPLICATE = "already ingested ({n} chunks)"
UPLOAD_STATUS_UNKNOWN_JOB = "? unknown status (job lost, restart the upload)"
UPLOAD_STATUS_UI_TIMEOUT = "⏳ UI timeout (ingest continues on the server)"
UPLOAD_STATUS_ERROR_PREFIX = "✗ "

# Ingest phases, in the order the server walks them. Duplicated here on purpose:
# the UI talks to the API over HTTP, so this is part of the wire vocabulary, not
# an import from src.api. Keep in sync with PHASES in src/api/ingest_jobs.py.
INGEST_PHASE_LABELS = {
    "loading": "Reading file",
    "chunking": "Splitting text",
    "embedding": "Computing embeddings",
    "indexing": "Building index",
    "saving": "Saving",
}

# ── Search status line ───────────────────────────────────────
SEARCH_STATUS_RETRIEVING = "⏳ Retrieving documents…"
SEARCH_STATUS_GENERATING = "⏳ Generating answer…"
SEARCH_STATUS_IDLE = ""
SEARCH_STATUS_VARIANT = "search-status"

# ── Documents table ──────────────────────────────────────────
DOCUMENTS_TABLE_HEADERS = ["Filename", "Type", "Created", ""]

# Delete action — two-click arm/confirm, no native Gradio dialog available.
DELETE_LABEL_IDLE = "🗑"
DELETE_LABEL_ARMED = "Confirm delete?"
DELETE_STATUS_SUCCESS = "Deleted {filename}."
DELETE_STATUS_ERROR_PREFIX = "✗ Failed to delete: "

# ── Collections table ────────────────────────────────────────
COLLECTIONS_TABLE_HEADERS = ["ID", "Name", "Description", "Created", ""]

# Same two-click arm/confirm as the documents table, but the delete cascades
# over every document in the collection — the armed label has to say so.
DELETE_COLLECTION_LABEL_ARMED = "Confirm delete (+ all documents)?"
DELETE_COLLECTION_STATUS_SUCCESS = "Deleted collection {name}."

# ── Labels ───────────────────────────────────────────────────
NO_COLLECTION_PLACEHOLDER = "—"
DOWNLOAD_FILE_PREFIX = "ragbook_answer_"
