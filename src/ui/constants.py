"""UI constants for the Gradio interface."""

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

# ── Warm-up banner ───────────────────────────────────────
HEALTH_POLL_SECONDS = 10
BANNER_DOWNLOADING = "⏳ Downloading model… {percent}% ({done} / {total} GB)"
BANNER_LOADING = "⏳ Loading model into memory…"
BANNER_ERROR = "❌ Model loading failed — check server logs."
BANNER_VARIANT_INFO = "warmup-banner-info"
BANNER_VARIANT_ERROR = "warmup-banner-error"

# ── Search status line ───────────────────────────────────────
SEARCH_STATUS_GENERATING = "⏳ Generating answer…"
SEARCH_STATUS_IDLE = ""
SEARCH_STATUS_VARIANT = "search-status"

# ── Labels ───────────────────────────────────────────────────
NO_COLLECTION_PLACEHOLDER = "—"
DOWNLOAD_FILE_PREFIX = "ragbook_answer_"
