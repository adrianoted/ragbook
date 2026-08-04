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

# ── HTTP timeouts (seconds) ──────────────────────────────────
TIMEOUT_DEFAULT = 10
TIMEOUT_INGEST = 120
TIMEOUT_SEARCH = 120

# ── Labels ───────────────────────────────────────────────────
NO_COLLECTION_PLACEHOLDER = "—"
DOWNLOAD_FILE_PREFIX = "ragbook_answer_"
