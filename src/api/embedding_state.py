"""In-process warm-up state for the embedding model.

Thread-safe: mutated by huggingface_hub download threads,
read by the asyncio event loop via /api/health.
Valid only with a single uvicorn worker.
"""
import threading

PHASE_DOWNLOADING = "downloading"
PHASE_LOADING = "loading"

_lock = threading.Lock()
_state: dict = {
    "status": "warming",
    "phase": None,
    "downloaded_bytes": 0,
    "total_bytes": 0,
}


def report_download_progress(downloaded_bytes: int, total_bytes: int) -> None:
    with _lock:
        _state["phase"] = PHASE_DOWNLOADING
        _state["downloaded_bytes"] = downloaded_bytes
        _state["total_bytes"] = total_bytes


def mark_loading() -> None:
    with _lock:
        _state["phase"] = PHASE_LOADING
        _state["downloaded_bytes"] = 0
        _state["total_bytes"] = 0


def mark_ready() -> None:
    with _lock:
        _state["status"] = "ready"
        _state["phase"] = None
        _state["downloaded_bytes"] = 0
        _state["total_bytes"] = 0


def mark_error() -> None:
    with _lock:
        _state["status"] = "error"
        _state["phase"] = None
        _state["downloaded_bytes"] = 0
        _state["total_bytes"] = 0


def reset() -> None:
    with _lock:
        _state["status"] = "warming"
        _state["phase"] = None
        _state["downloaded_bytes"] = 0
        _state["total_bytes"] = 0


def health_payload() -> dict:
    with _lock:
        status = _state["status"]
        phase = _state["phase"]
        dl = _state["downloaded_bytes"]
        total = _state["total_bytes"]

    if phase == PHASE_DOWNLOADING and total > 0:
        percent = round(min(dl / total, 1.0) * 100, 1)
        progress = {"downloaded_bytes": dl, "total_bytes": total, "percent": percent}
    else:
        progress = None

    return {
        "status": "ok",
        "embedding_status": status,
        "embedding_phase": phase,
        "embedding_progress": progress,
    }
