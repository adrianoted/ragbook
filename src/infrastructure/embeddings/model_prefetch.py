"""Prefetch a Hugging Face model with byte-level progress.

Flow: ``prefetch_model`` → HF cache → load. The prefetch **precedes** the
``SentenceTransformer`` constructor: it fills the HF cache so the load finds
everything already downloaded, exposing a percentage of progress meanwhile.

Every error here is **non-fatal**: HF unreachable, ``dry_run`` failed, local
model, ``HF_HUB_OFFLINE`` set → logged at ``warning`` and returns ``False``,
letting warm-up proceed with the normal load. No exception propagates. This
is the only point in the project that talks to ``huggingface_hub`` (besides
the implicit download inside ``SentenceTransformer``).

The two primitives used — ``snapshot_download(dry_run=True)`` and
``snapshot_download(tqdm_class=...)`` — are not a stable public contract of
hub 1.x: if they change, the fallback degrades the feature to current
behavior without breaking warm-up.
"""

import fnmatch
import logging
import threading
from collections.abc import Callable
from typing import Any

from huggingface_hub import snapshot_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Receives (downloaded_bytes, total_bytes).
ProgressCallback = Callable[[int, int], None]

# Formats sentence-transformers never uses: excluded from download.
IGNORE_PATTERNS = [
    "*.onnx",
    "*.onnx_data",
    "onnx/*",
    "openvino/*",
    "*.gguf",
    "coreml/*",
    "*.tflite",
    "*.h5",
    "*.msgpack",
]

# Legacy-format weights: downloadable only if .safetensors don't exist.
LEGACY_WEIGHT_PATTERNS = ["*.bin", "*.pt"]


def select_files(files: list[Any]) -> tuple[list[str], int]:
    """Chooses which files to download and sums their sizes. Pure function.

    Each element of ``files`` has ``filename``, ``file_size`` and
    ``will_download`` (as returned by ``snapshot_download(dry_run=True)``).

    1. discards filenames matching ``IGNORE_PATTERNS``;
    2. if among the remaining files (cached included) there is at least one
       ``.safetensors``, also discards ``LEGACY_WEIGHT_PATTERNS`` (duplicate
       legacy weights);
    3. returns ``(filenames of those with will_download, sum of file_size)``.
    """
    remaining = [
        f
        for f in files
        if not any(fnmatch.fnmatch(f.filename, pat) for pat in IGNORE_PATTERNS)
    ]

    has_safetensors = any(f.filename.endswith(".safetensors") for f in remaining)
    if has_safetensors:
        remaining = [
            f
            for f in remaining
            if not any(
                fnmatch.fnmatch(f.filename, pat) for pat in LEGACY_WEIGHT_PATTERNS
            )
        ]

    pending = [f for f in remaining if f.will_download]
    allow = [f.filename for f in pending]
    total = sum(f.file_size for f in pending)
    return allow, total


def compute_percent(downloaded: int, total: int) -> float:
    """Downloaded percentage: ``0.0`` if ``total <= 0`` (warm cache), clamped
    to ``100.0``, rounded to 1 decimal."""
    if total <= 0:
        return 0.0
    percent = downloaded / total * 100
    return round(min(percent, 100.0), 1)


def _make_tqdm_class(
    on_progress: ProgressCallback, total_bytes: int
) -> tuple[type, dict[str, int]]:
    """Local ``tqdm`` subclass that forwards cumulative progress.

    The total comes from the ``dry_run`` (``total_bytes``), not from
    ``self.total`` on the bar: in hub 1.x the latter grows as download
    threads start, while the UI wants the final total right away.

    ``snapshot_download`` builds this class twice: once for the aggregate
    byte bar (``unit="B"``) and once for the file-count bar passed to
    ``thread_map`` (default ``unit``, counts completed files). Only the
    first represents the requested progress — the other must be ignored,
    otherwise the file count overwrites the downloaded bytes.

    ``self.n`` cannot be read: with the bar disabled (stderr not a tty, the
    normal case on servers and Docker) ``tqdm.update()`` returns before
    incrementing it, so it would stay 0. The ``n`` delta passed to
    ``update()`` arrives regardless, so accumulation happens in its own
    thread-safe state, returned together with the class so the caller can
    check its final value without widening ``prefetch_model``'s public
    signature.
    """
    lock = threading.Lock()
    state = {"downloaded": 0}

    class _ProgressTqdm(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # hub 1.x passes `name=`, unknown to tqdm: raises TqdmKeyError
            # and sinks the whole snapshot_download if we don't discard it.
            kwargs.pop("name", None)
            self._counts_bytes = kwargs.get("unit") == "B"
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> bool | None:
            displayed = super().update(n)
            if self._counts_bytes and n:
                with lock:
                    state["downloaded"] += int(n)
                    downloaded = state["downloaded"]
                on_progress(min(downloaded, total_bytes), total_bytes)
            return displayed

    return _ProgressTqdm, state


def prefetch_model(model_name: str, on_progress: ProgressCallback) -> bool:
    """Downloads ``model_name``'s weights into the HF cache ahead of time.

    Returns ``True`` if a download was performed, ``False`` if the cache was
    already complete or something went wrong (in the latter case the error is
    logged at ``warning`` and swallowed: warm-up proceeds with the load).
    """
    try:
        files = snapshot_download(
            repo_id=model_name,
            dry_run=True,
            ignore_patterns=IGNORE_PATTERNS,
        )
        allow, total = select_files(files)

        if not allow:
            logger.info("Prefetch: cache already complete for %s", model_name)
            return False

        # The UI shows 0% immediately, before the first byte.
        on_progress(0, total)

        tqdm_class, state = _make_tqdm_class(on_progress, total)
        snapshot_download(
            repo_id=model_name,
            allow_patterns=allow,
            tqdm_class=tqdm_class,
        )
        if state["downloaded"] < total:
            logger.warning(
                "Prefetch: accumulated progress (%s) below expected total "
                "(%s) for %s",
                state["downloaded"],
                total,
                model_name,
            )
        return True
    except Exception:
        logger.warning(
            "Prefetch skipped for %s, proceeding with load", model_name, exc_info=True
        )
        return False
