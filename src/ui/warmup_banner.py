"""Pure functions for mapping health payloads to warm-up banner state."""

from src.ui.constants import (
    BANNER_DOWNLOADING,
    BANNER_ERROR,
    BANNER_LOADING,
    BANNER_VARIANT_ERROR,
    BANNER_VARIANT_INFO,
)


def format_gb(num_bytes: int) -> str:
    """Convert bytes to GiB string with one decimal place."""
    return f"{num_bytes / (1024 ** 3):.1f}"


def banner_state(health: dict) -> tuple[str | None, bool, bool, str]:
    """Map a /api/health payload to warm-up banner state.

    Returns (text, timer_active, unchanged, variant) where:
    - text: markdown to display, or None to hide the banner
    - timer_active: whether the polling timer should keep running
    - unchanged: when True the caller leaves the banner as-is (gr.update() with no
                 arguments); this signals that the server is unreachable and we
                 should preserve whatever was already visible
    - variant: CSS class selecting the banner color (info or error)
    """
    if not health:
        return None, True, True, BANNER_VARIANT_INFO

    status = health.get("embedding_status")

    if status == "ready":
        return None, False, False, BANNER_VARIANT_INFO

    if status == "error":
        return BANNER_ERROR, False, False, BANNER_VARIANT_ERROR

    # warming (or any unknown status — treat as warming)
    phase = health.get("embedding_phase")
    progress = health.get("embedding_progress")

    if phase == "downloading" and progress is not None:
        text = BANNER_DOWNLOADING.format(
            percent=progress["percent"],
            done=format_gb(progress["downloaded_bytes"]),
            total=format_gb(progress["total_bytes"]),
        )
        return text, True, False, BANNER_VARIANT_INFO

    return BANNER_LOADING, True, False, BANNER_VARIANT_INFO
