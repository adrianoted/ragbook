"""Pure event/polling logic for the Upload tab.

No ``gradio`` import: the polling loop and the status-text mapping live here so
they can be tested in isolation (mirrors ``search_events`` / ``warmup_banner``).
The ``tabs/upload_tab.py`` module only wires these into the Gradio widgets.

``sleep`` and ``now`` are injectable so tests never wait on the real clock.
"""
import time
from collections.abc import Iterator

from src.ui.constants import (
    INGEST_MAX_WAIT,
    INGEST_PHASE_LABELS,
    INGEST_POLL_INTERVAL,
    UPLOAD_STATUS_DONE,
    UPLOAD_STATUS_ERROR_PREFIX,
    UPLOAD_STATUS_QUEUED,
    UPLOAD_STATUS_UI_TIMEOUT,
    UPLOAD_STATUS_UNKNOWN_JOB,
)


def poll_ingest_job(
    client,
    job_id: str,
    *,
    interval: float = INGEST_POLL_INTERVAL,
    max_wait: float = INGEST_MAX_WAIT,
    sleep=time.sleep,
    now=time.monotonic,
) -> Iterator[dict]:
    """Poll ``GET /ingest/jobs/{job_id}`` and yield every state read.

    Yields each job dict as it is read, sleeping ``interval`` seconds *between*
    reads (never before the first). Terminates when the job reaches ``done`` or
    ``error``. A missing job (``get_ingest_job`` returns ``None``) yields a final
    ``{"status": "unknown"}`` and stops. Exceeding ``max_wait`` yields a final
    ``{"status": "ui_timeout"}`` and stops — the server keeps ingesting.
    """
    start = now()
    while True:
        job = client.get_ingest_job(job_id)
        if job is None:
            yield {"status": "unknown"}
            return
        yield job
        if job.get("status") in ("done", "error"):
            return
        if now() - start > max_wait:
            yield {"status": "ui_timeout"}
            return
        sleep(interval)


def job_status_text(job: dict, elapsed: float = 0.0) -> str:
    """Map a job dict to the English status text shown in the Upload table.

    ``elapsed`` is the seconds since the caller started polling this job; the
    running branch renders it as the "in progress" signal (a timer needs no
    animation, unlike the spinner it replaced). Chunk counts are deliberately
    absent — the total lives in the table's own Chunks column, and
    ``chunks_done`` only moves once per batch, so showing it here reads as
    stuck.
    """
    status = job.get("status")
    if status == "pending":
        return UPLOAD_STATUS_QUEUED
    if status == "running":
        phase = job.get("phase") or ""
        timer = f"{int(elapsed)}s"
        label = INGEST_PHASE_LABELS.get(phase)
        if label is not None:
            return f"{label}… {timer}"
        if phase:
            # Unknown phase (server ahead of the UI): show it raw rather than blank.
            return f"{phase}… {timer}"
        return timer
    if status == "done":
        return UPLOAD_STATUS_DONE
    if status == "error":
        return UPLOAD_STATUS_ERROR_PREFIX + (job.get("error") or "")
    if status == "unknown":
        return UPLOAD_STATUS_UNKNOWN_JOB
    if status == "ui_timeout":
        return UPLOAD_STATUS_UI_TIMEOUT
    return ""
