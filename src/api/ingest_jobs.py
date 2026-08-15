"""In-process job registry for asynchronous ingest operations.

Thread-safe via threading.Lock. Valid only with a single uvicorn worker —
with multiple workers a GET on a job may land on the wrong process
(same limitation as embedding_state.py).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

JOB_TTL_SECONDS = 3600
PHASES = ("loading", "chunking", "embedding", "indexing", "saving")

_lock = threading.Lock()
_registry: dict[str, IngestJob] = {}


@dataclass
class IngestJob:
    id: str
    filename: str
    collection_id: str
    status: str = "pending"
    phase: str | None = None
    chunks_done: int = 0
    chunks_total: int = 0
    document_id: str | None = None
    num_chunks: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    task: Any = None


def create_job(filename: str, collection_id: str) -> IngestJob:
    job = IngestJob(
        id=uuid.uuid4().hex,
        filename=filename,
        collection_id=collection_id,
    )
    with _lock:
        _registry[job.id] = job
    cleanup_expired()
    return job


def get_job(job_id: str) -> IngestJob | None:
    with _lock:
        return _registry.get(job_id)


def mark_running(job_id: str) -> IngestJob | None:
    with _lock:
        job = _registry.get(job_id)
        if job is None:
            return None
        job.status = "running"
        return job


def update_progress(
    job_id: str, phase: str, chunks_done: int, chunks_total: int
) -> IngestJob | None:
    with _lock:
        job = _registry.get(job_id)
        if job is None:
            return None
        job.phase = phase
        job.chunks_done = chunks_done
        job.chunks_total = chunks_total
        return job


def mark_done(job_id: str, document_id: str, num_chunks: int) -> IngestJob | None:
    with _lock:
        job = _registry.get(job_id)
        if job is None:
            return None
        job.status = "done"
        job.document_id = document_id
        job.num_chunks = num_chunks
        job.finished_at = time.monotonic()
        return job


def mark_error(job_id: str, message: str) -> IngestJob | None:
    with _lock:
        job = _registry.get(job_id)
        if job is None:
            return None
        job.status = "error"
        job.error = message
        job.finished_at = time.monotonic()
        return job


def cleanup_expired(ttl: float = JOB_TTL_SECONDS, now: float | None = None) -> int:
    if now is None:
        now = time.monotonic()
    to_remove = []
    with _lock:
        for job_id, job in _registry.items():
            if job.status in ("done", "error") and job.finished_at is not None:
                if now - job.finished_at > ttl:
                    to_remove.append(job_id)
        for job_id in to_remove:
            del _registry[job_id]
    return len(to_remove)


def reset_registry() -> None:
    with _lock:
        _registry.clear()
