import asyncio
import hashlib
import logging
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.dependencies import EmbeddingDimensionDeps, MetadataStoreDeps, SettingsDeps, get_ingest_use_case
from src.api.guards import check_model_compatibility
from src.api.ingest_jobs import (
    create_job,
    get_job,
    mark_done,
    mark_error,
    mark_running,
    update_progress,
)
from src.application.ingest_use_case import IngestUseCase
from src.domain.enums import DocumentType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

EXTENSION_MAP = {
    ".txt": DocumentType.TEXT,
    ".md": DocumentType.TEXT,
    ".pdf": DocumentType.PDF,
    ".csv": DocumentType.CSV,
    ".png": DocumentType.IMAGE,
    ".jpg": DocumentType.IMAGE,
    ".jpeg": DocumentType.IMAGE,
}


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    collection_id: str
    already_ingested: bool = False


class IngestJobAcceptedResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    collection_id: str


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str | None = None
    chunks_done: int = 0
    chunks_total: int = 0
    document_id: str | None = None
    filename: str
    num_chunks: int = 0
    error: str | None = None


class PreparedIngest(NamedTuple):
    """Result of the fast, synchronous part of an ingest: the file is already
    on disk and validated. Only ``IngestUseCase.execute`` remains."""

    file_path: str
    document_type: DocumentType
    content_hash: str
    filename: str


async def _prepare_ingest(
    file: UploadFile,
    collection_id: str,
    settings: SettingsDeps,
    metadata_store: MetadataStoreDeps,
    embedding_dim: int,
) -> tuple[IngestResponse | None, PreparedIngest | None]:
    """Run every fast, synchronous step shared by both ingest endpoints:
    collection lookup, model compatibility, extension mapping, read + hash,
    dedup, and writing the file to disk.

    Returns a discriminated pair: ``(IngestResponse, None)`` when the upload is
    a duplicate (nothing left to do), or ``(None, PreparedIngest)`` on the new
    path. Raises ``HTTPException`` (404/400/409) for invalid input — the same
    status codes for both the sync and the async endpoint.
    """
    collection = await metadata_store.get_collection(collection_id)
    if collection is None:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_id}' not found. Create a collection first.",
        )
    check_model_compatibility(collection, settings.embedding_model, embedding_dim)

    ext = Path(file.filename).suffix.lower()
    document_type = EXTENSION_MAP.get(ext)
    if document_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    # Dedup on (content hash, collection) before touching disk: same bytes
    # already ingested in this collection → skip, return the existing document.
    existing = await metadata_store.find_document_by_content_hash(collection_id, content_hash)
    if existing is not None:
        existing_chunks = await metadata_store.get_chunks_by_document(str(existing.id))
        duplicate = IngestResponse(
            document_id=str(existing.id),
            filename=file.filename,
            num_chunks=len(existing_chunks),
            collection_id=collection_id,
            already_ingested=True,
        )
        return duplicate, None

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{uuid4().hex}_{file.filename}"
    file_path.write_bytes(content)

    prepared = PreparedIngest(
        file_path=str(file_path),
        document_type=document_type,
        content_hash=content_hash,
        filename=file.filename,
    )
    return None, prepared


@router.post("", response_model=IngestResponse)
async def ingest_document(
    settings: SettingsDeps,
    metadata_store: MetadataStoreDeps,
    embedding_dim: EmbeddingDimensionDeps,
    file: UploadFile = File(...),
    collection_id: str = Form(...),
):
    duplicate, prepared = await _prepare_ingest(
        file, collection_id, settings, metadata_store, embedding_dim
    )
    if duplicate is not None:
        return duplicate

    use_case = get_ingest_use_case(prepared.document_type)
    document, chunks = await use_case.execute(
        file_path=prepared.file_path,
        document_type=prepared.document_type,
        collection_id=collection_id,
        content_hash=prepared.content_hash,
    )

    return IngestResponse(
        document_id=str(document.id),
        filename=prepared.filename,
        num_chunks=len(chunks),
        collection_id=collection_id,
    )


async def _run_ingest_job(
    job_id: str,
    use_case: IngestUseCase,
    prepared: PreparedIngest,
    collection_id: str,
) -> None:
    """Background driver: run the slow ``execute`` step and reflect its progress
    and terminal state into the job registry."""
    mark_running(job_id)
    try:
        document, chunks = await use_case.execute(
            file_path=prepared.file_path,
            document_type=prepared.document_type,
            collection_id=collection_id,
            content_hash=prepared.content_hash,
            on_progress=lambda phase, done, total: update_progress(job_id, phase, done, total),
        )
        mark_done(job_id, str(document.id), len(chunks))
    except asyncio.CancelledError:
        # Cancellation is not a job failure — let it propagate untouched.
        raise
    except Exception as e:  # noqa: BLE001 — surface any failure through the job
        logger.exception("Ingest job %s failed", job_id)
        mark_error(job_id, str(e))


@router.post("/async", status_code=202, response_model=None)
async def ingest_document_async(
    settings: SettingsDeps,
    metadata_store: MetadataStoreDeps,
    embedding_dim: EmbeddingDimensionDeps,
    file: UploadFile = File(...),
    collection_id: str = Form(...),
):
    duplicate, prepared = await _prepare_ingest(
        file, collection_id, settings, metadata_store, embedding_dim
    )
    if duplicate is not None:
        # A duplicate has no work to schedule — answer like the sync endpoint.
        return JSONResponse(status_code=200, content=duplicate.model_dump())

    # Resolve the use case before scheduling: it is a direct call patched by the
    # tests, and adapter-construction errors must surface in this HTTP response,
    # not inside the background task.
    use_case = get_ingest_use_case(prepared.document_type)
    job = create_job(filename=prepared.filename, collection_id=collection_id)
    # Keep a strong reference to the task — asyncio only holds a weak one.
    job.task = asyncio.create_task(
        _run_ingest_job(job.id, use_case, prepared, collection_id)
    )

    return IngestJobAcceptedResponse(
        job_id=job.id,
        status="pending",
        filename=prepared.filename,
        collection_id=collection_id,
    )


@router.get("/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def get_ingest_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Ingest job '{job_id}' not found (unknown or expired).",
        )
    return IngestJobStatusResponse(
        job_id=job.id,
        status=job.status,
        phase=job.phase,
        chunks_done=job.chunks_done,
        chunks_total=job.chunks_total,
        document_id=job.document_id,
        filename=job.filename,
        num_chunks=job.num_chunks,
        error=job.error,
    )
