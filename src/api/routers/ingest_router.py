import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import EmbeddingDimensionDeps, MetadataStoreDeps, SettingsDeps, get_ingest_use_case
from src.api.guards import check_model_compatibility
from src.domain.enums import DocumentType

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


@router.post("", response_model=IngestResponse)
async def ingest_document(
    settings: SettingsDeps,
    metadata_store: MetadataStoreDeps,
    embedding_dim: EmbeddingDimensionDeps,
    file: UploadFile = File(...),
    collection_id: str = Form(...),
):
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
        return IngestResponse(
            document_id=str(existing.id),
            filename=file.filename,
            num_chunks=len(existing_chunks),
            collection_id=collection_id,
            already_ingested=True,
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{uuid4().hex}_{file.filename}"
    file_path.write_bytes(content)

    use_case = get_ingest_use_case(document_type)
    document, chunks = await use_case.execute(
        file_path=str(file_path),
        document_type=document_type,
        collection_id=collection_id,
        content_hash=content_hash,
    )

    return IngestResponse(
        document_id=str(document.id),
        filename=file.filename,
        num_chunks=len(chunks),
        collection_id=collection_id,
    )
