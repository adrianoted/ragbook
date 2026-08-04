from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.dependencies import CollectionUseCaseDeps, DocumentUseCaseDeps

router = APIRouter(prefix="/api/collections/{collection_id}/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    filename: str
    document_type: str
    collection_id: str | None
    created_at: str


class DeleteResponse(BaseModel):
    detail: str


async def _resolve_collection_id(
    collection_id: str,
    collections: CollectionUseCaseDeps,
) -> str:
    """Validate the collection_id: 400 if malformed, 404 if not found.

    Without this the endpoints answer as if the collection were simply empty,
    while search and ingest return 404 for the same id.
    """
    try:
        parsed = UUID(collection_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid collection_id: {collection_id!r}"
        ) from None
    # Canonical UUID string: the metadata store matches ids exactly against
    # str(uuid4()), so the raw path value would 404 a collection that exists.
    canonical = str(parsed)
    if await collections.get(canonical) is None:
        raise HTTPException(
            status_code=404, detail=f"Collection {collection_id} not found."
        )
    return canonical


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    collection_id: str,
    use_case: DocumentUseCaseDeps,
    collections: CollectionUseCaseDeps,
):
    resolved = await _resolve_collection_id(collection_id, collections)
    documents = await use_case.list_documents(resolved)
    return [
        DocumentResponse(
            id=str(d.id),
            filename=d.filename,
            document_type=d.document_type.value,
            collection_id=str(d.collection_id) if d.collection_id else None,
            created_at=d.created_at.isoformat(),
        )
        for d in documents
    ]


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    collection_id: str,
    document_id: str,
    use_case: DocumentUseCaseDeps,
    collections: CollectionUseCaseDeps,
):
    await _resolve_collection_id(collection_id, collections)
    document = await use_case.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    await use_case.delete(document_id)
    return DeleteResponse(detail=f"Document {document_id} deleted.")
