from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.dependencies import CollectionUseCaseDeps

router = APIRouter(prefix="/api/collections", tags=["collections"])


class CreateCollectionRequest(BaseModel):
    name: str
    description: str = ""


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str


class DeleteResponse(BaseModel):
    detail: str


@router.post("", response_model=CollectionResponse)
async def create_collection(
    request: CreateCollectionRequest,
    use_case: CollectionUseCaseDeps,
):
    collection = await use_case.create(request.name, request.description)
    return CollectionResponse(
        id=str(collection.id),
        name=collection.name,
        description=collection.description,
        created_at=collection.created_at.isoformat(),
    )


@router.get("", response_model=list[CollectionResponse])
async def list_collections(
    use_case: CollectionUseCaseDeps,
):
    collections = await use_case.list_collections()
    return [
        CollectionResponse(
            id=str(c.id),
            name=c.name,
            description=c.description,
            created_at=c.created_at.isoformat(),
        )
        for c in collections
    ]


@router.delete("/{collection_id}", response_model=DeleteResponse)
async def delete_collection(
    collection_id: str,
    use_case: CollectionUseCaseDeps,
):
    collection = await use_case.get(collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found.")
    await use_case.delete(collection_id)
    return DeleteResponse(detail=f"Collection {collection_id} deleted.")
