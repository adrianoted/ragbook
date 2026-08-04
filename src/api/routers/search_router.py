from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import CollectionUseCaseDeps, EmbeddingDimensionDeps, SearchUseCaseDeps, SettingsDeps
from src.api.guards import check_model_compatibility
from src.api.sse import delta_event, done_event, error_event, sources_event
from src.config.settings import Settings
from src.domain.entities import SearchQuery, SearchResult
from src.domain.enums import SearchStrategy

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    collection_id: str
    # ge=1: top_k reaches the adapters unchanged (FAISS `min(top_k, ntotal)`,
    # sklearn slicing) — 0 or negative crashes the vector branch.
    top_k: int = Field(5, ge=1, le=100)
    strategy: SearchStrategy | None = None
    min_score: float | None = None


class SourceItem(BaseModel):
    chunk_content: str
    score: float
    document_filename: str


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class RawSearchResponse(BaseModel):
    sources: list[SourceItem]


async def _resolve_collection_id(
    collection_id: str,
    collections: CollectionUseCaseDeps,
    current_model: str,
    current_dimension: int,
) -> UUID:
    """Validate the collection_id: 400 if malformed, 404 if not found, 409 if model mismatch."""
    try:
        parsed = UUID(collection_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid collection_id: {collection_id!r}"
        ) from None
    # Look up with the canonical UUID string: UUID() accepts non-canonical forms
    # (uppercase, braces, urn:, no dashes) but the metadata store matches ids
    # exactly against str(uuid4()) — lookup with the raw string would 404 a
    # collection that exists.
    collection = await collections.get(str(parsed))
    if collection is None:
        raise HTTPException(
            status_code=404, detail=f"Collection {collection_id} not found."
        )
    check_model_compatibility(collection, current_model, current_dimension)
    return parsed


def _build_search_query(
    request: SearchRequest, collection_id: UUID, current_settings: Settings
) -> SearchQuery:
    if request.min_score is not None:
        min_score = request.min_score
    elif current_settings.reranker_enabled:
        # The filter runs on cross-encoder scores (see SearchUseCase._retrieve)
        min_score = current_settings.rerank_min_score
    else:
        min_score = current_settings.min_score

    return SearchQuery(
        query=request.query,
        collection_id=collection_id,
        top_k=request.top_k,
        strategy=request.strategy,
        min_score=min_score,
    )


def _to_source_item(result: SearchResult) -> SourceItem:
    return SourceItem(
        chunk_content=result.chunk.content,
        score=result.score,
        document_filename=result.chunk.metadata.get("filename", "unknown"),
    )


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    use_case: SearchUseCaseDeps,
    collections: CollectionUseCaseDeps,
    current_settings: SettingsDeps,
    embedding_dim: EmbeddingDimensionDeps,
):
    collection_id = await _resolve_collection_id(
        request.collection_id, collections, current_settings.embedding_model, embedding_dim
    )
    search_query = _build_search_query(request, collection_id, current_settings)
    answer, results = await use_case.execute(search_query)
    return SearchResponse(
        answer=answer,
        sources=[_to_source_item(r) for r in results],
    )


@router.post("/stream")
async def search_stream(
    request: SearchRequest,
    use_case: SearchUseCaseDeps,
    collections: CollectionUseCaseDeps,
    current_settings: SettingsDeps,
    embedding_dim: EmbeddingDimensionDeps,
):
    collection_id = await _resolve_collection_id(
        request.collection_id, collections, current_settings.embedding_model, embedding_dim
    )
    search_query = _build_search_query(request, collection_id, current_settings)

    async def event_generator():
        try:
            results, token_iter = await use_case.execute_stream(search_query)
            yield sources_event([_to_source_item(r).model_dump() for r in results])
            async for token in token_iter:
                yield delta_event(token)
            yield done_event()
        except Exception as e:
            yield error_event(str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/raw", response_model=RawSearchResponse)
async def search_raw(
    request: SearchRequest,
    use_case: SearchUseCaseDeps,
    collections: CollectionUseCaseDeps,
    current_settings: SettingsDeps,
    embedding_dim: EmbeddingDimensionDeps,
):
    collection_id = await _resolve_collection_id(
        request.collection_id, collections, current_settings.embedding_model, embedding_dim
    )
    search_query = _build_search_query(request, collection_id, current_settings)
    results = await use_case.execute_raw(search_query)
    return RawSearchResponse(
        sources=[_to_source_item(r) for r in results],
    )
