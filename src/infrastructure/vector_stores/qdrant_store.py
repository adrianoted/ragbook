import logging
from typing import Callable
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.domain.entities import Chunk, SearchResult
from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStorePort):
    """Vector store backed by Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        vector_size: int | Callable[[], int] = 384,
    ) -> None:
        self._url = url
        # Accepts a callable so the caller can defer resolving the dimension:
        # reading it from the embedding model loads it, and only collection
        # creation needs it.
        self._vector_size_source = vector_size
        self._vector_size: int | None = vector_size if isinstance(vector_size, int) else None
        self._client = QdrantClient(
            url=url,
            api_key=api_key if api_key else None,
        )

    def _get_vector_size(self) -> int:
        if self._vector_size is None:
            self._vector_size = self._vector_size_source()  # type: ignore[operator]
        return self._vector_size

    def _ensure_collection(self, collection_id: str) -> None:
        if not self._client.collection_exists(collection_id):
            self._client.create_collection(
                collection_name=collection_id,
                vectors_config=VectorParams(
                    size=self._get_vector_size(),
                    distance=Distance.COSINE,
                ),
            )

    async def add(self, chunks: list[Chunk], collection_id: str) -> None:
        self._ensure_collection(collection_id)

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Chunk %s has no embedding, skipping.", chunk.id)
                continue

            payload = {
                "content": chunk.content,
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.index,
                **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
            }

            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        if not points:
            return

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=collection_id,
                points=points[i : i + batch_size],
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_id: str | None = None,
    ) -> list[SearchResult]:
        if collection_id is None:
            raise ValueError("collection_id is required for Qdrant search.")

        # Searching must not create collections as a side effect
        if not self._client.collection_exists(collection_id):
            return []

        hits = self._client.query_points(
            collection_name=collection_id,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = Chunk(
                id=UUID(hit.id),
                document_id=UUID(payload.get("document_id", "00000000-0000-0000-0000-000000000000")),
                content=payload.get("content", ""),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in ("content", "document_id", "chunk_index")
                },
                index=int(payload.get("chunk_index", 0)),
            )
            results.append(SearchResult(chunk=chunk, score=hit.score, source="vector"))

        return results

    async def delete_document(self, document_id: str, collection_id: str) -> None:
        if not self._client.collection_exists(collection_id):
            return

        self._client.delete(
            collection_name=collection_id,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

    async def delete_collection(self, collection_id: str) -> None:
        if self._client.collection_exists(collection_id):
            self._client.delete_collection(collection_id)

    async def save(self) -> None:
        """No-op: Qdrant persists automatically."""

    async def load(self) -> None:
        """No-op: Qdrant persists automatically."""
