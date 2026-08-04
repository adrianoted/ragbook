import logging
from uuid import UUID

from pinecone import Pinecone, ServerlessSpec, exceptions as pinecone_exceptions

from src.domain.entities import Chunk, SearchResult
from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)

METADATA_MAX_BYTES = 40_000


def _truncate_content(content: str, max_bytes: int = METADATA_MAX_BYTES) -> str:
    """Truncate content to fit within Pinecone metadata size limits."""
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class PineconeVectorStore(VectorStorePort):
    """Vector store backed by Pinecone (cloud-managed)."""

    def __init__(
        self,
        api_key: str,
        index_name: str = "ragbook",
        dimension: int = 1536,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Pinecone api_key must be provided and non-empty.")

        self._api_key = api_key
        self._index_name = index_name
        self._client = Pinecone(api_key=api_key)
        self._ensure_index(index_name, dimension, cloud, region)
        self._index = self._client.Index(index_name)

    def _ensure_index(self, index_name: str, dimension: int, cloud: str, region: str) -> None:
        if not self._client.has_index(index_name):
            logger.info("Pinecone index '%s' not found — creating serverless index (dim=%d).", index_name, dimension)
            self._client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        else:
            index_model = self._client.describe_index(index_name)
            existing_dim = index_model.dimension
            if existing_dim != dimension:
                raise ValueError(
                    f"Pinecone index '{index_name}' has dimension {existing_dim}, "
                    f"embedding model produces {dimension} — "
                    "recreate the index or change EMBEDDING_MODEL."
                )

    async def add(self, chunks: list[Chunk], collection_id: str) -> None:
        vectors = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Chunk %s has no embedding, skipping.", chunk.id)
                continue

            metadata = {
                "content": _truncate_content(chunk.content),
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.index,
                **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
            }

            vectors.append({
                "id": str(chunk.id),
                "values": chunk.embedding,
                "metadata": metadata,
            })

        if not vectors:
            return

        batch_size = 100
        try:
            for i in range(0, len(vectors), batch_size):
                self._index.upsert(
                    vectors=vectors[i : i + batch_size],
                    namespace=collection_id,
                )
        except pinecone_exceptions.PineconeApiException as exc:
            logger.error("Pinecone upsert failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error during Pinecone upsert: %s", exc)
            raise

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_id: str | None = None,
    ) -> list[SearchResult]:
        if collection_id is None:
            raise ValueError("collection_id is required for vector search.")

        try:
            response = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=collection_id,
            )
        except pinecone_exceptions.PineconeApiException as exc:
            logger.error("Pinecone query failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error during Pinecone query: %s", exc)
            raise

        results: list[SearchResult] = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            chunk = Chunk(
                id=UUID(match["id"]),
                document_id=UUID(meta.get("document_id", "00000000-0000-0000-0000-000000000000")),
                content=meta.get("content", ""),
                metadata={
                    k: v
                    for k, v in meta.items()
                    if k not in ("content", "document_id", "chunk_index")
                },
                index=int(meta.get("chunk_index", 0)),
                embedding=match.get("values"),
            )
            results.append(SearchResult(chunk=chunk, score=match["score"], source="vector"))

        return results

    async def delete_document(self, document_id: str, collection_id: str) -> None:
        try:
            self._index.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=collection_id,
            )
        except pinecone_exceptions.PineconeApiException as exc:
            logger.error("Pinecone delete document failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error during Pinecone delete document: %s", exc)
            raise

    async def delete_collection(self, collection_id: str) -> None:
        try:
            self._index.delete(delete_all=True, namespace=collection_id)
        except pinecone_exceptions.PineconeApiException as exc:
            logger.error("Pinecone delete namespace failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error during Pinecone delete: %s", exc)
            raise

    async def save(self) -> None:
        """No-op: Pinecone is cloud-managed."""

    async def load(self) -> None:
        """No-op: Pinecone is cloud-managed."""
