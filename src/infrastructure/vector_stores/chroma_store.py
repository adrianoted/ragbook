import json
import logging
from uuid import UUID

import chromadb

from src.domain.entities import Chunk, SearchResult
from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStorePort):
    def __init__(self, persist_directory: str = "data/chroma_store") -> None:
        self._persist_directory = persist_directory
        self._client: chromadb.ClientAPI | None = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._persist_directory)

    def _get_collection(self, collection_id: str) -> chromadb.Collection:
        self._ensure_client()
        # Cosine space so scores are on the same scale as the other stores
        # (FAISS uses normalized inner product = cosine similarity).
        return self._client.get_or_create_collection(
            name=collection_id,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, chunks: list[Chunk], collection_id: str) -> None:
        if not chunks:
            return

        collection = self._get_collection(collection_id)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Skipping chunk %s with no embedding", chunk.id)
                continue

            ids.append(str(chunk.id))
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            metadatas.append({
                "document_id": str(chunk.document_id),
                "index": chunk.index,
                "metadata": json.dumps(chunk.metadata),
            })

        if not ids:
            return

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_id: str | None = None,
    ) -> list[SearchResult]:
        if collection_id is None:
            raise ValueError("collection_id is required for vector search.")

        try:
            self._ensure_client()
            collection = self._client.get_collection(name=collection_id)
        except Exception:
            logger.warning("Collection '%s' not found", collection_id)
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        search_results: list[SearchResult] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings_list = results.get("embeddings")
        embeddings_row = embeddings_list[0] if embeddings_list else [None] * len(ids)

        for i, doc_id in enumerate(ids):
            stored_meta = metadatas[i] if metadatas[i] else {}
            chunk_metadata = json.loads(stored_meta.get("metadata", "{}"))

            chunk = Chunk(
                id=UUID(doc_id),
                document_id=UUID(stored_meta.get("document_id", "00000000-0000-0000-0000-000000000000")),
                content=documents[i],
                metadata=chunk_metadata,
                index=int(stored_meta.get("index", 0)),
                embedding=embeddings_row[i] if embeddings_row[i] is not None else None,
            )

            # Chroma returns cosine *distance* (1 - cosine similarity) in the
            # cosine space configured above; convert back to similarity so the
            # score scale matches FAISS.
            distance = distances[i]
            score = 1.0 - distance

            search_results.append(SearchResult(chunk=chunk, score=score))

        return search_results

    async def delete_document(self, document_id: str, collection_id: str) -> None:
        try:
            self._ensure_client()
            collection = self._client.get_collection(name=collection_id)
            collection.delete(where={"document_id": document_id})
        except Exception:
            logger.warning(
                "Failed to delete document '%s' from collection '%s'",
                document_id,
                collection_id,
            )

    async def delete_collection(self, collection_id: str) -> None:
        self._ensure_client()
        try:
            self._client.delete_collection(name=collection_id)
        except Exception:
            logger.warning("Collection '%s' not found for deletion", collection_id)

    async def save(self) -> None:
        # ChromaDB PersistentClient handles persistence automatically.
        pass

    async def load(self) -> None:
        # Re-initialize the client to pick up any persisted data.
        self._client = None
        self._ensure_client()
