import logging
from typing import Callable
from uuid import UUID

from src.domain.entities import Chunk, Document
from src.domain.ports.chunker_port import ChunkerPort
from src.domain.ports.document_loader_port import DocumentLoaderPort
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.tfidf_port import TfidfPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.enums import DocumentType
from src.domain.ports.metadata_store_port import MetadataStorePort

logger = logging.getLogger(__name__)

INGEST_BATCH_SIZE = 500

ProgressCallback = Callable[[str, int, int], None]


def _notify(on_progress: ProgressCallback | None, phase: str, done: int, total: int) -> None:
    if on_progress is not None:
        on_progress(phase, done, total)


class IngestUseCase:
    def __init__(
        self,
        loader: DocumentLoaderPort,
        chunker: ChunkerPort,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        tfidf: TfidfPort,
        metadata_store: MetadataStorePort,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedding = embedding
        self._vector_store = vector_store
        self._tfidf = tfidf
        self._metadata_store = metadata_store

    async def execute(
        self,
        file_path: str,
        document_type: DocumentType,
        collection_id: str,
        content_hash: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[Document, list[Chunk]]:
        _notify(on_progress, "loading", 0, 0)
        document = await self._loader.load(file_path, document_type)
        document.collection_id = UUID(collection_id)
        if content_hash is not None:
            document.metadata["content_hash"] = content_hash

        _notify(on_progress, "chunking", 0, 0)
        chunks = await self._chunker.chunk(document)
        # Free document content — no longer needed
        document.content = ""

        coll_id = collection_id
        total = len(chunks)
        logger.info("Ingesting %d chunks in batches of %d", total, INGEST_BATCH_SIZE)

        _notify(on_progress, "embedding", 0, total)

        # Embed and index in batches to limit memory usage
        for i in range(0, total, INGEST_BATCH_SIZE):
            batch = chunks[i : i + INGEST_BATCH_SIZE]
            texts = [chunk.content for chunk in batch]
            embeddings = await self._embedding.embed(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

            await self._vector_store.add(batch, coll_id)

            # Clear embeddings from chunks — vector store already has them
            for chunk in batch:
                chunk.embedding = None

            _notify(on_progress, "embedding", min(i + INGEST_BATCH_SIZE, total), total)
            logger.info("Batch %d/%d done", i // INGEST_BATCH_SIZE + 1, (total + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE)

        _notify(on_progress, "indexing", total, total)
        await self._vector_store.save()
        await self._tfidf.fit(chunks, coll_id)

        _notify(on_progress, "saving", total, total)
        await self._metadata_store.save_document(document)
        await self._metadata_store.save_chunks(chunks)

        return document, chunks
