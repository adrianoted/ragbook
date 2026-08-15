from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.application.ingest_use_case import IngestUseCase, INGEST_BATCH_SIZE
from src.domain.entities import Chunk
from src.domain.enums import DocumentType


@pytest.fixture
def ingest_use_case(
    mock_loader,
    mock_chunker,
    mock_embedding,
    mock_vector_store,
    mock_tfidf,
    mock_metadata_store,
):
    return IngestUseCase(
        loader=mock_loader,
        chunker=mock_chunker,
        embedding=mock_embedding,
        vector_store=mock_vector_store,
        tfidf=mock_tfidf,
        metadata_store=mock_metadata_store,
    )


@pytest.mark.asyncio
async def test_execute_success(
    ingest_use_case, mock_loader, mock_chunker, mock_embedding,
    mock_vector_store, mock_tfidf, mock_metadata_store,
    sample_document, sample_chunks,
):
    coll_id = str(uuid4())
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = sample_chunks
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    doc, chunks = await ingest_use_case.execute("/tmp/test.txt", DocumentType.TEXT, collection_id=coll_id)

    mock_loader.load.assert_called_once_with("/tmp/test.txt", DocumentType.TEXT)
    mock_chunker.chunk.assert_called_once_with(sample_document)
    mock_embedding.embed.assert_called_once()
    mock_vector_store.add.assert_called_once()
    mock_vector_store.save.assert_called_once()
    mock_tfidf.fit.assert_called_once()
    mock_metadata_store.save_document.assert_called_once()
    mock_metadata_store.save_chunks.assert_called_once()
    assert doc == sample_document
    assert str(doc.collection_id) == coll_id
    assert chunks == sample_chunks


@pytest.mark.asyncio
async def test_execute_with_collection_id(
    ingest_use_case, mock_loader, mock_chunker,
    mock_metadata_store, sample_document, sample_chunks,
):
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = sample_chunks
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    coll_id = str(uuid4())
    doc, _ = await ingest_use_case.execute("/tmp/test.txt", DocumentType.TEXT, collection_id=coll_id)

    assert str(doc.collection_id) == coll_id


@pytest.mark.asyncio
async def test_execute_sets_embeddings_on_chunks(
    ingest_use_case, mock_loader, mock_chunker, mock_embedding,
    mock_vector_store, mock_metadata_store, sample_document, sample_chunks,
):
    mock_loader.load.return_value = sample_document
    for c in sample_chunks:
        c.embedding = None
    mock_chunker.chunk.return_value = sample_chunks
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    await ingest_use_case.execute("/tmp/test.txt", DocumentType.TEXT, collection_id=str(uuid4()))

    # embed() must be called with the chunk texts
    mock_embedding.embed.assert_called_once()
    call_args = mock_embedding.embed.call_args[0][0]
    assert call_args == [c.content for c in sample_chunks]

    # vector_store.add() must be called with chunks that had embeddings at that point
    mock_vector_store.add.assert_called_once()
    # After ingestion, embeddings are cleared from chunks (memory optimization)
    for chunk in sample_chunks:
        assert chunk.embedding is None


@pytest.mark.asyncio
async def test_execute_calls_save_on_vector_store(
    ingest_use_case, mock_loader, mock_chunker,
    mock_metadata_store, mock_vector_store,
    sample_document, sample_chunks,
):
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = sample_chunks
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    await ingest_use_case.execute("/tmp/test.txt", DocumentType.TEXT, collection_id=str(uuid4()))

    mock_vector_store.save.assert_called_once()


@pytest.mark.asyncio
async def test_execute_persists_content_hash_in_metadata(
    ingest_use_case, mock_loader, mock_chunker,
    mock_metadata_store, sample_document, sample_chunks,
):
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = sample_chunks
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    content_hash = "a" * 64
    doc, _ = await ingest_use_case.execute(
        "/tmp/test.txt", DocumentType.TEXT, collection_id=str(uuid4()),
        content_hash=content_hash,
    )

    assert doc.metadata["content_hash"] == content_hash
    saved_doc = mock_metadata_store.save_document.call_args[0][0]
    assert saved_doc.metadata["content_hash"] == content_hash


@pytest.mark.asyncio
async def test_execute_without_content_hash_leaves_metadata_untouched(
    ingest_use_case, mock_loader, mock_chunker,
    mock_metadata_store, sample_document, sample_chunks,
):
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = sample_chunks
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    doc, _ = await ingest_use_case.execute(
        "/tmp/test.txt", DocumentType.TEXT, collection_id=str(uuid4()),
    )

    assert "content_hash" not in doc.metadata


# ── on_progress callback tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_progress_complete_sequence(
    ingest_use_case, mock_loader, mock_chunker, mock_embedding,
    mock_metadata_store, sample_document, sample_chunks,
):
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = sample_chunks  # 3 chunks, one batch
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    received: list[tuple[str, int, int]] = []
    await ingest_use_case.execute(
        "/tmp/test.txt", DocumentType.TEXT, collection_id=str(uuid4()),
        on_progress=lambda phase, done, total: received.append((phase, done, total)),
    )

    total = len(sample_chunks)
    assert received == [
        ("loading", 0, 0),
        ("chunking", 0, 0),
        ("embedding", 0, total),
        ("embedding", total, total),
        ("indexing", total, total),
        ("saving", total, total),
    ]


@pytest.mark.asyncio
async def test_on_progress_batch_advancement(
    ingest_use_case, mock_loader, mock_chunker, mock_metadata_store,
    sample_document, mock_embedding,
):
    total_chunks = 5
    doc_id = sample_document.id
    big_chunks = [
        Chunk(document_id=doc_id, content=f"chunk {i}", metadata={}, index=i)
        for i in range(total_chunks)
    ]
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = big_chunks
    mock_embedding.embed.return_value = [[0.1] * 4] * 2  # will be called per batch
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = big_chunks

    received: list[tuple[str, int, int]] = []

    with patch("src.application.ingest_use_case.INGEST_BATCH_SIZE", 2):
        await ingest_use_case.execute(
            "/tmp/test.txt", DocumentType.TEXT, collection_id=str(uuid4()),
            on_progress=lambda phase, done, total: received.append((phase, done, total)),
        )

    embedding_calls = [(d, t) for phase, d, t in received if phase == "embedding"]
    # first call is start (0, total), then one per batch
    assert embedding_calls[0] == (0, total_chunks)
    chunks_done_values = [d for d, _ in embedding_calls[1:]]
    chunks_total_values = [t for _, t in embedding_calls[1:]]
    assert all(t == total_chunks for t in chunks_total_values)
    assert chunks_done_values == sorted(chunks_done_values)
    assert chunks_done_values[-1] == total_chunks


@pytest.mark.asyncio
async def test_execute_without_on_progress_works(
    ingest_use_case, mock_loader, mock_chunker, mock_embedding,
    mock_vector_store, mock_tfidf, mock_metadata_store,
    sample_document, sample_chunks,
):
    mock_loader.load.return_value = sample_document
    mock_chunker.chunk.return_value = sample_chunks
    mock_metadata_store.save_document.return_value = sample_document
    mock_metadata_store.save_chunks.return_value = sample_chunks

    doc, chunks = await ingest_use_case.execute(
        "/tmp/test.txt", DocumentType.TEXT, collection_id=str(uuid4()),
    )

    assert doc == sample_document
    assert chunks == sample_chunks
