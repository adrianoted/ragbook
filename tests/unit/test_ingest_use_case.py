from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.application.ingest_use_case import IngestUseCase
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
