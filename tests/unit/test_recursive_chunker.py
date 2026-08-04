import pytest

from src.domain.entities import Document
from src.domain.enums import DocumentType
from src.infrastructure.chunkers.recursive_chunker import RecursiveChunker


def _make_document(content: str, filename: str = "test.txt") -> Document:
    return Document(
        filename=filename,
        document_type=DocumentType.TEXT,
        content=content,
    )


@pytest.mark.asyncio
async def test_chunk_short_document():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    doc = _make_document("Short content.")

    chunks = await chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].content == "Short content."


@pytest.mark.asyncio
async def test_chunk_long_document():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = _make_document("word " * 200)

    chunks = await chunker.chunk(doc)

    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_chunk_preserves_document_id():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = _make_document("word " * 200)

    chunks = await chunker.chunk(doc)

    for chunk in chunks:
        assert chunk.document_id == doc.id


@pytest.mark.asyncio
async def test_chunk_indexes_are_sequential():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = _make_document("word " * 200)

    chunks = await chunker.chunk(doc)

    assert [c.index for c in chunks] == list(range(len(chunks)))


@pytest.mark.asyncio
async def test_chunk_custom_size():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = _make_document("word " * 200)

    chunks = await chunker.chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 50
