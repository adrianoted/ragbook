from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.domain.entities import (
    Chunk,
    Collection,
    Document,
    SearchQuery,
    SearchResult,
)
from src.domain.enums import DocumentType, SearchStrategy


@pytest.fixture
def sample_collection() -> Collection:
    return Collection(
        name="test-collection",
        description="A test collection",
        embedding_model="",
        embedding_dimension=0,
    )


@pytest.fixture
def sample_document(sample_collection: Collection) -> Document:
    return Document(
        filename="test.txt",
        document_type=DocumentType.TEXT,
        content="This is a test document with some content for testing purposes.",
        metadata={"size": 62, "encoding": "utf-8"},
        collection_id=sample_collection.id,
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    doc_id = sample_document.id
    return [
        Chunk(
            document_id=doc_id,
            content="This is chunk zero.",
            metadata={"position": 0},
            index=0,
            embedding=[0.1, 0.2, 0.3, 0.4],
        ),
        Chunk(
            document_id=doc_id,
            content="This is chunk one.",
            metadata={"position": 1},
            index=1,
            embedding=[0.5, 0.6, 0.7, 0.8],
        ),
        Chunk(
            document_id=doc_id,
            content="This is chunk two.",
            metadata={"position": 2},
            index=2,
            embedding=[0.9, 0.1, 0.2, 0.3],
        ),
    ]


@pytest.fixture
def sample_search_query() -> SearchQuery:
    return SearchQuery(query="What is the meaning of life?", top_k=5)


@pytest.fixture
def sample_search_results(sample_chunks: list[Chunk]) -> list[SearchResult]:
    return [
        SearchResult(chunk=sample_chunks[0], score=0.95, source="vector"),
        SearchResult(chunk=sample_chunks[1], score=0.80, source="vector"),
    ]


@pytest.fixture
def mock_loader() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_chunker() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_embedding() -> AsyncMock:
    mock = AsyncMock()
    mock.embed.return_value = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return mock


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_tfidf() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_llm() -> AsyncMock:
    mock = AsyncMock()
    mock.generate.return_value = "This is the generated answer."
    return mock


@pytest.fixture
def mock_search() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_metadata_store() -> AsyncMock:
    mock = AsyncMock()
    # Default: no content-hash match, so ingest takes the normal (non-dedup) path
    mock.find_document_by_content_hash.return_value = None
    return mock


# ── API test fixtures ────────────────────────────────────────


@pytest.fixture
def mock_ingest_use_case(sample_document, sample_chunks) -> AsyncMock:
    mock = AsyncMock()
    mock.execute.return_value = (sample_document, sample_chunks)
    return mock


@pytest.fixture
def mock_search_use_case(sample_search_results) -> AsyncMock:
    mock = AsyncMock()
    mock.execute.return_value = ("This is the generated answer.", sample_search_results)
    mock.execute_raw.return_value = sample_search_results
    return mock


@pytest.fixture
def mock_collection_use_case(sample_collection) -> AsyncMock:
    mock = AsyncMock()
    mock.create.return_value = sample_collection
    mock.list_collections.return_value = [sample_collection]
    mock.delete.return_value = None
    return mock


