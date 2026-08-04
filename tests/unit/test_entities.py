from datetime import datetime
from uuid import UUID

import pytest

from src.domain.entities import (
    Chunk,
    Collection,
    Document,
    SearchQuery,
    SearchResult,
)
from src.domain.enums import DocumentType, SearchStrategy


class TestDocument:
    def test_create_with_all_fields(self):
        doc = Document(
            filename="report.pdf",
            document_type=DocumentType.PDF,
            content="PDF content here",
            metadata={"pages": 10},
        )
        assert isinstance(doc.id, UUID)
        assert doc.filename == "report.pdf"
        assert doc.document_type == DocumentType.PDF
        assert doc.content == "PDF content here"
        assert doc.metadata == {"pages": 10}
        assert isinstance(doc.created_at, datetime)
        assert doc.collection_id is None

    def test_create_with_minimal_fields(self):
        doc = Document(
            filename="file.txt",
            document_type=DocumentType.TEXT,
            content="hello",
        )
        assert isinstance(doc.id, UUID)
        assert isinstance(doc.created_at, datetime)
        assert doc.metadata == {}
        assert doc.collection_id is None

    def test_unique_ids(self):
        d1 = Document(filename="a.txt", document_type=DocumentType.TEXT, content="a")
        d2 = Document(filename="b.txt", document_type=DocumentType.TEXT, content="b")
        assert d1.id != d2.id


class TestChunk:
    def test_create_with_embedding(self):
        doc_id = UUID("12345678-1234-5678-1234-567812345678")
        chunk = Chunk(
            document_id=doc_id,
            content="chunk content",
            index=0,
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.document_id == doc_id
        assert chunk.content == "chunk content"
        assert chunk.index == 0
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_create_without_embedding(self):
        doc_id = UUID("12345678-1234-5678-1234-567812345678")
        chunk = Chunk(document_id=doc_id, content="text", index=0)
        assert chunk.embedding is None


class TestSearchResult:
    def test_create(self, sample_chunks):
        result = SearchResult(chunk=sample_chunks[0], score=0.95)
        assert result.score == 0.95
        assert result.source == "vector"

    def test_custom_source(self, sample_chunks):
        result = SearchResult(chunk=sample_chunks[0], score=0.5, source="tfidf")
        assert result.source == "tfidf"


class TestCollection:
    def test_create(self):
        coll = Collection(name="my-collection", description="Test")
        assert isinstance(coll.id, UUID)
        assert coll.name == "my-collection"
        assert coll.description == "Test"
        assert isinstance(coll.created_at, datetime)

    def test_default_description(self):
        coll = Collection(name="c")
        assert coll.description == ""


class TestSearchQuery:
    def test_default_strategy_none(self):
        sq = SearchQuery(query="test")
        assert sq.strategy is None
        assert sq.top_k == 5
        assert sq.collection_id is None

    def test_explicit_strategy(self):
        sq = SearchQuery(query="test", strategy=SearchStrategy.HYBRID, top_k=10)
        assert sq.strategy == SearchStrategy.HYBRID
        assert sq.top_k == 10


class TestEnums:
    def test_document_type_values(self):
        assert DocumentType.TEXT.value == "text"
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.CSV.value == "csv"
        assert DocumentType.IMAGE.value == "image"
        assert len(DocumentType) == 4

    def test_search_strategy_values(self):
        assert SearchStrategy.VECTOR.value == "vector"
        assert SearchStrategy.TFIDF.value == "tfidf"
        assert SearchStrategy.HYBRID.value == "hybrid"
        assert len(SearchStrategy) == 3
