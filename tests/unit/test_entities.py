from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.domain.entities import (
    Chunk,
    Collection,
    Document,
    LlmOptions,
    RetrievalTuning,
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


class TestRetrievalTuning:
    def test_default_is_all_none(self):
        rt = RetrievalTuning()
        assert rt.fusion is None
        assert rt.vector_weight is None
        assert rt.max_results_per_document is None
        assert rt.reranker_enabled is None

    def test_falsy_vector_weight_preserved(self):
        rt = RetrievalTuning(vector_weight=0.0)
        assert rt.vector_weight == 0.0
        assert rt.vector_weight is not None

    def test_false_reranker_enabled_preserved(self):
        rt = RetrievalTuning(reranker_enabled=False)
        assert rt.reranker_enabled is False

    def test_invalid_fusion_raises(self):
        with pytest.raises(ValidationError):
            RetrievalTuning(fusion="invalid")


class TestLlmOptions:
    def test_default_is_all_none(self):
        opts = LlmOptions()
        assert opts.temperature is None
        assert opts.think is None
        assert opts.num_ctx is None

    def test_falsy_values_preserved(self):
        opts = LlmOptions(temperature=0.0, think=False)
        assert opts.temperature == 0.0
        assert opts.temperature is not None
        assert opts.think is False


class TestSearchQueryTuning:
    def test_defaults_are_empty_vos_not_none(self):
        sq = SearchQuery(query="x")
        assert isinstance(sq.tuning, RetrievalTuning)
        assert isinstance(sq.llm_options, LlmOptions)
        assert sq.tuning.fusion is None
        assert sq.tuning.vector_weight is None
        assert sq.tuning.max_results_per_document is None
        assert sq.tuning.reranker_enabled is None
        assert sq.llm_options.temperature is None
        assert sq.llm_options.think is None
        assert sq.llm_options.num_ctx is None

    def test_model_copy_preserves_tuning(self):
        sq = SearchQuery(
            query="x",
            tuning=RetrievalTuning(fusion="rrf"),
            llm_options=LlmOptions(temperature=0.7),
        )
        copied = sq.model_copy(update={"top_k": 20})
        assert copied.top_k == 20
        assert copied.tuning.fusion == "rrf"
        assert copied.llm_options.temperature == 0.7
