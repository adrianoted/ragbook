from collections import defaultdict
from uuid import uuid4

from src.domain.entities import Chunk, SearchResult
from src.application.result_diversifier import diversify_results

# Stable UUID per logical document name
_DOC_IDS = defaultdict(uuid4)


def _make_result(doc_id: str, score: float, content: str | None = None) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            document_id=_DOC_IDS[doc_id],
            content=content or f"{doc_id} chunk at {score}",
            index=0,
            metadata={"document_id": doc_id},
        ),
        score=score,
        source="vector",
    )


def test_all_results_from_same_document():
    results = [_make_result("doc-1", 0.9 - i * 0.1) for i in range(5)]
    diversified = diversify_results(results, max_per_document=2, top_k=5)
    assert len(diversified) == 2
    assert all(
        r.chunk.metadata["document_id"] == "doc-1" for r in diversified
    )


def test_results_from_different_documents():
    results = [
        _make_result("doc-1", 0.9),
        _make_result("doc-2", 0.8),
        _make_result("doc-3", 0.7),
        _make_result("doc-4", 0.6),
        _make_result("doc-5", 0.5),
    ]
    diversified = diversify_results(results, max_per_document=2, top_k=5)
    assert len(diversified) == 5


def test_duplicate_content_dropped():
    results = [
        _make_result("doc-1", 0.9, content="same text"),
        _make_result("doc-2", 0.8, content="same text"),
        _make_result("doc-3", 0.7, content="other text"),
    ]
    diversified = diversify_results(results, max_per_document=2, top_k=5)
    assert len(diversified) == 2
    assert [r.chunk.metadata["document_id"] for r in diversified] == ["doc-1", "doc-3"]


def test_top_k_limit():
    results = [_make_result(f"doc-{i}", 0.9 - i * 0.05) for i in range(10)]
    diversified = diversify_results(results, max_per_document=2, top_k=3)
    assert len(diversified) == 3


def test_empty_input():
    assert diversify_results([], max_per_document=2, top_k=5) == []
