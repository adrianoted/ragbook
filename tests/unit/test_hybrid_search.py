from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.domain.entities import Chunk, RetrievalTuning, SearchQuery, SearchResult
from src.infrastructure.search.hybrid_search import (
    HybridSearch,
    _fuse_rrf,
    _fuse_weighted,
    _normalize_scores,
    validate_fusion,
)


def _make_chunk(**kwargs):
    defaults = dict(document_id=uuid4(), content="c", metadata={}, index=0)
    defaults.update(kwargs)
    return Chunk(**defaults)


def _make_result(score=0.9, source="vector", chunk=None):
    if chunk is None:
        chunk = _make_chunk()
    return SearchResult(chunk=chunk, score=score, source=source)


@pytest.mark.asyncio
async def test_hybrid_combines_results():
    chunk_v = _make_chunk(content="vector chunk")
    chunk_t = _make_chunk(content="tfidf chunk")

    mock_store = AsyncMock()
    mock_store.search.return_value = [SearchResult(chunk=chunk_v, score=0.9, source="vector")]

    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk_t, score=0.8, source="tfidf")]

    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7)
    results = await hs.search(SearchQuery(query="q", top_k=5), mock_embed)

    assert len(results) == 2
    assert all(r.source == "hybrid" for r in results)


@pytest.mark.asyncio
async def test_hybrid_respects_weights():
    chunk = _make_chunk()

    mock_store = AsyncMock()
    mock_store.search.return_value = [SearchResult(chunk=chunk, score=0.9, source="vector")]

    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = []

    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, vector_weight=1.0)
    results = await hs.search(SearchQuery(query="q", top_k=5), mock_embed)

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id


@pytest.mark.asyncio
async def test_hybrid_deduplicates_by_chunk_id():
    shared_chunk = _make_chunk(content="shared")

    mock_store = AsyncMock()
    mock_store.search.return_value = [SearchResult(chunk=shared_chunk, score=0.9, source="vector")]

    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=shared_chunk, score=0.8, source="tfidf")]

    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7)
    results = await hs.search(SearchQuery(query="q", top_k=5), mock_embed)

    assert len(results) == 1
    assert results[0].chunk.id == shared_chunk.id
    # Combined score: 0.7 * 1.0 (normalized vector) + 0.3 * 1.0 (normalized tfidf) = 1.0
    assert results[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_hybrid_fallback_single_source():
    chunk = _make_chunk()

    # Only tfidf returns results
    mock_store = AsyncMock()
    mock_store.search.return_value = []

    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk, score=0.8, source="tfidf")]

    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7)
    results = await hs.search(SearchQuery(query="q", top_k=5), mock_embed)

    assert len(results) == 1
    assert results[0].source == "hybrid"


def test_normalize_scores():
    chunks = [_make_chunk() for _ in range(3)]
    results = [
        SearchResult(chunk=chunks[0], score=0.2, source="v"),
        SearchResult(chunk=chunks[1], score=0.6, source="v"),
        SearchResult(chunk=chunks[2], score=1.0, source="v"),
    ]
    normalized = _normalize_scores(results)
    assert normalized == pytest.approx([0.2, 0.6, 1.0])


def test_normalize_scores_same_value():
    chunk = _make_chunk()
    results = [SearchResult(chunk=chunk, score=0.5, source="v")]
    assert _normalize_scores(results) == [1.0]


def test_normalize_scores_empty():
    assert _normalize_scores([]) == []


# --- Fusion functions -------------------------------------------------------


def test_fuse_weighted_parity():
    # Same case as test_hybrid_deduplicates_by_chunk_id but on the pure fn:
    # shared chunk sums the two normalized contributions to 1.0.
    shared = _make_chunk(content="shared")
    vector = [SearchResult(chunk=shared, score=0.9, source="vector")]
    tfidf = [SearchResult(chunk=shared, score=0.8, source="tfidf")]

    fused = _fuse_weighted(vector, tfidf, vector_weight=0.7)

    assert len(fused) == 1
    assert fused[0].chunk.id == shared.id
    assert fused[0].source == "hybrid"
    assert fused[0].score == pytest.approx(1.0)


def test_fuse_rrf_sums_contributions_for_shared_chunk():
    # A chunk in both lists (rank 1 in each) must sum 1/(60+1) twice and
    # outrank a chunk present in only one list at the same rank.
    shared = _make_chunk(content="shared")
    only_vector = _make_chunk(content="only-v")
    vector = [
        SearchResult(chunk=shared, score=0.5, source="vector"),
        SearchResult(chunk=only_vector, score=0.4, source="vector"),
    ]
    tfidf = [SearchResult(chunk=shared, score=0.5, source="tfidf")]

    fused = _fuse_rrf(vector, tfidf, k=60)

    assert fused[0].chunk.id == shared.id
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[1].chunk.id == only_vector.id
    assert fused[1].score == pytest.approx(1 / 62)  # rank 2 in vector list
    assert all(r.source == "hybrid" for r in fused)


def test_fuse_rrf_ranks_by_rank_not_raw_score():
    # Chunk with low raw scores but rank 1 in both lists beats a chunk with a
    # high raw score present in only one list.
    ranked_high = _make_chunk(content="rank-high")
    raw_high = _make_chunk(content="raw-high")
    vector = [
        SearchResult(chunk=ranked_high, score=0.01, source="vector"),
        SearchResult(chunk=raw_high, score=0.99, source="vector"),
    ]
    tfidf = [SearchResult(chunk=ranked_high, score=0.01, source="tfidf")]

    fused = _fuse_rrf(vector, tfidf, k=60)

    assert fused[0].chunk.id == ranked_high.id  # 2/61 > 1/61


def test_fuse_rrf_stable_on_ties():
    # Two chunks each appearing once at rank 1 → equal RRF score → stable order,
    # no exception.
    a = _make_chunk(content="a")
    b = _make_chunk(content="b")
    vector = [SearchResult(chunk=a, score=0.9, source="vector")]
    tfidf = [SearchResult(chunk=b, score=0.9, source="tfidf")]

    fused = _fuse_rrf(vector, tfidf, k=60)

    assert len(fused) == 2
    assert {r.chunk.id for r in fused} == {a.id, b.id}
    assert fused[0].score == pytest.approx(fused[1].score)
    assert fused[0].chunk.id == a.id  # vector list processed first → stable


def test_fuse_rrf_keeps_first_search_result_on_merge():
    shared = _make_chunk(content="shared")
    vector = [SearchResult(chunk=shared, score=0.9, source="vector")]
    tfidf = [SearchResult(chunk=shared, score=0.8, source="tfidf")]

    fused = _fuse_rrf(vector, tfidf, k=60)

    assert len(fused) == 1
    assert fused[0].chunk.content == "shared"


@pytest.mark.asyncio
async def test_hybrid_rrf_fusion_end_to_end():
    shared = _make_chunk(content="shared")
    only_vector = _make_chunk(content="only-v")

    mock_store = AsyncMock()
    mock_store.search.return_value = [
        SearchResult(chunk=shared, score=0.5, source="vector"),
        SearchResult(chunk=only_vector, score=0.4, source="vector"),
    ]
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [
        SearchResult(chunk=shared, score=0.5, source="tfidf")
    ]
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(
        vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7, fusion="rrf"
    )
    results = await hs.search(SearchQuery(query="q", top_k=5), mock_embed)

    assert results[0].chunk.id == shared.id
    assert results[0].score == pytest.approx(2 / 61)
    assert all(r.source == "hybrid" for r in results)


@pytest.mark.asyncio
async def test_hybrid_rrf_empty_list_degrades_without_fusion():
    # One empty list → degrade path (passthrough), fusion never runs even in rrf.
    chunk = _make_chunk()
    mock_store = AsyncMock()
    mock_store.search.return_value = []
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [
        SearchResult(chunk=chunk, score=0.8, source="tfidf")
    ]
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(
        vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7, fusion="rrf"
    )
    results = await hs.search(SearchQuery(query="q", top_k=5), mock_embed)

    assert len(results) == 1
    assert results[0].source == "hybrid"
    assert results[0].score == pytest.approx(0.8)  # raw passthrough, no RRF


def test_hybrid_invalid_fusion_raises():
    with pytest.raises(ValueError):
        HybridSearch(
            vector_store=AsyncMock(),
            tfidf=AsyncMock(),
            fusion="nonsense",
        )


def test_hybrid_fusion_defaults_to_weighted():
    hs = HybridSearch(vector_store=AsyncMock(), tfidf=AsyncMock())
    assert hs._fusion == "weighted"


# --- Per-query override tests -----------------------------------------------


def test_validate_fusion_accepts_valid():
    assert validate_fusion("weighted") == "weighted"
    assert validate_fusion("rrf") == "rrf"


def test_validate_fusion_rejects_invalid():
    with pytest.raises(ValueError, match="Unknown fusion 'bad'"):
        validate_fusion("bad")


@pytest.mark.asyncio
async def test_per_query_override_fusion_rrf():
    # Ctor uses weighted; query overrides to rrf → chunk_b (appears in both lists) wins.
    chunk_a = _make_chunk(content="a")
    chunk_b = _make_chunk(content="b")

    mock_store = AsyncMock()
    mock_store.search.return_value = [
        SearchResult(chunk=chunk_a, score=0.9, source="vector"),
        SearchResult(chunk=chunk_b, score=0.4, source="vector"),
    ]
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk_b, score=0.8, source="tfidf")]
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, fusion="weighted")
    query = SearchQuery(query="q", top_k=5, tuning=RetrievalTuning(fusion="rrf"))
    results = await hs.search(query, mock_embed)

    # RRF: chunk_b appears at rank 2 in vector + rank 1 in tfidf → 1/62 + 1/61 > 1/61
    assert results[0].chunk.id == chunk_b.id
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)


@pytest.mark.asyncio
async def test_per_query_override_fusion_weighted():
    # Ctor uses rrf; query overrides to weighted → chunk_a (rank 1 vector) wins.
    chunk_a = _make_chunk(content="a")
    chunk_b = _make_chunk(content="b")

    mock_store = AsyncMock()
    mock_store.search.return_value = [
        SearchResult(chunk=chunk_a, score=0.9, source="vector"),
        SearchResult(chunk=chunk_b, score=0.4, source="vector"),
    ]
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk_b, score=0.8, source="tfidf")]
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, fusion="rrf")
    query = SearchQuery(query="q", top_k=5, tuning=RetrievalTuning(fusion="weighted"))
    results = await hs.search(query, mock_embed)

    # Weighted (default 0.7): chunk_a score = 0.7*1.0 = 0.7, chunk_b = 0.7*0.444 + 0.3*1.0 = 0.611
    assert results[0].chunk.id == chunk_a.id


@pytest.mark.asyncio
async def test_per_query_vector_weight_zero():
    # vector_weight=0.0 per-query → ranking follows only the lexical list.
    chunk_vector = _make_chunk(content="vector-only")
    chunk_tfidf = _make_chunk(content="tfidf-only")

    mock_store = AsyncMock()
    mock_store.search.return_value = [SearchResult(chunk=chunk_vector, score=0.9, source="vector")]
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk_tfidf, score=0.8, source="tfidf")]
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7)
    query = SearchQuery(query="q", top_k=5, tuning=RetrievalTuning(vector_weight=0.0))
    results = await hs.search(query, mock_embed)

    # vector_weight=0.0 → chunk_vector gets 0.0*1.0=0.0, chunk_tfidf gets 1.0*1.0=1.0
    assert results[0].chunk.id == chunk_tfidf.id


@pytest.mark.asyncio
async def test_per_query_vector_weight_one():
    # vector_weight=1.0 per-query → ranking follows only the vector list.
    chunk_vector = _make_chunk(content="vector-only")
    chunk_tfidf = _make_chunk(content="tfidf-only")

    mock_store = AsyncMock()
    mock_store.search.return_value = [SearchResult(chunk=chunk_vector, score=0.9, source="vector")]
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk_tfidf, score=0.8, source="tfidf")]
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7)
    query = SearchQuery(query="q", top_k=5, tuning=RetrievalTuning(vector_weight=1.0))
    results = await hs.search(query, mock_embed)

    # vector_weight=1.0 → chunk_tfidf gets 0.0*1.0=0.0, chunk_vector gets 1.0*1.0=1.0
    assert results[0].chunk.id == chunk_vector.id


@pytest.mark.asyncio
async def test_per_query_no_override_regression():
    # No tuning override → behaviour identical to ctor (weighted, default weight).
    chunk_a = _make_chunk(content="a")
    chunk_b = _make_chunk(content="b")

    mock_store = AsyncMock()
    mock_store.search.return_value = [
        SearchResult(chunk=chunk_a, score=0.9, source="vector"),
        SearchResult(chunk=chunk_b, score=0.4, source="vector"),
    ]
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk_b, score=0.8, source="tfidf")]
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    hs = HybridSearch(vector_store=mock_store, tfidf=mock_tfidf, vector_weight=0.7, fusion="weighted")
    results_no_tuning = await hs.search(SearchQuery(query="q", top_k=5), mock_embed)

    mock_store.search.reset_mock()
    mock_tfidf.search.reset_mock()
    mock_embed.embed_query.reset_mock()
    mock_store.search.return_value = [
        SearchResult(chunk=chunk_a, score=0.9, source="vector"),
        SearchResult(chunk=chunk_b, score=0.4, source="vector"),
    ]
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk_b, score=0.8, source="tfidf")]
    mock_embed.embed_query.return_value = [0.1]

    results_empty_tuning = await hs.search(
        SearchQuery(query="q", top_k=5, tuning=RetrievalTuning()), mock_embed
    )

    assert [r.chunk.id for r in results_no_tuning] == [r.chunk.id for r in results_empty_tuning]
    assert [r.score for r in results_no_tuning] == pytest.approx(
        [r.score for r in results_empty_tuning]
    )


