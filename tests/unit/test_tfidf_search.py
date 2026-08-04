from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.domain.entities import Chunk, SearchQuery, SearchResult
from src.infrastructure.search.tfidf_search import TfidfSearch


def _make_chunk(**kwargs):
    defaults = dict(document_id=uuid4(), content="c", metadata={}, index=0)
    defaults.update(kwargs)
    return Chunk(**defaults)


@pytest.mark.asyncio
async def test_search_calls_tfidf_port():
    chunk = _make_chunk()
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk, score=0.8, source="tfidf")]

    mock_embed = AsyncMock()

    ts = TfidfSearch(tfidf_port=mock_tfidf)
    query = SearchQuery(query="hello", top_k=3)
    await ts.search(query, mock_embed)

    mock_tfidf.search.assert_called_once_with(query="hello", top_k=3, collection_id=None)


@pytest.mark.asyncio
async def test_search_sets_source_tfidf():
    chunk = _make_chunk()
    mock_tfidf = AsyncMock()
    mock_tfidf.search.return_value = [SearchResult(chunk=chunk, score=0.8, source="vector")]

    ts = TfidfSearch(tfidf_port=mock_tfidf)
    results = await ts.search(SearchQuery(query="q"), AsyncMock())

    assert all(r.source == "tfidf" for r in results)
