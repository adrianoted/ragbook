from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.domain.entities import Chunk, SearchQuery, SearchResult
from src.infrastructure.search.vector_search import VectorSearch


def _make_chunk(**kwargs):
    defaults = dict(document_id=uuid4(), content="c", metadata={}, index=0)
    defaults.update(kwargs)
    return Chunk(**defaults)


@pytest.mark.asyncio
async def test_search_calls_embedding_and_store():
    mock_store = AsyncMock()
    chunk = _make_chunk()
    mock_store.search.return_value = [SearchResult(chunk=chunk, score=0.9, source="vector")]

    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]

    vs = VectorSearch(vector_store=mock_store)
    query = SearchQuery(query="hello", top_k=3)
    await vs.search(query, mock_embed)

    mock_embed.embed_query.assert_called_once_with("hello")
    mock_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3], top_k=3, collection_id=None
    )


@pytest.mark.asyncio
async def test_search_sets_source_vector():
    mock_store = AsyncMock()
    chunk = _make_chunk()
    mock_store.search.return_value = [SearchResult(chunk=chunk, score=0.9, source="other")]

    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1]

    vs = VectorSearch(vector_store=mock_store)
    results = await vs.search(SearchQuery(query="q"), mock_embed)

    assert all(r.source == "vector" for r in results)
