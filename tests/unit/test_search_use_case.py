from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.search_use_case import SearchUseCase, _NO_RESULTS_MSG
from src.domain.entities import SearchQuery, SearchResult


@pytest.fixture
def search_use_case(mock_search, mock_embedding, mock_llm):
    return SearchUseCase(search=mock_search, embedding=mock_embedding, llm=mock_llm)


@pytest.mark.asyncio
async def test_execute_returns_answer_and_results(
    search_use_case, mock_search, sample_search_query, sample_search_results,
):
    mock_search.search.return_value = sample_search_results

    answer, results = await search_use_case.execute(sample_search_query)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_execute_passes_context_to_llm(
    search_use_case, mock_search, mock_llm, sample_search_query, sample_search_results,
):
    mock_search.search.return_value = sample_search_results

    await search_use_case.execute(sample_search_query)

    mock_llm.generate.assert_called_once()
    call_args = mock_llm.generate.call_args
    context = call_args[0][1]  # second positional arg
    assert len(context) == len(sample_search_results)
    for result in sample_search_results:
        assert result.chunk.content in context


@pytest.mark.asyncio
async def test_execute_no_results(
    search_use_case, mock_search, mock_llm, sample_search_query,
):
    mock_search.search.return_value = []

    answer, results = await search_use_case.execute(sample_search_query)

    assert results == []
    assert "No sufficiently relevant results found." in answer
    mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_min_score_filters_cross_encoder_scores_after_rerank(
    mock_search, mock_embedding, mock_llm, sample_chunks,
):
    """With the reranker enabled, min_score applies to the (sigmoid)
    cross-encoder scores, and the reranker receives ALL candidates."""
    relevant = SearchResult(chunk=sample_chunks[0], score=0.95, source="hybrid")
    garbage = SearchResult(chunk=sample_chunks[1], score=0.02, source="hybrid")

    mock_search = AsyncMock()
    mock_search.search.return_value = [relevant, garbage]
    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = [relevant, garbage]

    use_case = SearchUseCase(
        search=mock_search,
        embedding=mock_embedding,
        llm=mock_llm,
        reranker=mock_reranker,
    )
    query = SearchQuery(query="anything", top_k=5, min_score=0.3)

    results = await use_case.execute_raw(query)

    assert results == [relevant]
    # No cut before diversification: the full candidate list is reranked
    assert mock_reranker.rerank.call_args.kwargs["top_k"] == 2


@pytest.mark.asyncio
async def test_execute_raw_returns_results_without_llm(
    search_use_case, mock_search, mock_llm, sample_search_query, sample_search_results,
):
    mock_search.search.return_value = sample_search_results

    results = await search_use_case.execute_raw(sample_search_query)

    assert results == sample_search_results
    mock_llm.generate.assert_not_called()


async def _async_tokens(*tokens):
    for token in tokens:
        yield token


@pytest.mark.asyncio
async def test_execute_stream_with_results_returns_sources_and_tokens(
    search_use_case, mock_search, mock_llm, sample_search_query, sample_search_results,
):
    mock_search.search.return_value = sample_search_results
    expected_tokens = ["Hello", " world", "!"]
    mock_llm.generate_stream = MagicMock(return_value=_async_tokens(*expected_tokens))

    sources, stream = await search_use_case.execute_stream(sample_search_query)

    assert sources == sample_search_results
    collected = [t async for t in stream]
    assert collected == expected_tokens


@pytest.mark.asyncio
async def test_execute_stream_no_results_yields_fallback(
    search_use_case, mock_search, mock_llm, sample_search_query,
):
    mock_search.search.return_value = []
    mock_llm.generate_stream = MagicMock()

    sources, stream = await search_use_case.execute_stream(sample_search_query)

    assert sources == []
    collected = [t async for t in stream]
    assert collected == [_NO_RESULTS_MSG]
    mock_llm.generate_stream.assert_not_called()


@pytest.mark.asyncio
async def test_execute_stream_passes_query_and_context_to_generate_stream(
    search_use_case, mock_search, mock_llm, sample_search_query, sample_search_results,
):
    mock_search.search.return_value = sample_search_results
    mock_llm.generate_stream = MagicMock(return_value=_async_tokens("ok"))

    sources, stream = await search_use_case.execute_stream(sample_search_query)
    # consume the stream to trigger the call
    [t async for t in stream]

    mock_llm.generate_stream.assert_called_once_with(
        sample_search_query.query,
        [r.chunk.content for r in sample_search_results],
    )


@pytest.mark.asyncio
async def test_execute_stream_propagates_llm_exception_during_iteration(
    search_use_case, mock_search, mock_llm, sample_search_query, sample_search_results,
):
    mock_search.search.return_value = sample_search_results

    async def _failing_gen():
        yield "first"
        raise RuntimeError("LLM error mid-stream")

    mock_llm.generate_stream = MagicMock(return_value=_failing_gen())

    sources, stream = await search_use_case.execute_stream(sample_search_query)
    assert sources == sample_search_results

    with pytest.raises(RuntimeError, match="LLM error mid-stream"):
        async for _ in stream:
            pass
