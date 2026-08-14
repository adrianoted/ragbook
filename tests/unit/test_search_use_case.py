from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.search_use_case import SearchUseCase, _NO_RESULTS_MSG
from src.domain.entities import (
    LlmOptions,
    RetrievalTuning,
    SearchQuery,
    SearchResult,
)


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
        options=sample_search_query.llm_options,
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


# ── Per-request override resolution (task 2) ──────────────────


@pytest.mark.asyncio
async def test_reranker_enabled_per_request_overrides_disabled_default(
    mock_search, mock_embedding, mock_llm, sample_search_results,
):
    """Reranker off by default in the ctor, but the request turns it on:
    rerank runs and the over-fetch factor is 4x."""
    mock_search.search.return_value = sample_search_results
    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = sample_search_results

    use_case = SearchUseCase(
        search=mock_search,
        embedding=mock_embedding,
        llm=mock_llm,
        reranker=mock_reranker,
        reranker_enabled=False,
    )
    query = SearchQuery(
        query="anything", top_k=5, tuning=RetrievalTuning(reranker_enabled=True)
    )

    await use_case.execute_raw(query)

    mock_reranker.rerank.assert_called_once()
    overfetch_query = mock_search.search.call_args[0][0]
    assert overfetch_query.top_k == 5 * 4


@pytest.mark.asyncio
async def test_reranker_disabled_per_request_overrides_enabled_default(
    mock_search, mock_embedding, mock_llm, sample_search_results,
):
    """Reranker on by default in the ctor, but the request turns it off:
    rerank does not run and the over-fetch factor is 3x."""
    mock_search.search.return_value = sample_search_results
    mock_reranker = AsyncMock()

    use_case = SearchUseCase(
        search=mock_search,
        embedding=mock_embedding,
        llm=mock_llm,
        reranker=mock_reranker,
        reranker_enabled=True,
    )
    query = SearchQuery(
        query="anything", top_k=5, tuning=RetrievalTuning(reranker_enabled=False)
    )

    await use_case.execute_raw(query)

    mock_reranker.rerank.assert_not_called()
    overfetch_query = mock_search.search.call_args[0][0]
    assert overfetch_query.top_k == 5 * 3


@pytest.mark.asyncio
async def test_no_override_keeps_default_reranker_behavior(
    mock_search, mock_embedding, mock_llm, sample_search_results,
):
    """No per-request override: reranker follows the ctor default (on),
    rerank runs and the over-fetch factor is 4x."""
    mock_search.search.return_value = sample_search_results
    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = sample_search_results

    use_case = SearchUseCase(
        search=mock_search,
        embedding=mock_embedding,
        llm=mock_llm,
        reranker=mock_reranker,
    )
    query = SearchQuery(query="anything", top_k=5)

    await use_case.execute_raw(query)

    mock_reranker.rerank.assert_called_once()
    overfetch_query = mock_search.search.call_args[0][0]
    assert overfetch_query.top_k == 5 * 4


@pytest.mark.asyncio
async def test_reranker_override_ignored_when_no_reranker_wired(
    mock_search, mock_embedding, mock_llm, sample_search_results,
):
    """No reranker instance wired: a per-request enable cannot conjure one,
    no crash, over-fetch stays 3x."""
    mock_search.search.return_value = sample_search_results

    use_case = SearchUseCase(
        search=mock_search,
        embedding=mock_embedding,
        llm=mock_llm,
        reranker=None,
    )
    query = SearchQuery(
        query="anything", top_k=5, tuning=RetrievalTuning(reranker_enabled=True)
    )

    results = await use_case.execute_raw(query)

    assert isinstance(results, list)
    overfetch_query = mock_search.search.call_args[0][0]
    assert overfetch_query.top_k == 5 * 3


@pytest.mark.asyncio
async def test_max_results_per_document_override_reaches_diversifier(
    mock_search, mock_embedding, mock_llm, sample_chunks,
):
    """A per-request cap of 1 overrides the ctor default of 3: at most one
    chunk per document survives even though all candidates share a document."""
    candidates = [
        SearchResult(chunk=sample_chunks[0], score=0.95, source="vector"),
        SearchResult(chunk=sample_chunks[1], score=0.90, source="vector"),
        SearchResult(chunk=sample_chunks[2], score=0.85, source="vector"),
    ]
    mock_search.search.return_value = candidates

    use_case = SearchUseCase(
        search=mock_search,
        embedding=mock_embedding,
        llm=mock_llm,
        max_results_per_document=3,
    )
    query = SearchQuery(
        query="anything", top_k=5, tuning=RetrievalTuning(max_results_per_document=1)
    )

    results = await use_case.execute_raw(query)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_execute_forwards_llm_options_to_generate(
    mock_search, mock_embedding, mock_llm, sample_search_results,
):
    mock_search.search.return_value = sample_search_results
    options = LlmOptions(temperature=0.1, think=False, num_ctx=4096)
    query = SearchQuery(query="anything", top_k=5, llm_options=options)

    use_case = SearchUseCase(search=mock_search, embedding=mock_embedding, llm=mock_llm)
    await use_case.execute(query)

    assert mock_llm.generate.call_args.kwargs["options"] == options


@pytest.mark.asyncio
async def test_execute_stream_forwards_llm_options_to_generate_stream(
    mock_search, mock_embedding, mock_llm, sample_search_results,
):
    mock_search.search.return_value = sample_search_results
    mock_llm.generate_stream = MagicMock(return_value=_async_tokens("ok"))
    options = LlmOptions(temperature=0.7)
    query = SearchQuery(query="anything", top_k=5, llm_options=options)

    use_case = SearchUseCase(search=mock_search, embedding=mock_embedding, llm=mock_llm)
    sources, stream = await use_case.execute_stream(query)
    [t async for t in stream]

    assert mock_llm.generate_stream.call_args.kwargs["options"] == options


@pytest.mark.asyncio
async def test_execute_forwards_empty_llm_options_when_none_specified(
    mock_search, mock_embedding, mock_llm, sample_search_results,
):
    """A SearchQuery with no explicit llm_options passes an empty LlmOptions
    (all fields None) to the LLM, never None."""
    mock_search.search.return_value = sample_search_results
    query = SearchQuery(query="anything", top_k=5)

    use_case = SearchUseCase(search=mock_search, embedding=mock_embedding, llm=mock_llm)
    await use_case.execute(query)

    forwarded = mock_llm.generate.call_args.kwargs["options"]
    assert forwarded == LlmOptions()
    assert forwarded is not None
