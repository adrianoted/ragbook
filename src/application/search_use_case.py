from collections.abc import AsyncIterator

from src.domain.entities import SearchQuery, SearchResult
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.llm_port import LlmPort
from src.domain.ports.reranker_port import RerankerPort
from src.domain.ports.search_port import SearchPort
from src.domain.query_mode import parse_mode
from src.application.result_diversifier import diversify_results

_NO_RESULTS_MSG = "No sufficiently relevant results found."


async def _fallback_stream() -> AsyncIterator[str]:
    yield _NO_RESULTS_MSG


class SearchUseCase:
    def __init__(
        self,
        search: SearchPort,
        embedding: EmbeddingPort,
        llm: LlmPort,
        reranker: RerankerPort | None = None,
        max_results_per_document: int = 2,
        reranker_enabled: bool = True,
    ) -> None:
        self._search = search
        self._embedding = embedding
        self._llm = llm
        self._reranker = reranker
        self._max_results_per_document = max_results_per_document
        self._reranker_enabled = reranker_enabled

    async def _retrieve(self, query: SearchQuery) -> list[SearchResult]:
        """Shared retrieval pipeline: over-fetch → rerank → filter → diversify."""
        # Strip any /mode prefix so retrieval scores the actual question,
        # not the slash-command (the LLM adapters re-parse it for the answer style)
        _, clean_query = parse_mode(query.query)

        # Resolve the effective reranker flag before over-fetch: a per-request
        # override wins over the ctor default, but `self._reranker is not None`
        # stays the safety guard — no override can switch on a reranker that
        # was never wired.
        tuning = query.tuning
        requested = tuning.reranker_enabled
        use_reranker = self._reranker is not None and (
            requested if requested is not None else self._reranker_enabled
        )

        # Over-fetch: 4x if reranker enabled, 3x otherwise
        overfetch_factor = 4 if use_reranker else 3
        overfetch_query = query.model_copy(
            update={"query": clean_query, "top_k": query.top_k * overfetch_factor}
        )
        results = await self._search.search(overfetch_query, self._embedding)

        # Rerank ALL candidates (no cut here): the diversifier below needs the
        # full reordered list to backfill when a document exceeds its cap
        if use_reranker and results:
            results = await self._reranker.rerank(
                clean_query, results, top_k=len(results)
            )

        # Filter AFTER reranking: cross-encoder sigmoid scores are on an
        # absolute, query-independent scale. Retrieval scores are
        # max-normalized per batch (top ≈ 1 even for off-topic queries), so a
        # threshold on them cannot detect the "nothing relevant" case.
        # The scale (sigmoid vs raw) depends on the request's *effective*
        # reranker (`use_reranker`), not the global config: without a reranker
        # for this request the filter falls back to raw retrieval scores.
        if query.min_score is not None:
            results = [r for r in results if r.score >= query.min_score]

        max_per_doc = (
            tuning.max_results_per_document
            if tuning.max_results_per_document is not None
            else self._max_results_per_document
        )
        return diversify_results(
            results,
            top_k=query.top_k,
            max_per_document=max_per_doc,
        )

    async def execute(self, query: SearchQuery) -> tuple[str, list[SearchResult]]:
        results = await self._retrieve(query)

        if not results:
            return _NO_RESULTS_MSG, []

        context = [result.chunk.content for result in results]
        answer = await self._llm.generate(
            query.query, context, options=query.llm_options
        )

        return answer, results

    async def execute_stream(
        self, query: SearchQuery
    ) -> tuple[list[SearchResult], AsyncIterator[str]]:
        results = await self._retrieve(query)

        if not results:
            return [], _fallback_stream()

        return results, self._llm.generate_stream(
            query.query,
            [r.chunk.content for r in results],
            options=query.llm_options,
        )

    async def execute_raw(self, query: SearchQuery) -> list[SearchResult]:
        return await self._retrieve(query)
