import asyncio
import logging
from uuid import UUID

from src.domain.entities import SearchQuery, SearchResult
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.search_port import SearchPort
from src.domain.ports.tfidf_port import TfidfPort
from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class HybridSearch(SearchPort):
    def __init__(
        self,
        vector_store: VectorStorePort,
        tfidf: TfidfPort,
        vector_weight: float = 0.7,
        fusion: str = "weighted",
    ) -> None:
        if fusion not in ("weighted", "rrf"):
            raise ValueError(
                f"Unknown fusion '{fusion}'; expected 'weighted' or 'rrf'"
            )
        self._vector_store = vector_store
        self._tfidf = tfidf
        self._vector_weight = vector_weight
        self._fusion = fusion

    async def search(
        self, query: SearchQuery, embedding_port: EmbeddingPort
    ) -> list[SearchResult]:
        collection_id = (
            str(query.collection_id) if query.collection_id else None
        )

        async def _vector_search() -> list[SearchResult]:
            try:
                query_embedding = await embedding_port.embed_query(query.query)
                return await self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=query.top_k,
                    collection_id=collection_id,
                )
            except ValueError:
                # Misconfiguration (e.g. missing collection_id) is a programming
                # error, not a transient failure — surface it, don't degrade.
                raise
            except Exception:
                logger.exception(
                    "Vector search failed; hybrid degrades to TF-IDF only"
                )
                return []

        async def _tfidf_search() -> list[SearchResult]:
            try:
                return await self._tfidf.search(
                    query=query.query,
                    top_k=query.top_k,
                    collection_id=collection_id,
                )
            except ValueError:
                # Misconfiguration (e.g. missing collection_id) is a programming
                # error, not a transient failure — surface it, don't degrade.
                raise
            except Exception:
                logger.exception(
                    "TF-IDF search failed; hybrid degrades to vector only"
                )
                return []

        vector_results, tfidf_results = await asyncio.gather(
            _vector_search(), _tfidf_search()
        )

        if not vector_results and not tfidf_results:
            return []

        if not vector_results:
            return [
                SearchResult(chunk=r.chunk, score=r.score, source="hybrid")
                for r in tfidf_results[: query.top_k]
            ]

        if not tfidf_results:
            return [
                SearchResult(chunk=r.chunk, score=r.score, source="hybrid")
                for r in vector_results[: query.top_k]
            ]

        if self._fusion == "rrf":
            fused = _fuse_rrf(vector_results, tfidf_results)
        else:
            fused = _fuse_weighted(
                vector_results, tfidf_results, self._vector_weight
            )

        return fused[: query.top_k]


def _fuse_weighted(
    vector_results: list[SearchResult],
    tfidf_results: list[SearchResult],
    vector_weight: float,
) -> list[SearchResult]:
    """Max-normalize each list, then sum weighted scores merged by chunk id.

    Returns the full fused list ordered desc (source="hybrid"); the top_k cut
    stays in the caller.
    """
    vector_scores = _normalize_scores(vector_results)
    tfidf_scores = _normalize_scores(tfidf_results)

    combined: dict[UUID, tuple[SearchResult, float]] = {}

    for result, norm_score in zip(vector_results, vector_scores):
        score = vector_weight * norm_score
        combined[result.chunk.id] = (result, score)

    for result, norm_score in zip(tfidf_results, tfidf_scores):
        tfidf_score = (1 - vector_weight) * norm_score
        chunk_id = result.chunk.id
        if chunk_id in combined:
            existing_result, existing_score = combined[chunk_id]
            combined[chunk_id] = (existing_result, existing_score + tfidf_score)
        else:
            combined[chunk_id] = (result, tfidf_score)

    sorted_results = sorted(
        combined.values(), key=lambda x: x[1], reverse=True
    )

    return [
        SearchResult(chunk=result.chunk, score=score, source="hybrid")
        for result, score in sorted_results
    ]


def _fuse_rrf(
    vector_results: list[SearchResult],
    tfidf_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion (Cormack et al.), k=60, rank 1-based.

    Each chunk's score accrues ``1 / (k + rank)`` from every list it appears in;
    merged by chunk id (the first SearchResult seen is kept, as in weighted).
    Returns the full fused list ordered desc (source="hybrid"). k is a fixed
    constant, not configurable from settings.
    """
    combined: dict[UUID, tuple[SearchResult, float]] = {}

    for results in (vector_results, tfidf_results):
        for rank, result in enumerate(results, start=1):
            contribution = 1 / (k + rank)
            chunk_id = result.chunk.id
            if chunk_id in combined:
                existing_result, existing_score = combined[chunk_id]
                combined[chunk_id] = (
                    existing_result,
                    existing_score + contribution,
                )
            else:
                combined[chunk_id] = (result, contribution)

    sorted_results = sorted(
        combined.values(), key=lambda x: x[1], reverse=True
    )

    return [
        SearchResult(chunk=result.chunk, score=score, source="hybrid")
        for result, score in sorted_results
    ]


def _normalize_scores(results: list[SearchResult]) -> list[float]:
    if not results:
        return []

    scores = [r.score for r in results]
    max_score = max(scores)

    if max_score == 0:
        return [0.0] * len(scores)

    return [s / max_score for s in scores]
