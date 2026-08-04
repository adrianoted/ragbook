from src.domain.entities import SearchQuery, SearchResult
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.search_port import SearchPort
from src.domain.ports.tfidf_port import TfidfPort


class TfidfSearch(SearchPort):
    def __init__(self, tfidf_port: TfidfPort) -> None:
        self._tfidf_port = tfidf_port

    async def search(
        self, query: SearchQuery, embedding_port: EmbeddingPort
    ) -> list[SearchResult]:
        collection_id = (
            str(query.collection_id) if query.collection_id else None
        )
        results = await self._tfidf_port.search(
            query=query.query,
            top_k=query.top_k,
            collection_id=collection_id,
        )
        return [
            SearchResult(chunk=r.chunk, score=r.score, source="tfidf")
            for r in results
        ]
