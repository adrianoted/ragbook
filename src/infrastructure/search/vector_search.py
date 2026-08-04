from src.domain.entities import SearchQuery, SearchResult
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.search_port import SearchPort
from src.domain.ports.vector_store_port import VectorStorePort


class VectorSearch(SearchPort):
    def __init__(self, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

    async def search(
        self, query: SearchQuery, embedding_port: EmbeddingPort
    ) -> list[SearchResult]:
        query_embedding = await embedding_port.embed_query(query.query)

        collection_id = str(query.collection_id) if query.collection_id else None

        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            collection_id=collection_id,
        )

        for result in results:
            result.source = "vector"

        return results
