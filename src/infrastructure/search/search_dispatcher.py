from src.domain.entities import SearchQuery, SearchResult
from src.domain.enums import SearchStrategy
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.search_port import SearchPort


class SearchDispatcher(SearchPort):
    def __init__(
        self,
        strategies: dict[SearchStrategy, SearchPort],
        default: SearchStrategy,
    ) -> None:
        self._strategies = strategies
        self._default = default

    async def search(
        self, query: SearchQuery, embedding_port: EmbeddingPort
    ) -> list[SearchResult]:
        strategy = query.strategy or self._default
        search_impl = self._strategies[strategy]
        return await search_impl.search(query, embedding_port)
