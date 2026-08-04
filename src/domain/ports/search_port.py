from abc import ABC, abstractmethod

from src.domain.entities import SearchQuery, SearchResult
from src.domain.ports.embedding_port import EmbeddingPort


class SearchPort(ABC):
    @abstractmethod
    async def search(
        self, query: SearchQuery, embedding_port: EmbeddingPort
    ) -> list[SearchResult]:
        ...
