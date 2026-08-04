from abc import ABC, abstractmethod

from src.domain.entities import SearchResult


class RerankerPort(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]: ...
