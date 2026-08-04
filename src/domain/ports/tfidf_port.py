from abc import ABC, abstractmethod

from src.domain.entities import Chunk, SearchResult


class TfidfPort(ABC):
    @abstractmethod
    async def fit(self, chunks: list[Chunk], collection_id: str) -> None:
        ...

    @abstractmethod
    async def search(
        self, query: str, top_k: int, collection_id: str | None = None
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete_collection(self, collection_id: str) -> None:
        ...

    @abstractmethod
    async def delete_document(self, document_id: str, collection_id: str) -> None:
        ...
