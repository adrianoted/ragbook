from abc import ABC, abstractmethod

from src.domain.entities import Chunk, SearchResult


class VectorStorePort(ABC):
    @abstractmethod
    async def add(self, chunks: list[Chunk], collection_id: str) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_id: str | None = None,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete_collection(self, collection_id: str) -> None:
        ...

    @abstractmethod
    async def delete_document(self, document_id: str, collection_id: str) -> None:
        ...

    @abstractmethod
    async def save(self) -> None:
        ...

    @abstractmethod
    async def load(self) -> None:
        ...
