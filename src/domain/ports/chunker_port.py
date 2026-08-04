from abc import ABC, abstractmethod

from src.domain.entities import Chunk, Document


class ChunkerPort(ABC):
    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        ...
