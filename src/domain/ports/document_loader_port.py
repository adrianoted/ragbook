from abc import ABC, abstractmethod

from src.domain.entities import Document
from src.domain.enums import DocumentType


class DocumentLoaderPort(ABC):
    @abstractmethod
    async def load(self, file_path: str, document_type: DocumentType) -> Document:
        ...
