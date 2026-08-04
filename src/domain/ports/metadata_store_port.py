from abc import ABC, abstractmethod

from src.domain.entities import Chunk, Collection, Document


class MetadataStorePort(ABC):
    @abstractmethod
    async def init_db(self) -> None: ...

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "",
        embedding_dimension: int = 0,
    ) -> Collection: ...

    @abstractmethod
    async def list_collections(self) -> list[Collection]: ...

    @abstractmethod
    async def get_collection(self, collection_id: str) -> Collection | None: ...

    @abstractmethod
    async def delete_collection(self, collection_id: str) -> None: ...

    @abstractmethod
    async def save_document(self, document: Document) -> Document: ...

    @abstractmethod
    async def save_chunks(self, chunks: list[Chunk]) -> list[Chunk]: ...

    @abstractmethod
    async def get_documents_by_collection(self, collection_id: str) -> list[Document]: ...

    @abstractmethod
    async def get_chunks_by_document(self, document_id: str) -> list[Chunk]: ...

    @abstractmethod
    async def get_chunk_by_id(self, chunk_id: str) -> Chunk | None: ...

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None: ...

    @abstractmethod
    async def find_document_by_content_hash(
        self, collection_id: str, content_hash: str
    ) -> Document | None: ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> None: ...
