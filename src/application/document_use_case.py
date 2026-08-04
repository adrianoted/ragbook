from pathlib import Path

from src.domain.entities import Document
from src.domain.ports.metadata_store_port import MetadataStorePort
from src.domain.ports.tfidf_port import TfidfPort
from src.domain.ports.vector_store_port import VectorStorePort


class DocumentUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        tfidf: TfidfPort,
        metadata_store: MetadataStorePort,
        upload_dir: str = "data/uploads",
    ) -> None:
        self._vector_store = vector_store
        self._tfidf = tfidf
        self._metadata_store = metadata_store
        self._upload_dir = Path(upload_dir)

    async def list_documents(self, collection_id: str) -> list[Document]:
        return await self._metadata_store.get_documents_by_collection(collection_id)

    async def get_document(self, document_id: str) -> Document | None:
        return await self._metadata_store.get_document(document_id)

    async def delete(self, document_id: str) -> None:
        document = await self._metadata_store.get_document(document_id)
        if document is None:
            return

        collection_id = str(document.collection_id) if document.collection_id else str(document.id)

        await self._vector_store.delete_document(document_id, collection_id)
        await self._vector_store.save()
        await self._tfidf.delete_document(document_id, collection_id)
        await self._metadata_store.delete_document(document_id)

        upload_file = self._upload_dir / document.filename
        if upload_file.exists():
            upload_file.unlink()
