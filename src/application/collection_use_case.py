import asyncio
from pathlib import Path
from uuid import UUID

from src.domain.entities import Collection
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.ports.metadata_store_port import MetadataStorePort
from src.domain.ports.tfidf_port import TfidfPort


class CollectionUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        tfidf: TfidfPort,
        metadata_store: MetadataStorePort,
        upload_dir: str = "data/uploads",
        embedding_model: str = "",
        embedding: EmbeddingPort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._tfidf = tfidf
        self._metadata_store = metadata_store
        self._upload_dir = Path(upload_dir)
        self._embedding_model = embedding_model
        # The port, not its dimension: `dimension()` loads the model, and only
        # `create` needs it — list/get/delete must not pay for it.
        self._embedding = embedding

    async def create(self, name: str, description: str = "") -> Collection:
        dimension = 0
        if self._embedding is not None:
            # Sync call that may load the model: keep it off the event loop.
            dimension = await asyncio.to_thread(self._embedding.dimension)
        return await self._metadata_store.create_collection(
            name, description,
            embedding_model=self._embedding_model,
            embedding_dimension=dimension,
        )

    async def list_collections(self) -> list[Collection]:
        return await self._metadata_store.list_collections()

    async def get(self, collection_id: str) -> Collection | None:
        return await self._metadata_store.get_collection(collection_id)

    async def delete(self, collection_id: str) -> None:
        documents = await self._metadata_store.get_documents_by_collection(
            collection_id
        )

        await self._vector_store.delete_collection(collection_id)
        await self._tfidf.delete_collection(collection_id)
        await self._metadata_store.delete_collection(collection_id)

        for doc in documents:
            upload_file = self._upload_dir / doc.filename
            if upload_file.exists():
                upload_file.unlink()
