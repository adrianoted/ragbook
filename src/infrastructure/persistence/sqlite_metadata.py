import json
import os
from datetime import datetime
from uuid import UUID, uuid4

import aiosqlite

from src.domain.entities import Collection, Chunk, Document
from src.domain.enums import DocumentType
from src.domain.ports.metadata_store_port import MetadataStorePort


class SqliteMetadataStore(MetadataStorePort):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    embedding_model TEXT DEFAULT '',
                    embedding_dimension INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    collection_id TEXT REFERENCES collections(id),
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    content TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    chunk_index INTEGER NOT NULL
                )
                """
            )
            await db.commit()

    async def create_collection(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "",
        embedding_dimension: int = 0,
    ) -> Collection:
        collection = Collection(
            id=uuid4(),
            name=name,
            description=description,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            created_at=datetime.now(),
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO collections (id, name, description, embedding_model, embedding_dimension, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(collection.id),
                    collection.name,
                    collection.description,
                    collection.embedding_model,
                    collection.embedding_dimension,
                    collection.created_at.isoformat(),
                ),
            )
            await db.commit()
        return collection

    async def list_collections(self) -> list[Collection]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM collections")
            rows = await cursor.fetchall()
        return [
            Collection(
                id=UUID(row["id"]),
                name=row["name"],
                description=row["description"],
                embedding_model=row["embedding_model"] or "",
                embedding_dimension=row["embedding_dimension"] or 0,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def get_collection(self, collection_id: str) -> Collection | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return Collection(
            id=UUID(row["id"]),
            name=row["name"],
            description=row["description"],
            embedding_model=row["embedding_model"] or "",
            embedding_dimension=row["embedding_dimension"] or 0,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def delete_collection(self, collection_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            doc_cursor = await db.execute(
                "SELECT id FROM documents WHERE collection_id = ?", (collection_id,)
            )
            doc_rows = await doc_cursor.fetchall()
            for doc_row in doc_rows:
                await db.execute("DELETE FROM chunks WHERE document_id = ?", (doc_row[0],))
            await db.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            await db.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            await db.commit()

    async def save_document(self, document: Document) -> Document:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO documents (id, filename, document_type, collection_id, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(document.id),
                    document.filename,
                    document.document_type.value,
                    str(document.collection_id) if document.collection_id else None,
                    json.dumps(document.metadata),
                    document.created_at.isoformat(),
                ),
            )
            await db.commit()
        return document

    async def save_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "INSERT INTO chunks (id, document_id, content, metadata_json, chunk_index) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        str(chunk.id),
                        str(chunk.document_id),
                        chunk.content,
                        json.dumps(chunk.metadata),
                        chunk.index,
                    )
                    for chunk in chunks
                ],
            )
            await db.commit()
        return chunks

    async def get_documents_by_collection(self, collection_id: str) -> list[Document]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE collection_id = ?", (collection_id,)
            )
            rows = await cursor.fetchall()
        return [
            Document(
                id=UUID(row["id"]),
                filename=row["filename"],
                document_type=DocumentType(row["document_type"]),
                content="",
                metadata=json.loads(row["metadata_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                collection_id=UUID(row["collection_id"]) if row["collection_id"] else None,
            )
            for row in rows
        ]

    async def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM chunks WHERE document_id = ?", (document_id,)
            )
            rows = await cursor.fetchall()
        return [
            Chunk(
                id=UUID(row["id"]),
                document_id=UUID(row["document_id"]),
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                index=row["chunk_index"],
                embedding=None,
            )
            for row in rows
        ]

    async def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return Chunk(
            id=UUID(row["id"]),
            document_id=UUID(row["document_id"]),
            content=row["content"],
            metadata=json.loads(row["metadata_json"]),
            index=row["chunk_index"],
            embedding=None,
        )

    async def get_document(self, document_id: str) -> Document | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return Document(
            id=UUID(row["id"]),
            filename=row["filename"],
            document_type=DocumentType(row["document_type"]),
            content="",
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            collection_id=UUID(row["collection_id"]) if row["collection_id"] else None,
        )

    async def find_document_by_content_hash(
        self, collection_id: str, content_hash: str
    ) -> Document | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents "
                "WHERE collection_id = ? "
                "AND json_extract(metadata_json, '$.content_hash') = ?",
                (collection_id, content_hash),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return Document(
            id=UUID(row["id"]),
            filename=row["filename"],
            document_type=DocumentType(row["document_type"]),
            content="",
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            collection_id=UUID(row["collection_id"]) if row["collection_id"] else None,
        )

    async def delete_document(self, document_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            await db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            await db.commit()
