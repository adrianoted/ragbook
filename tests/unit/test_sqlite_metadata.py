from uuid import uuid4

import pytest
import pytest_asyncio

from src.domain.entities import Document
from src.domain.enums import DocumentType
from src.infrastructure.persistence.sqlite_metadata import SqliteMetadataStore


@pytest_asyncio.fixture
async def store(tmp_path) -> SqliteMetadataStore:
    s = SqliteMetadataStore(str(tmp_path / "meta.db"))
    await s.init_db()
    return s


@pytest.mark.asyncio
async def test_find_document_by_content_hash_returns_match(store):
    coll_id = uuid4()
    content_hash = "b" * 64
    doc = Document(
        filename="test.txt",
        document_type=DocumentType.TEXT,
        content="",
        metadata={"content_hash": content_hash},
        collection_id=coll_id,
    )
    await store.save_document(doc)

    found = await store.find_document_by_content_hash(str(coll_id), content_hash)

    assert found is not None
    assert found.id == doc.id


@pytest.mark.asyncio
async def test_find_document_by_content_hash_scoped_to_collection(store):
    content_hash = "c" * 64
    doc = Document(
        filename="test.txt",
        document_type=DocumentType.TEXT,
        content="",
        metadata={"content_hash": content_hash},
        collection_id=uuid4(),
    )
    await store.save_document(doc)

    found = await store.find_document_by_content_hash(str(uuid4()), content_hash)

    assert found is None


@pytest.mark.asyncio
async def test_find_document_by_content_hash_legacy_no_false_positive(store):
    """A document with no content_hash in its metadata must never match a lookup."""
    coll_id = uuid4()
    legacy = Document(
        filename="legacy.txt",
        document_type=DocumentType.TEXT,
        content="",
        metadata={"size": 10},
        collection_id=coll_id,
    )
    await store.save_document(legacy)

    found = await store.find_document_by_content_hash(str(coll_id), "d" * 64)

    assert found is None
