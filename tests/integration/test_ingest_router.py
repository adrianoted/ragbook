from unittest.mock import patch, AsyncMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_metadata_store
from src.domain.entities import Collection


@pytest.mark.asyncio
async def test_ingest_txt_file(test_app, mock_metadata_store, sample_document, sample_chunks, sample_collection):
    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = (sample_document, sample_chunks)
    mock_metadata_store.get_collection.return_value = sample_collection

    with patch(
        "src.api.routers.ingest_router.get_ingest_use_case",
        return_value=mock_use_case,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ingest",
                files={"file": ("test.txt", b"hello world", "text/plain")},
                data={"collection_id": str(sample_collection.id)},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == str(sample_document.id)
    assert data["filename"] == "test.txt"
    assert data["num_chunks"] == 3


@pytest.mark.asyncio
async def test_ingest_unsupported_extension(test_app, mock_metadata_store, sample_collection):
    mock_metadata_store.get_collection.return_value = sample_collection

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
            data={"collection_id": str(sample_collection.id)},
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_with_collection_id(test_app, mock_metadata_store, sample_document, sample_chunks, sample_collection):
    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = (sample_document, sample_chunks)
    mock_metadata_store.get_collection.return_value = sample_collection

    coll_id = str(sample_collection.id)

    with patch(
        "src.api.routers.ingest_router.get_ingest_use_case",
        return_value=mock_use_case,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ingest",
                files={"file": ("doc.pdf", b"pdf content", "application/pdf")},
                data={"collection_id": coll_id},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["collection_id"] == coll_id
    mock_use_case.execute.assert_called_once()
    call_kwargs = mock_use_case.execute.call_args[1]
    assert call_kwargs["collection_id"] == coll_id


@pytest.mark.asyncio
async def test_ingest_without_collection_id_returns_422(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_with_nonexistent_collection_returns_404(test_app, mock_metadata_store):
    mock_metadata_store.get_collection.return_value = None

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            data={"collection_id": str(uuid4())},
        )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ── guardia 409 (B10b) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_model_mismatch_returns_409(test_app, mock_metadata_store):
    from src.api.dependencies import get_embedding_dimension

    coll = Collection(
        name="old",
        embedding_model="text-embedding-ada-002",
        embedding_dimension=1536,
    )
    mock_metadata_store.get_collection.return_value = coll
    test_app.dependency_overrides[get_embedding_dimension] = lambda: 128

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            data={"collection_id": str(coll.id)},
        )

    assert response.status_code == 409
    assert "re-ingest required" in response.json()["detail"]


# ── dedup su content hash (B11a) ──────────────────────────────


@pytest.mark.asyncio
async def test_ingest_duplicate_content_returns_already_ingested(
    test_app, mock_metadata_store, sample_document, sample_chunks, sample_collection
):
    mock_metadata_store.get_collection.return_value = sample_collection
    mock_metadata_store.find_document_by_content_hash.return_value = sample_document
    mock_metadata_store.get_chunks_by_document.return_value = sample_chunks

    mock_use_case = AsyncMock()

    with patch(
        "src.api.routers.ingest_router.get_ingest_use_case",
        return_value=mock_use_case,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ingest",
                files={"file": ("reupload.txt", b"hello world", "text/plain")},
                data={"collection_id": str(sample_collection.id)},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["already_ingested"] is True
    assert data["document_id"] == str(sample_document.id)
    # the response echoes the just-uploaded filename, not the stored (uuid-prefixed) one
    assert data["filename"] == "reupload.txt"
    # num_chunks is the real count of the existing document's chunks
    assert data["num_chunks"] == len(sample_chunks)
    # no new ingest happened
    mock_use_case.execute.assert_not_called()
    mock_metadata_store.get_chunks_by_document.assert_called_once_with(str(sample_document.id))


@pytest.mark.asyncio
async def test_ingest_same_content_different_collection_ingests(
    test_app, mock_metadata_store, sample_document, sample_chunks, sample_collection
):
    mock_metadata_store.get_collection.return_value = sample_collection
    # dedup key includes the collection → no match in this collection
    mock_metadata_store.find_document_by_content_hash.return_value = None

    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = (sample_document, sample_chunks)

    with patch(
        "src.api.routers.ingest_router.get_ingest_use_case",
        return_value=mock_use_case,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ingest",
                files={"file": ("test.txt", b"hello world", "text/plain")},
                data={"collection_id": str(sample_collection.id)},
            )

    assert response.status_code == 200
    assert response.json()["already_ingested"] is False
    mock_use_case.execute.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_different_content_same_filename_ingests(
    test_app, mock_metadata_store, sample_document, sample_chunks, sample_collection
):
    mock_metadata_store.get_collection.return_value = sample_collection
    # filename is not part of the dedup key → different content, no match
    mock_metadata_store.find_document_by_content_hash.return_value = None

    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = (sample_document, sample_chunks)

    with patch(
        "src.api.routers.ingest_router.get_ingest_use_case",
        return_value=mock_use_case,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ingest",
                files={"file": ("test.txt", b"brand new content", "text/plain")},
                data={"collection_id": str(sample_collection.id)},
            )

    assert response.status_code == 200
    assert response.json()["already_ingested"] is False
    mock_use_case.execute.assert_called_once()
    # the hash of the actual bytes is forwarded to the use case
    assert mock_use_case.execute.call_args[1]["content_hash"] is not None
