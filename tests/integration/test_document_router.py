import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_list_documents(test_app, sample_collection, sample_document):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/collections/{sample_collection.id}/documents")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == str(sample_document.id)
    assert data[0]["filename"] == sample_document.filename


@pytest.mark.asyncio
async def test_list_documents_unknown_collection_returns_404(
    test_app, sample_collection, mock_collection_use_case
):
    mock_collection_use_case.get.return_value = None

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/collections/{sample_collection.id}/documents")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_documents_malformed_collection_id_returns_400(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/api/collections/not-a-uuid/documents")

    assert response.status_code == 400
    assert "Invalid collection_id" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_document(
    test_app, sample_collection, sample_document, mock_document_use_case
):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete(
            f"/api/collections/{sample_collection.id}/documents/{sample_document.id}"
        )

    assert response.status_code == 200
    assert "deleted" in response.json()["detail"]
    mock_document_use_case.delete.assert_called_once_with(str(sample_document.id))


@pytest.mark.asyncio
async def test_delete_document_unknown_collection_returns_404(
    test_app, sample_collection, sample_document, mock_collection_use_case,
    mock_document_use_case,
):
    mock_collection_use_case.get.return_value = None

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete(
            f"/api/collections/{sample_collection.id}/documents/{sample_document.id}"
        )

    assert response.status_code == 404
    mock_document_use_case.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_unknown_document_returns_404(
    test_app, sample_collection, mock_document_use_case
):
    mock_document_use_case.get_document.return_value = None

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete(
            f"/api/collections/{sample_collection.id}/documents/missing-doc-id"
        )

    assert response.status_code == 404
    mock_document_use_case.delete.assert_not_called()
