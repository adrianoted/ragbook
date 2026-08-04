import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_create_collection(test_app, sample_collection):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/collections",
            json={"name": "my-collection", "description": "Test desc"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(sample_collection.id)
    assert data["name"] == sample_collection.name
    assert "created_at" in data


@pytest.mark.asyncio
async def test_list_collections(test_app, sample_collection):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/api/collections")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == sample_collection.name


@pytest.mark.asyncio
async def test_delete_collection(test_app, mock_collection_use_case):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete("/api/collections/some-collection-id")

    assert response.status_code == 200
    assert "deleted" in response.json()["detail"]
    mock_collection_use_case.delete.assert_called_once_with("some-collection-id")
