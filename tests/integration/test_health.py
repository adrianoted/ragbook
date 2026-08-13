import pytest
from httpx import ASGITransport, AsyncClient

from src.api import embedding_state


async def _get_health(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        return await client.get("/api/health")


@pytest.mark.asyncio
async def test_health_reports_download_progress_while_warming(test_app):
    embedding_state.report_download_progress(4_000, 10_000)

    response = await _get_health(test_app)

    assert response.status_code == 200
    body = response.json()
    assert body["embedding_status"] == "warming"
    assert body["embedding_phase"] == "downloading"
    assert body["embedding_progress"] == {
        "downloaded_bytes": 4_000,
        "total_bytes": 10_000,
        "percent": 40.0,
    }


@pytest.mark.asyncio
async def test_health_reports_loading_phase_without_progress(test_app):
    embedding_state.mark_loading()

    response = await _get_health(test_app)

    assert response.status_code == 200
    body = response.json()
    assert body["embedding_status"] == "warming"
    assert body["embedding_phase"] == "loading"
    assert body["embedding_progress"] is None


@pytest.mark.asyncio
async def test_health_reports_ready(test_app):
    embedding_state.mark_ready()

    response = await _get_health(test_app)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedding_status": "ready",
        "embedding_phase": None,
        "embedding_progress": None,
    }


@pytest.mark.asyncio
async def test_health_reports_error_with_200_status(test_app):
    embedding_state.mark_error()

    response = await _get_health(test_app)

    assert response.status_code == 200
    body = response.json()
    assert body["embedding_status"] == "error"
    assert body["embedding_phase"] is None
    assert body["embedding_progress"] is None


@pytest.mark.asyncio
async def test_health_reports_warming_before_any_progress(test_app):
    response = await _get_health(test_app)

    assert response.status_code == 200
    body = response.json()
    assert body["embedding_status"] == "warming"
    assert body["embedding_phase"] is None
    assert body["embedding_progress"] is None
