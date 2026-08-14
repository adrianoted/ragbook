import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_settings
from src.api.routers.config_router import router as config_router
from src.config.settings import Settings

_EIGHT_DEFAULT_KEYS = {
    "min_score",
    "fusion",
    "hybrid_vector_weight",
    "max_results_per_document",
    "reranker_enabled",
    "llm_temperature",
    "llm_think",
    "llm_num_ctx",
}


def _make_app(**settings_kwargs) -> FastAPI:
    app = FastAPI()
    app.include_router(config_router)
    override_settings = Settings(**settings_kwargs)
    app.dependency_overrides[get_settings] = lambda: override_settings
    return app


@pytest.mark.asyncio
async def test_config_returns_200_with_three_top_level_keys():
    async with AsyncClient(
        transport=ASGITransport(app=_make_app()), base_url="http://test"
    ) as client:
        response = await client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "llm_provider" in data
    assert "defaults" in data
    assert "ranges" in data


@pytest.mark.asyncio
async def test_defaults_contains_exactly_eight_params():
    async with AsyncClient(
        transport=ASGITransport(app=_make_app()), base_url="http://test"
    ) as client:
        response = await client.get("/api/config")
    defaults = response.json()["defaults"]
    assert set(defaults.keys()) == _EIGHT_DEFAULT_KEYS


@pytest.mark.asyncio
async def test_defaults_min_score_uses_rerank_min_score_when_reranker_enabled():
    app = _make_app(reranker_enabled=True, rerank_min_score=0.42, min_score=0.1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/config")
    assert response.json()["defaults"]["min_score"] == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_defaults_min_score_uses_min_score_when_reranker_disabled():
    app = _make_app(reranker_enabled=False, rerank_min_score=0.42, min_score=0.1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/config")
    assert response.json()["defaults"]["min_score"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_ranges_llm_num_ctx_values():
    async with AsyncClient(
        transport=ASGITransport(app=_make_app()), base_url="http://test"
    ) as client:
        response = await client.get("/api/config")
    assert response.json()["ranges"]["llm_num_ctx"] == {"min": 2048, "max": 32768, "step": 2048}


@pytest.mark.asyncio
async def test_payload_contains_no_secrets():
    async with AsyncClient(
        transport=ASGITransport(app=_make_app()), base_url="http://test"
    ) as client:
        response = await client.get("/api/config")
    serialized = response.text
    for forbidden in ("api_key", "path", "url"):
        assert forbidden not in serialized, f"Leaked field containing {forbidden!r}"
