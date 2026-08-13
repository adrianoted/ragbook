"""Integration test fixtures for API layer."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Only stub modules genuinely missing from the environment
_MISSING_MODULES = [
    "langchain_community",
    "langchain_community.chat_models",
]

for mod_name in _MISSING_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


@pytest.fixture(autouse=True)
def _reset_embedding_state():
    """Isolate warm-up state between tests: the module-level dict in
    embedding_state is shared, so a test that mutates it must not leak into
    the next one."""
    from src.api import embedding_state

    embedding_state.reset()
    yield
    embedding_state.reset()


@pytest.fixture
def mock_ingest_use_case(sample_document, sample_chunks) -> AsyncMock:
    mock = AsyncMock()
    mock.execute.return_value = (sample_document, sample_chunks)
    return mock


@pytest.fixture
def mock_search_use_case(sample_search_results) -> AsyncMock:
    mock = AsyncMock()
    mock.execute.return_value = ("This is the generated answer.", sample_search_results)
    mock.execute_raw.return_value = sample_search_results
    return mock


@pytest.fixture
def mock_collection_use_case(sample_collection) -> AsyncMock:
    mock = AsyncMock()
    mock.create.return_value = sample_collection
    mock.list_collections.return_value = [sample_collection]
    mock.get.return_value = sample_collection
    mock.delete.return_value = None
    return mock


@pytest.fixture
def mock_document_use_case(sample_document) -> AsyncMock:
    mock = AsyncMock()
    mock.list_documents.return_value = [sample_document]
    mock.get_document.return_value = sample_document
    mock.delete.return_value = None
    return mock


@pytest.fixture
def test_app(
    mock_ingest_use_case,
    mock_search_use_case,
    mock_collection_use_case,
    mock_document_use_case,
    mock_metadata_store,
    mock_vector_store,
    tmp_path,
):
    """Create a FastAPI test app with dependency overrides (no Gradio)."""
    from fastapi import FastAPI
    from src.api.routers.ingest_router import router as ingest_router
    from src.api.routers.search_router import router as search_router
    from src.api.routers.collection_router import router as collection_router
    from src.api.routers.document_router import router as document_router
    from src.api.dependencies import (
        get_search_use_case,
        get_collection_use_case,
        get_document_use_case,
        get_metadata_store,
        get_vector_store,
        get_settings,
        get_embedding_dimension,
    )
    from src.config.settings import Settings

    app = FastAPI(title="RAGBook API Test")

    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(collection_router)
    app.include_router(document_router)

    from src.api import embedding_state

    @app.get("/api/health")
    async def health_check():
        return embedding_state.health_payload()

    test_settings = Settings(
        upload_dir=str(tmp_path / "uploads"),
        sqlite_db_path=":memory:",
    )

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_search_use_case] = lambda: mock_search_use_case
    app.dependency_overrides[get_collection_use_case] = lambda: mock_collection_use_case
    app.dependency_overrides[get_document_use_case] = lambda: mock_document_use_case
    app.dependency_overrides[get_metadata_store] = lambda: mock_metadata_store
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_embedding_dimension] = lambda: 128

    yield app

    app.dependency_overrides.clear()
