import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from src.domain.entities import Collection


def _parse_sse(text: str) -> list[dict]:
    events = []
    current: dict = {}
    for line in text.splitlines():
        if line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[6:])
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


@pytest.mark.asyncio
async def test_search_returns_answer_and_sources(test_app, sample_collection):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={
                "query": "What is testing?",
                "collection_id": str(sample_collection.id),
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is the generated answer."
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 2
    assert "chunk_content" in data["sources"][0]
    assert "score" in data["sources"][0]
    assert "document_filename" in data["sources"][0]


@pytest.mark.asyncio
async def test_search_raw_returns_sources_only(test_app, sample_collection):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search/raw",
            json={
                "query": "What is testing?",
                "collection_id": str(sample_collection.id),
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" not in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 2


@pytest.mark.asyncio
async def test_search_with_strategy(test_app, mock_search_use_case, sample_collection):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "strategy": "vector",
            },
        )

    assert response.status_code == 200
    call_args = mock_search_use_case.execute.call_args
    search_query = call_args[0][0]
    assert search_query.strategy.value == "vector"


@pytest.mark.asyncio
async def test_search_with_unknown_strategy_returns_422(test_app, sample_collection):
    """strategy is typed as the enum: unknown values are rejected by Pydantic,
    not by SearchStrategy() raising ValueError inside the handler (→ 500)."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search/raw",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "strategy": "bm25",
            },
        )

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, -3, 101])
@pytest.mark.asyncio
async def test_search_with_out_of_range_top_k_returns_422(
    test_app, sample_collection, top_k
):
    """top_k reaches FAISS unvalidated: 0 crashes the vector branch with a 500."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search/raw",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "top_k": top_k,
                "strategy": "vector",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_without_collection_id_is_rejected(test_app):
    """collection_id is required (§3.7): a search always targets one collection."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post("/api/search", json={"query": "test"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_with_malformed_collection_id_returns_400(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": "not-a-uuid"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_search_with_unknown_collection_returns_404(
    test_app, mock_collection_use_case
):
    from uuid import uuid4

    mock_collection_use_case.get.return_value = None
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(uuid4())},
        )

    assert response.status_code == 404


# ── /api/search/stream ────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_stream_happy_path(
    test_app, mock_search_use_case, sample_search_results, sample_collection
):
    async def token_gen():
        for tok in ["tok1", "tok2", "tok3"]:
            yield tok

    mock_search_use_case.execute_stream.return_value = (sample_search_results, token_gen())

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search/stream",
            json={"query": "test", "collection_id": str(sample_collection.id)},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert len(events) == 5  # sources + 3 delta + done
    assert events[0]["event"] == "sources"
    assert len(events[0]["data"]) == 2
    assert "chunk_content" in events[0]["data"][0]
    assert events[1] == {"event": "delta", "data": {"text": "tok1"}}
    assert events[2] == {"event": "delta", "data": {"text": "tok2"}}
    assert events[3] == {"event": "delta", "data": {"text": "tok3"}}
    assert events[4] == {"event": "done", "data": {}}


@pytest.mark.asyncio
async def test_search_stream_malformed_collection_returns_400(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search/stream",
            json={"query": "test", "collection_id": "not-a-uuid"},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_search_stream_unknown_collection_returns_404(test_app, mock_collection_use_case):
    mock_collection_use_case.get.return_value = None
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search/stream",
            json={"query": "test", "collection_id": str(uuid4())},
        )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_search_stream_error_mid_stream(
    test_app, mock_search_use_case, sample_search_results, sample_collection
):
    async def error_gen():
        yield "first token"
        raise RuntimeError("LLM crashed")

    mock_search_use_case.execute_stream.return_value = (sample_search_results, error_gen())

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search/stream",
            json={"query": "test", "collection_id": str(sample_collection.id)},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0]["event"] == "sources"
    assert events[1] == {"event": "delta", "data": {"text": "first token"}}
    assert events[2]["event"] == "error"
    assert "LLM crashed" in events[2]["data"]["detail"]
    assert not any(e["event"] == "done" for e in events)


@pytest.mark.asyncio
async def test_search_stream_zero_results(
    test_app, mock_search_use_case, sample_collection
):
    async def fallback_gen():
        yield "No sufficiently relevant results found."

    mock_search_use_case.execute_stream.return_value = ([], fallback_gen())

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search/stream",
            json={"query": "test", "collection_id": str(sample_collection.id)},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0] == {"event": "sources", "data": []}
    assert events[1] == {"event": "delta", "data": {"text": "No sufficiently relevant results found."}}
    assert events[2] == {"event": "done", "data": {}}


# ── guardia 409 (B10b) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_model_mismatch_returns_409(test_app, mock_collection_use_case):
    from src.api.dependencies import get_embedding_dimension

    coll = Collection(
        name="old",
        embedding_model="text-embedding-ada-002",
        embedding_dimension=1536,
    )
    mock_collection_use_case.get.return_value = coll
    test_app.dependency_overrides[get_embedding_dimension] = lambda: 128

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(coll.id)},
        )

    assert response.status_code == 409
    assert "re-ingest required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_dimension_mismatch_returns_409(test_app, mock_collection_use_case):
    from src.api.dependencies import get_embedding_dimension

    coll = Collection(
        name="old-dim",
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dimension=512,
    )
    mock_collection_use_case.get.return_value = coll
    test_app.dependency_overrides[get_embedding_dimension] = lambda: 128

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(coll.id)},
        )

    assert response.status_code == 409
    assert "re-ingest required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_matching_model_returns_200(test_app, mock_collection_use_case):
    from src.api.dependencies import get_embedding_dimension

    coll = Collection(
        name="current",
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dimension=128,
    )
    mock_collection_use_case.get.return_value = coll
    test_app.dependency_overrides[get_embedding_dimension] = lambda: 128

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(coll.id)},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_search_legacy_collection_passes_guard(test_app, mock_collection_use_case):
    from src.api.dependencies import get_embedding_dimension

    coll = Collection(
        name="legacy",
        embedding_model="",
        embedding_dimension=0,
    )
    mock_collection_use_case.get.return_value = coll
    test_app.dependency_overrides[get_embedding_dimension] = lambda: 128

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(coll.id)},
        )

    assert response.status_code == 200
