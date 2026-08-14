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


# ── tuning parameters (task 6) ────────────────────────────────


@pytest.mark.asyncio
async def test_search_without_tuning_defaults_to_rerank_min_score_and_empty_vos(
    test_app, mock_search_use_case, sample_collection
):
    """Regression: no tuning params → empty VOs and min_score == rerank_min_score (0.3)."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(sample_collection.id)},
        )

    assert response.status_code == 200
    search_query = mock_search_use_case.execute.call_args[0][0]
    assert search_query.min_score == 0.3
    assert search_query.tuning.fusion is None
    assert search_query.tuning.vector_weight is None
    assert search_query.tuning.max_results_per_document is None
    assert search_query.tuning.reranker_enabled is None
    assert search_query.llm_options.temperature is None
    assert search_query.llm_options.think is None
    assert search_query.llm_options.num_ctx is None


@pytest.mark.asyncio
async def test_search_reranker_disabled_defaults_to_raw_min_score(
    test_app, mock_search_use_case, sample_collection
):
    """reranker_enabled=False without min_score → raw-scale settings.min_score (0.15)."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "reranker_enabled": False,
            },
        )

    assert response.status_code == 200
    search_query = mock_search_use_case.execute.call_args[0][0]
    assert search_query.min_score == 0.15
    assert search_query.tuning.reranker_enabled is False


@pytest.mark.asyncio
async def test_search_reranker_enabled_defaults_to_rerank_min_score(
    test_app, mock_search_use_case, sample_collection
):
    """reranker_enabled=True without min_score → sigmoid-scale rerank_min_score (0.3)."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "reranker_enabled": True,
            },
        )

    assert response.status_code == 200
    search_query = mock_search_use_case.execute.call_args[0][0]
    assert search_query.min_score == 0.3
    assert search_query.tuning.reranker_enabled is True


@pytest.mark.parametrize("reranker_enabled", [True, False, None])
@pytest.mark.asyncio
async def test_search_explicit_min_score_overrides_defaults(
    test_app, mock_search_use_case, sample_collection, reranker_enabled
):
    """An explicit min_score wins over both defaults, whatever the reranker flag is."""
    body = {
        "query": "test",
        "collection_id": str(sample_collection.id),
        "min_score": 0.42,
    }
    if reranker_enabled is not None:
        body["reranker_enabled"] = reranker_enabled

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post("/api/search", json=body)

    assert response.status_code == 200
    search_query = mock_search_use_case.execute.call_args[0][0]
    assert search_query.min_score == 0.42


@pytest.mark.asyncio
async def test_search_propagates_all_tuning_params(
    test_app, mock_search_use_case, sample_collection
):
    """All eight params land in the right VO fields; hybrid_vector_weight → tuning.vector_weight."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "min_score": 0.2,
                "fusion": "rrf",
                "hybrid_vector_weight": 0.4,
                "max_results_per_document": 3,
                "reranker_enabled": True,
                "llm_temperature": 0.7,
                "llm_think": True,
                "llm_num_ctx": 8192,
            },
        )

    assert response.status_code == 200
    sq = mock_search_use_case.execute.call_args[0][0]
    assert sq.min_score == 0.2
    assert sq.tuning.fusion == "rrf"
    assert sq.tuning.vector_weight == 0.4  # rename: hybrid_vector_weight → vector_weight
    assert sq.tuning.max_results_per_document == 3
    assert sq.tuning.reranker_enabled is True
    assert sq.llm_options.temperature == 0.7
    assert sq.llm_options.think is True
    assert sq.llm_options.num_ctx == 8192


@pytest.mark.parametrize(
    "field,value",
    [
        ("hybrid_vector_weight", 1.5),
        ("min_score", -0.1),
        ("llm_temperature", 3.0),
        ("llm_num_ctx", 1024),
        ("llm_num_ctx", 65536),
        ("max_results_per_document", 0),
        ("fusion", "banana"),
    ],
)
@pytest.mark.asyncio
async def test_search_out_of_range_tuning_returns_422(
    test_app, sample_collection, field, value
):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(sample_collection.id), field: value},
        )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("hybrid_vector_weight", 0.0),
        ("llm_temperature", 0.0),
        ("llm_num_ctx", 2048),
        ("llm_num_ctx", 32768),
        ("max_results_per_document", 1),
    ],
)
@pytest.mark.asyncio
async def test_search_boundary_tuning_accepted(
    test_app, sample_collection, field, value
):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search",
            json={"query": "test", "collection_id": str(sample_collection.id), field: value},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_search_raw_accepts_tuning_params(
    test_app, mock_search_use_case, sample_collection
):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search/raw",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "fusion": "weighted",
                "hybrid_vector_weight": 0.6,
                "llm_temperature": 0.5,
            },
        )

    assert response.status_code == 200
    sq = mock_search_use_case.execute_raw.call_args[0][0]
    assert sq.tuning.fusion == "weighted"
    assert sq.tuning.vector_weight == 0.6
    assert sq.llm_options.temperature == 0.5


@pytest.mark.asyncio
async def test_search_stream_accepts_tuning_params(
    test_app, mock_search_use_case, sample_search_results, sample_collection
):
    async def token_gen():
        yield "tok"

    mock_search_use_case.execute_stream.return_value = (sample_search_results, token_gen())

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post(
            "/api/search/stream",
            json={
                "query": "test",
                "collection_id": str(sample_collection.id),
                "reranker_enabled": False,
                "llm_num_ctx": 4096,
            },
        )

    assert response.status_code == 200
    sq = mock_search_use_case.execute_stream.call_args[0][0]
    assert sq.tuning.reranker_enabled is False
    assert sq.llm_options.num_ctx == 4096


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
