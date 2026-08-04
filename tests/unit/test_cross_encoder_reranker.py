import math
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from src.domain.entities import Chunk, SearchResult
from src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker


def _make_result(content: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            document_id=uuid4(),
            content=content,
            index=0,
            metadata={"document_id": "doc-1"},
        ),
        score=score,
        source="vector",
    )


@pytest.fixture
def mock_cross_encoder():
    with patch(
        "src.infrastructure.rerankers.cross_encoder_reranker.CrossEncoder"
    ) as mock_cls:
        model = MagicMock()
        mock_cls.return_value = model
        yield model


@pytest.fixture
def reranker(mock_cross_encoder):
    return CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


@pytest.mark.asyncio
async def test_pairs_constructed_correctly(reranker, mock_cross_encoder):
    mock_cross_encoder.predict.return_value = np.array([0.9, 0.1])
    results = [
        _make_result("first chunk", 0.5),
        _make_result("second chunk", 0.4),
    ]

    await reranker.rerank("my query", results, top_k=2)

    pairs = mock_cross_encoder.predict.call_args[0][0]
    assert pairs == [("my query", "first chunk"), ("my query", "second chunk")]


@pytest.mark.asyncio
async def test_results_reordered_by_cross_encoder_score(reranker, mock_cross_encoder):
    mock_cross_encoder.predict.return_value = np.array([0.1, 0.9, 0.5])
    results = [
        _make_result("low", 0.9),
        _make_result("high", 0.3),
        _make_result("mid", 0.5),
    ]

    reranked = await reranker.rerank("query", results, top_k=3)

    assert reranked[0].chunk.content == "high"
    assert reranked[1].chunk.content == "mid"
    assert reranked[2].chunk.content == "low"
    # Scores are sigmoid-normalized to [0, 1]
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    assert reranked[0].score == pytest.approx(sigmoid(0.9))
    assert reranked[1].score == pytest.approx(sigmoid(0.5))
    assert reranked[2].score == pytest.approx(sigmoid(0.1))


@pytest.mark.asyncio
async def test_top_k_respected(reranker, mock_cross_encoder):
    mock_cross_encoder.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6])
    results = [_make_result(f"chunk-{i}", 0.5) for i in range(4)]

    reranked = await reranker.rerank("query", results, top_k=2)

    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_empty_input_returns_empty(reranker):
    reranked = await reranker.rerank("query", [], top_k=5)

    assert reranked == []
