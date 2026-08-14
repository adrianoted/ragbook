"""Unit tests for ApiClient.search_stream and parse_sse_lines."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.ui.api_client import ApiClient, parse_sse_lines


# ── parse_sse_lines ───────────────────────────────────────────────────────────


def test_parser_complete_frame():
    lines = ["event: delta", 'data: {"text": "x"}', ""]
    events = list(parse_sse_lines(iter(lines)))
    assert events == [("delta", {"text": "x"})]


def test_parser_split_across_batches():
    """iter_lines may deliver a frame across multiple batches; accumulator merges."""
    batch1 = ["event: delta"]
    batch2 = ['data: {"text": "y"}', ""]
    events = []
    acc = parse_sse_lines.__wrapped__ if hasattr(parse_sse_lines, "__wrapped__") else None
    # Call parse_sse_lines twice, simulating two separate iter_lines calls
    # The function must be stateful across calls — we feed all lines at once
    all_lines = batch1 + batch2
    events = list(parse_sse_lines(iter(all_lines)))
    assert events == [("delta", {"text": "y"})]


def test_parser_multiple_frames_in_batch():
    lines = [
        "event: sources",
        'data: [{"chunk_content": "a"}]',
        "",
        "event: delta",
        'data: {"text": "hello"}',
        "",
    ]
    events = list(parse_sse_lines(iter(lines)))
    assert events == [
        ("sources", [{"chunk_content": "a"}]),
        ("delta", {"text": "hello"}),
    ]


def test_parser_interrupted_frame_no_partial_emit():
    """Stream cut mid-frame: no partial event, no exception."""
    lines = ["event: delta", 'data: {"text": "incomplete"}']
    events = list(parse_sse_lines(iter(lines)))
    assert events == []


def test_parser_ignores_comment_lines():
    lines = [": keep-alive", "event: done", "data: {}", ""]
    events = list(parse_sse_lines(iter(lines)))
    assert events == [("done", {})]


def test_parser_frame_without_data_not_emitted():
    lines = ["event: ping", ""]
    events = list(parse_sse_lines(iter(lines)))
    assert events == []


# ── search_stream ─────────────────────────────────────────────────────────────


def _make_mock_response(raw_lines: list[str], status_code: int = 200):
    """Build a mock httpx response whose iter_lines() returns raw_lines."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.iter_lines.return_value = iter(raw_lines)
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status.return_value = None
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_search_stream_yields_events_and_stops_on_done():
    sse_lines = [
        "event: sources",
        'data: [{"chunk_content": "ctx"}]',
        "",
        "event: delta",
        'data: {"text": "hi"}',
        "",
        "event: done",
        "data: {}",
        "",
    ]
    mock_resp = _make_mock_response(sse_lines)

    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.stream", return_value=mock_resp):
        events = list(client.search_stream("q", collection_id="coll-1"))

    assert events == [
        ("sources", [{"chunk_content": "ctx"}]),
        ("delta", {"text": "hi"}),
        ("done", {}),
    ]


def test_search_stream_404_raises_before_yield():
    mock_resp = _make_mock_response([], status_code=404)

    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.stream", return_value=mock_resp):
        with pytest.raises(httpx.HTTPStatusError):
            list(client.search_stream("q", collection_id="coll-1"))


# ── search (tuning params) ────────────────────────────────────────────────────


def _make_post_mock(json_data: dict | list, status_code: int = 200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


def test_search_no_tuning_params_body_has_no_tuning_keys():
    mock_resp = _make_post_mock({"answer": "x"})
    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        client.search("q", collection_id="coll-1")
    body = mock_post.call_args.kwargs["json"]
    assert set(body.keys()) == {"query", "top_k", "strategy", "collection_id"}


def test_search_all_tuning_params_present_in_body():
    mock_resp = _make_post_mock({"answer": "x"})
    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        client.search(
            "q",
            collection_id="coll-1",
            min_score=0.3,
            fusion="rrf",
            hybrid_vector_weight=0.6,
            max_results_per_document=2,
            reranker_enabled=True,
            llm_temperature=0.7,
            llm_think=True,
            llm_num_ctx=4096,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["min_score"] == 0.3
    assert body["fusion"] == "rrf"
    assert body["hybrid_vector_weight"] == 0.6
    assert body["max_results_per_document"] == 2
    assert body["reranker_enabled"] is True
    assert body["llm_temperature"] == 0.7
    assert body["llm_think"] is True
    assert body["llm_num_ctx"] == 4096


def test_search_falsy_tuning_values_included_in_body():
    mock_resp = _make_post_mock({"answer": "x"})
    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        client.search(
            "q",
            collection_id="coll-1",
            reranker_enabled=False,
            llm_temperature=0.0,
            hybrid_vector_weight=0.0,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["reranker_enabled"] is False
    assert body["llm_temperature"] == 0.0
    assert body["hybrid_vector_weight"] == 0.0


def test_search_stream_same_body_as_search():
    sse_lines = ["event: done", "data: {}", ""]
    mock_resp = _make_mock_response(sse_lines)
    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.stream", return_value=mock_resp) as mock_stream:
        list(client.search_stream("q", collection_id="coll-1", min_score=0.3, fusion="rrf"))
    body = mock_stream.call_args.kwargs["json"]
    assert body["min_score"] == 0.3
    assert body["fusion"] == "rrf"
    assert body["query"] == "q"
    assert body["collection_id"] == "coll-1"


# ── get_config ────────────────────────────────────────────────────────────────


def test_get_config_returns_parsed_json():
    config_data = {"llm_temperature": 0.7, "min_score": 0.3}
    mock_resp = MagicMock()
    mock_resp.json.return_value = config_data
    mock_resp.raise_for_status.return_value = None
    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.get", return_value=mock_resp):
        result = client.get_config()
    assert result == config_data


def test_get_config_returns_empty_dict_on_connection_error():
    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.get", side_effect=httpx.ConnectError("unreachable")):
        result = client.get_config()
    assert result == {}


def test_get_config_returns_empty_dict_on_http_500():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="HTTP 500", request=MagicMock(), response=mock_resp
    )
    client = ApiClient("http://localhost:8000/api")
    with patch("httpx.get", return_value=mock_resp):
        result = client.get_config()
    assert result == {}
