import json

import pytest

from src.api.sse import delta_event, done_event, error_event, format_sse_event, sources_event


def _parse_frame(frame: str) -> tuple[str, object]:
    lines = frame.split("\n")
    event = lines[0].removeprefix("event: ")
    data = json.loads(lines[1].removeprefix("data: "))
    return event, data


def test_format_sse_frame_ends_with_double_newline():
    assert format_sse_event("done", {}).endswith("\n\n")


def test_format_sse_event_structure():
    event, data = _parse_frame(format_sse_event("delta", {"text": "hello"}))
    assert event == "delta"
    assert data == {"text": "hello"}


def test_format_sse_event_newline_in_token_stays_on_one_data_line():
    # \n in token must be JSON-escaped, not a literal newline — literal breaks SSE framing
    frame = format_sse_event("delta", {"text": "hello\nworld"})
    lines = frame.split("\n")
    assert len(lines) == 4  # event:, data:, "", ""
    _, data = _parse_frame(frame)
    assert data == {"text": "hello\nworld"}


def test_sources_event():
    items = [{"chunk_content": "x", "score": 0.9, "document_filename": "f.pdf"}]
    event, data = _parse_frame(sources_event(items))
    assert event == "sources"
    assert data == items


def test_delta_event():
    event, data = _parse_frame(delta_event("tok"))
    assert event == "delta"
    assert data == {"text": "tok"}


def test_done_event():
    event, data = _parse_frame(done_event())
    assert event == "done"
    assert data == {}


def test_error_event():
    event, data = _parse_frame(error_event("LLM failed"))
    assert event == "error"
    assert data == {"detail": "LLM failed"}
