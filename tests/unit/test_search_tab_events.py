"""Tests for src/ui/search_events module."""
from src.ui.constants import (
    CHUNK_PREVIEW_MAX_CHARS,
    SEARCH_STATUS_GENERATING,
    SEARCH_STATUS_IDLE,
)
from src.ui.search_events import initial_outputs, next_outputs


_LONG_CONTENT = "A" * (CHUNK_PREVIEW_MAX_CHARS + 50)
_SOURCE_DATA = [
    {
        "chunk_content": _LONG_CONTENT,
        "score": 0.12345678,
        "document_filename": "report.pdf",
    }
]


class TestInitialOutputs:
    def test_should_return_empty_answer_when_created(self):
        assert initial_outputs().answer == ""

    def test_should_return_empty_sources_when_created(self):
        assert initial_outputs().sources_rows == []

    def test_should_return_not_finished_when_created(self):
        assert initial_outputs().finished is False

    def test_should_return_idle_status_when_created(self):
        assert initial_outputs().status == SEARCH_STATUS_IDLE


class TestSourcesEvent:
    def test_should_set_status_generating_when_sources_received(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert result.status == SEARCH_STATUS_GENERATING

    def test_should_set_answer_empty_when_sources_received(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert result.answer == ""

    def test_should_set_last_answer_empty_when_sources_received(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert result.last_answer == ""

    def test_should_not_finish_when_sources_received(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert result.finished is False

    def test_should_build_one_row_per_chunk_when_sources_received(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert len(result.sources_rows) == 1

    def test_should_truncate_chunk_preview_to_max_chars_when_content_exceeds_max(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert len(result.sources_rows[0][0]) == CHUNK_PREVIEW_MAX_CHARS

    def test_should_format_score_to_four_decimals_when_sources_received(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert result.sources_rows[0][1] == "0.1235"

    def test_should_include_filename_in_row_when_sources_received(self):
        result = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        assert result.sources_rows[0][2] == "report.pdf"


class TestDeltaEvent:
    def setup_method(self):
        self._prev = next_outputs("sources", _SOURCE_DATA, initial_outputs())

    def test_should_set_status_idle_when_delta_received(self):
        result = next_outputs("delta", {"text": "Hello"}, self._prev)
        assert result.status == SEARCH_STATUS_IDLE

    def test_should_append_token_to_answer_when_delta_received(self):
        result = next_outputs("delta", {"text": "Hello"}, self._prev)
        assert "Hello" in result.answer

    def test_should_preserve_sources_when_delta_received(self):
        result = next_outputs("delta", {"text": "Hello"}, self._prev)
        assert result.sources_rows == self._prev.sources_rows

    def test_should_not_finish_when_delta_received(self):
        result = next_outputs("delta", {"text": "Hello"}, self._prev)
        assert result.finished is False

    def test_should_accumulate_tokens_on_successive_deltas(self):
        state = self._prev
        state = next_outputs("delta", {"text": "Hello"}, state)
        state = next_outputs("delta", {"text": " World"}, state)
        assert state.answer == "Hello World"


class TestDoneEvent:
    def setup_method(self):
        state = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        self._prev = next_outputs("delta", {"text": "Answer text"}, state)

    def test_should_set_status_idle_when_done_received(self):
        result = next_outputs("done", {}, self._prev)
        assert result.status == SEARCH_STATUS_IDLE

    def test_should_set_last_answer_to_accumulated_when_done_received(self):
        result = next_outputs("done", {}, self._prev)
        assert result.last_answer == self._prev.answer

    def test_should_preserve_answer_when_done_received(self):
        result = next_outputs("done", {}, self._prev)
        assert result.answer == self._prev.answer

    def test_should_finish_when_done_received(self):
        result = next_outputs("done", {}, self._prev)
        assert result.finished is True

    def test_should_preserve_sources_when_done_received(self):
        result = next_outputs("done", {}, self._prev)
        assert result.sources_rows == self._prev.sources_rows


class TestErrorEvent:
    def setup_method(self):
        state = next_outputs("sources", _SOURCE_DATA, initial_outputs())
        self._prev = next_outputs("delta", {"text": "Partial"}, state)

    def test_should_append_error_block_when_error_received(self):
        result = next_outputs("error", {"detail": "timeout"}, self._prev)
        assert "**Error:** timeout" in result.answer

    def test_should_preserve_accumulated_answer_when_error_received(self):
        result = next_outputs("error", {"detail": "timeout"}, self._prev)
        assert "Partial" in result.answer

    def test_should_use_unknown_error_when_detail_missing(self):
        result = next_outputs("error", {}, self._prev)
        assert "unknown error" in result.answer

    def test_should_set_status_idle_when_error_received(self):
        result = next_outputs("error", {"detail": "timeout"}, self._prev)
        assert result.status == SEARCH_STATUS_IDLE

    def test_should_finish_when_error_received(self):
        result = next_outputs("error", {"detail": "timeout"}, self._prev)
        assert result.finished is True

    def test_should_preserve_sources_when_error_received(self):
        result = next_outputs("error", {"detail": "timeout"}, self._prev)
        assert result.sources_rows == self._prev.sources_rows

    def test_should_set_last_answer_empty_when_error_received(self):
        result = next_outputs("error", {"detail": "timeout"}, self._prev)
        assert result.last_answer == ""


class TestUnknownEvent:
    def test_should_return_none_when_unknown_event_received(self):
        result = next_outputs("heartbeat", {}, initial_outputs())
        assert result is None


class TestFullSequence:
    def test_should_transition_status_through_full_sequence(self):
        state = initial_outputs()
        state = next_outputs("sources", _SOURCE_DATA, state)
        assert state.status == SEARCH_STATUS_GENERATING
        state = next_outputs("delta", {"text": "tok1"}, state)
        assert state.status == SEARCH_STATUS_IDLE
        state = next_outputs("delta", {"text": "tok2"}, state)
        assert state.status == SEARCH_STATUS_IDLE
        state = next_outputs("done", {}, state)
        assert state.status == SEARCH_STATUS_IDLE
        assert state.finished is True


class TestEmptySourcesFollowedByDelta:
    def test_should_set_generating_status_when_empty_sources_received(self):
        state = initial_outputs()
        state = next_outputs("sources", [], state)
        assert state.status == SEARCH_STATUS_GENERATING

    def test_should_set_idle_status_when_delta_follows_empty_sources(self):
        state = initial_outputs()
        state = next_outputs("sources", [], state)
        state = next_outputs("delta", {"text": "fallback"}, state)
        assert state.status == SEARCH_STATUS_IDLE
