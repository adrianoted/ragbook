"""Tests for src/ui/upload_events module."""
from unittest.mock import Mock

from src.ui.constants import (
    INGEST_POLL_INTERVAL,
    UPLOAD_STATUS_DONE,
    UPLOAD_STATUS_ERROR_PREFIX,
    UPLOAD_STATUS_QUEUED,
    UPLOAD_STATUS_UI_TIMEOUT,
    UPLOAD_STATUS_UNKNOWN_JOB,
)
from src.ui.upload_events import job_status_text, poll_ingest_job


def _running(done: int, total: int) -> dict:
    return {
        "status": "running",
        "phase": "embedding",
        "chunks_done": done,
        "chunks_total": total,
    }


class TestPollIngestJob:
    def test_should_yield_each_state_and_sleep_between_reads_when_running_then_done(self):
        client = Mock()
        client.get_ingest_job.side_effect = [
            _running(1, 3),
            _running(2, 3),
            {"status": "done", "num_chunks": 3},
        ]
        sleep = Mock()
        results = list(
            poll_ingest_job(client, "j1", sleep=sleep, now=lambda: 0.0)
        )
        assert [r["status"] for r in results] == ["running", "running", "done"]
        assert sleep.call_count == 2
        assert all(call.args == (INGEST_POLL_INTERVAL,) for call in sleep.call_args_list)

    def test_should_yield_error_state_and_stop_when_job_errors(self):
        client = Mock()
        client.get_ingest_job.side_effect = [{"status": "error", "error": "boom"}]
        sleep = Mock()
        results = list(
            poll_ingest_job(client, "j1", sleep=sleep, now=lambda: 0.0)
        )
        assert results[-1]["status"] == "error"
        assert sleep.call_count == 0

    def test_should_yield_unknown_and_stop_when_job_missing(self):
        client = Mock()
        client.get_ingest_job.side_effect = [None]
        sleep = Mock()
        results = list(
            poll_ingest_job(client, "j1", sleep=sleep, now=lambda: 0.0)
        )
        assert results[-1]["status"] == "unknown"
        assert client.get_ingest_job.call_count == 1

    def test_should_yield_ui_timeout_when_max_wait_exceeded(self):
        client = Mock()
        client.get_ingest_job.return_value = _running(0, 3)
        now = Mock(side_effect=[0.0, 10.0])
        sleep = Mock()
        results = list(
            poll_ingest_job(client, "j1", max_wait=5, sleep=sleep, now=now)
        )
        assert results[-1]["status"] == "ui_timeout"

    def test_should_yield_once_and_not_sleep_when_job_already_done(self):
        client = Mock()
        client.get_ingest_job.side_effect = [{"status": "done", "num_chunks": 5}]
        sleep = Mock()
        results = list(
            poll_ingest_job(client, "j1", sleep=sleep, now=lambda: 0.0)
        )
        assert len(results) == 1
        assert sleep.call_count == 0


class TestJobStatusText:
    def test_should_show_phase_label_and_truncated_elapsed_seconds_when_running(self):
        assert job_status_text(_running(500, 1800), elapsed=42.7) == "Computing embeddings… 42s"

    def test_should_change_text_when_elapsed_differs(self):
        # The timer is what signals "in progress" now that the spinner is gone.
        job = _running(500, 1800)
        assert job_status_text(job, elapsed=1.0) != job_status_text(job, elapsed=2.0)

    def test_should_not_show_chunk_counts_when_batch_not_landed(self):
        # The count belongs to the table's Chunks column; a frozen 0/85 here
        # reads as stuck, which is exactly what this format avoids.
        assert job_status_text(_running(0, 85), elapsed=5.0) == "Computing embeddings… 5s"

    def test_should_show_reading_file_label_when_phase_is_loading(self):
        job = {
            "status": "running",
            "phase": "loading",
            "chunks_done": 0,
            "chunks_total": 0,
        }
        assert job_status_text(job, elapsed=0.0) == "Reading file… 0s"

    def test_should_show_raw_phase_and_elapsed_when_phase_unknown_to_the_ui(self):
        job = {"status": "running", "phase": "reticulating", "chunks_total": 0}
        assert job_status_text(job, elapsed=3.0) == "reticulating… 3s"

    def test_should_return_queued_when_pending(self):
        assert job_status_text({"status": "pending"}) == UPLOAD_STATUS_QUEUED

    def test_should_return_done_when_done(self):
        assert job_status_text({"status": "done", "num_chunks": 3}) == UPLOAD_STATUS_DONE

    def test_should_prefix_message_when_error(self):
        job = {"status": "error", "error": "boom"}
        assert job_status_text(job) == UPLOAD_STATUS_ERROR_PREFIX + "boom"

    def test_should_return_unknown_text_when_unknown(self):
        assert job_status_text({"status": "unknown"}) == UPLOAD_STATUS_UNKNOWN_JOB

    def test_should_return_ui_timeout_text_when_ui_timeout(self):
        assert job_status_text({"status": "ui_timeout"}) == UPLOAD_STATUS_UI_TIMEOUT
