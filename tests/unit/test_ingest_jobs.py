"""Tests for src/api/ingest_jobs module."""
import time

import pytest

import src.api.ingest_jobs as ij


@pytest.fixture(autouse=True)
def reset():
    ij.reset_registry()
    yield
    ij.reset_registry()


class TestCreateJob:
    def test_id_not_empty(self):
        job = ij.create_job("file.txt", "col-1")
        assert job.id

    def test_ids_unique(self):
        j1 = ij.create_job("a.txt", "col-1")
        j2 = ij.create_job("b.txt", "col-1")
        assert j1.id != j2.id

    def test_status_pending(self):
        job = ij.create_job("file.txt", "col-1")
        assert job.status == "pending"

    def test_phase_none(self):
        job = ij.create_job("file.txt", "col-1")
        assert job.phase is None

    def test_retrievable_via_get_job(self):
        job = ij.create_job("file.txt", "col-1")
        assert ij.get_job(job.id) is job


class TestGetJob:
    def test_unknown_id_returns_none(self):
        assert ij.get_job("nonexistent") is None

    def test_no_exception_on_unknown(self):
        ij.get_job("anything")  # must not raise


class TestHappyPath:
    def test_mark_running(self):
        job = ij.create_job("file.txt", "col-1")
        result = ij.mark_running(job.id)
        assert result is not None
        assert result.status == "running"

    def test_update_progress(self):
        job = ij.create_job("file.txt", "col-1")
        ij.mark_running(job.id)
        result = ij.update_progress(job.id, "embedding", 500, 1800)
        assert result is not None
        assert result.phase == "embedding"
        assert result.chunks_done == 500
        assert result.chunks_total == 1800

    def test_mark_done(self):
        job = ij.create_job("file.txt", "col-1")
        ij.mark_running(job.id)
        result = ij.mark_done(job.id, "doc-42", 12)
        assert result is not None
        assert result.status == "done"
        assert result.document_id == "doc-42"
        assert result.num_chunks == 12
        assert result.finished_at is not None


class TestErrorPath:
    def test_mark_error_status(self):
        job = ij.create_job("file.txt", "col-1")
        result = ij.mark_error(job.id, "boom")
        assert result.status == "error"

    def test_mark_error_message(self):
        job = ij.create_job("file.txt", "col-1")
        result = ij.mark_error(job.id, "boom")
        assert result.error == "boom"

    def test_mark_error_finished_at(self):
        job = ij.create_job("file.txt", "col-1")
        result = ij.mark_error(job.id, "boom")
        assert result.finished_at is not None


class TestUnknownIdTransitions:
    def test_mark_running_unknown_returns_none(self):
        assert ij.mark_running("nope") is None

    def test_update_progress_unknown_returns_none(self):
        assert ij.update_progress("nope", "embedding", 0, 0) is None

    def test_mark_done_unknown_returns_none(self):
        assert ij.mark_done("nope", "doc-1", 0) is None

    def test_mark_error_unknown_returns_none(self):
        assert ij.mark_error("nope", "msg") is None


class TestCleanupExpired:
    def test_removes_terminal_job_past_ttl(self):
        job = ij.create_job("file.txt", "col-1")
        ij.mark_done(job.id, "doc-1", 5)
        removed = ij.cleanup_expired(ttl=0, now=time.monotonic() + 1)
        assert removed == 1
        assert ij.get_job(job.id) is None

    def test_returns_count(self):
        job = ij.create_job("file.txt", "col-1")
        ij.mark_done(job.id, "doc-1", 5)
        removed = ij.cleanup_expired(ttl=0, now=time.monotonic() + 1)
        assert removed == 1

    def test_does_not_remove_running_job(self):
        job = ij.create_job("file.txt", "col-1")
        ij.mark_running(job.id)
        removed = ij.cleanup_expired(ttl=0, now=time.monotonic() + 9999)
        assert removed == 0
        assert ij.get_job(job.id) is not None

    def test_does_not_remove_terminal_within_ttl(self):
        job = ij.create_job("file.txt", "col-1")
        ij.mark_done(job.id, "doc-1", 5)
        removed = ij.cleanup_expired(ttl=3600, now=time.monotonic())
        assert removed == 0
        assert ij.get_job(job.id) is not None


class TestCreateJobTriggersCleanup:
    def test_expired_job_removed_on_new_create(self):
        old_job = ij.create_job("old.txt", "col-1")
        ij.mark_done(old_job.id, "doc-old", 5)
        old_job.finished_at = time.monotonic() - ij.JOB_TTL_SECONDS - 1
        ij.create_job("new.txt", "col-1")
        assert ij.get_job(old_job.id) is None
