"""Tests for src/api/embedding_state module."""
import threading
import time

import pytest

import src.api.embedding_state as es


@pytest.fixture(autouse=True)
def reset_state():
    es.reset()
    yield
    es.reset()


def _payload():
    return es.health_payload()


class TestInitialState:
    def test_status_ok(self):
        p = _payload()
        assert p["status"] == "ok"

    def test_embedding_status_warming(self):
        p = _payload()
        assert p["embedding_status"] == "warming"

    def test_embedding_phase_none(self):
        p = _payload()
        assert p["embedding_phase"] is None

    def test_embedding_progress_none(self):
        p = _payload()
        assert p["embedding_progress"] is None

    def test_always_four_keys(self):
        p = _payload()
        assert set(p.keys()) == {"status", "embedding_status", "embedding_phase", "embedding_progress"}


class TestReportDownloadProgress:
    def test_sets_phase_downloading(self):
        es.report_download_progress(0, 1000)
        p = _payload()
        assert p["embedding_phase"] == es.PHASE_DOWNLOADING

    def test_progress_zero_bytes(self):
        es.report_download_progress(0, 1000)
        p = _payload()
        assert p["embedding_progress"] == {"downloaded_bytes": 0, "total_bytes": 1000, "percent": 0.0}

    def test_progress_half(self):
        es.report_download_progress(500, 1000)
        p = _payload()
        assert p["embedding_progress"]["percent"] == 50.0

    def test_progress_full(self):
        es.report_download_progress(1000, 1000)
        p = _payload()
        assert p["embedding_progress"]["percent"] == 100.0

    def test_zero_total_no_exception(self):
        es.report_download_progress(0, 0)
        p = _payload()
        assert p["embedding_progress"] is None

    def test_embedding_status_still_warming(self):
        es.report_download_progress(100, 1000)
        assert _payload()["embedding_status"] == "warming"


class TestMarkLoading:
    def test_phase_loading_after_progress(self):
        es.report_download_progress(500, 1000)
        es.mark_loading()
        p = _payload()
        assert p["embedding_phase"] == es.PHASE_LOADING

    def test_progress_none_after_loading(self):
        es.report_download_progress(500, 1000)
        es.mark_loading()
        assert _payload()["embedding_progress"] is None

    def test_embedding_status_still_warming(self):
        es.mark_loading()
        assert _payload()["embedding_status"] == "warming"


class TestMarkReady:
    def test_status_ready(self):
        es.mark_ready()
        assert _payload()["embedding_status"] == "ready"

    def test_phase_none(self):
        es.mark_loading()
        es.mark_ready()
        assert _payload()["embedding_phase"] is None

    def test_progress_none(self):
        es.report_download_progress(500, 1000)
        es.mark_ready()
        assert _payload()["embedding_progress"] is None


class TestMarkError:
    def test_status_error(self):
        es.mark_error()
        assert _payload()["embedding_status"] == "error"

    def test_phase_none(self):
        es.mark_loading()
        es.mark_error()
        assert _payload()["embedding_phase"] is None

    def test_progress_none(self):
        es.report_download_progress(500, 1000)
        es.mark_error()
        assert _payload()["embedding_progress"] is None


class TestHealthPayload:
    def test_four_keys_in_all_states(self):
        expected = {"status", "embedding_status", "embedding_phase", "embedding_progress"}
        es.report_download_progress(100, 1000)
        assert set(_payload().keys()) == expected
        es.mark_loading()
        assert set(_payload().keys()) == expected
        es.mark_ready()
        assert set(_payload().keys()) == expected
        es.reset()
        es.mark_error()
        assert set(_payload().keys()) == expected

    def test_returned_dict_is_copy(self):
        p1 = _payload()
        p1["extra"] = "mutated"
        p2 = _payload()
        assert "extra" not in p2


class TestConcurrency:
    def test_no_exception_and_consistent_payloads(self):
        errors = []
        inconsistencies = []

        def writer():
            for i in range(200):
                es.report_download_progress(i * 5, 1000)
                time.sleep(0)

        def reader():
            for _ in range(300):
                try:
                    p = es.health_payload()
                    prog = p["embedding_progress"]
                    if prog is not None:
                        dl = prog["downloaded_bytes"]
                        total = prog["total_bytes"]
                        expected_pct = round(min(dl / total, 1.0) * 100, 1)
                        if prog["percent"] != expected_pct:
                            inconsistencies.append((dl, total, prog["percent"], expected_pct))
                except Exception as exc:
                    errors.append(exc)
                time.sleep(0)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"exceptions in threads: {errors}"
        assert not inconsistencies, f"inconsistent payloads: {inconsistencies}"
