"""Tests for src/ui/warmup_banner module."""
import pytest

from src.ui.warmup_banner import banner_state, format_gb
from src.ui.constants import (
    BANNER_DOWNLOADING,
    BANNER_ERROR,
    BANNER_LOADING,
    BANNER_VARIANT_ERROR,
    BANNER_VARIANT_INFO,
)


class TestFormatGb:
    def test_should_return_decimal_string_when_typical_bytes(self):
        assert format_gb(3435973836) == "3.2"

    def test_should_return_zero_string_when_zero_bytes(self):
        assert format_gb(0) == "0.0"

    def test_should_return_correct_value_for_total_bytes(self):
        assert format_gb(8053063680) == "7.5"


class TestBannerStateReady:
    def test_should_hide_banner_when_ready(self):
        health = {"embedding_status": "ready", "embedding_phase": None, "embedding_progress": None}
        text, timer_active, unchanged, variant = banner_state(health)
        assert text is None

    def test_should_deactivate_timer_when_ready(self):
        health = {"embedding_status": "ready", "embedding_phase": None, "embedding_progress": None}
        _, timer_active, _, _ = banner_state(health)
        assert timer_active is False

    def test_should_not_signal_unchanged_when_ready(self):
        health = {"embedding_status": "ready", "embedding_phase": None, "embedding_progress": None}
        _, _, unchanged, _ = banner_state(health)
        assert unchanged is False


class TestBannerStateDownloadingWithProgress:
    _health = {
        "embedding_status": "warming",
        "embedding_phase": "downloading",
        "embedding_progress": {
            "downloaded_bytes": 3435973836,
            "total_bytes": 8053063680,
            "percent": 42.7,
        },
    }

    def test_should_include_percent_in_banner_text(self):
        text, _, _, _ = banner_state(self._health)
        assert "42.7" in text

    def test_should_include_done_gb_in_banner_text(self):
        text, _, _, _ = banner_state(self._health)
        assert "3.2" in text

    def test_should_include_total_gb_in_banner_text(self):
        text, _, _, _ = banner_state(self._health)
        assert "7.5" in text

    def test_should_activate_timer_when_downloading(self):
        _, timer_active, _, _ = banner_state(self._health)
        assert timer_active is True

    def test_should_not_signal_unchanged_when_downloading(self):
        _, _, unchanged, _ = banner_state(self._health)
        assert unchanged is False

    def test_should_use_info_variant_when_downloading(self):
        _, _, _, variant = banner_state(self._health)
        assert variant == BANNER_VARIANT_INFO


class TestBannerStateDownloadingWithoutProgress:
    _health = {
        "embedding_status": "warming",
        "embedding_phase": "downloading",
        "embedding_progress": None,
    }

    def test_should_show_loading_text_when_progress_is_none(self):
        text, _, _, _ = banner_state(self._health)
        assert text == BANNER_LOADING

    def test_should_not_contain_none_literal_in_text(self):
        text, _, _, _ = banner_state(self._health)
        assert "None" not in text

    def test_should_not_contain_spurious_percent_in_text(self):
        text, _, _, _ = banner_state(self._health)
        assert "%" not in text

    def test_should_activate_timer_when_downloading_no_progress(self):
        _, timer_active, _, _ = banner_state(self._health)
        assert timer_active is True


class TestBannerStateLoading:
    def test_should_show_loading_text_when_phase_loading(self):
        health = {"embedding_status": "warming", "embedding_phase": "loading", "embedding_progress": None}
        text, _, _, _ = banner_state(health)
        assert text == BANNER_LOADING

    def test_should_activate_timer_when_loading(self):
        health = {"embedding_status": "warming", "embedding_phase": "loading", "embedding_progress": None}
        _, timer_active, _, _ = banner_state(health)
        assert timer_active is True

    def test_should_show_loading_text_when_phase_is_none(self):
        health = {"embedding_status": "warming", "embedding_phase": None, "embedding_progress": None}
        text, _, _, _ = banner_state(health)
        assert text == BANNER_LOADING

    def test_should_activate_timer_when_phase_is_none(self):
        health = {"embedding_status": "warming", "embedding_phase": None, "embedding_progress": None}
        _, timer_active, _, _ = banner_state(health)
        assert timer_active is True

    def test_should_use_info_variant_when_loading(self):
        health = {"embedding_status": "warming", "embedding_phase": "loading", "embedding_progress": None}
        _, _, _, variant = banner_state(health)
        assert variant == BANNER_VARIANT_INFO


class TestBannerStateError:
    def test_should_show_error_text_when_error(self):
        health = {"embedding_status": "error", "embedding_phase": None, "embedding_progress": None}
        text, _, _, _ = banner_state(health)
        assert text == BANNER_ERROR

    def test_should_deactivate_timer_when_error(self):
        health = {"embedding_status": "error", "embedding_phase": None, "embedding_progress": None}
        _, timer_active, _, _ = banner_state(health)
        assert timer_active is False

    def test_should_not_signal_unchanged_when_error(self):
        health = {"embedding_status": "error", "embedding_phase": None, "embedding_progress": None}
        _, _, unchanged, _ = banner_state(health)
        assert unchanged is False

    def test_should_use_error_variant_when_error(self):
        health = {"embedding_status": "error", "embedding_phase": None, "embedding_progress": None}
        _, _, _, variant = banner_state(health)
        assert variant == BANNER_VARIANT_ERROR


class TestBannerStateServerUnreachable:
    def test_should_signal_unchanged_when_empty_dict(self):
        _, _, unchanged, _ = banner_state({})
        assert unchanged is True

    def test_should_activate_timer_when_empty_dict(self):
        _, timer_active, _, _ = banner_state({})
        assert timer_active is True

    def test_should_return_none_text_when_empty_dict(self):
        text, _, _, _ = banner_state({})
        assert text is None
