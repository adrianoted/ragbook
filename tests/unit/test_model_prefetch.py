from types import SimpleNamespace

import pytest

from src.infrastructure.embeddings import model_prefetch
from src.infrastructure.embeddings.model_prefetch import (
    compute_percent,
    prefetch_model,
    select_files,
)


def _file(filename: str, file_size: int, will_download: bool):
    """Build a dry_run entry as huggingface_hub returns it."""
    return SimpleNamespace(
        filename=filename,
        file_size=file_size,
        is_cached=not will_download,
        will_download=will_download,
    )


# --- select_files ---------------------------------------------------------


def test_should_keep_all_safetensors_when_only_safetensors():
    files = [
        _file("model.safetensors", 100, True),
        _file("model2.safetensors", 200, True),
    ]
    allow, total = select_files(files)
    assert sorted(allow) == ["model.safetensors", "model2.safetensors"]
    assert total == 300


def test_should_exclude_bin_when_safetensors_present():
    files = [
        _file("model.safetensors", 100, True),
        _file("pytorch_model.bin", 999, True),
    ]
    allow, total = select_files(files)
    assert allow == ["model.safetensors"]
    assert total == 100


def test_should_exclude_bin_even_when_safetensors_already_cached():
    # The format choice looks at all files, not only the pending ones.
    files = [
        _file("model.safetensors", 100, False),  # cached
        _file("pytorch_model.bin", 999, True),  # pending
    ]
    allow, total = select_files(files)
    assert allow == []
    assert total == 0


def test_should_keep_bin_when_no_safetensors():
    files = [
        _file("pytorch_model.bin", 500, True),
        _file("extra.pt", 250, True),
    ]
    allow, total = select_files(files)
    assert sorted(allow) == ["extra.pt", "pytorch_model.bin"]
    assert total == 750


def test_should_exclude_ignored_formats():
    files = [
        _file("model.safetensors", 100, True),
        _file("model.onnx", 1, True),
        _file("onnx/model.onnx", 1, True),
        _file("model.gguf", 1, True),
        _file("openvino/model.xml", 1, True),
    ]
    allow, total = select_files(files)
    assert allow == ["model.safetensors"]
    assert total == 100


def test_should_ignore_cached_files_in_list_and_total():
    files = [
        _file("model.safetensors", 100, True),
        _file("tokenizer.json", 50, False),  # cached, not pending
    ]
    allow, total = select_files(files)
    assert allow == ["model.safetensors"]
    assert total == 100


# --- compute_percent ------------------------------------------------------


def test_should_compute_half_percent():
    assert compute_percent(500, 1000) == 50.0


def test_should_return_zero_when_total_is_zero():
    assert compute_percent(0, 0) == 0.0


def test_should_clamp_percent_to_hundred():
    assert compute_percent(1200, 1000) == 100.0


# --- prefetch_model -------------------------------------------------------


def test_should_download_pending_files_and_return_true(monkeypatch):
    dry = [
        _file("model.safetensors", 100, True),
        _file("config.json", 20, True),
    ]
    calls = []

    def fake_snapshot(*args, **kwargs):
        if kwargs.get("dry_run"):
            return dry
        calls.append(kwargs)
        return "/some/path"

    monkeypatch.setattr(model_prefetch, "snapshot_download", fake_snapshot)

    result = prefetch_model("some/model", lambda d, t: None)

    assert result is True
    assert len(calls) == 1
    assert sorted(calls[0]["allow_patterns"]) == ["config.json", "model.safetensors"]


def test_should_call_progress_with_zero_before_download(monkeypatch):
    dry = [_file("model.safetensors", 100, True)]
    progress = []

    def fake_snapshot(*args, **kwargs):
        if kwargs.get("dry_run"):
            return dry
        return "/some/path"

    monkeypatch.setattr(model_prefetch, "snapshot_download", fake_snapshot)

    prefetch_model("some/model", lambda d, t: progress.append((d, t)))

    assert progress[0] == (0, 100)


def _captured_tqdm_class(monkeypatch, total_bytes: int) -> tuple[type, list]:
    """Run prefetch_model against a monkeypatched snapshot_download and return
    the tqdm_class it built plus the list collecting on_progress calls."""
    dry = [_file("model.safetensors", total_bytes, True)]
    progress: list = []
    captured = {}

    def fake_snapshot(*args, **kwargs):
        if kwargs.get("dry_run"):
            return dry
        captured["tqdm_class"] = kwargs["tqdm_class"]
        return "/some/path"

    monkeypatch.setattr(model_prefetch, "snapshot_download", fake_snapshot)

    prefetch_model("some/model", lambda d, t: progress.append((d, t)))

    return captured["tqdm_class"], progress


def test_should_forward_progress_through_custom_tqdm_when_disabled(monkeypatch):
    # Exact construction hub uses for the byte bar on non-tty (server/Docker):
    # disable=True, plus the `name` kwarg tqdm itself doesn't know about.
    tqdm_class, progress = _captured_tqdm_class(monkeypatch, total_bytes=1000)

    bar = tqdm_class(
        desc="Fetching",
        disable=True,
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        name="huggingface_hub.snapshot_download",
    )
    bar.update(300)
    bar.update(300)
    bar.update(400)

    assert progress[-3:] == [(300, 1000), (600, 1000), (1000, 1000)]


def test_should_forward_progress_through_custom_tqdm_when_enabled(monkeypatch):
    # Same construction on tty (disable=False): must not raise TqdmKeyError
    # on the `name` kwarg, and must still forward cumulative byte counts.
    tqdm_class, progress = _captured_tqdm_class(monkeypatch, total_bytes=1000)

    bar = tqdm_class(
        desc="Fetching",
        disable=False,
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        name="huggingface_hub.snapshot_download",
    )
    bar.update(300)
    bar.update(300)

    assert progress[-2:] == [(300, 1000), (600, 1000)]


def test_should_not_forward_progress_from_file_count_bar(monkeypatch):
    # thread_map's own bar: unit defaults to "it", not "B" — counts files,
    # not bytes, and must never reach on_progress.
    tqdm_class, progress = _captured_tqdm_class(monkeypatch, total_bytes=1000)
    baseline = len(progress)

    bar = tqdm_class(iter(["a", "b"]), desc="Fetching files", total=2)
    list(bar)

    assert len(progress) == baseline


def test_should_forward_only_byte_updates_when_both_bars_coexist(monkeypatch):
    tqdm_class, progress = _captured_tqdm_class(monkeypatch, total_bytes=1000)
    baseline = len(progress)

    byte_bar = tqdm_class(
        disable=True, total=0, initial=0, unit="B", unit_scale=True
    )
    file_bar = tqdm_class(iter(["a", "b"]), desc="Fetching files", total=2)

    byte_bar.update(300)
    file_bar.update(1)
    byte_bar.update(200)

    new_progress = progress[baseline:]
    assert new_progress == [(300, 1000), (500, 1000)]


def test_should_clamp_progress_to_total_bytes(monkeypatch):
    tqdm_class, progress = _captured_tqdm_class(monkeypatch, total_bytes=1000)

    bar = tqdm_class(disable=True, total=0, initial=0, unit="B", unit_scale=True)
    bar.update(900)
    bar.update(900)

    assert progress[-1] == (1000, 1000)


def test_should_return_false_and_skip_download_when_cache_warm(monkeypatch):
    dry = [_file("model.safetensors", 100, False)]  # cached, nothing pending
    calls = []

    def fake_snapshot(*args, **kwargs):
        if kwargs.get("dry_run"):
            return dry
        calls.append(kwargs)
        return "/some/path"

    monkeypatch.setattr(model_prefetch, "snapshot_download", fake_snapshot)

    result = prefetch_model("some/model", lambda d, t: None)

    assert result is False
    assert calls == []


def test_should_return_false_when_dry_run_raises(monkeypatch):
    calls = []

    def fake_snapshot(*args, **kwargs):
        if kwargs.get("dry_run"):
            raise RuntimeError("network down")
        calls.append(kwargs)
        return "/some/path"

    monkeypatch.setattr(model_prefetch, "snapshot_download", fake_snapshot)

    result = prefetch_model("some/model", lambda d, t: None)

    assert result is False
    assert calls == []


def test_should_return_false_when_download_raises(monkeypatch):
    dry = [_file("model.safetensors", 100, True)]

    def fake_snapshot(*args, **kwargs):
        if kwargs.get("dry_run"):
            return dry
        raise RuntimeError("connection reset mid download")

    monkeypatch.setattr(model_prefetch, "snapshot_download", fake_snapshot)

    result = prefetch_model("some/model", lambda d, t: None)

    assert result is False
