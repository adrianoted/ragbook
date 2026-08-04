"""Unit tests for OllamaLlm — streaming + refactor invariance.

No respx: httpx streaming is faked with unittest.mock. ``client.stream(...)``
returns an async context manager whose response exposes ``raise_for_status``
and an async-generator ``aiter_lines``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.infrastructure.llm.ollama_llm import OllamaLlm


def _settings(*, think: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        ollama_base_url="http://localhost:11434",
        llm_model="llama3.2",
        llm_provider="ollama",
        llm_temperature=0.0,
        llm_num_ctx=8192,
        llm_think=think,
        llm_timeout=30.0,
    )


def _ndjson(*objs: dict) -> list[str]:
    return [json.dumps(o) for o in objs]


def _fake_stream_client(lines: list[str], *, raise_exc: Exception | None = None):
    """Build a fake httpx.AsyncClient whose ``stream`` yields *lines*."""

    async def _aiter_lines():
        for line in lines:
            yield line

    response = MagicMock()
    if raise_exc is not None:
        response.raise_for_status.side_effect = raise_exc
    else:
        response.raise_for_status.return_value = None
    response.aiter_lines = _aiter_lines

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=response)
    stream_cm.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.stream = MagicMock(return_value=stream_cm)
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    return client_cm, client


async def _collect(llm, prompt="hello", context=None):
    return [tok async for tok in llm.generate_stream(prompt, context or ["ctx"])]


@pytest.mark.asyncio
async def test_stream_yields_tokens_in_order():
    lines = _ndjson(
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " "}, "done": False},
        {"message": {"content": "world"}, "done": False},
        {"message": {"content": ""}, "done": True},
    )
    client_cm, _ = _fake_stream_client(lines)
    with patch("httpx.AsyncClient", return_value=client_cm):
        tokens = await _collect(OllamaLlm(_settings()))
    assert tokens == ["Hello", " ", "world"]


@pytest.mark.asyncio
async def test_stream_skips_malformed_line():
    lines = [
        json.dumps({"message": {"content": "a"}, "done": False}),
        "{not valid json",
        json.dumps({"message": {"content": "b"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    client_cm, _ = _fake_stream_client(lines)
    with patch("httpx.AsyncClient", return_value=client_cm):
        tokens = await _collect(OllamaLlm(_settings()))
    assert tokens == ["a", "b"]


@pytest.mark.asyncio
async def test_stream_ignores_thinking_only_chunks():
    lines = _ndjson(
        {"message": {"thinking": "hmm", "content": ""}, "done": False},
        {"message": {"content": "answer"}, "done": False},
        {"message": {"content": ""}, "done": True},
    )
    client_cm, _ = _fake_stream_client(lines)
    with patch("httpx.AsyncClient", return_value=client_cm):
        tokens = await _collect(OllamaLlm(_settings()))
    assert tokens == ["answer"]


@pytest.mark.asyncio
async def test_stream_connect_error_maps_to_runtime_error():
    client_cm, _ = _fake_stream_client(
        [], raise_exc=httpx.ConnectError("refused")
    )
    with patch("httpx.AsyncClient", return_value=client_cm):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            await _collect(OllamaLlm(_settings()))


@pytest.mark.asyncio
async def test_stream_payload_has_stream_true_and_think_conditional():
    lines = _ndjson({"message": {"content": "x"}, "done": True})

    # think disabled → no "think" key, stream true
    client_cm, client = _fake_stream_client(lines)
    with patch("httpx.AsyncClient", return_value=client_cm):
        await _collect(OllamaLlm(_settings(think=False)))
    payload = client.stream.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert "think" not in payload

    # think enabled → "think": True present
    client_cm, client = _fake_stream_client(lines)
    with patch("httpx.AsyncClient", return_value=client_cm):
        await _collect(OllamaLlm(_settings(think=True)))
    payload = client.stream.call_args.kwargs["json"]
    assert payload["think"] is True


@pytest.mark.asyncio
async def test_generate_unchanged_after_refactor():
    """generate still builds the /mode-stripped prompt and posts stream=False."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": "the answer"}}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=client_cm):
        out = await OllamaLlm(_settings()).generate("/summary q", ["ctx"])

    assert out == "the answer"
    payload = client.post.call_args.kwargs["json"]
    assert payload["stream"] is False
    # /mode prefix stripped from the user question in the final prompt
    sent = payload["messages"][0]["content"]
    assert "/summary" not in sent
    assert "concise summary" in sent
