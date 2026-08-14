"""Unit tests for GeminiLlm — streaming + refactor invariance.

``ChatGoogleGenerativeAI`` is patched so no real client is built; the fake
LLM exposes ``astream`` (async generator of chunks) and ``ainvoke``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.entities import LlmOptions
from src.infrastructure.llm.gemini_llm import GeminiLlm


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        google_api_key="key-abc",
        llm_model="gemini-2.0-flash",
        llm_temperature=0.0,
    )


def _chunk(content):
    return SimpleNamespace(content=content)


def _make_llm(fake_chat):
    with patch(
        "src.infrastructure.llm.gemini_llm.ChatGoogleGenerativeAI",
        return_value=fake_chat,
    ):
        return GeminiLlm(_settings())


@pytest.mark.asyncio
async def test_stream_yields_contents_in_order_skips_empty():
    async def _astream(_prompt):
        for c in ["Hello", "", " world"]:
            yield _chunk(c)

    fake_chat = SimpleNamespace(astream=_astream)
    llm = _make_llm(fake_chat)

    tokens = [t async for t in llm.generate_stream("q", ["ctx"])]
    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_exception_maps_to_runtime_error():
    async def _astream(_prompt):
        yield _chunk("partial")
        raise ValueError("boom")

    fake_chat = SimpleNamespace(astream=_astream)
    llm = _make_llm(fake_chat)

    with pytest.raises(RuntimeError, match="LLM generation failed"):
        _ = [t async for t in llm.generate_stream("q", ["ctx"])]


@pytest.mark.asyncio
async def test_generate_unchanged_after_refactor():
    fake_chat = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="the answer"))
    )
    llm = _make_llm(fake_chat)

    out = await llm.generate("/summary q", ["ctx"])

    assert out == "the answer"
    sent = fake_chat.ainvoke.call_args.args[0]
    assert "/summary" not in sent
    assert "concise summary" in sent


def _fake_astream():
    """MagicMock returning a fresh async generator; records call kwargs."""

    def _side_effect(*_args, **_kwargs):
        async def _gen():
            yield _chunk("Hello")

        return _gen()

    return MagicMock(side_effect=_side_effect)


@pytest.mark.asyncio
async def test_generate_without_options_omits_temperature_kwarg():
    fake_chat = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="a"))
    )
    llm = _make_llm(fake_chat)

    await llm.generate("q", ["ctx"])

    assert "temperature" not in fake_chat.ainvoke.call_args.kwargs


@pytest.mark.asyncio
async def test_generate_options_temperature_passed_to_ainvoke():
    fake_chat = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="a"))
    )
    llm = _make_llm(fake_chat)

    await llm.generate("q", ["ctx"], LlmOptions(temperature=0.9))

    assert fake_chat.ainvoke.call_args.kwargs["temperature"] == 0.9


@pytest.mark.asyncio
async def test_generate_options_temperature_zero_preserved():
    fake_chat = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="a"))
    )
    llm = _make_llm(fake_chat)

    await llm.generate("q", ["ctx"], LlmOptions(temperature=0.0))

    assert fake_chat.ainvoke.call_args.kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_generate_ollama_only_options_are_noop():
    fake_chat = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="a"))
    )
    llm = _make_llm(fake_chat)

    await llm.generate("q", ["ctx"], LlmOptions(think=True, num_ctx=4096))

    assert fake_chat.ainvoke.call_args.kwargs == {}


@pytest.mark.asyncio
async def test_stream_options_temperature_passed_to_astream():
    astream = _fake_astream()
    fake_chat = SimpleNamespace(astream=astream)
    llm = _make_llm(fake_chat)

    _ = [
        t
        async for t in llm.generate_stream(
            "q", ["ctx"], LlmOptions(temperature=0.9)
        )
    ]

    assert astream.call_args.kwargs["temperature"] == 0.9
