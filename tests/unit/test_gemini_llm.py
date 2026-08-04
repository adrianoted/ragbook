"""Unit tests for GeminiLlm — streaming + refactor invariance.

``ChatGoogleGenerativeAI`` is patched so no real client is built; the fake
LLM exposes ``astream`` (async generator of chunks) and ``ainvoke``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

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
