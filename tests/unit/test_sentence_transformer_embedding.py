import numpy as np
import pytest
from unittest.mock import MagicMock

from src.infrastructure.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)


@pytest.fixture
def model_with_prompt():
    m = MagicMock()
    m.prompts = {"query": "Instruct: Retrieve semantically similar text."}
    m.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    return m


@pytest.fixture
def model_without_prompt():
    m = MagicMock()
    m.prompts = {}
    m.encode.return_value = np.array([[0.4, 0.5, 0.6]])
    return m


@pytest.mark.asyncio
async def test_embed_query_uses_prompt_name_when_available(model_with_prompt):
    adapter = SentenceTransformerEmbedding("test-model")
    adapter._model = model_with_prompt

    await adapter.embed_query("cosa è RAG?")

    model_with_prompt.encode.assert_called_once_with(
        ["cosa è RAG?"], convert_to_numpy=True, prompt_name="query"
    )


@pytest.mark.asyncio
async def test_embed_query_no_prompt_name_when_not_available(model_without_prompt):
    adapter = SentenceTransformerEmbedding("test-model")
    adapter._model = model_without_prompt

    await adapter.embed_query("cosa è RAG?")

    model_without_prompt.encode.assert_called_once_with(
        ["cosa è RAG?"], convert_to_numpy=True
    )


@pytest.mark.asyncio
async def test_embed_passages_never_uses_prompt_name_with_prompt_model(model_with_prompt):
    model_with_prompt.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    adapter = SentenceTransformerEmbedding("test-model")
    adapter._model = model_with_prompt

    await adapter.embed(["passaggio uno", "passaggio due"])

    assert "prompt_name" not in model_with_prompt.encode.call_args.kwargs


@pytest.mark.asyncio
async def test_embed_passages_never_uses_prompt_name_without_prompt_model(model_without_prompt):
    model_without_prompt.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    adapter = SentenceTransformerEmbedding("test-model")
    adapter._model = model_without_prompt

    await adapter.embed(["passaggio uno", "passaggio due"])

    assert "prompt_name" not in model_without_prompt.encode.call_args.kwargs
