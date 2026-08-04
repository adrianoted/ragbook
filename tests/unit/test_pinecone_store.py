from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.vector_stores.pinecone_store import PineconeVectorStore


def _make_client(has_index: bool, dimension: int | None = None) -> MagicMock:
    """Fake Pinecone client."""
    client = MagicMock()
    client.has_index.return_value = has_index
    if dimension is not None:
        index_model = MagicMock()
        index_model.dimension = dimension
        client.describe_index.return_value = index_model
    return client


@patch("src.infrastructure.vector_stores.pinecone_store.Pinecone")
def test_create_index_when_absent(mock_pinecone_cls):
    client = _make_client(has_index=False)
    mock_pinecone_cls.return_value = client

    PineconeVectorStore(
        api_key="key-abc",
        index_name="my-index",
        dimension=1536,
        cloud="aws",
        region="us-east-1",
    )

    client.has_index.assert_called_once_with("my-index")
    client.create_index.assert_called_once()
    call_kwargs = client.create_index.call_args
    assert call_kwargs.kwargs.get("name") == "my-index" or call_kwargs.args[0] == "my-index"
    assert call_kwargs.kwargs.get("dimension") == 1536
    assert call_kwargs.kwargs.get("metric") == "cosine"
    spec = call_kwargs.kwargs.get("spec")
    assert spec is not None
    client.Index.assert_called_once_with("my-index")


@patch("src.infrastructure.vector_stores.pinecone_store.Pinecone")
def test_no_create_index_when_present_correct_dimension(mock_pinecone_cls):
    client = _make_client(has_index=True, dimension=1536)
    mock_pinecone_cls.return_value = client

    PineconeVectorStore(
        api_key="key-abc",
        index_name="my-index",
        dimension=1536,
        cloud="aws",
        region="us-east-1",
    )

    client.has_index.assert_called_once_with("my-index")
    client.create_index.assert_not_called()
    client.Index.assert_called_once_with("my-index")


@patch("src.infrastructure.vector_stores.pinecone_store.Pinecone")
def test_value_error_on_dimension_mismatch(mock_pinecone_cls):
    client = _make_client(has_index=True, dimension=768)
    mock_pinecone_cls.return_value = client

    with pytest.raises(ValueError) as exc_info:
        PineconeVectorStore(
            api_key="key-abc",
            index_name="my-index",
            dimension=1536,
            cloud="aws",
            region="us-east-1",
        )

    msg = str(exc_info.value)
    assert "my-index" in msg
    assert "768" in msg
    assert "1536" in msg


@pytest.mark.parametrize("bad_key", ["", "   ", "\t"])
@patch("src.infrastructure.vector_stores.pinecone_store.Pinecone")
def test_value_error_on_empty_api_key(mock_pinecone_cls, bad_key):
    with pytest.raises(ValueError, match="api_key"):
        PineconeVectorStore(
            api_key=bad_key,
            index_name="my-index",
            dimension=1536,
            cloud="aws",
            region="us-east-1",
        )

    mock_pinecone_cls.assert_not_called()
