from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.application.collection_use_case import CollectionUseCase
from src.domain.entities import Collection


@pytest.fixture
def mock_embedding():
    embedding = MagicMock()
    embedding.dimension.return_value = 384
    return embedding


@pytest.fixture
def collection_use_case_with_model(
    mock_vector_store, mock_tfidf, mock_metadata_store, mock_embedding
):
    return CollectionUseCase(
        vector_store=mock_vector_store,
        tfidf=mock_tfidf,
        metadata_store=mock_metadata_store,
        embedding_model="model-x",
        embedding=mock_embedding,
    )


@pytest.fixture
def collection_use_case(mock_vector_store, mock_tfidf, mock_metadata_store):
    return CollectionUseCase(
        vector_store=mock_vector_store,
        tfidf=mock_tfidf,
        metadata_store=mock_metadata_store,
    )


@pytest.mark.asyncio
async def test_create_collection(collection_use_case, mock_metadata_store, sample_collection):
    mock_metadata_store.create_collection.return_value = sample_collection

    result = await collection_use_case.create("test-collection", "A test collection")

    mock_metadata_store.create_collection.assert_called_once_with(
        "test-collection", "A test collection",
        embedding_model="", embedding_dimension=0,
    )
    assert result == sample_collection


@pytest.mark.asyncio
async def test_list_collections(collection_use_case, mock_metadata_store, sample_collection):
    mock_metadata_store.list_collections.return_value = [sample_collection]

    result = await collection_use_case.list_collections()

    mock_metadata_store.list_collections.assert_called_once()
    assert result == [sample_collection]


@pytest.mark.asyncio
async def test_delete_collection_cleans_all_stores(
    collection_use_case, mock_vector_store, mock_metadata_store,
):
    coll_id = str(uuid4())

    await collection_use_case.delete(coll_id)

    mock_vector_store.delete_collection.assert_called_once_with(coll_id)
    mock_metadata_store.delete_collection.assert_called_once_with(coll_id)


@pytest.mark.asyncio
async def test_create_collection_persists_embedding_metadata(
    collection_use_case_with_model, mock_metadata_store, sample_collection
):
    mock_metadata_store.create_collection.return_value = sample_collection

    await collection_use_case_with_model.create("test-collection", "A test collection")

    mock_metadata_store.create_collection.assert_called_once_with(
        "test-collection", "A test collection",
        embedding_model="model-x", embedding_dimension=384,
    )


@pytest.mark.asyncio
async def test_read_operations_never_load_the_embedding_model(
    collection_use_case_with_model, mock_metadata_store, mock_embedding, sample_collection
):
    """dimension() loads the model: only create() may pay that cost."""
    mock_metadata_store.list_collections.return_value = [sample_collection]
    mock_metadata_store.get_collection.return_value = sample_collection

    await collection_use_case_with_model.list_collections()
    await collection_use_case_with_model.get(str(sample_collection.id))
    await collection_use_case_with_model.delete(str(sample_collection.id))

    mock_embedding.dimension.assert_not_called()


@pytest.mark.asyncio
async def test_get_collection_returns_embedding_fields(collection_use_case, mock_metadata_store):
    expected = Collection(name="test", embedding_model="model-x", embedding_dimension=384)
    mock_metadata_store.get_collection.return_value = expected

    result = await collection_use_case.get(str(expected.id))

    assert result.embedding_model == "model-x"
    assert result.embedding_dimension == 384


@pytest.mark.asyncio
async def test_list_collections_returns_embedding_fields(collection_use_case, mock_metadata_store):
    expected = Collection(name="test", embedding_model="model-x", embedding_dimension=384)
    mock_metadata_store.list_collections.return_value = [expected]

    result = await collection_use_case.list_collections()

    assert result[0].embedding_model == "model-x"
    assert result[0].embedding_dimension == 384
