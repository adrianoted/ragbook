from uuid import uuid4

import pytest

from src.domain.entities import Chunk
from src.infrastructure.tfidf.sklearn_tfidf import SklearnTfidf


def _make_chunk(content: str) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        content=content,
        index=0,
    )


@pytest.fixture
def tfidf(tmp_path):
    return SklearnTfidf(index_path=str(tmp_path))


@pytest.fixture
def chunks():
    return [
        _make_chunk("Python is a popular programming language for data science"),
        _make_chunk("Java is widely used for enterprise backend development"),
        _make_chunk("Cooking pasta requires boiling water and adding salt"),
    ]


COLLECTION_ID = "test-collection"


@pytest.mark.asyncio
async def test_fit_and_search(tfidf, chunks):
    await tfidf.fit(chunks, COLLECTION_ID)

    results = await tfidf.search("python programming", top_k=3, collection_id=COLLECTION_ID)

    assert len(results) > 0
    assert results[0].chunk.content == chunks[0].content


@pytest.mark.asyncio
async def test_search_empty_collection(tfidf):
    results = await tfidf.search("python", top_k=3, collection_id="nonexistent-collection")

    assert results == []


@pytest.mark.asyncio
async def test_search_without_collection_id_raises(tfidf, chunks):
    await tfidf.fit(chunks, COLLECTION_ID)
    with pytest.raises(ValueError):
        await tfidf.search("python", top_k=3, collection_id=None)


@pytest.mark.asyncio
async def test_fit_creates_index_files(tfidf, chunks, tmp_path):
    await tfidf.fit(chunks, COLLECTION_ID)

    collection_dir = tmp_path / COLLECTION_ID
    assert (collection_dir / "vectorizer.pkl").exists()
    assert (collection_dir / "matrix.pkl").exists()
    assert (collection_dir / "chunks.pkl").exists()


@pytest.mark.asyncio
async def test_search_result_scores_normalized(tfidf, chunks):
    await tfidf.fit(chunks, COLLECTION_ID)

    results = await tfidf.search("python programming", top_k=3, collection_id=COLLECTION_ID)

    assert len(results) > 0
    for result in results:
        assert 0.0 <= result.score <= 1.0
