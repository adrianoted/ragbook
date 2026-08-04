from uuid import uuid4

import pytest

from src.domain.entities import Chunk
from src.infrastructure.tfidf.bm25 import Bm25Lexical


def _make_chunk(content: str, document_id=None) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=document_id or uuid4(),
        content=content,
        index=0,
    )


@pytest.fixture
def bm25(tmp_path):
    return Bm25Lexical(index_path=str(tmp_path))


@pytest.fixture
def chunks():
    return [
        _make_chunk("Python is a popular programming language for data science"),
        _make_chunk("Java is widely used for enterprise backend development"),
        _make_chunk("Cooking pasta requires boiling water and adding salt"),
    ]


COLLECTION_ID = "test-collection"


@pytest.mark.asyncio
async def test_fit_and_search(bm25, chunks):
    await bm25.fit(chunks, COLLECTION_ID)

    results = await bm25.search("python programming", top_k=3, collection_id=COLLECTION_ID)

    assert len(results) > 0
    assert results[0].chunk.content == chunks[0].content
    assert results[0].source == "bm25"
    # scores descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_fit_incremental_dedup(bm25, chunks):
    await bm25.fit(chunks, COLLECTION_ID)
    # refit with same chunks (same ids) + one new
    new_chunk = _make_chunk("Rust is a systems programming language")
    await bm25.fit(chunks + [new_chunk], COLLECTION_ID)

    results = await bm25.search("programming language", top_k=10, collection_id=COLLECTION_ID)
    contents = [r.chunk.content for r in results]
    # no duplicates of the original chunk
    assert contents.count(chunks[0].content) <= 1
    # the new chunk is searchable
    all_results = await bm25.search("rust systems", top_k=10, collection_id=COLLECTION_ID)
    assert any(r.chunk.content == new_chunk.content for r in all_results)


@pytest.mark.asyncio
async def test_search_without_collection_id_raises(bm25, chunks):
    await bm25.fit(chunks, COLLECTION_ID)
    with pytest.raises(ValueError):
        await bm25.search("python", top_k=3, collection_id=None)


@pytest.mark.asyncio
async def test_search_unknown_collection(bm25):
    results = await bm25.search("python", top_k=3, collection_id="nonexistent-collection")
    assert results == []


@pytest.mark.asyncio
async def test_search_no_matching_terms(bm25, chunks):
    await bm25.fit(chunks, COLLECTION_ID)
    results = await bm25.search("zzzznonexistentterm", top_k=3, collection_id=COLLECTION_ID)
    assert results == []


@pytest.mark.asyncio
async def test_delete_document_refit(bm25):
    doc_a = uuid4()
    doc_b = uuid4()
    doc_c = uuid4()
    doc_d = uuid4()
    chunks = [
        _make_chunk("Python programming for data science", document_id=doc_a),
        _make_chunk("Cooking pasta with salt and water", document_id=doc_b),
        _make_chunk("Java enterprise backend development", document_id=doc_c),
        _make_chunk("Rust systems memory safety", document_id=doc_d),
    ]
    await bm25.fit(chunks, COLLECTION_ID)

    await bm25.delete_document(str(doc_a), COLLECTION_ID)

    results = await bm25.search("cooking pasta", top_k=10, collection_id=COLLECTION_ID)
    contents = [r.chunk.content for r in results]
    assert chunks[1].content in contents
    # doc_a chunk gone
    py_results = await bm25.search("python", top_k=10, collection_id=COLLECTION_ID)
    assert all(r.chunk.content != chunks[0].content for r in py_results)


@pytest.mark.asyncio
async def test_delete_last_document_removes_dir(bm25, tmp_path):
    doc = uuid4()
    chunks = [_make_chunk("only document here", document_id=doc)]
    await bm25.fit(chunks, COLLECTION_ID)
    assert (tmp_path / COLLECTION_ID).exists()

    await bm25.delete_document(str(doc), COLLECTION_ID)

    assert not (tmp_path / COLLECTION_ID).exists()
    results = await bm25.search("document", top_k=3, collection_id=COLLECTION_ID)
    assert results == []


@pytest.mark.asyncio
async def test_delete_collection_removes_dir(bm25, chunks, tmp_path):
    await bm25.fit(chunks, COLLECTION_ID)
    assert (tmp_path / COLLECTION_ID).exists()

    await bm25.delete_collection(COLLECTION_ID)

    assert not (tmp_path / COLLECTION_ID).exists()
    results = await bm25.search("python", top_k=3, collection_id=COLLECTION_ID)
    assert results == []


@pytest.mark.asyncio
async def test_fit_creates_index_files(bm25, chunks, tmp_path):
    await bm25.fit(chunks, COLLECTION_ID)

    collection_dir = tmp_path / COLLECTION_ID
    assert (collection_dir / "chunks.pkl").exists()
    assert (collection_dir / "corpus.pkl").exists()


@pytest.mark.asyncio
async def test_persistence_across_instances(bm25, chunks, tmp_path):
    await bm25.fit(chunks, COLLECTION_ID)
    first = await bm25.search("python programming", top_k=3, collection_id=COLLECTION_ID)

    # new instance, same index_path — loads from pickle + rebuilds BM25Okapi
    reloaded = Bm25Lexical(index_path=str(tmp_path))
    second = await reloaded.search("python programming", top_k=3, collection_id=COLLECTION_ID)

    assert [r.chunk.content for r in first] == [r.chunk.content for r in second]
    assert [r.score for r in first] == [r.score for r in second]
