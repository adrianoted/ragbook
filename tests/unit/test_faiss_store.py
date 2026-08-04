import numpy as np
import pytest
from uuid import uuid4

from src.domain.entities import Chunk, SearchResult
from src.infrastructure.vector_stores.faiss_store import FaissVectorStore


def _make_chunks(n: int, dim: int = 8) -> list[Chunk]:
    doc_id = uuid4()
    chunks = []
    for i in range(n):
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        chunks.append(
            Chunk(
                document_id=doc_id,
                content=f"chunk {i}",
                metadata={"position": i},
                index=i,
                embedding=vec.tolist(),
            )
        )
    return chunks


@pytest.mark.asyncio
async def test_add_and_search(tmp_path):
    store = FaissVectorStore(index_path=str(tmp_path))
    chunks = _make_chunks(3)
    await store.add(chunks, "col1")

    query_vec = chunks[0].embedding
    results = await store.search(query_vec, top_k=2, collection_id="col1")

    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].chunk.id == chunks[0].id


@pytest.mark.asyncio
async def test_search_empty_index(tmp_path):
    store = FaissVectorStore(index_path=str(tmp_path))
    results = await store.search([0.1] * 8, top_k=5, collection_id="col1")
    assert results == []


@pytest.mark.asyncio
async def test_search_without_collection_id_raises(tmp_path):
    store = FaissVectorStore(index_path=str(tmp_path))
    with pytest.raises(ValueError):
        await store.search([0.1] * 8, top_k=5, collection_id=None)


@pytest.mark.asyncio
async def test_save_and_load(tmp_path):
    store = FaissVectorStore(index_path=str(tmp_path))
    chunks = _make_chunks(3)
    await store.add(chunks, "col1")
    await store.save()

    store2 = FaissVectorStore(index_path=str(tmp_path))
    await store2.load()

    results = await store2.search(chunks[0].embedding, top_k=2, collection_id="col1")
    assert len(results) > 0
    assert results[0].chunk.id == chunks[0].id


@pytest.mark.asyncio
async def test_delete_collection(tmp_path):
    store = FaissVectorStore(index_path=str(tmp_path))
    chunks = _make_chunks(2)
    await store.add(chunks, "col1")
    await store.delete_collection("col1")

    results = await store.search(chunks[0].embedding, top_k=5, collection_id="col1")
    assert results == []


@pytest.mark.asyncio
async def test_delete_document_after_ingest_clears_embeddings(tmp_path):
    """IngestUseCase clears chunk.embedding after add(); delete_document must
    recover the vectors from the index itself, keeping row/id alignment."""
    store = FaissVectorStore(index_path=str(tmp_path))
    chunks_a = _make_chunks(3)
    chunks_b = _make_chunks(3)
    query = list(chunks_b[0].embedding)

    await store.add(chunks_a, "col1")
    await store.add(chunks_b, "col1")
    for c in chunks_a + chunks_b:
        c.embedding = None  # mimic IngestUseCase memory cleanup

    await store.delete_document(str(chunks_a[0].document_id), "col1")

    results = await store.search(query, top_k=6, collection_id="col1")
    assert len(results) == 3
    assert all(r.chunk.document_id == chunks_b[0].document_id for r in results)
    assert results[0].chunk.id == chunks_b[0].id


@pytest.mark.asyncio
async def test_delete_document_removes_all_chunks(tmp_path):
    store = FaissVectorStore(index_path=str(tmp_path))
    chunks = _make_chunks(2)
    query = list(chunks[0].embedding)
    await store.add(chunks, "col1")
    for c in chunks:
        c.embedding = None

    await store.delete_document(str(chunks[0].document_id), "col1")

    assert await store.search(query, top_k=5, collection_id="col1") == []


@pytest.mark.asyncio
async def test_multiple_collections(tmp_path):
    store = FaissVectorStore(index_path=str(tmp_path))
    chunks_a = _make_chunks(2)
    chunks_b = _make_chunks(2)
    await store.add(chunks_a, "colA")
    await store.add(chunks_b, "colB")

    results_a = await store.search(chunks_a[0].embedding, top_k=5, collection_id="colA")
    results_b = await store.search(chunks_b[0].embedding, top_k=5, collection_id="colB")

    ids_a = {r.chunk.id for r in results_a}
    ids_b = {r.chunk.id for r in results_b}

    assert all(c.id in ids_a for c in chunks_a)
    assert all(c.id in ids_b for c in chunks_b)
    assert ids_a.isdisjoint(ids_b)
