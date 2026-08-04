import asyncio
import pickle
import shutil
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.domain.entities import Chunk, SearchResult
from src.domain.ports.tfidf_port import TfidfPort


def _tokenize(content: str) -> list[str]:
    # Deliberately simple: lowercase + whitespace split, no stemming/stopwords.
    # The choice is declared in the analysis so the eval baseline is comparable.
    return content.lower().split()


class Bm25Lexical(TfidfPort):
    def __init__(self, index_path: str = "data/bm25_indexes/") -> None:
        self._index_path = Path(index_path)
        self._chunks: dict[str, list[Chunk]] = {}
        self._corpus: dict[str, list[list[str]]] = {}
        self._bm25: dict[str, BM25Okapi] = {}
        self._load_all()

    def _collection_dir(self, collection_id: str) -> Path:
        return self._index_path / collection_id

    def _load_all(self) -> None:
        if not self._index_path.exists():
            return
        for child in self._index_path.iterdir():
            if child.is_dir():
                self._load_collection(child.name)

    def _load_collection(self, collection_id: str) -> None:
        cdir = self._collection_dir(collection_id)
        chunks_path = cdir / "chunks.pkl"
        corpus_path = cdir / "corpus.pkl"
        if chunks_path.exists() and corpus_path.exists():
            with open(chunks_path, "rb") as f:
                self._chunks[collection_id] = pickle.load(f)
            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)
            self._corpus[collection_id] = corpus
            # BM25Okapi is not picklable — rebuild it from the tokenized corpus.
            self._bm25[collection_id] = BM25Okapi(corpus)

    def _save_collection(self, collection_id: str) -> None:
        cdir = self._collection_dir(collection_id)
        cdir.mkdir(parents=True, exist_ok=True)
        with open(cdir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks[collection_id], f)
        with open(cdir / "corpus.pkl", "wb") as f:
            pickle.dump(self._corpus[collection_id], f)

    def _rebuild(self, collection_id: str, chunks: list[Chunk]) -> None:
        corpus = [_tokenize(c.content) for c in chunks]
        self._chunks[collection_id] = chunks
        self._corpus[collection_id] = corpus
        self._bm25[collection_id] = BM25Okapi(corpus)
        self._save_collection(collection_id)

    def _cleanup(self, collection_id: str) -> None:
        self._chunks.pop(collection_id, None)
        self._corpus.pop(collection_id, None)
        self._bm25.pop(collection_id, None)
        cdir = self._collection_dir(collection_id)
        if cdir.exists():
            shutil.rmtree(cdir)

    def _fit_sync(self, chunks: list[Chunk], collection_id: str) -> None:
        existing = self._chunks.get(collection_id, [])
        existing_ids = {c.id for c in existing}
        new_chunks = [c for c in chunks if c.id not in existing_ids]
        all_chunks = existing + new_chunks

        if not all_chunks:
            return

        self._rebuild(collection_id, all_chunks)

    def _search_sync(
        self, query: str, top_k: int, collection_id: str | None = None
    ) -> list[SearchResult]:
        # A single collection = a single BM25 index. Searching across collections
        # would mix incomparable scores, so collection_id is required; the API
        # layer validates it before we get here.
        if collection_id is None:
            raise ValueError("collection_id is required for BM25 search.")
        if collection_id not in self._bm25:
            return []

        bm25 = self._bm25[collection_id]
        chunks = self._chunks[collection_id]

        scores = bm25.get_scores(_tokenize(query))

        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        results: list[SearchResult] = []
        for idx in ranked[:top_k]:
            score = float(scores[idx])
            if score > 0:
                results.append(
                    SearchResult(chunk=chunks[idx], score=score, source="bm25")
                )

        return results

    async def fit(self, chunks: list[Chunk], collection_id: str) -> None:
        await asyncio.to_thread(self._fit_sync, chunks, collection_id)

    async def search(
        self, query: str, top_k: int, collection_id: str | None = None
    ) -> list[SearchResult]:
        return await asyncio.to_thread(self._search_sync, query, top_k, collection_id)

    async def delete_document(self, document_id: str, collection_id: str) -> None:
        if collection_id not in self._chunks:
            return

        existing = self._chunks[collection_id]
        remaining = [c for c in existing if str(c.document_id) != document_id]

        if len(remaining) == len(existing):
            return  # nothing to remove

        if remaining:
            await asyncio.to_thread(self._rebuild, collection_id, remaining)
        else:
            # No chunks left — clean up entirely
            await asyncio.to_thread(self._cleanup, collection_id)

    async def delete_collection(self, collection_id: str) -> None:
        await asyncio.to_thread(self._cleanup, collection_id)
