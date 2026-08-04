import asyncio
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.domain.entities import Chunk, SearchResult
from src.domain.ports.tfidf_port import TfidfPort


class SklearnTfidf(TfidfPort):
    def __init__(self, index_path: str = "data/tfidf_indexes/") -> None:
        self._index_path = Path(index_path)
        self._vectorizers: dict[str, TfidfVectorizer] = {}
        self._matrices: dict[str, object] = {}
        self._chunks: dict[str, list[Chunk]] = {}
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
        vectorizer_path = cdir / "vectorizer.pkl"
        matrix_path = cdir / "matrix.pkl"
        chunks_path = cdir / "chunks.pkl"
        if vectorizer_path.exists() and matrix_path.exists() and chunks_path.exists():
            with open(vectorizer_path, "rb") as f:
                self._vectorizers[collection_id] = pickle.load(f)
            with open(matrix_path, "rb") as f:
                self._matrices[collection_id] = pickle.load(f)
            with open(chunks_path, "rb") as f:
                self._chunks[collection_id] = pickle.load(f)

    def _save_collection(self, collection_id: str) -> None:
        cdir = self._collection_dir(collection_id)
        cdir.mkdir(parents=True, exist_ok=True)
        with open(cdir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self._vectorizers[collection_id], f)
        with open(cdir / "matrix.pkl", "wb") as f:
            pickle.dump(self._matrices[collection_id], f)
        with open(cdir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks[collection_id], f)

    def _fit_sync(self, chunks: list[Chunk], collection_id: str) -> None:
        existing = self._chunks.get(collection_id, [])
        existing_ids = {c.id for c in existing}
        new_chunks = [c for c in chunks if c.id not in existing_ids]
        all_chunks = existing + new_chunks

        if not all_chunks:
            return

        vectorizer = TfidfVectorizer()
        corpus = [chunk.content for chunk in all_chunks]
        matrix = vectorizer.fit_transform(corpus)

        self._vectorizers[collection_id] = vectorizer
        self._matrices[collection_id] = matrix
        self._chunks[collection_id] = all_chunks
        self._save_collection(collection_id)

    def _search_sync(
        self, query: str, top_k: int, collection_id: str | None = None
    ) -> list[SearchResult]:
        # A single collection = a single vectorizer/IDF space. Searching across
        # collections would mix incomparable TF-IDF scores, so collection_id is
        # required; the API layer validates it before we get here.
        if collection_id is None:
            raise ValueError("collection_id is required for TF-IDF search.")
        if collection_id not in self._vectorizers:
            return []

        vectorizer = self._vectorizers[collection_id]
        matrix = self._matrices[collection_id]
        chunks = self._chunks[collection_id]

        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, matrix).flatten()

        # argsort()[::-1][:top_k] already yields the top_k indices in descending
        # score order for this single collection — no extra sort/slice needed.
        results: list[SearchResult] = []
        for idx in similarities.argsort()[::-1][:top_k]:
            score = float(similarities[idx])
            if score > 0:
                results.append(
                    SearchResult(chunk=chunks[idx], score=score, source="tfidf")
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
            vectorizer = TfidfVectorizer()
            corpus = [chunk.content for chunk in remaining]
            matrix = vectorizer.fit_transform(corpus)
            self._vectorizers[collection_id] = vectorizer
            self._matrices[collection_id] = matrix
            self._chunks[collection_id] = remaining
            self._save_collection(collection_id)
        else:
            # No chunks left — clean up entirely
            self._vectorizers.pop(collection_id, None)
            self._matrices.pop(collection_id, None)
            self._chunks.pop(collection_id, None)
            cdir = self._collection_dir(collection_id)
            if cdir.exists():
                import shutil

                shutil.rmtree(cdir)

    async def delete_collection(self, collection_id: str) -> None:
        self._vectorizers.pop(collection_id, None)
        self._matrices.pop(collection_id, None)
        self._chunks.pop(collection_id, None)
        cdir = self._collection_dir(collection_id)
        if cdir.exists():
            import shutil

            shutil.rmtree(cdir)
