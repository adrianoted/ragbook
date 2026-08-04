import json
import shutil
import threading
from pathlib import Path

import faiss
import numpy as np

from src.domain.entities import Chunk, SearchResult
from src.domain.ports.vector_store_port import VectorStorePort


class FaissVectorStore(VectorStorePort):
    def __init__(self, index_path: str = "data/faiss_indexes") -> None:
        self._base_path = Path(index_path)
        self._indexes: dict[str, faiss.IndexFlatIP] = {}
        self._mappings: dict[str, dict[str, Chunk]] = {}
        self._id_lists: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def _collection_path(self, collection_id: str) -> Path:
        return self._base_path / collection_id

    async def add(self, chunks: list[Chunk], collection_id: str) -> None:
        if not chunks:
            return

        vectors = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            vectors.append(chunk.embedding)

        arr = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(arr)

        with self._lock:
            if collection_id not in self._indexes:
                dim = arr.shape[1]
                self._indexes[collection_id] = faiss.IndexFlatIP(dim)
                self._mappings[collection_id] = {}
                self._id_lists[collection_id] = []

            self._indexes[collection_id].add(arr)
            for i, chunk in enumerate(chunks):
                chunk_id = str(chunk.id)
                self._mappings[collection_id][chunk_id] = chunk
                self._id_lists[collection_id].append(chunk_id)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_id: str | None = None,
    ) -> list[SearchResult]:
        if collection_id is None:
            raise ValueError("collection_id is required for vector search.")
        if collection_id not in self._indexes:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        with self._lock:
            index = self._indexes[collection_id]
            if index.ntotal == 0:
                return []
            k = min(top_k, index.ntotal)
            distances, indices = index.search(query, k)

        results: list[SearchResult] = []
        id_list = self._id_lists[collection_id]
        mapping = self._mappings[collection_id]

        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = id_list[idx]
            chunk = mapping.get(chunk_id)
            if chunk is not None:
                results.append(SearchResult(chunk=chunk, score=float(score), source="vector"))

        return results

    async def delete_document(self, document_id: str, collection_id: str) -> None:
        if collection_id not in self._indexes:
            return

        with self._lock:
            mapping = self._mappings.get(collection_id, {})
            id_list = self._id_lists.get(collection_id, [])
            old_index = self._indexes[collection_id]

            # Rows to keep (exclude those belonging to the document). Row i of
            # the index corresponds to id_list[i], so we work by position.
            keep_rows = [
                i for i, cid in enumerate(id_list)
                if str(mapping[cid].document_id) != document_id
            ]

            if len(keep_rows) == len(id_list):
                return  # nothing to remove

            # Rebuild the index from the vectors stored in the index itself:
            # the Chunk objects have embedding=None (cleared after ingest),
            # while index.reconstruct returns the original (already
            # L2-normalized) vectors.
            new_index = faiss.IndexFlatIP(old_index.d)
            if keep_rows:
                arr = np.array(
                    [old_index.reconstruct(i) for i in keep_rows],
                    dtype=np.float32,
                )
                new_index.add(arr)

            new_id_list = [id_list[i] for i in keep_rows]
            self._indexes[collection_id] = new_index
            self._mappings[collection_id] = {
                cid: mapping[cid] for cid in new_id_list
            }
            self._id_lists[collection_id] = new_id_list

    async def delete_collection(self, collection_id: str) -> None:
        with self._lock:
            self._indexes.pop(collection_id, None)
            self._mappings.pop(collection_id, None)
            self._id_lists.pop(collection_id, None)

        path = self._collection_path(collection_id)
        if path.exists():
            shutil.rmtree(path)

    async def save(self) -> None:
        with self._lock:
            for collection_id, index in self._indexes.items():
                col_path = self._collection_path(collection_id)
                col_path.mkdir(parents=True, exist_ok=True)

                faiss.write_index(index, str(col_path / "index.faiss"))

                mapping = self._mappings.get(collection_id, {})
                serialized = {
                    cid: chunk.model_dump(mode="json") for cid, chunk in mapping.items()
                }
                (col_path / "mappings.json").write_text(json.dumps(serialized))

                id_list = self._id_lists.get(collection_id, [])
                (col_path / "id_list.json").write_text(json.dumps(id_list))

    async def load(self) -> None:
        if not self._base_path.exists():
            return

        with self._lock:
            for col_path in self._base_path.iterdir():
                if not col_path.is_dir():
                    continue

                collection_id = col_path.name
                index_file = col_path / "index.faiss"
                mappings_file = col_path / "mappings.json"
                id_list_file = col_path / "id_list.json"

                if not index_file.exists():
                    continue

                self._indexes[collection_id] = faiss.read_index(str(index_file))

                if mappings_file.exists():
                    raw = json.loads(mappings_file.read_text())
                    self._mappings[collection_id] = {
                        cid: Chunk.model_validate(data) for cid, data in raw.items()
                    }
                else:
                    self._mappings[collection_id] = {}

                if id_list_file.exists():
                    self._id_lists[collection_id] = json.loads(id_list_file.read_text())
                else:
                    self._id_lists[collection_id] = []
