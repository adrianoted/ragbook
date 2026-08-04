import asyncio
import threading

from sentence_transformers import SentenceTransformer

from src.domain.ports.embedding_port import EmbeddingPort

BATCH_SIZE = 256


class SentenceTransformerEmbedding(EmbeddingPort):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch

                    if torch.cuda.is_available():
                        device = "cuda"
                    elif torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                    self._model = SentenceTransformer(self._model_name, device=device)
        return self._model

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        results: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = model.encode(batch, convert_to_numpy=True)
            results.extend(row.tolist() for row in embeddings)
        return results

    def _encode_query_sync(self, text: str) -> list[float]:
        model = self._get_model()
        kwargs: dict = {"convert_to_numpy": True}
        if "query" in model.prompts:
            kwargs["prompt_name"] = "query"
        embeddings = model.encode([text], **kwargs)
        return embeddings[0].tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode_query_sync, text)

    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()
