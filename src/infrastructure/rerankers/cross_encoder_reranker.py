import asyncio
import math
import threading

from sentence_transformers import CrossEncoder

from src.domain.entities import SearchResult
from src.domain.ports.reranker_port import RerankerPort


class CrossEncoderReranker(RerankerPort):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> CrossEncoder:
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
                    self._model = CrossEncoder(self._model_name, device=device)
        return self._model

    def _rerank_sync(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []

        model = self._get_model()
        pairs = [(query, result.chunk.content) for result in results]
        scores = model.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = []
        for result, score in scored[:top_k]:
            normalized_score = 1 / (1 + math.exp(-float(score)))
            reranked.append(
                result.model_copy(update={"score": normalized_score})
            )
        return reranked

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        return await asyncio.to_thread(self._rerank_sync, query, results, top_k)
