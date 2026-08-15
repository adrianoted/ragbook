"""HTTP client for communicating with the RAGBook API."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from src.ui.constants import (
    TIMEOUT_DEFAULT,
    TIMEOUT_INGEST,
    TIMEOUT_SEARCH,
)


def _build_search_body(
    query: str,
    collection_id: str | None,
    top_k: int,
    strategy: str,
    **tuning: Any,
) -> dict:
    body: dict = {"query": query, "top_k": top_k, "strategy": strategy}
    if collection_id:
        body["collection_id"] = collection_id
    body.update({k: v for k, v in tuning.items() if v is not None})
    return body


def parse_sse_lines(lines: Iterator[str]) -> Iterator[tuple[str, Any]]:
    """Parse SSE lines into (event, data) pairs.

    Accumulates event/data fields across lines; emits a complete frame on blank
    line. Ignores comment lines (`:` prefix) and frames missing a data field.
    Partial frames at stream end are silently dropped.
    """
    event: str | None = None
    data: str | None = None
    for line in lines:
        if line.startswith(":"):
            continue
        if line == "":
            if event is not None and data is not None:
                yield event, json.loads(data)
            event = None
            data = None
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data = line[len("data:"):].strip()


class ApiClient:
    """Thin wrapper around httpx for RAGBook API calls."""

    def __init__(self, base_url: str) -> None:
        self._base = base_url

    # ── Collections ──────────────────────────────────────────

    def list_collections(self) -> list[dict]:
        try:
            resp = httpx.get(f"{self._base}/collections", timeout=TIMEOUT_DEFAULT)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []

    def create_collection(self, name: str, description: str) -> dict:
        resp = httpx.post(
            f"{self._base}/collections",
            json={"name": name, "description": description},
            timeout=TIMEOUT_DEFAULT,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_collection(self, collection_id: str) -> None:
        resp = httpx.delete(
            f"{self._base}/collections/{collection_id}", timeout=TIMEOUT_DEFAULT
        )
        resp.raise_for_status()

    # ── Ingest ───────────────────────────────────────────────

    def ingest(self, file_path: str, collection_id: str | None = None) -> dict:
        filename = Path(file_path).name
        data = {}
        if collection_id:
            data["collection_id"] = collection_id
        with open(file_path, "rb") as f:
            resp = httpx.post(
                f"{self._base}/ingest",
                files={"file": (filename, f)},
                data=data,
                timeout=TIMEOUT_INGEST,
            )
        resp.raise_for_status()
        return resp.json()

    def ingest_async(self, file_path: str, collection_id: str) -> dict:
        filename = Path(file_path).name
        with open(file_path, "rb") as f:
            resp = httpx.post(
                f"{self._base}/ingest/async",
                files={"file": (filename, f)},
                data={"collection_id": collection_id},
                timeout=TIMEOUT_INGEST,
            )
        resp.raise_for_status()
        return resp.json()

    def get_ingest_job(self, job_id: str) -> dict | None:
        try:
            resp = httpx.get(f"{self._base}/ingest/jobs/{job_id}", timeout=TIMEOUT_DEFAULT)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    def list_documents(self, collection_id: str) -> list[dict]:
        try:
            resp = httpx.get(
                f"{self._base}/collections/{collection_id}/documents",
                timeout=TIMEOUT_DEFAULT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []

    def delete_document(self, collection_id: str, document_id: str) -> None:
        resp = httpx.delete(
            f"{self._base}/collections/{collection_id}/documents/{document_id}",
            timeout=TIMEOUT_DEFAULT,
        )
        resp.raise_for_status()

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        collection_id: str | None = None,
        top_k: int = 5,
        strategy: str = "hybrid",
        min_score: float | None = None,
        fusion: str | None = None,
        hybrid_vector_weight: float | None = None,
        max_results_per_document: int | None = None,
        reranker_enabled: bool | None = None,
        llm_temperature: float | None = None,
        llm_think: bool | None = None,
        llm_num_ctx: int | None = None,
    ) -> dict:
        body = _build_search_body(
            query, collection_id, top_k, strategy,
            min_score=min_score, fusion=fusion,
            hybrid_vector_weight=hybrid_vector_weight,
            max_results_per_document=max_results_per_document,
            reranker_enabled=reranker_enabled,
            llm_temperature=llm_temperature,
            llm_think=llm_think,
            llm_num_ctx=llm_num_ctx,
        )
        resp = httpx.post(f"{self._base}/search", json=body, timeout=TIMEOUT_SEARCH)
        resp.raise_for_status()
        return resp.json()

    def search_stream(
        self,
        query: str,
        *,
        collection_id: str | None = None,
        top_k: int = 5,
        strategy: str = "hybrid",
        min_score: float | None = None,
        fusion: str | None = None,
        hybrid_vector_weight: float | None = None,
        max_results_per_document: int | None = None,
        reranker_enabled: bool | None = None,
        llm_temperature: float | None = None,
        llm_think: bool | None = None,
        llm_num_ctx: int | None = None,
    ) -> Iterator[tuple[str, Any]]:
        body = _build_search_body(
            query, collection_id, top_k, strategy,
            min_score=min_score, fusion=fusion,
            hybrid_vector_weight=hybrid_vector_weight,
            max_results_per_document=max_results_per_document,
            reranker_enabled=reranker_enabled,
            llm_temperature=llm_temperature,
            llm_think=llm_think,
            llm_num_ctx=llm_num_ctx,
        )
        with httpx.stream(
            "POST", f"{self._base}/search/stream", json=body, timeout=TIMEOUT_SEARCH
        ) as resp:
            resp.raise_for_status()
            for event, data in parse_sse_lines(resp.iter_lines()):
                yield event, data
                if event in ("done", "error"):
                    return

    # ── Config ───────────────────────────────────────────────

    def get_config(self) -> dict:
        try:
            resp = httpx.get(f"{self._base}/config", timeout=TIMEOUT_DEFAULT)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    # ── Health ───────────────────────────────────────────────

    def health(self) -> dict:
        """Return the /health payload, or {} when the server is unreachable."""
        try:
            resp = httpx.get(f"{self._base}/health", timeout=TIMEOUT_DEFAULT)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    # ── Helpers ──────────────────────────────────────────────

    def collection_choices(self) -> list[tuple[str, str]]:
        return [(c["name"], c["id"]) for c in self.list_collections()]
