from uuid import UUID

from src.domain.entities import SearchResult


def diversify_results(
    results: list[SearchResult],
    top_k: int = 5,
    max_per_document: int = 2,
) -> list[SearchResult]:
    """Cap results per document and drop duplicate chunk contents.

    Results are assumed sorted by relevance; the first (best) chunks of
    each document win. Preserves input order.
    """
    seen_contents: set[str] = set()
    per_document: dict[UUID, int] = {}
    diversified: list[SearchResult] = []

    for result in results:
        content = result.chunk.content.strip()
        if content in seen_contents:
            continue
        doc_id = result.chunk.document_id
        if per_document.get(doc_id, 0) >= max_per_document:
            continue
        seen_contents.add(content)
        per_document[doc_id] = per_document.get(doc_id, 0) + 1
        diversified.append(result)
        if len(diversified) >= top_k:
            break

    return diversified
