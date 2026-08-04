"""Pure metric functions for retrieval evaluation."""


def mrr(ranked_filenames: list[str], expected_docs: set[str] | list[str]) -> float:
    """Reciprocal rank of the first relevant result (1-based). Substring match."""
    if not ranked_filenames or not expected_docs:
        return 0.0
    expected = set(expected_docs)
    for rank, filename in enumerate(ranked_filenames, start=1):
        if any(exp in filename for exp in expected):
            return 1.0 / rank
    return 0.0


def recall(result_filenames: set[str], expected_docs: set[str]) -> float:
    """Fraction of expected docs found in results. Substring match. 0.0 if expected empty."""
    if not expected_docs:
        return 0.0
    matched = sum(
        1 for exp in expected_docs if any(exp in rf for rf in result_filenames)
    )
    return matched / len(expected_docs)


def keyword_hit_rate(chunk_contents: list[str], expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present (case-insensitive) in joined chunk contents."""
    if not expected_keywords:
        return 0.0
    all_content = " ".join(chunk_contents).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in all_content)
    return hits / len(expected_keywords)
