import pytest

from evaluation.evaluate import compute_aggregates
from evaluation.metrics import (
    keyword_hit_rate,
    mrr,
    recall,
)


# --- mrr ---


def test_mrr_hit_at_rank_1():
    assert mrr(["doc_a.md"], {"doc_a.md"}) == pytest.approx(1.0)


def test_mrr_hit_at_rank_2():
    assert mrr(["other.md", "doc_a.md"], {"doc_a.md"}) == pytest.approx(0.5)


def test_mrr_hit_at_rank_n():
    filenames = ["x.md", "y.md", "z.md", "doc_a.md"]
    assert mrr(filenames, {"doc_a.md"}) == pytest.approx(1 / 4)


def test_mrr_no_hit():
    assert mrr(["other.md", "another.md"], {"doc_a.md"}) == pytest.approx(0.0)


def test_mrr_empty_filenames():
    assert mrr([], {"doc_a.md"}) == pytest.approx(0.0)


def test_mrr_empty_expected():
    assert mrr(["doc_a.md"], set()) == pytest.approx(0.0)


def test_mrr_substring_match_uuid_prefix():
    assert mrr(["abc123_doc_a.md"], {"doc_a.md"}) == pytest.approx(1.0)


def test_mrr_uses_first_hit_rank():
    # Two expected docs; first appearing is at rank 2
    filenames = ["irrelevant.md", "doc_b.md", "doc_a.md"]
    assert mrr(filenames, {"doc_a.md", "doc_b.md"}) == pytest.approx(0.5)


# --- recall ---


def test_recall_all_found():
    assert recall({"doc_a.md", "doc_b.md"}, {"doc_a.md", "doc_b.md"}) == pytest.approx(1.0)


def test_recall_partial():
    assert recall({"doc_a.md", "other.md"}, {"doc_a.md", "doc_b.md"}) == pytest.approx(0.5)


def test_recall_none_found():
    assert recall({"other.md"}, {"doc_a.md"}) == pytest.approx(0.0)


def test_recall_empty_expected():
    assert recall({"doc_a.md"}, set()) == pytest.approx(0.0)


def test_recall_substring_match():
    # UUID-prefixed filename matches the bare expected doc name
    assert recall({"abc123_doc_a.md"}, {"doc_a.md"}) == pytest.approx(1.0)


# --- keyword_hit_rate ---


def test_keyword_hit_rate_all_present():
    contents = ["The graph state is updated", "LangGraph node"]
    assert keyword_hit_rate(contents, ["graph state", "langgraph"]) == pytest.approx(1.0)


def test_keyword_hit_rate_partial():
    contents = ["The graph state is updated"]
    assert keyword_hit_rate(contents, ["graph state", "missing_kw"]) == pytest.approx(0.5)


def test_keyword_hit_rate_none_present():
    contents = ["irrelevant content"]
    assert keyword_hit_rate(contents, ["graph state"]) == pytest.approx(0.0)


def test_keyword_hit_rate_empty_keywords():
    assert keyword_hit_rate(["some content"], []) == pytest.approx(0.0)


def test_keyword_hit_rate_case_insensitive():
    contents = ["LangGraph is a framework"]
    assert keyword_hit_rate(contents, ["langgraph"]) == pytest.approx(1.0)


# --- compute_aggregates ---


def _positive(query, **overrides):
    base = {
        "query": query,
        "avg_score": 0.5,
        "recall": 1.0,
        "keyword_hit_rate": 1.0,
        "unique_docs_ratio": 0.8,
        "mrr": 1.0,
        "latency_ms": 100.0,
    }
    base.update(overrides)
    return base


def _negative(query, **overrides):
    base = {
        "query": query,
        "negative": True,
        "negative_pass": 1.0,
        "latency_ms": 100.0,
    }
    base.update(overrides)
    return base


def test_compute_aggregates_empty():
    assert compute_aggregates([]) == {}


def test_compute_aggregates_positive_only():
    results = [
        _positive("q1", avg_score=0.4, recall=1.0, mrr=1.0),
        _positive("q2", avg_score=0.6, recall=0.0, mrr=0.5),
    ]
    aggs = compute_aggregates(results)
    assert aggs["avg_score"] == pytest.approx(0.5)
    assert aggs["avg_recall"] == pytest.approx(0.5)
    assert aggs["avg_mrr"] == pytest.approx(0.75)
    assert aggs["avg_latency_ms"] == pytest.approx(100.0)
    assert "negative_pass_rate" not in aggs


def test_compute_aggregates_negative_only_omits_positive_keys():
    results = [_negative("n1", negative_pass=1.0), _negative("n2", negative_pass=0.0)]
    aggs = compute_aggregates(results)
    assert aggs["negative_pass_rate"] == pytest.approx(0.5)
    assert "avg_score" not in aggs
    assert "avg_mrr" not in aggs
    assert aggs["avg_latency_ms"] == pytest.approx(100.0)


def test_compute_aggregates_mixed_excludes_negatives_from_positive_avgs():
    # avg_score must average only the two positives, not the negative
    results = [
        _positive("q1", avg_score=0.4),
        _positive("q2", avg_score=0.6),
        _negative("n1", negative_pass=1.0),
    ]
    aggs = compute_aggregates(results)
    assert aggs["avg_score"] == pytest.approx(0.5)  # not divided by 3
    assert aggs["negative_pass_rate"] == pytest.approx(1.0)
    # latency averages across all results
    assert aggs["avg_latency_ms"] == pytest.approx(100.0)
