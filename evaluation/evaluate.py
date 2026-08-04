"""Evaluation framework for retrieval quality.

Usage:
    python -m evaluation.evaluate --collection-id <ID> --api-url http://localhost:8000 --top-k 5 --strategy hybrid
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from evaluation.metrics import (
    keyword_hit_rate,
    mrr,
    recall,
)

GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_golden_queries() -> list[dict]:
    with open(GOLDEN_QUERIES_PATH) as f:
        return json.load(f)


def evaluate_query(
    client: httpx.Client,
    api_url: str,
    query_entry: dict,
    top_k: int,
    strategy: str | None,
    collection_id: str | None = None,
) -> dict:
    payload: dict = {"query": query_entry["query"], "top_k": top_k}
    if strategy:
        payload["strategy"] = strategy
    if collection_id:
        payload["collection_id"] = collection_id

    start = time.perf_counter()
    response = client.post(f"{api_url}/api/search/raw", json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    data = response.json()
    sources = data["sources"]
    total_results = len(sources)

    # Ordered list required for MRR (set used only for recall/unique_docs)
    ordered_filenames = [s["document_filename"] for s in sources]
    result_filenames = set(ordered_filenames)

    if query_entry.get("negative"):
        return {
            "query": query_entry["query"],
            "negative": True,
            "negative_pass": 1.0 if total_results == 0 else 0.0,
            "latency_ms": round(latency_ms, 1),
            "num_results": total_results,
            "result_filenames": sorted(result_filenames),
        }

    scores = [s["score"] for s in sources]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    expected_docs = set(query_entry.get("expected_documents", []))
    chunk_contents = [s["chunk_content"] for s in sources]
    expected_keywords = query_entry.get("expected_keywords", [])

    unique_docs = len(result_filenames)
    unique_docs_ratio = unique_docs / total_results if total_results else 0.0

    return {
        "query": query_entry["query"],
        "avg_score": round(avg_score, 4),
        "recall": round(recall(result_filenames, expected_docs), 4),
        "keyword_hit_rate": round(keyword_hit_rate(chunk_contents, expected_keywords), 4),
        "unique_docs_ratio": round(unique_docs_ratio, 4),
        "mrr": round(mrr(ordered_filenames, expected_docs), 4),
        "latency_ms": round(latency_ms, 1),
        "num_results": total_results,
        "result_filenames": sorted(result_filenames),
    }


def print_results(results: list[dict], aggregates: dict) -> None:
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)

    for r in results:
        print(f"\nQuery: {r['query']}")
        if r.get("negative"):
            print("  [NEGATIVE QUERY]")
            print(f"  negative_pass:    {r['negative_pass']:.4f}")
            print(f"  latency_ms:       {r['latency_ms']:.1f}")
            print(f"  num_results:      {r['num_results']}")
            print(f"  files:            {', '.join(r['result_filenames'])}")
        else:
            print(f"  avg_score:        {r['avg_score']:.4f}")
            print(f"  recall:           {r['recall']:.4f}")
            print(f"  keyword_hit_rate: {r['keyword_hit_rate']:.4f}")
            print(f"  unique_docs_ratio:{r['unique_docs_ratio']:.4f}")
            print(f"  mrr:              {r['mrr']:.4f}")
            print(f"  latency_ms:       {r['latency_ms']:.1f}")
            print(f"  num_results:      {r['num_results']}")
            print(f"  files:            {', '.join(r['result_filenames'])}")

    print("\n" + "-" * 80)
    print("AGGREGATES")
    print("-" * 80)
    for key, value in aggregates.items():
        print(f"  {key}: {value:.4f}")
    print()


def compute_aggregates(results: list[dict]) -> dict:
    if not results:
        return {}

    positive = [r for r in results if not r.get("negative")]
    negative = [r for r in results if r.get("negative")]

    aggs: dict = {}

    if positive:
        n = len(positive)
        aggs["avg_score"] = round(sum(r["avg_score"] for r in positive) / n, 4)
        aggs["avg_recall"] = round(sum(r["recall"] for r in positive) / n, 4)
        aggs["avg_keyword_hit_rate"] = round(
            sum(r["keyword_hit_rate"] for r in positive) / n, 4
        )
        aggs["avg_unique_docs_ratio"] = round(
            sum(r["unique_docs_ratio"] for r in positive) / n, 4
        )
        aggs["avg_mrr"] = round(sum(r["mrr"] for r in positive) / n, 4)

    if negative:
        aggs["negative_pass_rate"] = round(
            sum(r["negative_pass"] for r in negative) / len(negative), 4
        )

    aggs["avg_latency_ms"] = round(sum(r["latency_ms"] for r in results) / len(results), 1)

    return aggs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="Base URL of the API"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["vector", "tfidf", "hybrid"],
        help="Search strategy",
    )
    parser.add_argument(
        "--collection-id",
        required=True,
        help="Collection ID to search in (required: a search targets one collection)",
    )
    args = parser.parse_args()

    golden_queries = load_golden_queries()
    print(f"Loaded {len(golden_queries)} golden queries")
    print(f"API: {args.api_url} | top_k: {args.top_k} | strategy: {args.strategy} | collection: {args.collection_id}")

    results = []
    with httpx.Client(timeout=30.0) as client:
        for entry in golden_queries:
            result = evaluate_query(
                client, args.api_url, entry, args.top_k, args.strategy, args.collection_id
            )
            results.append(result)

    aggregates = compute_aggregates(results)
    print_results(results, aggregates)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "config": {
            "api_url": args.api_url,
            "top_k": args.top_k,
            "strategy": args.strategy,
            "collection_id": args.collection_id,
        },
        "results": results,
        "aggregates": aggregates,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
