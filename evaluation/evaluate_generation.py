"""Evaluation framework for generation faithfulness using an LLM-as-judge.

Usage:
    python -m evaluation.evaluate_generation --collection-id <ID> [--api-url URL] [--top-k N] [--strategy STRATEGY] [--judge-model MODEL]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from evaluation.generation_metrics import (
    build_judge_prompt,
    compute_faithfulness,
    compute_negative_refusal,
    extract_message_text,
    is_fallback_answer,
    parse_judge_response,
)

GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_RETRY_DELAY_S = 5.0


def load_golden_queries() -> list[dict]:
    with open(GOLDEN_QUERIES_PATH) as f:
        return json.load(f)


def evaluate_query(
    client: httpx.Client,
    api_url: str,
    query_entry: dict,
    top_k: int,
    strategy: str | None,
    collection_id: str,
    judge: ChatGoogleGenerativeAI,
) -> dict:
    payload: dict = {
        "query": query_entry["query"],
        "top_k": top_k,
        "collection_id": collection_id,
    }
    if strategy:
        payload["strategy"] = strategy

    start = time.perf_counter()
    response = client.post(f"{api_url}/api/search", json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    if response.is_error:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(f"HTTP {response.status_code} from /api/search: {detail}")

    data = response.json()
    answer = data["answer"]
    sources = data.get("sources", [])
    chunk_contents = [s["chunk_content"] for s in sources]

    base: dict = {
        "query": query_entry["query"],
        "latency_ms": round(latency_ms, 1),
    }

    if query_entry.get("negative"):
        base["type"] = "negative"
        base["negative_refusal"] = compute_negative_refusal(answer)
        return base

    base["type"] = "positive"

    if is_fallback_answer(answer):
        base["faithfulness"] = 1.0
        base["fallback"] = True
        base["num_claims_supported"] = 0
        base["num_claims_total"] = 0
        return base

    prompt = build_judge_prompt(query_entry["query"], answer, chunk_contents)
    claims = None
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = extract_message_text(judge.invoke(prompt).content)
            claims = parse_judge_response(raw)
            break
        except Exception as exc:  # parse errors, rate limits, network failures
            last_error = exc
            if attempt == 0:
                time.sleep(JUDGE_RETRY_DELAY_S)

    if claims is None:
        base["judge_error"] = True
        base["judge_error_reason"] = f"{type(last_error).__name__}: {last_error}"
        base["faithfulness"] = None
        base["num_claims_supported"] = None
        base["num_claims_total"] = None
        return base

    if not claims:
        # No factual claim to verify: faithfulness is undefined, not perfect.
        # Scoring it 1.0 would make the metric rise when retrieval degrades and
        # the LLM answers "the documents do not contain this information".
        base["no_claims"] = True
        base["faithfulness"] = None
        base["num_claims_supported"] = 0
        base["num_claims_total"] = 0
        return base

    supported = sum(1 for c in claims if c["supported"])
    base["faithfulness"] = round(compute_faithfulness(claims), 4)
    base["num_claims_supported"] = supported
    base["num_claims_total"] = len(claims)
    return base


def compute_aggregates(results: list[dict]) -> dict:
    if not results:
        return {}

    positive = [r for r in results if r.get("type") == "positive"]
    negative = [r for r in results if r.get("type") == "negative"]

    aggs: dict = {}

    judged = [
        r
        for r in positive
        if not r.get("fallback")
        and not r.get("judge_error")
        and not r.get("no_claims")
    ]
    if judged:
        aggs["avg_faithfulness"] = round(
            sum(r["faithfulness"] for r in judged) / len(judged), 4
        )
    aggs["faithfulness_n"] = len(judged)

    if negative:
        aggs["negative_refusal_rate"] = round(
            sum(r["negative_refusal"] for r in negative) / len(negative), 4
        )

    aggs["fallback_count"] = sum(1 for r in positive if r.get("fallback"))
    aggs["no_claims_count"] = sum(1 for r in positive if r.get("no_claims"))
    aggs["judge_error_count"] = sum(1 for r in positive if r.get("judge_error"))
    aggs["avg_latency_ms"] = round(
        sum(r["latency_ms"] for r in results) / len(results), 1
    )

    return aggs


def print_results(results: list[dict], aggregates: dict) -> None:
    print("\n" + "=" * 80)
    print("GENERATION FAITHFULNESS EVALUATION RESULTS")
    print("=" * 80)

    for r in results:
        print(f"\nQuery: {r['query']}")
        if r["type"] == "negative":
            print("  [NEGATIVE QUERY]")
            print(f"  negative_refusal: {r['negative_refusal']:.4f}")
            print(f"  latency_ms:       {r['latency_ms']:.1f}")
        elif r.get("fallback"):
            print("  [FALLBACK — no results]")
            print(f"  faithfulness:     1.0000 (no claims)")
            print(f"  latency_ms:       {r['latency_ms']:.1f}")
        elif r.get("no_claims"):
            print("  [NO CLAIMS — excluded from avg_faithfulness]")
            print(f"  latency_ms:       {r['latency_ms']:.1f}")
        elif r.get("judge_error"):
            print("  [JUDGE ERROR]")
            print(f"  reason:           {r.get('judge_error_reason', 'unknown')}")
            print(f"  latency_ms:       {r['latency_ms']:.1f}")
        else:
            print(f"  faithfulness:     {r['faithfulness']:.4f}")
            print(f"  claims:           {r['num_claims_supported']}/{r['num_claims_total']}")
            print(f"  latency_ms:       {r['latency_ms']:.1f}")

    print("\n" + "-" * 80)
    print("AGGREGATES")
    print("-" * 80)
    for key, value in aggregates.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation faithfulness")
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
        help="Collection ID to search in",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-2.5-flash",
        help="Gemini model to use as faithfulness judge (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help=(
            "HTTP timeout in seconds for /api/search; must cover full LLM "
            "generation, which is not streamed here (default: 300)"
        ),
    )
    args = parser.parse_args()

    # The key may live in the project .env (as for the app) rather than the shell.
    load_dotenv(Path(__file__).parent.parent / ".env")

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is set neither in the environment nor in .env")
        print("Set it before running: export GOOGLE_API_KEY=<your-key>")
        sys.exit(1)

    golden_queries = load_golden_queries()
    print(f"Loaded {len(golden_queries)} golden queries")
    print(
        f"API: {args.api_url} | top_k: {args.top_k} | strategy: {args.strategy} "
        f"| collection: {args.collection_id} | judge: {args.judge_model} "
        f"| timeout: {args.timeout}s"
    )

    judge = ChatGoogleGenerativeAI(model=args.judge_model, temperature=0)

    results = []
    with httpx.Client(timeout=args.timeout) as client:
        for entry in golden_queries:
            result = evaluate_query(
                client,
                args.api_url,
                entry,
                args.top_k,
                args.strategy,
                args.collection_id,
                judge,
            )
            results.append(result)

    aggregates = compute_aggregates(results)
    print_results(results, aggregates)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"gen_{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "config": {
            "api_url": args.api_url,
            "top_k": args.top_k,
            "strategy": args.strategy,
            "collection_id": args.collection_id,
            "judge_model": args.judge_model,
            "timeout": args.timeout,
        },
        "results": results,
        "aggregates": aggregates,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
