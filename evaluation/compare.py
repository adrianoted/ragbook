"""Compare two evaluation result files side by side.

Usage:
    python -m evaluation.compare results/before.json results/after.json
"""

import argparse
import json
import sys
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

HIGHER_IS_BETTER = {
    "avg_score",
    "recall",
    "keyword_hit_rate",
    "unique_docs_ratio",
    "avg_recall",
    "avg_keyword_hit_rate",
    "avg_unique_docs_ratio",
    "mrr",
    "avg_mrr",
    "negative_pass",
    "negative_pass_rate",
}
LOWER_IS_BETTER = {"latency_ms", "avg_latency_ms"}

POSITIVE_METRICS = ["avg_score", "recall", "keyword_hit_rate", "unique_docs_ratio", "mrr", "latency_ms"]
NEGATIVE_METRICS = ["negative_pass", "latency_ms"]


def colorize(before: float, after: float, metric: str) -> str:
    diff = after - before
    if abs(diff) < 0.0001:
        return f"{after:.4f}"

    if metric in HIGHER_IS_BETTER:
        color = GREEN if diff > 0 else RED
    elif metric in LOWER_IS_BETTER:
        color = GREEN if diff < 0 else RED
    else:
        color = GREEN if diff > 0 else RED

    sign = "+" if diff > 0 else ""
    return f"{color}{after:.4f} ({sign}{diff:.4f}){RESET}"


def compare(before_path: str, after_path: str) -> None:
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f"\n{BOLD}Comparison: {Path(before_path).name} vs {Path(after_path).name}{RESET}")
    print("=" * 90)

    # Per-query comparison
    before_by_query = {r["query"]: r for r in before["results"]}
    after_by_query = {r["query"]: r for r in after["results"]}

    common_queries = set(before_by_query) & set(after_by_query)

    for query in sorted(common_queries):
        b = before_by_query[query]
        a = after_by_query[query]
        is_negative = b.get("negative") or a.get("negative")
        metrics = NEGATIVE_METRICS if is_negative else POSITIVE_METRICS
        label = f" {BOLD}[negative]{RESET}" if is_negative else ""
        print(f"\n{BOLD}Query:{RESET} {query}{label}")
        print(f"  {'Metric':<22} {'Before':>10}  {'After':>30}")
        print(f"  {'-' * 65}")
        for m in metrics:
            bv = b.get(m, 0.0)
            av = a.get(m, 0.0)
            after_str = colorize(bv, av, m)
            print(f"  {m:<22} {bv:>10.4f}  {after_str:>30}")

    # Aggregates comparison
    print(f"\n{'=' * 90}")
    print(f"{BOLD}AGGREGATES{RESET}")
    print(f"{'=' * 90}")
    print(f"  {'Metric':<25} {'Before':>10}  {'After':>30}")
    print(f"  {'-' * 68}")

    ba = before.get("aggregates", {})
    aa = after.get("aggregates", {})
    for m in sorted(set(ba) | set(aa)):
        bv = ba.get(m, 0.0)
        av = aa.get(m, 0.0)
        after_str = colorize(bv, av, m)
        print(f"  {m:<25} {bv:>10.4f}  {after_str:>30}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation result files")
    parser.add_argument("before", help="Path to the 'before' results JSON")
    parser.add_argument("after", help="Path to the 'after' results JSON")
    args = parser.parse_args()

    if not Path(args.before).exists():
        print(f"Error: {args.before} not found", file=sys.stderr)
        sys.exit(1)
    if not Path(args.after).exists():
        print(f"Error: {args.after} not found", file=sys.stderr)
        sys.exit(1)

    compare(args.before, args.after)


if __name__ == "__main__":
    main()
