"""
Build the Postman collection from individual JSON fragments.

Usage:
    python postman/build_collection.py

Reads fragments from postman/src/ and produces
postman/ragbook.postman_collection.json
"""

import json
from pathlib import Path

# Ordered list of fragment files (each is a top-level folder in the collection).
#
# The order is not cosmetic: running the whole collection executes the folders
# top to bottom, so every folder must leave the next one something to work with.
# Cleanup goes last because it deletes the collection the others still need.
ITEM_FRAGMENTS = [
    "health.json",
    "e2e.json",
    "collections.json",
    "documents.json",
    "ingest.json",
    "search.json",
    "config.json",
    "cleanup.json",
]

def build() -> dict:
    src_dir = Path(__file__).parent / "src"

    # Load collection metadata (info + variables)
    with open(src_dir / "_meta.json", encoding="utf-8") as f:
        collection = json.load(f)

    # Load each item fragment in order
    collection["item"] = []
    for filename in ITEM_FRAGMENTS:
        with open(src_dir / filename, encoding="utf-8") as f:
            collection["item"].append(json.load(f))

    return collection


def main() -> None:
    collection = build()

    output_path = Path(__file__).parent / "ragbook.postman_collection.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)

    print(f"Collection built → {output_path}")
    print(f"  {len(collection['item'])} top-level folders")
    total_requests = sum(
        len(folder.get("item", []))
        for folder in collection["item"]
    )
    print(f"  ~{total_requests} direct items (some contain sub-folders)")


if __name__ == "__main__":
    main()
