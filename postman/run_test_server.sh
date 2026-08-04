#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Start the RAGBook server with an isolated test database.
#
# All data (SQLite, FAISS indexes, TF-IDF indexes, uploads)
# is stored under data/test/ instead of data/, so development
# data is never affected by Postman E2E runs.
#
# It also listens on a different port (8010) than the dev server
# (8000), so the Postman collection cannot hit dev data by mistake.
#
# Usage:
#   bash postman/run_test_server.sh        (from project root)
#   ./postman/run_test_server.sh           (if executable)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# Ensure we run from the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Override data paths → isolated test directory.
# Every path setting that writes to data/ must be listed here, whatever
# VECTOR_STORE / LEXICAL_BACKEND the .env selects.
export SQLITE_DB_PATH="data/test/ragbook_test.db"
export FAISS_INDEX_PATH="data/test/faiss_indexes_test"
export CHROMA_PERSIST_DIR="data/test/chroma_store_test"
export TFIDF_INDEX_PATH="data/test/tfidf_indexes_test"
export BM25_INDEX_PATH="data/test/bm25_indexes_test"
export UPLOAD_DIR="data/test/uploads"

# Distinct port → the collection's base_url points here, not at dev
export PORT="${TEST_PORT:-8010}"

echo "─── RAGBook Test Server ───"
echo "  Port:    $PORT"
echo "  SQLite:  $SQLITE_DB_PATH"
echo "  FAISS:   $FAISS_INDEX_PATH"
echo "  Chroma:  $CHROMA_PERSIST_DIR"
echo "  TF-IDF:  $TFIDF_INDEX_PATH"
echo "  BM25:    $BM25_INDEX_PATH"
echo "  Uploads: $UPLOAD_DIR"
echo "──────────────────────────────"

python run.py
