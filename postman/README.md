# Postman Collection — RAGBook

This folder contains the Postman collection for testing RAGBook APIs,
organized as **modular fragments** assembled by a build script.

## Structure

```
postman/
├── src/                              # Source fragments (edit these)
│   ├── _meta.json                    # Collection info + shared variables
│   ├── health.json                   # Health API (server smoke test)
│   ├── e2e.json                      # E2E Flow (sequential happy path)
│   ├── collections.json              # Collections API (CRUD)
│   ├── documents.json                # Documents API (list, delete)
│   ├── ingest.json                   # Ingest API (file upload)
│   ├── search.json                   # Search API (LLM + Raw)
│   └── cleanup.json                  # Cleanup — deliberately last, see below
├── build_collection.py               # Script that assembles the final JSON
├── run_test_server.sh                # Starts the server with isolated test data
└── ragbook.postman_collection.json # Generated output (import this in Postman)
```

The files uploaded by the ingestion requests are **not** in this folder — they live in
`resources/` at the repo root, shared with the retrieval evaluation:

```
resources/
├── default-file-streaming.md         # Test file for ingestion (E2E)
├── default-file-memory.md            # Test file for ingestion (alternative)
└── products-100.csv                  # CSV sample (exercises the CsvChunker)
```

The two `.md` files are also the corpus expected by `evaluation/golden_queries.json`,
which is why they are kept outside `postman/`.

## How it works

The files in `src/` are independent JSON fragments:

- **`_meta.json`** — contains `info` (name, description, schema) and `variable` (collection variables such as `base_url`, `collection_id`, etc.)
- **Every other file** — represents a top-level folder in the Postman collection, with its requests and related test scripts

The order of `ITEM_FRAGMENTS` in `build_collection.py` is not cosmetic: running the whole
collection executes the folders top to bottom, so each one must leave the next something
to work with. `cleanup.json` is last because it deletes the collection the others still
need. Keep that in mind before reordering the list.

The `build_collection.py` script reads `_meta.json`, then loads the fragments in the
order defined inside it, and produces the `ragbook.postman_collection.json` file.

## Workflow

### Editing an API

1. Open the corresponding fragment in `src/` (e.g. `src/search.json` for Search APIs)
2. Edit the request, body, test scripts, etc.
3. Rebuild the collection:

```bash
python postman/build_collection.py
```

4. Import the generated file in Postman (see below)

### Adding a new API

1. If the new API belongs to an existing group, add it to the corresponding file in `src/`
2. If a new group is needed, create a new JSON file in `src/` with this structure:

```json
{
  "name": "Folder Name",
  "description": "Folder description",
  "item": [
    {
      "name": "Request Name",
      "request": { ... }
    }
  ]
}
```

3. Add the filename to the `ITEM_FRAGMENTS` list in `build_collection.py`
4. Rebuild: `python postman/build_collection.py`

### Editing collection variables

Edit `src/_meta.json` and rebuild.

## Importing in Postman

1. Open Postman
2. Click **Import** (top left)
3. Select `postman/ragbook.postman_collection.json`
4. If the collection already exists, Postman will ask whether to replace it — confirm with **Replace**

## File upload for Ingestion (important)

File upload requests (`Ingest Document`) require selecting the file **manually** in
Postman. This is because Postman resolves file paths relative to its **Working Directory**,
not the project folder.

### Option 1 — Set the Working Directory (recommended)

1. In Postman, go to **Settings** (gear icon) → **General**
2. Scroll to **Working directory** — the current path is displayed there
3. Set it to the project's `resources/` folder, for example:
   ```
   /path/to/ragbook/resources
   ```
4. The `src` field in formdata requests (`default-file-streaming.md`) will now be
   resolved automatically

> **Tip:** not sure where the Working Directory is right now? Open Postman Settings →
> General and look for the "Working directory" field — it shows the full current path.
> You can also copy the test files (`default-file-streaming.md`, `default-file-memory.md`)
> into whatever folder is already set as the Working Directory, instead of changing the
> setting. Postman will find them either way.

> With the Working Directory configured, both manual execution and the Collection Runner
> will find the file without having to select it each time.

> **Symptom when it is not configured:** the ingest request answers
> `400 {"detail": "Unsupported file type: "}` — note the empty extension. Postman could
> not resolve the file, so it sent the field empty and the server saw a nameless upload.

### Option 2 — Select the file manually

If you don't want to change the Working Directory:

1. Open the `Ingest Document` request (both the E2E and the standalone one)
2. In the **Body** tab, click **Select Files** next to the `file` field
3. Navigate to the `resources/` folder and select `default-file-streaming.md`

> **Note:** this selection is lost when you reimport the collection. You will need to
> repeat this step every time you import an updated version.

### Why can't I use an absolute path?

Postman **does not support absolute paths** in the `src` field of formdata files.
The `src` value is always relative to the Working Directory set in Settings.
If `src` contains only a filename (e.g. `default-file-streaming.md`), Postman looks
for it in the Working Directory. If the Working Directory is not set or does not contain
that file, the file will not be found and the request will fail.

## Running E2E tests

### Using the test server (recommended)

To avoid mixing test data with development data, start the server with isolated storage:

```bash
bash postman/run_test_server.sh
```

This overrides the data paths so that all data (SQLite, FAISS, TF-IDF, uploads) is stored
under `data/test/` instead of `data/`. Your development database is never touched.

The test server uses the same `.env` configuration (API keys, LLM provider, etc.) but
redirects all storage to the isolated directory.

### Running the whole collection

Running everything top-to-bottom used to cascade into 404s: the cleanup wipe ran in the
middle and removed the E2E collection, then `Collections → Delete Collection` removed the
next one, and every request after that hit a `collection_id` that no longer existed.

Two structural changes fixed it, so a full run now works with no configuration at all:

- **Cleanup runs last.** It is its own top-level folder at the bottom of the collection,
  and inside it the two targeted deletes come before the prefix wipe — the wipe used to
  destroy the very resources they were about to delete.
- **The `Collections` folder owns its resource.** Its create/delete pair works on
  `standalone_collection_id`, a separate variable, so it no longer clobbers or deletes the
  `collection_id` that Documents, Ingest and Search still depend on.

The resulting chain: E2E creates a collection and a document → Collections exercises its
own throwaway collection → Documents lists, then deletes the E2E document → Ingest puts it
back → Search queries it → Cleanup removes everything.

#### `run_mode` — the short run

`run_mode` is an optional shortcut, not a requirement:

| `run_mode` | behaviour |
|---|---|
| `manual` (default) | nothing is skipped — the full chain above runs, and single requests behave normally when fired by hand |
| `full` | skips the standalone folders and the two targeted cleanup deletes, leaving just **Health → E2E Flow → prefix wipe** |

Use `full` when you only want the smoke test. Note that it relies on
`pm.execution.skipRequest()`, available in reasonably recent Postman versions; on older
ones the requests simply run instead of being skipped, which is still fine now that the
ordering itself is sound.

### Running the tests in Postman

1. Start the test server (see above) or the regular server (`python run.py`)
2. Run **Health → Health Check** first: the `embedding model ready` test must pass.
   If `embedding_status` is `warming`, the model is still loading (on first boot it is
   downloaded from Hugging Face) — wait and retry, otherwise ingest and search will hang
   or fail.
3. Set the Working Directory as described above
4. In Postman, select the **E2E Flow** folder (or the whole collection — see above)
5. Click **Run** (Collection Runner)
6. Tests will run sequentially:
   - Create Collection → Ingest Document → List Documents → Search (LLM) → Search Raw
7. Variables (`collection_id`, `document_id`, etc.) are saved automatically between steps

### Cleanup

The **Cleanup** folder is the last one in the collection and contains three requests,
in this order:

- **Delete Document** — deletes the single document from the last E2E run
- **Delete Collection** — deletes the single collection from the last E2E run
- **Delete Test Collections (prefix wipe)** — deletes the collections whose name starts
  with the `test_prefix` variable (default `[postman_test]`), together with their
  documents. Collections without that prefix are left untouched and counted in the
  console log. The pre-request script does the deleting; the request itself verifies
  that no prefixed collection remains.

The wipe comes last on purpose: run first, it deleted the resources the two targeted
deletes were about to remove, and they answered 404.

If an E2E run fails midway (e.g. ingestion step), run **Delete Test Collections** to
clean up any orphaned resources before retrying.

### Two safeguards against wiping real data

The cleanup request used to delete *every* collection on `base_url`, and the test server
listened on the same port as the dev server — so pointing the collection at dev and
running cleanup destroyed real data. Two guards now prevent that:

1. **Port separation** — `run_test_server.sh` listens on **8010** (override with
   `TEST_PORT`), the dev server on 8000. The collection's `base_url` defaults to
   `http://localhost:8010`, so hitting dev data requires editing the variable on purpose.
2. **Prefix-scoped deletion** — the wipe only touches collections whose name starts with
   `test_prefix`. The E2E flow already creates its collection as
   `[postman_test] Streaming Docs`, so it cleans up after itself. Emptying `test_prefix`
   makes the request delete nothing at all.

Keep the `[postman_test]` prefix on any collection you create by hand for testing —
that is what makes it eligible for cleanup.
