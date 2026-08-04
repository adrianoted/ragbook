# From File to Chunks: Loading and Splitting Documents

Before RAGBook can search anything, it needs to **read** your documents and **break them into pieces** small enough to be indexed and compared. This chapter explains how that happens, step by step.

## 3.1 Supported formats and how they are read

Each file format has its own challenges. RAGBook handles them through dedicated **loaders** — classes that implement the same abstract interface (`DocumentLoaderPort`) but know how to deal with one specific format.

The factory function `get_loader()` in `src/infrastructure/loaders/__init__.py` maps a `DocumentType` to the right loader automatically:

```python
loader = get_loader(DocumentType.PDF)
document = await loader.load("report.pdf", DocumentType.PDF)
```

Every loader returns a `Document` entity with the extracted text, the original filename, and format-specific metadata.

### 3.1.1 Text and Markdown

The simplest case. `TextLoader` reads `.txt` and `.md` files as UTF-8 strings. If the encoding is wrong, it raises a clear error instead of producing garbled text.

Metadata captured: `filename`, `size` (bytes), `encoding`.

### 3.1.2 PDF: the challenges of text extraction

PDFs are containers, not text files. The same PDF might store text as selectable characters, as images of text, or as a mix of both.

`PdfLoader` uses `pypdf` to extract text page by page and concatenate the results:

```python
reader = PdfReader(file_path)
for page in reader.pages:
    text = page.extract_text()
```

This works well for text-based PDFs. Scanned PDFs (images) will produce empty or near-empty text — for those, you'd use the image loader with OCR instead.

Metadata captured: `filename`, `num_pages`, `page_range`.

### 3.1.3 CSV: turning tables into readable text

Tabular data doesn't lend itself to semantic search as-is. A row like `Alice,Engineering,72000` is meaningless without column headers.

#### Automatic delimiter detection

CSV files come in many flavours: some use commas, others semicolons (`;`), tabs, or pipes (`|`). Rather than assuming a single format, `CsvLoader` uses Python's `csv.Sniffer` to **auto-detect the delimiter** from the first 8 KB of the file:

```python
sample = f.read(8192)
dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
```

This makes the loader work out of the box with Anki exports (`;`-separated), European CSVs, TSV files, and standard comma-separated files.

#### Header detection

`csv.Sniffer().has_header()` determines whether the first row contains column names or data:

- **With header**: each row is converted into a human-readable key-value string:
  ```python
  # Input row: {"name": "Alice", "dept": "Engineering", "salary": "72000"}
  # Output:    "name: Alice, dept: Engineering, salary: 72000"
  ```
- **Without header** (e.g. Anki Q&A exports): values are joined with `|`:
  ```python
  # Input row: ["What is X?", "X is ...", "tag::subtag"]
  # Output:    "What is X? | X is ... | tag::subtag"
  ```

All rows are joined with newlines to form the document content.

Metadata captured: `filename`, `num_rows`, `columns` (list of column names, empty if headerless), `has_header`.

### 3.1.4 Images: OCR and LLM description

Images are the trickiest format. `ImageLoader` supports two strategies, chosen at construction time:

- **OCR** (default): uses `pytesseract` + `Pillow` to extract text from the image. Works well for photos of documents, screenshots, scanned pages.
- **LLM**: passes the image's *textual metadata* — filename, dimensions, format — to a language model (`LlmPort`) and asks it to describe the contents.

> ⚠️ **Known limitation.** The LLM strategy does **not** send the actual image pixels. In `ImageLoader.load` it only forwards a string like `"Image: diagram.png, dimensions: 1920x1080, format: PNG"` as context. The model receives no visual input, so it cannot genuinely describe a diagram, chart, or photo. Treat LLM mode as a stub, not a working vision pipeline — OCR is the only strategy that reads real content from the image.

```python
# OCR strategy (default)
loader = ImageLoader(strategy="ocr")

# LLM strategy (requires an LlmPort instance)
loader = ImageLoader(strategy="llm", llm_port=my_llm)
```

If the LLM strategy is selected but no `LlmPort` is provided, the loader falls back to OCR silently.

Metadata captured: `filename`, `dimensions` (e.g. `1920x1080`), `format` (e.g. `PNG`), `strategy_used`. Note that `strategy_used` records the *configured* strategy (`"llm"`) even when the silent OCR fallback above actually ran — so this field can misreport what really happened.

## 3.2 Chunking: why we split documents into pieces

Once a document is loaded, it's typically too long to be useful as a single unit. If you ask "what is the return policy?" and the entire 50-page manual is one block, the search engine can't tell you *where* in the manual the answer is. Worse, feeding 50 pages to an LLM would exceed its context window.

The solution: split the document into **chunks** — smaller passages that each capture a self-contained idea.

### 3.2.1 The trade-off: large chunks vs. small chunks

| | Small chunks (~200 chars) | Large chunks (~1000 chars) |
|---|---|---|
| **Precision** | High — each chunk is focused | Lower — relevant text mixed with noise |
| **Context** | Low — may cut mid-sentence | High — more surrounding context |
| **Search quality** | Better for specific questions | Better for broad questions |

RAGBook defaults to **1000 characters** (`CHUNK_SIZE`) — a middle ground that works well for most use cases. You can tune this via `CHUNK_SIZE` in settings.

### 3.2.2 Overlap: don't lose context at the edges

Imagine cutting a book into pages, but the last sentence of each page continues on the next one. If you only read one page, you miss half a sentence.

**Overlap** solves this by repeating a small portion of text between consecutive chunks. With the default `CHUNK_OVERLAP=200`, the last 200 characters of chunk N also appear at the beginning of chunk N+1. This ensures that sentences spanning a chunk boundary are fully captured in at least one chunk.

### 3.2.3 The default chunker: SemanticChunker

By default (`CHUNKER=semantic`) RAGBook uses `SemanticChunker`, which respects a document's structure instead of blindly counting characters. It picks a splitting strategy from the **file extension**:

| File type | Split strategy |
|---|---|
| `.md` / `.markdown` | on headings — `\n## `, `\n### `, `\n#### `, then blank lines, newlines, spaces |
| `.py` | on Python function / class boundaries (LangChain `Language.PYTHON`) |
| `.js` / `.ts` | on JS/TS function / class boundaries (`Language.JS`) |
| everything else | falls back to the character-based `RecursiveCharacterTextSplitter` |

The idea: a Markdown section or a function is a self-contained unit of meaning, so keeping it whole yields cleaner chunks than an arbitrary character cut.

Size and overlap come from settings — `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=200` — passed at the wiring layer (`get_chunker` in `src/api/dependencies.py`). The class's own constructor defaults (`1500` / `200`) are never used at runtime, because the wiring always passes the settings values.

To use the simpler character-based chunker instead, set `CHUNKER=recursive`.

### 3.2.4 The alternative: RecursiveChunker

`RecursiveChunker` (selected with `CHUNKER=recursive`) wraps LangChain's `RecursiveCharacterTextSplitter` directly. That same splitter is also the **fallback** inside `SemanticChunker` for any non-structured file, so it's worth understanding. It tries to split text at natural boundaries in this priority order:

1. **Double newline** (`\n\n`) — paragraph breaks
2. **Single newline** (`\n`) — line breaks
3. **Space** (` `) — word boundaries
4. **Character** — last resort

So the splitter prefers to break between paragraphs. If a paragraph is still too long, it breaks between lines, then between words. It only splits mid-word as a last resort.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # from CHUNK_SIZE
    chunk_overlap=200,  # from CHUNK_OVERLAP
)
```

> The `RecursiveChunker` constructor defaults are `chunk_size=500` / `chunk_overlap=50`, but the composition root always passes the settings values, so at runtime it too uses 1000 / 200.

Both chunkers produce `Chunk` entities with a progressive index and metadata linking back to the source document:

```python
Chunk(
    document_id=document.id,
    content="...",
    index=0,  # position in the document
    metadata={"document_id": "...", "position": 0, "filename": "report.pdf"},
)
```

### 3.2.5 CSV-specific chunking: one row, one chunk

The `RecursiveCharacterTextSplitter` is designed for prose: it splits by character count and can break across row boundaries. For CSV data — where each row is a self-contained record (e.g. a Q&A pair) — this produces garbled chunks that mix content from different rows.

`CsvChunker` solves this by treating **each row as an independent chunk**:

```python
class CsvChunker(ChunkerPort):
    async def chunk(self, document: Document) -> list[Chunk]:
        rows = [line for line in document.content.split("\n") if line.strip()]
        return [
            Chunk(document_id=document.id, content=row, index=i, ...)
            for i, row in enumerate(rows)
        ]
```

This ensures that a search for "What is active recall?" returns the complete Q&A pair, not a fragment mixed with unrelated rows. The `CsvChunker` is automatically selected when ingesting CSV files — no configuration needed.

## 3.3 Metadata: what we save and why it matters

Every `Document` and `Chunk` carries a `metadata` dictionary. This isn't just bookkeeping — it's essential for the search experience:

- **Source attribution**: when RAGBook shows search results, the metadata tells the user *which file* and *which part* the answer came from.
- **Filtering**: collection-based search uses metadata to restrict results to specific document groups.
- **Debugging**: if results seem wrong, metadata helps trace back to the original content.

The metadata flows through the entire pipeline: loader adds file-level metadata, chunker adds position metadata, and later the search engine passes it through to the final results.

All metadata is persisted to **SQLite** via `SqliteMetadataStore`. During ingest, both the `Document` and its `Chunk` records are saved to the database asynchronously (using `aiosqlite`). The three tables — `collections`, `documents`, and `chunks` — are linked by foreign keys. Deleting a collection removes its documents and their chunks, but this cascade is done **manually**: `delete_collection` issues explicit `DELETE` statements for the chunks, then the documents, then the collection. The foreign keys are declared *without* `ON DELETE CASCADE`, and `PRAGMA foreign_keys` is not enabled, so SQLite does not cascade on its own — the code does it by hand. This persistence layer ensures that metadata survives application restarts and enables collection management features like listing documents or retrieving chunk details by ID.

---

**Next**: in [Chapter 4](04-embeddings-and-vector-store.md), we'll see how these chunks are transformed into numerical vectors that enable semantic search.
