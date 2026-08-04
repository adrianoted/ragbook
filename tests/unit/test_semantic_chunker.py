import pytest

from src.domain.entities import Document
from src.domain.enums import DocumentType
from src.infrastructure.chunkers.semantic_chunker import SemanticChunker


def _make_document(content: str, filename: str = "test.txt") -> Document:
    return Document(
        filename=filename,
        document_type=DocumentType.TEXT,
        content=content,
    )


@pytest.mark.asyncio
async def test_markdown_splits_on_headings():
    content = (
        "# Introduction\n\n"
        "Some intro text.\n\n"
        "## Section One\n\n"
        "Content of section one. " * 20 + "\n\n"
        "## Section Two\n\n"
        "Content of section two. " * 20
    )
    chunker = SemanticChunker(max_chunk_size=200, chunk_overlap=20)
    doc = _make_document(content, filename="guide.md")

    chunks = await chunker.chunk(doc)

    assert len(chunks) > 1
    # At least one chunk should contain a section heading indicator
    all_content = " ".join(c.content for c in chunks)
    assert "Section One" in all_content
    assert "Section Two" in all_content


@pytest.mark.asyncio
async def test_python_splits_on_function_boundaries():
    content = (
        "def foo():\n"
        "    return 'foo'\n\n"
        "def bar():\n"
        "    return 'bar'\n\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    chunker = SemanticChunker(max_chunk_size=50, chunk_overlap=5)
    doc = _make_document(content, filename="module.py")

    chunks = await chunker.chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == doc.id


@pytest.mark.asyncio
async def test_plain_text_falls_back_to_default():
    content = "word " * 300
    chunker = SemanticChunker(max_chunk_size=100, chunk_overlap=10)
    doc = _make_document(content, filename="notes.txt")

    chunks = await chunker.chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 100


@pytest.mark.asyncio
async def test_chunk_metadata_contains_filename():
    content = "Some content here."
    chunker = SemanticChunker(max_chunk_size=500, chunk_overlap=50)
    doc = _make_document(content, filename="readme.md")

    chunks = await chunker.chunk(doc)

    for chunk in chunks:
        assert chunk.metadata["filename"] == "readme.md"
        assert chunk.metadata["document_id"] == str(doc.id)


@pytest.mark.asyncio
async def test_chunk_indexes_are_sequential():
    content = "sentence. " * 200
    chunker = SemanticChunker(max_chunk_size=100, chunk_overlap=10)
    doc = _make_document(content, filename="data.txt")

    chunks = await chunker.chunk(doc)

    assert [c.index for c in chunks] == list(range(len(chunks)))


@pytest.mark.asyncio
async def test_js_file_uses_js_splitter():
    content = (
        "function hello() {\n"
        "  return 'hello';\n"
        "}\n\n"
        "function world() {\n"
        "  return 'world';\n"
        "}\n"
    )
    chunker = SemanticChunker(max_chunk_size=40, chunk_overlap=5)
    doc = _make_document(content, filename="app.js")

    chunks = await chunker.chunk(doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.document_id == doc.id
