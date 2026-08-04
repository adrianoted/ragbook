import pytest

from src.domain.enums import DocumentType
from src.infrastructure.loaders.text_loader import TextLoader


@pytest.fixture
def loader():
    return TextLoader()


@pytest.mark.asyncio
async def test_load_txt_file(loader, tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("Hello, world!", encoding="utf-8")

    document = await loader.load(str(file), DocumentType.TEXT)

    assert document.content == "Hello, world!"


@pytest.mark.asyncio
async def test_load_md_file(loader, tmp_path):
    file = tmp_path / "sample.md"
    file.write_text("# Heading\n\nSome markdown content.", encoding="utf-8")

    document = await loader.load(str(file), DocumentType.TEXT)

    assert document.content == "# Heading\n\nSome markdown content."


@pytest.mark.asyncio
async def test_load_file_not_found(loader):
    with pytest.raises(FileNotFoundError):
        await loader.load("/nonexistent/path/file.txt", DocumentType.TEXT)


@pytest.mark.asyncio
async def test_metadata_contains_filename_and_size(loader, tmp_path):
    file = tmp_path / "meta.txt"
    content = "metadata test"
    file.write_text(content, encoding="utf-8")

    document = await loader.load(str(file), DocumentType.TEXT)

    assert "filename" in document.metadata
    assert "size" in document.metadata
    assert "encoding" in document.metadata
    assert document.metadata["filename"] == "meta.txt"
    assert document.metadata["size"] == len(content.encode("utf-8"))
