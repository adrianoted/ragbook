import pytest

from src.domain.enums import DocumentType
from src.infrastructure.loaders.csv_loader import CsvLoader


@pytest.fixture
def loader():
    return CsvLoader()


@pytest.mark.asyncio
async def test_load_csv(loader, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25\nCharlie,40\n", encoding="utf-8")

    doc = await loader.load(str(csv_file), DocumentType.CSV)

    assert doc.content == "name: Alice, age: 30\nname: Bob, age: 25\nname: Charlie, age: 40"


@pytest.mark.asyncio
async def test_load_csv_metadata(loader, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25\nCharlie,40\n", encoding="utf-8")

    doc = await loader.load(str(csv_file), DocumentType.CSV)

    assert doc.metadata["num_rows"] == 3
    assert doc.metadata["columns"] == ["name", "age"]
    assert doc.metadata["filename"] == "data.csv"


@pytest.mark.asyncio
async def test_load_empty_csv(loader, tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("name,age\n", encoding="utf-8")

    doc = await loader.load(str(csv_file), DocumentType.CSV)

    assert doc.content == ""
    assert doc.metadata["num_rows"] == 0
