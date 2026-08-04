import csv
from pathlib import Path

from src.domain.entities import Document
from src.domain.enums import DocumentType
from src.domain.ports.document_loader_port import DocumentLoaderPort


class CsvLoader(DocumentLoaderPort):
    SUPPORTED_EXTENSIONS = {".csv"}

    async def load(self, file_path: str, document_type: DocumentType) -> Document:
        path = Path(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported extensions: {self.SUPPORTED_EXTENSIONS}"
            )

        try:
            with path.open(encoding="utf-8", newline="") as f:
                sample = f.read(8192)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel
                has_header = csv.Sniffer().has_header(sample)
                f.seek(0)

                if has_header:
                    reader = csv.DictReader(f, dialect=dialect)
                    columns = reader.fieldnames or []
                    rows = []
                    for row in reader:
                        line = ", ".join(
                            f"{col}: {val}" for col, val in row.items()
                        )
                        rows.append(line)
                else:
                    reader = csv.reader(f, dialect=dialect)
                    columns = []
                    rows = []
                    for row in reader:
                        line = " | ".join(
                            val.strip() for val in row if val and val.strip()
                        )
                        if line:
                            rows.append(line)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except csv.Error as e:
            raise csv.Error(f"Failed to parse CSV file '{file_path}': {e}")

        content = "\n".join(rows)

        return Document(
            filename=path.name,
            document_type=document_type,
            content=content,
            metadata={
                "filename": path.name,
                "num_rows": len(rows),
                "columns": list(columns),
                "has_header": has_header,
            },
        )
