import os
from pathlib import Path

from src.domain.entities import Document
from src.domain.enums import DocumentType
from src.domain.ports.document_loader_port import DocumentLoaderPort


class TextLoader(DocumentLoaderPort):
    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    async def load(self, file_path: str, document_type: DocumentType) -> Document:
        path = Path(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported extensions: {self.SUPPORTED_EXTENSIONS}"
            )

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except UnicodeDecodeError:
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                f"Failed to decode file '{file_path}' as UTF-8. "
                f"Ensure the file is encoded in UTF-8.",
            )

        file_size = os.path.getsize(file_path)

        return Document(
            filename=path.name,
            document_type=document_type,
            content=content,
            metadata={
                "filename": path.name,
                "size": file_size,
                "encoding": "utf-8",
            },
        )
