import os

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.domain.entities import Document
from src.domain.enums import DocumentType
from src.domain.ports.document_loader_port import DocumentLoaderPort


class PdfLoader(DocumentLoaderPort):
    async def load(self, file_path: str, document_type: DocumentType) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            reader = PdfReader(file_path)
        except PdfReadError as e:
            raise ValueError(f"Failed to read PDF file '{file_path}': {e}")

        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        num_pages = len(reader.pages)
        filename = os.path.basename(file_path)

        return Document(
            filename=filename,
            document_type=document_type,
            content="\n".join(pages_text),
            metadata={
                "filename": filename,
                "num_pages": num_pages,
                "page_range": f"1-{num_pages}" if num_pages > 1 else "1",
            },
        )
