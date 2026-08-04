from src.domain.enums import DocumentType
from src.domain.ports.document_loader_port import DocumentLoaderPort


def get_loader(document_type: DocumentType) -> DocumentLoaderPort:
    """Map DocumentType to the correct loader."""
    if document_type == DocumentType.TEXT:
        from .text_loader import TextLoader
        return TextLoader()
    elif document_type == DocumentType.PDF:
        from .pdf_loader import PdfLoader
        return PdfLoader()
    elif document_type == DocumentType.CSV:
        from .csv_loader import CsvLoader
        return CsvLoader()
    elif document_type == DocumentType.IMAGE:
        from .image_loader import ImageLoader
        return ImageLoader()
    else:
        raise ValueError(f"Unknown document type: {document_type}")
