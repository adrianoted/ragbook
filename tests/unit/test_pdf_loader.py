from unittest.mock import MagicMock, patch

import pytest

from src.domain.enums import DocumentType
from src.infrastructure.loaders.pdf_loader import PdfLoader


class TestPdfLoader:
    @pytest.mark.asyncio
    @patch("src.infrastructure.loaders.pdf_loader.PdfReader")
    async def test_load_pdf(self, mock_pdf_reader_cls):
        page1 = MagicMock()
        page1.extract_text.return_value = "Page one content"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page two content"

        mock_reader = MagicMock()
        mock_reader.pages = [page1, page2]
        mock_pdf_reader_cls.return_value = mock_reader

        loader = PdfLoader()

        with patch("src.infrastructure.loaders.pdf_loader.os.path.exists", return_value=True):
            doc = await loader.load("/fake/path/report.pdf", DocumentType.PDF)

        assert doc.filename == "report.pdf"
        assert doc.document_type == DocumentType.PDF
        assert doc.content == "Page one content\nPage two content"
        assert doc.metadata["filename"] == "report.pdf"
        assert doc.metadata["num_pages"] == 2
        assert doc.metadata["page_range"] == "1-2"

    @pytest.mark.asyncio
    async def test_load_pdf_file_not_found(self):
        loader = PdfLoader()

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            await loader.load("/nonexistent/path/missing.pdf", DocumentType.PDF)
