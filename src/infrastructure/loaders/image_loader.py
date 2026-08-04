import os

from PIL import Image
import pytesseract

from src.domain.entities import Document
from src.domain.enums import DocumentType
from src.domain.ports.document_loader_port import DocumentLoaderPort
from src.domain.ports.llm_port import LlmPort


class ImageLoader(DocumentLoaderPort):
    def __init__(self, strategy: str = "ocr", llm_port: LlmPort | None = None):
        self._strategy = strategy
        self._llm_port = llm_port

    async def load(self, file_path: str, document_type: DocumentType) -> Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            image = Image.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open image file '{file_path}': {e}")

        width, height = image.size
        image_format = image.format
        filename = os.path.basename(file_path)

        try:
            if self._strategy == "llm" and self._llm_port is not None:
                content = await self._llm_port.generate(
                    prompt="Describe the contents of this image in detail.",
                    context=[f"Image: {filename}, dimensions: {width}x{height}, format: {image_format}"],
                )
            else:
                content = pytesseract.image_to_string(image)
        except Exception as e:
            raise ValueError(f"Failed to process image '{file_path}' with strategy '{self._strategy}': {e}")

        return Document(
            filename=filename,
            document_type=document_type,
            content=content.strip(),
            metadata={
                "filename": filename,
                "dimensions": f"{width}x{height}",
                "format": image_format,
                "strategy_used": self._strategy,
            },
        )
