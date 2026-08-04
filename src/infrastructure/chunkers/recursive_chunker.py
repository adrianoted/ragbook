from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.domain.entities import Chunk, Document
from src.domain.ports.chunker_port import ChunkerPort


class RecursiveChunker(ChunkerPort):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def chunk(self, document: Document) -> list[Chunk]:
        texts = self._splitter.split_text(document.content)
        return [
            Chunk(
                document_id=document.id,
                content=text,
                index=index,
                metadata={
                    "document_id": str(document.id),
                    "position": index,
                    "filename": document.filename,
                },
            )
            for index, text in enumerate(texts)
        ]
