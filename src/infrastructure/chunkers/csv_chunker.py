from src.domain.entities import Chunk, Document
from src.domain.ports.chunker_port import ChunkerPort


class CsvChunker(ChunkerPort):
    """Treats each row of a CSV document as a separate chunk.

    Unlike RecursiveChunker, this preserves the semantic integrity of
    each row (e.g. a Q&A pair) instead of splitting by character count.
    """

    async def chunk(self, document: Document) -> list[Chunk]:
        rows = [line for line in document.content.split("\n") if line.strip()]
        return [
            Chunk(
                document_id=document.id,
                content=row,
                index=index,
                metadata={
                    "document_id": str(document.id),
                    "position": index,
                    "filename": document.filename,
                },
            )
            for index, row in enumerate(rows)
        ]
