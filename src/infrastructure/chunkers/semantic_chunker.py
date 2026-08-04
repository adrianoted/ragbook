from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.domain.entities import Chunk, Document
from src.domain.ports.chunker_port import ChunkerPort


class SemanticChunker(ChunkerPort):
    """Chunker that respects document structure.

    - .md / .markdown → splits on headings (##, ###, ####)
    - .py             → splits on Python function/class boundaries
    - .js / .ts       → splits on JS/TS function/class boundaries
    - everything else → standard RecursiveCharacterTextSplitter
    """

    def __init__(self, max_chunk_size: int = 1500, chunk_overlap: int = 200):
        self._max_size = max_chunk_size
        self._overlap = chunk_overlap

        self._markdown_splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._js_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS,
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def chunk(self, document: Document) -> list[Chunk]:
        filename = document.filename.lower()

        if filename.endswith((".md", ".markdown")):
            splitter = self._markdown_splitter
        elif filename.endswith(".py"):
            splitter = self._python_splitter
        elif filename.endswith((".js", ".ts")):
            splitter = self._js_splitter
        else:
            splitter = self._default_splitter

        texts = splitter.split_text(document.content)
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
