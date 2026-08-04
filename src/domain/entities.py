from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.domain.enums import DocumentType, SearchStrategy


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    document_type: DocumentType
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    collection_id: UUID | None = None


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    metadata: dict = Field(default_factory=dict)
    index: int
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    source: str = "vector"


class Collection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class SearchQuery(BaseModel):
    query: str
    collection_id: UUID | None = None
    top_k: int = 5
    strategy: SearchStrategy | None = None
    min_score: float | None = None
