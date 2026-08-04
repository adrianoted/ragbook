from enum import Enum


class DocumentType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    CSV = "csv"
    IMAGE = "image"


class SearchStrategy(str, Enum):
    VECTOR = "vector"
    TFIDF = "tfidf"
    HYBRID = "hybrid"
