from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.config.settings import Settings, settings
from src.domain.enums import DocumentType
from src.domain.ports.chunker_port import ChunkerPort
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.llm_port import LlmPort
from src.domain.ports.search_port import SearchPort
from src.domain.ports.tfidf_port import TfidfPort
from src.domain.ports.metadata_store_port import MetadataStorePort
from src.domain.ports.reranker_port import RerankerPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.infrastructure.chunkers.recursive_chunker import RecursiveChunker
from src.infrastructure.chunkers.semantic_chunker import SemanticChunker
from src.infrastructure.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from src.infrastructure.loaders import get_loader
from src.infrastructure.persistence.sqlite_metadata import SqliteMetadataStore
from src.infrastructure.tfidf.sklearn_tfidf import SklearnTfidf
from src.application.collection_use_case import CollectionUseCase
from src.application.document_use_case import DocumentUseCase
from src.application.ingest_use_case import IngestUseCase
from src.application.search_use_case import SearchUseCase


@lru_cache
def get_settings() -> Settings:
    return settings


@lru_cache
def get_embedding_port() -> EmbeddingPort:
    return SentenceTransformerEmbedding(model_name=settings.embedding_model)


@lru_cache
def get_vector_store() -> VectorStorePort:
    store_type = settings.vector_store
    if store_type == "faiss":
        from src.infrastructure.vector_stores.faiss_store import FaissVectorStore

        return FaissVectorStore(index_path=settings.faiss_index_path)
    elif store_type == "chroma":
        from src.infrastructure.vector_stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore(persist_directory=settings.chroma_persist_dir)
    elif store_type == "pinecone":
        from src.infrastructure.vector_stores.pinecone_store import PineconeVectorStore

        return PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            # Index dimension must match the embedding model. Known limitation:
            # PineconeVectorStore creates/validates the index in __init__, so
            # this loads the model synchronously inside the lifespan and delays
            # the port opening, defeating the background warm-up. Making it lazy
            # requires moving _ensure_index to first use (see docs 0501).
            dimension=get_embedding_port().dimension(),
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    elif store_type == "qdrant":
        from src.infrastructure.vector_stores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            # Collection dimension must match the embedding model. Passed as a
            # callable: Qdrant only needs it when creating a collection, so the
            # model is not loaded while building the store.
            vector_size=get_embedding_dimension,
        )
    else:
        raise ValueError(f"Unknown vector store: {store_type}")


@lru_cache
def get_tfidf_port() -> TfidfPort:
    backend = settings.lexical_backend
    if backend == "tfidf":
        return SklearnTfidf(index_path=settings.tfidf_index_path)
    elif backend == "bm25":
        from src.infrastructure.tfidf.bm25 import Bm25Lexical

        return Bm25Lexical(index_path=settings.bm25_index_path)
    else:
        raise ValueError(f"Unknown lexical backend: {backend}")


@lru_cache
def get_llm_port() -> LlmPort:
    provider = settings.llm_provider
    if provider == "gemini":
        from src.infrastructure.llm.gemini_llm import GeminiLlm

        return GeminiLlm(settings=settings)
    elif provider == "ollama":
        from src.infrastructure.llm.ollama_llm import OllamaLlm

        return OllamaLlm(settings=settings)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@lru_cache
def get_search_port() -> SearchPort:
    from src.domain.enums import SearchStrategy
    from src.infrastructure.search.hybrid_search import HybridSearch
    from src.infrastructure.search.search_dispatcher import SearchDispatcher
    from src.infrastructure.search.tfidf_search import TfidfSearch
    from src.infrastructure.search.vector_search import VectorSearch

    strategies: dict[SearchStrategy, SearchPort] = {
        SearchStrategy.VECTOR: VectorSearch(vector_store=get_vector_store()),
        SearchStrategy.TFIDF: TfidfSearch(tfidf_port=get_tfidf_port()),
        SearchStrategy.HYBRID: HybridSearch(
            vector_store=get_vector_store(),
            tfidf=get_tfidf_port(),
            vector_weight=settings.hybrid_vector_weight,
            fusion=settings.fusion,
        ),
    }
    default = SearchStrategy(settings.search_strategy)
    return SearchDispatcher(strategies=strategies, default=default)


@lru_cache
def get_metadata_store() -> MetadataStorePort:
    return SqliteMetadataStore(db_path=settings.sqlite_db_path)


@lru_cache
def get_chunker() -> ChunkerPort:
    if settings.chunker == "semantic":
        return SemanticChunker(settings.chunk_size, settings.chunk_overlap)
    return RecursiveChunker(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


@lru_cache
def get_reranker_port() -> RerankerPort:
    # The ctor does NOT load the model — loading is lazy inside _get_model() under a lock,
    # so constructing this unconditionally adds no startup cost or memory overhead.
    from src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker(model_name=settings.reranker_model)


@lru_cache
def get_search_use_case() -> SearchUseCase:
    return SearchUseCase(
        search=get_search_port(),
        embedding=get_embedding_port(),
        llm=get_llm_port(),
        reranker=get_reranker_port(),
        reranker_enabled=settings.reranker_enabled,
        max_results_per_document=settings.max_results_per_document,
    )


@lru_cache
def get_collection_use_case() -> CollectionUseCase:
    return CollectionUseCase(
        vector_store=get_vector_store(),
        tfidf=get_tfidf_port(),
        metadata_store=get_metadata_store(),
        upload_dir=settings.upload_dir,
        embedding_model=settings.embedding_model,
        # The port, not `dimension()`: resolving it here would load the model
        # on the first request to /api/collections, list included.
        embedding=get_embedding_port(),
    )


@lru_cache
def get_document_use_case() -> DocumentUseCase:
    return DocumentUseCase(
        vector_store=get_vector_store(),
        tfidf=get_tfidf_port(),
        metadata_store=get_metadata_store(),
        upload_dir=settings.upload_dir,
    )


@lru_cache
def get_embedding_dimension() -> int:
    return get_embedding_port().dimension()


def get_ingest_use_case(document_type: DocumentType) -> IngestUseCase:
    loader = get_loader(document_type)

    if document_type == DocumentType.CSV:
        from src.infrastructure.chunkers.csv_chunker import CsvChunker

        chunker: ChunkerPort = CsvChunker()
    else:
        chunker = get_chunker()

    return IngestUseCase(
        loader=loader,
        chunker=chunker,
        embedding=get_embedding_port(),
        vector_store=get_vector_store(),
        tfidf=get_tfidf_port(),
        metadata_store=get_metadata_store(),
    )


# Type aliases for FastAPI DI
SettingsDeps = Annotated[Settings, Depends(get_settings)]
MetadataStoreDeps = Annotated[MetadataStorePort, Depends(get_metadata_store)]
SearchUseCaseDeps = Annotated[SearchUseCase, Depends(get_search_use_case)]
CollectionUseCaseDeps = Annotated[CollectionUseCase, Depends(get_collection_use_case)]
DocumentUseCaseDeps = Annotated[DocumentUseCase, Depends(get_document_use_case)]
EmbeddingDimensionDeps = Annotated[int, Depends(get_embedding_dimension)]
