from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional


class ConfigurationError(Exception):
    pass


_VALID_LLM_PROVIDERS = {"gemini", "ollama"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "development"
    log_level: str = "info"

    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    vector_store: str = "faiss"  # "faiss" | "chroma" | "pinecone" | "qdrant"
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: float = 0.3
    llm_num_ctx: int = 8192
    llm_think: bool = False
    llm_timeout: float = 300.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_strategy: str = "hybrid"
    fusion: Literal["weighted", "rrf"] = "weighted"
    hybrid_vector_weight: float = 0.7
    min_score: float = 0.15
    # Relevance threshold applied to cross-encoder (sigmoid) scores after
    # reranking; used instead of min_score when the reranker is enabled
    rerank_min_score: float = 0.3
    max_results_per_document: int = 2
    chunker: str = "semantic"  # "recursive" | "semantic"
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    lexical_backend: Literal["tfidf", "bm25"] = "tfidf"

    faiss_index_path: str = "data/faiss_indexes"
    chroma_persist_dir: str = "data/chroma_store"
    tfidf_index_path: str = "data/tfidf_indexes"
    bm25_index_path: str = "data/bm25_indexes"
    sqlite_db_path: str = "data/ragbook.db"
    upload_dir: str = "data/uploads"

    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "ragbook"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # CORS
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.debug:
            self.log_level = "DEBUG"
        if self.is_development:
            self._print_startup_info()

    def validate_llm(self) -> None:
        errors = []
        if not self.llm_provider:
            errors.append("LLM_PROVIDER is not set (e.g. 'gemini', 'ollama')")
        elif self.llm_provider not in _VALID_LLM_PROVIDERS:
            errors.append(
                f"LLM_PROVIDER '{self.llm_provider}' is unknown; "
                f"choose from {sorted(_VALID_LLM_PROVIDERS)}"
            )
        if not self.llm_model:
            errors.append("LLM_MODEL is not set (e.g. 'gemini-2.0-flash')")
        if errors:
            raise ConfigurationError(
                "Invalid LLM configuration:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def _print_startup_info(self) -> None:
        print("=" * 50)
        print("🚀 RAGBook")
        print(f"🌍 Environment: {self.environment}")
        print(f"🔧 Debug mode: {self.debug}")
        print(f"📡 Server: {self.host}:{self.port}")
        print(f"📝 Log level: {self.log_level}")
        print(f"🤖 LLM: {self.llm_provider} / {self.llm_model}")
        print(f"🧠 Embedding: {self.embedding_model}")
        print(f"📉 Min_score: {self.min_score}")
        print(f"🗄️  Vector store: {self.vector_store}")
        print(f"🔍 Search strategy: {self.search_strategy}")
        print("=" * 50)


settings = Settings()
