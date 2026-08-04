import asyncio
import logging
import os
from contextlib import asynccontextmanager

# Disable Gradio telemetry: its analytics thread is non-daemon and can hang
# shutdown, leaving an orphan process holding the terminal on Ctrl+C.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_embedding_dimension, get_metadata_store, get_vector_store
from src.api.routers.ingest_router import router as ingest_router
from src.api.routers.search_router import router as search_router
from src.api.routers.collection_router import router as collection_router
from src.api.routers.document_router import router as document_router
from src.config.settings import settings
from src.ui.constants import GRADIO_MOUNT_PATH
from src.ui.gradio_app import create_gradio_app

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Silence noisy third-party loggers ("filelock" spams DEBUG lines during
# Hugging Face downloads, burying the progress bar)
for _name in ("httpcore", "httpx", "aiosqlite", "gradio", "urllib3", "matplotlib", "filelock"):
    logging.getLogger(_name).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# Embedding model state, exposed by /api/health: "warming" until the model is
# downloaded/loaded, then "ready" (or "error" if loading fails).
embedding_status: dict[str, str] = {"status": "warming"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_llm()
    metadata_store = get_metadata_store()
    vector_store = get_vector_store()
    await metadata_store.init_db()
    await vector_store.load()
    # The embedding model load (on first boot: a multi-GB download from
    # Hugging Face) runs in the background: the port opens immediately and
    # /api/health reports the state instead of leaving the user in the dark.
    warmup_task = asyncio.create_task(_warmup_embedding_model(metadata_store))
    try:
        yield
    finally:
        warmup_task.cancel()


async def _warmup_embedding_model(metadata_store) -> None:
    logger.info(
        "⏳ Loading embedding model %r... On first boot the model is downloaded "
        "from Hugging Face (large models can take several GB and many minutes). "
        "The API is already reachable: /api/health returns "
        "embedding_status=\"warming\" until the model is ready.",
        settings.embedding_model,
    )
    try:
        dimension = await asyncio.to_thread(get_embedding_dimension)
    except asyncio.CancelledError:
        raise
    except Exception:
        embedding_status["status"] = "error"
        logger.exception(
            "❌ Failed to load embedding model %r", settings.embedding_model
        )
        return
    embedding_status["status"] = "ready"
    logger.info(
        "✅ Embedding model %r ready (dim %d)", settings.embedding_model, dimension
    )
    await _warn_incompatible_collections(metadata_store, dimension)


async def _warn_incompatible_collections(metadata_store, current_dimension: int) -> None:
    current_model = settings.embedding_model
    for collection in await metadata_store.list_collections():
        if not collection.embedding_model or not collection.embedding_dimension:
            continue
        if (
            collection.embedding_model != current_model
            or collection.embedding_dimension != current_dimension
        ):
            logger.warning(
                "Startup: collection %s ('%s') was created with model %r (dim %d), "
                "server uses %r (dim %d) — re-ingest required",
                collection.id,
                collection.name,
                collection.embedding_model,
                collection.embedding_dimension,
                current_model,
                current_dimension,
            )


app = FastAPI(title="RAGBook API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(collection_router)
app.include_router(document_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "embedding_status": embedding_status["status"]}


# Mount Gradio UI
api_base_url = f"http://{settings.host}:{settings.port}/api"
gradio_app = create_gradio_app(api_base_url=api_base_url)
app = gr.mount_gradio_app(app, gradio_app, path=GRADIO_MOUNT_PATH)
