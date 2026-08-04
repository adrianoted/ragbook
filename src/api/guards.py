import logging

from fastapi import HTTPException

from src.domain.entities import Collection

logger = logging.getLogger(__name__)


def check_model_compatibility(
    collection: Collection,
    current_model: str,
    current_dimension: int,
) -> None:
    if not collection.embedding_model or not collection.embedding_dimension:
        logger.warning(
            "Collection %s ('%s') has no model metadata (legacy) — skipping compatibility check",
            collection.id,
            collection.name,
        )
        return
    if (
        collection.embedding_model != current_model
        or collection.embedding_dimension != current_dimension
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Collection created with model {collection.embedding_model!r} "
                f"(dim {collection.embedding_dimension}), server configured with "
                f"{current_model!r} (dim {current_dimension}) — re-ingest required"
            ),
        )
