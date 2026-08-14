from fastapi import APIRouter
from pydantic import BaseModel

from src.api.dependencies import SettingsDeps

router = APIRouter(prefix="/api/config", tags=["config"])


class RangeSpec(BaseModel):
    min: float
    max: float
    step: float


class TuningDefaults(BaseModel):
    min_score: float
    fusion: str
    hybrid_vector_weight: float
    max_results_per_document: int
    reranker_enabled: bool
    llm_temperature: float
    llm_think: bool
    llm_num_ctx: int


class ConfigResponse(BaseModel):
    llm_provider: str
    defaults: TuningDefaults
    ranges: dict[str, RangeSpec]


# These ranges mirror the Field(ge=..., le=...) constraints in SearchRequest (search_router.py).
# A future change to validation bounds must update both places.
_RANGES: dict[str, RangeSpec] = {
    "min_score":                RangeSpec(min=0.0,  max=1.0,   step=0.05),
    "hybrid_vector_weight":     RangeSpec(min=0.0,  max=1.0,   step=0.05),
    "max_results_per_document": RangeSpec(min=1,    max=10,    step=1),
    "llm_temperature":          RangeSpec(min=0.0,  max=2.0,   step=0.1),
    # Range is fixed and conservative: the server does not know the context-window
    # limit of the currently loaded Ollama model.
    "llm_num_ctx":              RangeSpec(min=2048, max=32768, step=2048),
}


@router.get("", response_model=ConfigResponse)
async def get_config(settings: SettingsDeps) -> ConfigResponse:
    # Mirror search_router._build_search_query: the effective reranker flag
    # determines which threshold the server would apply, so the UI shows the
    # correct starting value without duplicating this logic client-side.
    effective_min_score = (
        settings.rerank_min_score if settings.reranker_enabled else settings.min_score
    )

    return ConfigResponse(
        llm_provider=settings.llm_provider or "",
        defaults=TuningDefaults(
            min_score=effective_min_score,
            fusion=settings.fusion,
            hybrid_vector_weight=settings.hybrid_vector_weight,
            max_results_per_document=settings.max_results_per_document,
            reranker_enabled=settings.reranker_enabled,
            llm_temperature=settings.llm_temperature,
            llm_think=settings.llm_think,
            llm_num_ctx=settings.llm_num_ctx,
        ),
        ranges=_RANGES,
    )
