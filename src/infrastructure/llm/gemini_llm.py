import logging
from collections.abc import AsyncIterator

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import Settings
from src.domain.entities import LlmOptions
from src.domain.ports.llm_port import LlmPort
from src.domain.query_mode import parse_mode
from src.infrastructure.llm.prompts import (
    build_rag_prompt,
    truncate_context,
)

logger = logging.getLogger(__name__)


class GeminiLlm(LlmPort):
    def __init__(self, settings: Settings) -> None:
        if not settings.google_api_key:
            raise ValueError(
                "google_api_key is required for Gemini LLM. "
                "Set GOOGLE_API_KEY in your .env file."
            )
        self._llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
        )

    def _build_full_prompt(self, prompt: str, context: list[str]) -> str:
        mode, clean_prompt = parse_mode(prompt)
        context_text = truncate_context(context)
        return build_rag_prompt(clean_prompt, context_text, mode)

    def _call_kwargs(self, options: LlmOptions | None) -> dict:
        # Only `temperature` is passed per-call; `think` and `num_ctx` are
        # Ollama-specific and are no-ops on Gemini (intentionally ignored, not a
        # bug). langchain-google-genai reads `temperature` from the invoke
        # kwargs, so no bind()/per-call client and no extra I/O is needed.
        call_kwargs: dict = {}
        if options and options.temperature is not None:
            call_kwargs["temperature"] = options.temperature
        return call_kwargs

    async def generate(
        self,
        prompt: str,
        context: list[str],
        options: LlmOptions | None = None,
    ) -> str:
        full_prompt = self._build_full_prompt(prompt, context)
        try:
            response = await self._llm.ainvoke(
                full_prompt, **self._call_kwargs(options)
            )
            return str(response.content)
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            raise RuntimeError(f"LLM generation failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        context: list[str],
        options: LlmOptions | None = None,
    ) -> AsyncIterator[str]:
        full_prompt = self._build_full_prompt(prompt, context)
        try:
            async for chunk in self._llm.astream(
                full_prompt, **self._call_kwargs(options)
            ):
                content = str(chunk.content)
                if content:
                    yield content
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            raise RuntimeError(f"LLM generation failed: {e}") from e
