import json
import logging
from collections.abc import AsyncIterator

import httpx

from src.config.settings import Settings
from src.domain.entities import LlmOptions
from src.domain.ports.llm_port import LlmPort
from src.domain.query_mode import parse_mode
from src.infrastructure.llm.prompts import (
    build_rag_prompt,
    truncate_context,
)

logger = logging.getLogger(__name__)


class OllamaLlm(LlmPort):
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = (
            settings.llm_model
            if settings.llm_provider == "ollama"
            else "llama3.2"
        )
        self._temperature = settings.llm_temperature
        self._num_ctx = settings.llm_num_ctx
        self._think = settings.llm_think
        self._timeout = settings.llm_timeout

    def _build_full_prompt(self, prompt: str, context: list[str]) -> str:
        mode, clean_prompt = parse_mode(prompt)
        context_text = truncate_context(context)
        return build_rag_prompt(clean_prompt, context_text, mode)

    def _build_payload(
        self,
        full_prompt: str,
        stream: bool,
        options: LlmOptions | None = None,
    ) -> dict:
        # Field-by-field merge with `is not None` (never `or`): temperature=0.0
        # and think=False are legitimate overrides, not falsy fall-throughs.
        opts = options or LlmOptions()
        temperature = (
            opts.temperature if opts.temperature is not None else self._temperature
        )
        num_ctx = opts.num_ctx if opts.num_ctx is not None else self._num_ctx
        think = opts.think if opts.think is not None else self._think
        payload: dict = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": full_prompt},
            ],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        # Only send "think" when enabled: older Ollama versions reject the field
        if think:
            payload["think"] = True
        return payload

    async def generate(
        self,
        prompt: str,
        context: list[str],
        options: LlmOptions | None = None,
    ) -> str:
        full_prompt = self._build_full_prompt(prompt, context)
        payload = self._build_payload(full_prompt, stream=False, options=options)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat", json=payload
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s: %s", self._base_url, e)
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running (ollama serve)."
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error("Ollama request failed: %s", e)
            raise RuntimeError(f"Ollama request failed: {e}") from e
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise RuntimeError(f"LLM generation failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        context: list[str],
        options: LlmOptions | None = None,
    ) -> AsyncIterator[str]:
        full_prompt = self._build_full_prompt(prompt, context)
        payload = self._build_payload(full_prompt, stream=True, options=options)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", f"{self._base_url}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed Ollama line: %s", line)
                            continue
                        # Ignore "thinking" chunks: only stream visible content
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break
        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s: %s", self._base_url, e)
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running (ollama serve)."
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error("Ollama request failed: %s", e)
            raise RuntimeError(f"Ollama request failed: {e}") from e
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise RuntimeError(f"LLM generation failed: {e}") from e
