import json
import logging
from collections.abc import AsyncIterator

import httpx

from src.config.settings import Settings
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

    def _build_payload(self, full_prompt: str, stream: bool) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": full_prompt},
            ],
            "stream": stream,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
            },
        }
        # Only send "think" when enabled: older Ollama versions reject the field
        if self._think:
            payload["think"] = True
        return payload

    async def generate(self, prompt: str, context: list[str]) -> str:
        full_prompt = self._build_full_prompt(prompt, context)
        payload = self._build_payload(full_prompt, stream=False)
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
        self, prompt: str, context: list[str]
    ) -> AsyncIterator[str]:
        full_prompt = self._build_full_prompt(prompt, context)
        payload = self._build_payload(full_prompt, stream=True)
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
