from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.domain.entities import LlmOptions


class LlmPort(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: list[str], options: LlmOptions | None = None) -> str:
        ...

    @abstractmethod
    def generate_stream(
        self, prompt: str, context: list[str], options: LlmOptions | None = None
    ) -> AsyncIterator[str]:
        """Stream the answer token by token.

        Returns an async iterator yielding incremental text fragments.
        Implementations must not include reasoning/"thinking" content.
        When provided, `options` overrides the adapter's instance-level defaults field by field —
        only non-None fields in `options` take effect.
        """
        ...
