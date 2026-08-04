from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LlmPort(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: list[str]) -> str:
        ...

    @abstractmethod
    def generate_stream(
        self, prompt: str, context: list[str]
    ) -> AsyncIterator[str]:
        """Stream the answer token by token.

        Returns an async iterator yielding incremental text fragments.
        Implementations must not include reasoning/"thinking" content.
        """
        ...
