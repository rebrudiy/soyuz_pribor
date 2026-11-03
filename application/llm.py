from collections.abc import Mapping
from typing import Protocol


class LLMError(RuntimeError):
    """The model request could not produce a usable response."""


class JsonLLMClient(Protocol):
    """Application boundary for a model that accepts and returns JSON objects."""

    async def generate(
        self,
        data: Mapping[str, object],
        *,
        instructions: str,
        schema: Mapping[str, object],
    ) -> dict[str, object]: ...
