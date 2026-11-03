import json
from collections.abc import Mapping

from openai import AsyncOpenAI


class OpenAIJsonClient:
    """OpenAI transport adapter; prompts and output schemas belong to the caller."""

    def __init__(self, sdk: AsyncOpenAI, *, model: str) -> None:
        self._sdk = sdk
        self._model = model

    async def generate(
        self,
        data: Mapping[str, object],
        *,
        instructions: str,
        schema: Mapping[str, object],
    ) -> dict[str, object]:
        """Send supplied facts unchanged, including incomplete or empty data."""
        payload = json.dumps(dict(data), ensure_ascii=False)

        response = await self._sdk.responses.create(
            model=self._model,
            instructions=instructions,
            input=payload,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "strict": True,
                    "schema": dict(schema),
                }
            },
            max_output_tokens=4000,
            store=False,
        )

        return json.loads(response.output_text)
