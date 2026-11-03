import json
from collections.abc import Mapping

from openai import APIError, AsyncOpenAI

from application.llm import LLMError


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("The model returned repeated JSON fields.")
    return result


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
        try:
            payload = json.dumps(dict(data), ensure_ascii=False, allow_nan=False)
            payload.encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise LLMError("Input must contain supported JSON values.") from exc

        try:
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
        except APIError as exc:
            raise LLMError("The LLM request failed.") from exc

        if response.status != "completed":
            raise LLMError("The LLM response did not complete.")
        if any(
            part.type == "refusal"
            for item in response.output
            if item.type == "message"
            for part in item.content
        ):
            raise LLMError("The LLM refused the request.")
        try:
            result = json.loads(response.output_text, object_pairs_hook=_unique_object)
            if not isinstance(result, dict):
                raise ValueError("Expected an object.")
            json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise LLMError("The LLM response is not a valid JSON object.") from exc
        return result
