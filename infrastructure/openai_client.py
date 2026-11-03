import json
import logging
import os
from collections.abc import Mapping

from openai import APIError, APIStatusError, AsyncOpenAI
from openai.types.responses import Response

from application.llm import LLMError

logger = logging.getLogger(__name__)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("The model returned repeated JSON fields.")
    return result


class OpenAIJsonClient:
    """OpenAI transport adapter; prompts and output schemas belong to the caller."""

    def __init__(self, sdk: AsyncOpenAI, *, model: str) -> None:
        if not model.strip():
            raise LLMError("A model name is required.")
        self._sdk = sdk
        self._model = model

    @classmethod
    def from_env(cls) -> "OpenAIJsonClient":
        """Create a client from environment variables; close it after use."""
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "").strip()
        if not key or not model:
            raise LLMError("Set OPENAI_API_KEY and OPENAI_MODEL.")
        return cls(AsyncOpenAI(api_key=key, timeout=30.0, max_retries=1), model=model)

    async def generate(
        self,
        data: Mapping[str, object],
        *,
        instructions: str,
        schema: Mapping[str, object],
    ) -> dict[str, object]:
        """Send supplied facts unchanged, including incomplete or empty data."""
        payload = self._encode_input(data)
        response = await self._request(
            payload, instructions=instructions, schema=schema
        )
        return self._decode_response(response)

    @staticmethod
    def _decode_response(response: Response) -> dict[str, object]:
        try:
            if response.status != "completed":
                raise LLMError("The LLM response did not complete.")
            if any(
                part.type == "refusal"
                for item in response.output
                if item.type == "message"
                for part in item.content
            ):
                raise LLMError("The LLM refused the request.")
            result = json.loads(response.output_text, object_pairs_hook=_unique_object)
            if not isinstance(result, dict):
                raise ValueError("Expected an object.")
            json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (AttributeError, TypeError, ValueError, RecursionError) as exc:
            raise LLMError("The LLM response is not a valid JSON object.") from exc
        return result

    @staticmethod
    def _encode_input(data: Mapping[str, object]) -> str:
        try:
            payload = json.dumps(dict(data), ensure_ascii=False, allow_nan=False)
            payload.encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise LLMError("Input must contain supported JSON values.") from exc
        return payload

    async def _request(
        self,
        payload: str,
        *,
        instructions: str,
        schema: Mapping[str, object],
    ) -> Response:
        try:
            return await self._sdk.responses.create(
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
        except (
            APIError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            RecursionError,
        ) as exc:
            status = exc.status_code if isinstance(exc, APIStatusError) else None
            logger.warning(
                "LLM request failed: %s (HTTP status: %s).", type(exc).__name__, status
            )
            raise LLMError("The LLM request failed.") from exc

    async def close(self) -> None:
        """Release the SDK's HTTP connections."""
        await self._sdk.close()
