import asyncio
import json
from collections.abc import Callable

import httpx2
import pytest
from openai import AsyncOpenAI

from infrastructure.openai_client import OpenAIJsonClient

SCHEMA = {
    "type": "object",
    "properties": {"seo_title": {"type": "string"}},
    "required": ["seo_title"],
    "additionalProperties": False,
}
INSTRUCTIONS = "Return a draft using only the supplied product facts."


def _response(text: str = '{"seo_title":"Черновик"}') -> dict[str, object]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def _sdk(handler: Callable[[httpx2.Request], httpx2.Response]) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


@pytest.mark.parametrize(
    "data",
    [
        {},
        {
            "name": "Манометр",
            "manufacturer": None,
            "description": "",
            "specs": {"range": "0-1,6 МПа", "offset": 0, "enabled": False},
        },
    ],
)
def test_sends_supplied_json_and_returns_a_completed_object(
    data: dict[str, object],
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=_response(), request=request)

    async def run() -> dict[str, object]:
        async with _sdk(handler) as sdk:
            client = OpenAIJsonClient(sdk, model="test-model")
            return await client.generate(data, instructions=INSTRUCTIONS, schema=SCHEMA)

    assert asyncio.run(run()) == {"seo_title": "Черновик"}
    assert len(requests) == 1
    sent = requests[0]
    assert sent["model"] == "test-model"
    assert sent["instructions"] == INSTRUCTIONS
    assert json.loads(sent["input"]) == data
    assert sent["store"] is False
    assert sent["max_output_tokens"] == 4000
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["strict"] is True
    assert sent["text"]["format"]["schema"] == SCHEMA
