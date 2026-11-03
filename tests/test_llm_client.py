import asyncio
import json
from collections.abc import Callable

import httpx2
import pytest
from openai import AsyncOpenAI

from application.llm import LLMError
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


@pytest.mark.parametrize("status", [401, 429, 500])
def test_wraps_api_errors_without_exposing_provider_text(status: int) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={"error": {"message": "Provider error containing test-key"}},
            request=request,
        )

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                {}, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError) as caught:
        asyncio.run(run())

    assert "test-key" not in str(caught.value)


def test_wraps_sdk_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("Timeout containing test-key", request=request)

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                {}, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError) as caught:
        asyncio.run(run())

    assert "test-key" not in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nonfinite-input"),
        pytest.param({"not JSON"}, id="unserializable-input"),
    ],
)
def test_invalid_input_is_rejected_before_a_model_request(value: object) -> None:
    request_count = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        return httpx2.Response(200, json=_response(), request=request)

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                {"value": value}, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError):
        asyncio.run(run())

    assert request_count == 0


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            _response() | {"status": "incomplete"}, id="incomplete-with-valid-text"
        ),
        pytest.param(_response() | {"status": "failed"}, id="failed-with-valid-text"),
        pytest.param(
            _response()
            | {
                "output": [
                    {
                        "id": "msg_refusal",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "refusal", "refusal": "Refusal with test-key"},
                            {
                                "type": "output_text",
                                "text": '{"seo_title":"Draft"}',
                                "annotations": [],
                            },
                        ],
                    }
                ]
            },
            id="refusal-with-valid-text",
        ),
        pytest.param(_response("not JSON"), id="invalid-json"),
        pytest.param(_response("[]"), id="non-object-json"),
        pytest.param(_response('{"value":NaN}'), id="nonfinite-json"),
        pytest.param(
            _response('{"specs":{"range":"0-1 MPa","range":"0-2 MPa"}}'),
            id="duplicate-specification",
        ),
    ],
)
def test_rejects_unsuccessful_or_invalid_model_output(body: dict[str, object]) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body, request=request)

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                {}, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError) as caught:
        asyncio.run(run())

    assert "test-key" not in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"status": "completed"}, id="missing-output"),
        pytest.param(_response() | {"output": None}, id="null-output"),
        pytest.param(
            _response() | {"output": [{"type": "message", "content": None}]},
            id="null-message-content",
        ),
        pytest.param("Provider error containing test-key", id="non-json-body"),
    ],
)
def test_wraps_malformed_provider_responses(body: object) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if isinstance(body, str):
            return httpx2.Response(200, text=body, request=request)
        return httpx2.Response(200, json=body, request=request)

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                {}, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError) as caught:
        asyncio.run(run())

    assert "test-key" not in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"{broken test-key", id="invalid-json"),
        pytest.param(b"\xff test-key", id="invalid-utf8"),
    ],
)
def test_wraps_provider_json_body_decoding_errors(body: bytes) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
            request=request,
        )

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                {}, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError, match=r"^The LLM request failed\.$"):
        asyncio.run(run())


@pytest.mark.parametrize("source", ["provider-envelope", "model-text"])
def test_wraps_json_nesting_that_exceeds_parser_capacity(source: str) -> None:
    nested_json = "[" * 10000 + "0" + "]" * 10000
    if source == "provider-envelope":
        body = nested_json.encode()
    else:
        body = json.dumps(_response('{"value":' + nested_json + "}")).encode()

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
            request=request,
        )

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                {}, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError):
        asyncio.run(run())


def test_rejects_input_nesting_before_a_model_request() -> None:
    data: dict[str, object] = {}
    for _ in range(10000):
        data = {"nested": data}
    request_count = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        return httpx2.Response(200, json=_response(), request=request)

    async def run() -> None:
        async with _sdk(handler) as sdk:
            await OpenAIJsonClient(sdk, model="test-model").generate(
                data, instructions=INSTRUCTIONS, schema=SCHEMA
            )

    with pytest.raises(LLMError, match=r"^Input must contain supported JSON values\.$"):
        asyncio.run(run())

    assert request_count == 0


def test_close_closes_the_injected_sdk_http_client() -> None:
    async def run() -> None:
        http_client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda request: httpx2.Response(500))
        )
        sdk = AsyncOpenAI(api_key="test-key", http_client=http_client, max_retries=0)
        try:
            await OpenAIJsonClient(sdk, model="test-model").close()
            assert http_client.is_closed
        finally:
            await sdk.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("OPENAI_API_KEY", None),
        ("OPENAI_MODEL", None),
        ("OPENAI_MODEL", " \t"),
    ],
)
def test_from_env_reports_missing_configuration_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: str | None,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    if value is None:
        monkeypatch.delenv(field)
    else:
        monkeypatch.setenv(field, value)

    with pytest.raises(LLMError) as caught:
        OpenAIJsonClient.from_env()

    captured = capsys.readouterr()
    assert "test-key" not in str(caught.value)
    assert "test-key" not in captured.out + captured.err
