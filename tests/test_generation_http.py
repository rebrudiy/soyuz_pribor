from collections.abc import Mapping

from fastapi.testclient import TestClient

from application.llm import LLMError
from main import create_app


class StubLLM:
    def __init__(self) -> None:
        self.inputs: list[dict[str, object]] = []
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def generate(
        self,
        data: Mapping[str, object],
        *,
        instructions: str,
        schema: Mapping[str, object],
    ) -> dict[str, object]:
        self.inputs.append(dict(data))
        return {
            "seo_title": "Черновик карточки",
            "seo_description": "Описание прибора. " * 55,
            "search_queries": ["прибор", "измерения", "оборудование"],
            "commercial_block": "Для задач измерения.",
            "clarifications": [],
        }


def test_http_calls_generation_for_incomplete_input() -> None:
    llm = StubLLM()
    with TestClient(create_app(llm=llm)) as client:
        response = client.post("/products", json={})

    assert response.status_code == 200
    assert llm.inputs == [{}]
    assert response.json()["draft"]["seo_title"] == "Черновик карточки"
    assert response.json()["missing_fields"] == [
        "name",
        "manufacturer",
        "specs",
        "description",
    ]


def test_http_translates_model_failure_without_leaking_details() -> None:
    class FailedLLM(StubLLM):
        async def generate(self, data, *, instructions, schema):
            raise LLMError("private provider details")

    with TestClient(create_app(llm=FailedLLM())) as client:
        response = client.post("/products", json={})

    assert response.status_code == 502
    assert "private provider details" not in response.text


def test_missing_configuration_is_reported_without_a_model_call(monkeypatch) -> None:
    def missing_client():
        raise LLMError("Set OPENAI_API_KEY and OPENAI_MODEL.")

    monkeypatch.setattr("main.OpenAIJsonClient.from_env", missing_client)
    with TestClient(create_app()) as client:
        response = client.post("/products", json={})

    assert response.status_code == 503


def test_app_closes_the_client_it_creates(monkeypatch) -> None:
    llm = StubLLM()
    monkeypatch.setattr("main.OpenAIJsonClient.from_env", lambda: llm)
    with TestClient(create_app()) as client:
        assert client.post("/products", json={}).status_code == 200
        assert not llm.closed
    assert llm.closed


def test_app_does_not_close_a_borrowed_client() -> None:
    llm = StubLLM()
    with TestClient(create_app(llm=llm)) as client:
        assert client.post("/products", json={}).status_code == 200
    assert not llm.closed
