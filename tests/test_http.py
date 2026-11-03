import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from main import create_app


class StubLLM:
    def __init__(self) -> None:
        self.inputs: list[dict[str, object]] = []

    async def generate(self, data, *, instructions, schema):
        self.inputs.append(dict(data))
        if "issues" in schema["properties"]:
            return {"issues": []}
        return {
            "seo_title": "Черновик",
            "seo_description": "Описание прибора. " * 55,
            "search_queries": ["прибор", "измерения", "оборудование"],
            "commercial_block": "Для задач измерения.",
            "clarifications": [],
        }

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def configured_llm(monkeypatch):
    monkeypatch.setattr("main.OpenAIJsonClient.from_env", lambda: StubLLM())


@pytest.mark.parametrize(
    "product",
    [
        {"name": 42},
        {"name": False},
        {"name": []},
        {"name": {}},
    ],
)
def test_rejects_wrong_product_name_type(product: dict[str, object]) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/products", json=product)
    assert response.status_code == 422
    assert "name" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "product", [{}, {"name": ""}, {"name": " \n\t"}, {"name": None}]
)
def test_accepts_incomplete_product_data(product: dict[str, object]) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/products", json=product)
    assert response.status_code == 200
    assert response.json()["product"] == product
    assert response.json()["missing_fields"] == [
        "name",
        "manufacturer",
        "specs",
        "description",
    ]
    assert "draft" in response.json()


def test_receives_product_json_without_changing_technical_values():
    product = {
        "name": "Манометр показывающий DM 02",
        "manufacturer": "Манотомь",
        "specs": {
            "диапазон измерений": "0–1,6 МПа",  # noqa: RUF001 - preserve input text
            "класс точности": "1,5",
            "присоединение": "G1/2",
            "диаметр корпуса": "100 мм",
        },
        "description": "Прибор предназначен для измерения избыточного давления...",
    }

    with TestClient(create_app()) as client:
        response = client.post("/products", json=product)

    assert response.status_code == 200
    assert response.json()["product"] == product
    assert response.json()["missing_fields"] == []
    assert response.json()["draft"]["key_characteristics"] == [
        {"name": key, "value": value} for key, value in product["specs"].items()
    ]


def test_reports_missing_facts_and_preserves_supplied_fields() -> None:
    product = {
        "name": "DM 02",
        "specs": {"range": "0-1,6 МПа", "power": None},
        "custom_note": "Keep this source text",
    }
    with TestClient(create_app()) as client:
        response = client.post("/products", json=product)

    assert response.status_code == 200
    assert response.json()["product"] == product
    assert response.json()["missing_fields"] == [
        "manufacturer",
        "specs.power",
        "description",
    ]
    assert response.json()["draft"]["key_characteristics"] == [
        {"name": "range", "value": "0-1,6 МПа"},
        {"name": "power", "value": "данных нет"},
    ]


def test_rejects_malformed_json():
    with TestClient(create_app()) as client:
        response = client.post(
            "/products",
            content='{"name":',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        "null",
        '"DM 02"',
        "42",
        "true",
        "NaN",
        "Infinity",
        "-Infinity",
        "[NaN]",
        '"\\ud800"',
        '["\\ud800"]',
    ],
)
def test_rejects_json_that_is_not_an_object(body):
    with TestClient(create_app()) as client:
        response = client.post(
            "/products", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e400}',
        '{"specs": {"values": [NaN]}}',
        '{"name": "\\ud800"}',
    ],
)
def test_rejects_values_that_cannot_be_returned_as_json(body):
    with TestClient(create_app()) as client:
        response = client.post(
            "/products", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        '{"name": "DM 02", "name": "Another product"}',
        '{"specs": {"range": "0-1 MPa", "range": "0-2 MPa"}}',
        '{"specs": {"range": "0-1 MPa", "\\u0072ange": "0-2 MPa"}}',
    ],
)
def test_rejects_repeated_json_fields(body: str) -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/products", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("content_type", [None, "text/plain"])
def test_requires_json_content_type(content_type: str | None) -> None:
    headers = {} if content_type is None else {"Content-Type": content_type}

    with TestClient(create_app()) as client:
        response = client.post(
            "/products", content='{"name": "DM 02"}', headers=headers
        )

    assert response.status_code == 415
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "literal",
    [
        "0.1234567890123456789",
        "1e-400",
        "-1e-400",
        "1e400",
        "1e-9999999999999999999999",
    ],
)
def test_rejects_number_rounding_before_a_model_call(literal: str) -> None:
    llm = StubLLM()
    body = '{"specs":{"value":' + literal + "}}"

    with TestClient(create_app(llm=llm)) as client:
        response = client.post(
            "/products", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 422
    assert "quote" in response.json()["detail"].lower()
    assert llm.inputs == []


def test_rejects_rounding_in_nested_additional_source_data() -> None:
    llm = StubLLM()
    body = '{"extra":{"values":[0.1234567890123456789]}}'

    with TestClient(create_app(llm=llm)) as client:
        response = client.post(
            "/products", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 422
    assert "quote" in response.json()["detail"].lower()
    assert llm.inputs == []


def test_preserves_supported_numbers_and_quoted_exact_values() -> None:
    llm = StubLLM()
    body = (
        '{"specs":{"fraction":0.1,"trailing":1.2300,"exponent":1.25e2,'
        '"zero":0,"disabled":false,"nested":{"value":0.125},'
        '"exact":"0.1234567890123456789 МПа"}}'
    )

    with TestClient(create_app(llm=llm)) as client:
        response = client.post(
            "/products", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 200
    result = json.loads(response.text, parse_float=Decimal)
    assert result["product"] == json.loads(body, parse_float=Decimal)
    assert result["product"]["specs"]["disabled"] is False
    assert result["product"]["specs"]["zero"] == 0
    assert len(llm.inputs) == 2


def test_rejects_excessive_json_nesting_before_a_model_call() -> None:
    llm = StubLLM()
    body = '{"specs":{"nested":' + "[" * 10000 + "0" + "]" * 10000 + "}}"

    with TestClient(create_app(llm=llm)) as client:
        response = client.post(
            "/products", content=body, headers={"Content-Type": "application/json"}
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "JSON nesting is too deep."
    assert llm.inputs == []
