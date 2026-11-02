import pytest
from fastapi.testclient import TestClient

from main import create_app


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
    assert response.json() == product


def test_rejects_malformed_json():
    with TestClient(create_app()) as client:
        response = client.post(
            "/products",
            content='{"name":',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("body", ["[]", "null", '"DM 02"', "42", "true"])
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
