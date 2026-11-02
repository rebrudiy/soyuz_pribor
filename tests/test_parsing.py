from collections import UserDict
from copy import deepcopy

import pytest

from application.validate_product import validate_product
from domain.product import InvalidProduct


@pytest.fixture
def complete_data() -> dict[str, object]:
    return {
        "name": " Манометр DM 02 ",
        "manufacturer": " Манотомь ",
        "specs": {"range": "0–1,6 МПа"},  # noqa: RUF001 - preserve supplied text
        "description": " Описание прибора.\n",
    }


def test_preserves_all_supplied_fields(complete_data: dict[str, object]) -> None:
    product = validate_product(complete_data)

    assert product.name == complete_data["name"]
    assert product.manufacturer == complete_data["manufacturer"]
    assert product.specs == complete_data["specs"]
    assert product.description == complete_data["description"]
    assert product.missing_fields() == []


def test_empty_input_reports_missing_fields_without_inventing_values() -> None:
    product = validate_product({})

    assert product.name is None
    assert product.manufacturer is None
    assert product.specs is None
    assert product.description is None
    assert set(product.missing_fields()) == {
        "name",
        "manufacturer",
        "specs",
        "description",
    }


@pytest.mark.parametrize("field", ["name", "manufacturer", "specs", "description"])
def test_reports_an_absent_field(complete_data: dict[str, object], field: str) -> None:
    complete_data.pop(field)

    product = validate_product(complete_data)

    assert getattr(product, field) is None
    assert product.missing_fields() == [field]


@pytest.mark.parametrize("field", ["name", "manufacturer", "description"])
@pytest.mark.parametrize("value", [None, "", " \n\t"])
def test_accepts_and_reports_missing_text(
    complete_data: dict[str, object], field: str, value: str | None
) -> None:
    complete_data[field] = value

    product = validate_product(complete_data)

    assert getattr(product, field) == value
    assert product.missing_fields() == [field]


@pytest.mark.parametrize("specs", [None, {}])
def test_accepts_and_reports_empty_specifications(
    complete_data: dict[str, object], specs: object
) -> None:
    complete_data["specs"] = specs

    product = validate_product(complete_data)

    assert product.specs == specs
    assert product.missing_fields() == ["specs"]


@pytest.mark.parametrize("value", [None, "", " \n\t"])
def test_reports_missing_specification_values(
    complete_data: dict[str, object], value: str | None
) -> None:
    specs = {"range": "0-1 MPa", "power": value}
    complete_data["specs"] = specs

    product = validate_product(complete_data)

    assert product.specs == specs
    assert product.missing_fields() == ["specs.power"]


@pytest.mark.parametrize("value", [0, False])
def test_zero_and_false_are_present_specification_values(
    complete_data: dict[str, object], value: object
) -> None:
    complete_data["specs"] = {"value": value}

    product = validate_product(complete_data)

    assert product.specs is not None
    assert product.specs["value"] == value
    assert type(product.specs["value"]) is type(value)
    assert product.missing_fields() == []


@pytest.mark.parametrize("field", ["name", "manufacturer", "description"])
@pytest.mark.parametrize("value", [42, False, [], {}])
def test_rejects_wrong_text_types(
    complete_data: dict[str, object], field: str, value: object
) -> None:
    complete_data[field] = value

    with pytest.raises(InvalidProduct):
        validate_product(complete_data)


@pytest.mark.parametrize("specs", [42, False, "range", [], [("range", "0-1 MPa")]])
def test_rejects_wrong_specifications_types(
    complete_data: dict[str, object], specs: object
) -> None:
    complete_data["specs"] = specs

    with pytest.raises(InvalidProduct):
        validate_product(complete_data)


def test_rejects_nontext_specification_keys(complete_data: dict[str, object]) -> None:
    complete_data["specs"] = {1: "0-1 MPa"}

    with pytest.raises(InvalidProduct):
        validate_product(complete_data)


def test_accepts_a_specifications_mapping(complete_data: dict[str, object]) -> None:
    specs = UserDict({"range": "0-1 MPa"})
    complete_data["specs"] = specs

    product = validate_product(complete_data)

    assert product.specs == specs
    assert product.missing_fields() == []


def test_parsing_and_missing_field_checks_do_not_mutate_input(
    complete_data: dict[str, object],
) -> None:
    complete_data["name"] = " \t"
    complete_data["specs"] = {
        "range": "0-1 MPa",
        "power": None,
        "offset": 0,
        "enabled": False,
        "interfaces": ["RS485"],
    }
    complete_data["extra"] = {"tags": ["supplied", "unchanged"]}
    original = deepcopy(complete_data)

    product = validate_product(complete_data)
    assert set(product.missing_fields()) == {"name", "specs.power"}

    assert complete_data == original
