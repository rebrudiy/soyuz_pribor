import pytest

from domain.product import Product


@pytest.mark.parametrize("name", [None, "", " ", "\n\t"])
def test_product_reports_missing_name_without_rejecting_it(name: str | None) -> None:
    product = Product(name=name)
    assert product.missing_fields() == ["name", "manufacturer", "specs", "description"]


def test_product_keeps_supplied_name() -> None:
    product = Product(name=" Манометр DM 02 ")
    assert product.name == " Манометр DM 02 "
    assert product.missing_fields() == ["manufacturer", "specs", "description"]
