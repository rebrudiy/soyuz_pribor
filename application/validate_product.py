from collections.abc import Mapping

from domain.product import InvalidProduct, Product


def _optional_text(value: object, field: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise InvalidProduct(f"Product {field} must be text or null.")


def validate_product(data: Mapping[str, object]) -> Product:
    """Build a product from input data without changing supplied values."""
    name = _optional_text(data.get("name"), "name")
    manufacturer = _optional_text(data.get("manufacturer"), "manufacturer")
    description = _optional_text(data.get("description"), "description")
    specs = data.get("specs")
    if specs is not None and (
        not isinstance(specs, Mapping) or any(not isinstance(key, str) for key in specs)
    ):
        raise InvalidProduct("Product specs must be an object with text keys or null.")
    return Product(
        name=name,
        manufacturer=manufacturer,
        specs=specs,
        description=description,
    )
