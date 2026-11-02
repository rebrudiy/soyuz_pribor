from collections.abc import Mapping
from dataclasses import dataclass


class InvalidProduct(ValueError):
    """Product data does not satisfy a domain rule."""


@dataclass(frozen=True, slots=True)
class Product:
    """Partial supplied product data; absent facts are not invented."""

    name: str | None = None
    manufacturer: str | None = None
    specs: Mapping[str, object] | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("name", self.name),
            ("manufacturer", self.manufacturer),
            ("description", self.description),
        ):
            if value is not None and not isinstance(value, str):
                raise InvalidProduct(f"Product {field} must be text or null.")
        if self.specs is not None and (
            not isinstance(self.specs, Mapping)
            or any(not isinstance(key, str) for key in self.specs)
        ):
            raise InvalidProduct(
                "Product specs must be an object with text keys or null."
            )

    def missing_fields(self) -> list[str]:
        """Report absent facts without blocking generation."""
        missing = []
        for field, value in (("name", self.name), ("manufacturer", self.manufacturer)):
            if value is None or not value.strip():
                missing.append(field)
        if not self.specs:
            missing.append("specs")
        else:
            missing.extend(
                f"specs.{key}"
                for key, value in self.specs.items()
                if value is None or (isinstance(value, str) and not value.strip())
            )
        if self.description is None or not self.description.strip():
            missing.append("description")
        return missing
