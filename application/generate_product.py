from collections.abc import Mapping

from application.check_factuality import check_factuality
from application.llm import JsonLLMClient, LLMError, LLMNotConfigured
from application.product_prompt import DRAFT_SCHEMA, PRODUCT_PROMPT
from application.validate_product import validate_product
from domain.product import Product


async def generate_product(
    data: Mapping[str, object], *, client: JsonLLMClient | None
) -> dict[str, object]:
    """Generate an honest draft, including when supplied facts are incomplete."""
    product = validate_product(data)
    if client is None:
        raise LLMNotConfigured("The LLM client is not configured.")
    draft = await client.generate(
        data, instructions=PRODUCT_PROMPT, schema=DRAFT_SCHEMA
    )
    _check_draft(draft)
    factuality = await check_factuality(data, draft, client=client)
    return {
        "product": dict(data),
        "missing_fields": product.missing_fields(),
        "draft": {**draft, "key_characteristics": _characteristics(product)},
        "warnings": _quality_warnings(draft),
        "factuality": factuality,
    }


def _check_draft(draft: dict[str, object]) -> None:
    text_fields = ("seo_title", "seo_description", "commercial_block")
    list_fields = ("search_queries", "clarifications")
    if set(draft) != set(text_fields + list_fields):
        raise LLMError("The generated draft has unexpected or missing fields.")
    for field in text_fields:
        if not isinstance(draft[field], str):
            raise LLMError(f"The generated {field} must be text.")
    for field in list_fields:
        value = draft[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise LLMError(f"The generated {field} must be a list of text values.")


def _characteristics(product: Product) -> list[dict[str, object]]:
    return [
        {
            "name": key,
            "value": "данных нет"
            if value is None or (isinstance(value, str) and not value.strip())
            else value,
        }
        for key, value in (product.specs or {}).items()
    ]


def _quality_warnings(draft: dict[str, object]) -> list[str]:
    warnings = [
        f"{field} is blank."
        for field in ("seo_title", "seo_description", "commercial_block")
        if not draft[field].strip()
    ]
    if any(not query.strip() for query in draft["search_queries"]):
        warnings.append("Search queries contain blank entries.")
    normalized_queries = [query.strip().casefold() for query in draft["search_queries"]]
    if len(set(normalized_queries)) != len(normalized_queries):
        warnings.append("Search queries contain repeated entries.")
    if len(draft["seo_title"]) > 70:
        warnings.append("SEO title exceeds 70 characters.")
    if not 800 <= len(draft["seo_description"]) <= 1200:
        warnings.append("SEO description must contain 800-1200 characters.")
    if not 3 <= len(draft["search_queries"]) <= 5:
        warnings.append(
            "Provide 3-5 search queries when the source data supports them."
        )
    return warnings
