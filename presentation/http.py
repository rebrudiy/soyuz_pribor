import json
from decimal import Decimal, InvalidOperation
from typing import Never

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from application.generate_product import generate_product
from application.llm import LLMError, LLMNotConfigured

router = APIRouter()


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a repeated field name.")
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    raise ValueError(f"Unsupported JSON number: {value}.")


def _parse_float(literal: str) -> float:
    """Reject numeric values changed by the normal JSON float round-trip."""
    message = "Quote this number to preserve its exact value."
    try:
        number = float(literal)
        preserved = Decimal(literal) == Decimal(str(number))
    except InvalidOperation:
        raise ValueError(message) from None
    if not preserved:
        raise ValueError(message)
    return number


@router.post(
    "/products",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"type": "object"}}},
        }
    },
)
async def receive_product(request: Request) -> JSONResponse:
    try:
        product = await _read_json_object(request)
        result = await generate_product(product, client=request.app.state.llm)
        return JSONResponse(content=result)
    except LLMNotConfigured:
        raise HTTPException(
            status_code=503, detail="Set OPENAI_API_KEY and OPENAI_MODEL in .env."
        ) from None
    except LLMError:
        raise HTTPException(
            status_code=502, detail="Product generation failed."
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


async def _read_json_object(request: Request) -> dict[str, object]:
    content_type = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    if content_type != "application/json" and not (
        content_type.startswith("application/") and content_type.endswith("+json")
    ):
        raise HTTPException(
            status_code=415, detail="Use Content-Type: application/json."
        )

    body = await request.body()
    product = json.loads(
        body,
        object_pairs_hook=_unique_fields,
        parse_constant=_reject_constant,
        parse_float=_parse_float,
    )
    if not isinstance(product, dict):
        raise ValueError("JSON must be an object.")
    json.dumps(product, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return product
