import json
from typing import Never

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from application.validate_product import validate_product

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
    try:
        product = json.loads(
            body, object_pairs_hook=_unique_fields, parse_constant=_reject_constant
        )
        if not isinstance(product, dict):
            raise ValueError("JSON must be an object.")
        parsed = validate_product(product)
        return JSONResponse(
            content={"product": product, "missing_fields": parsed.missing_fields()}
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
