from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/products")
async def receive_product(product: dict[str, object]) -> JSONResponse:
    try:
        return JSONResponse(content=product)
    except (ValueError, UnicodeEncodeError):
        raise HTTPException(
            status_code=422,
            detail="Body contains an invalid or unsupported JSON value.",
        ) from None
