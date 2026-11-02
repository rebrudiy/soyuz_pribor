import uvicorn
from fastapi import FastAPI

from presentation.http import router


def create_app() -> FastAPI:
    app = FastAPI(title="Soyuz Pribor")
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
