from argparse import ArgumentParser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from application.llm import JsonLLMClient, LLMError
from infrastructure.openai_client import OpenAIJsonClient
from presentation.http import router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if app.state.llm is not None:
        yield
        return
    load_dotenv(Path(__file__).with_name(".env"))
    try:
        client = OpenAIJsonClient.from_env()
    except LLMError:
        yield
        return
    app.state.llm = client
    try:
        yield
    finally:
        await client.close()
        app.state.llm = None


def create_app(llm: JsonLLMClient | None = None) -> FastAPI:
    """Wire the app; injected clients remain owned by their caller."""
    app = FastAPI(title="Soyuz Pribor", lifespan=_lifespan)
    app.state.llm = llm
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--codex", action="store_true", help="Use the local Codex ChatGPT login."
    )
    if parser.parse_args().codex:
        from infrastructure.codex_client import CodexJsonClient

        app = create_app(CodexJsonClient())
    uvicorn.run(app, host="127.0.0.1", port=8000)
