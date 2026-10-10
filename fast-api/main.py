import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from google.genai import types

from routers.quiz_router import router as quiz_router


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not key or not model:
        raise RuntimeError(
            "GOOGLE_API_KEY와 GEMINI_MODEL 환경변수를 설정해주세요."
        )

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=150_000),
    )

    app.state.ai = client.aio
    app.state.settings = SimpleNamespace(
        model=model,
        token=os.getenv("QUIZ_AI_TOKEN", ""),
    )

    try:
        yield
    finally:
        await client.aio.aclose()
        client.close()


app = FastAPI(
    title="Quiz AI Server",
    lifespan=lifespan,
)

app.include_router(quiz_router)


@app.get("/health")
def health():
    return {"status": "ok"}