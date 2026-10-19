import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

from routers.quiz_router import router as quiz_router


load_dotenv(Path(__file__).resolve().parent / ".env")

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

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.0.81:3000",
        # 팀원들이 사용하는 프론트엔드 주소 추가
        "http://192.168.0.45:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quiz_router)


@app.get("/health")
def health():
    return {"status": "ok"}