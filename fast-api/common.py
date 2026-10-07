# -*- coding: utf-8 -*-
"""
common.py — 모든 실습 공통 파일

목적:
  - .env 의 API 키를 한 곳에서 로드한다.
  - Gemini(주력) / OpenAI(보조) 모델 객체를 일관되게 생성한다.
  - 실습 데이터(data/) 경로를 쉽게 찾는다.

각 강의 실습 코드 맨 위에서 다음처럼 불러 씁니다.
    from common import get_chat, get_genai_client, DATA
"""
import os
import pathlib
from dotenv import load_dotenv

# 프로젝트 루트
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = DATA / "docs"

# .env 로드 (루트의 .env 를 읽음)
load_dotenv(ROOT / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")


def require_key(name: str) -> str:
    """환경변수 키가 없으면 종료."""
    val = os.getenv(name)
    if not val or val.startswith("여기에"):
        raise SystemExit(
            f"[설정 필요] {name} 가 .env 에 없습니다.\n"
            f" 1) cp .env.example .env\n"
            f" 2) .env 파일을 열어 {name} 값을 채우세요."
        )
    return val


# ---------- raw SDK (원리 학습용) ----------
def get_genai_client():
    """google-genai 클라이언트 (from google import genai)."""
    from google import genai
    return genai.Client(api_key=require_key("GOOGLE_API_KEY"))


# ---------- LangChain Chat 모델 (현업용) ----------
def get_chat(provider: str = "gemini", temperature: float = 0.0):
    """LangChain ChatModel 반환. provider: 'gemini'(기본) | 'openai'."""
    if provider == "gemini":
        require_key("GOOGLE_API_KEY")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
    elif provider == "openai":
        require_key("OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    raise ValueError(f"알 수 없는 provider: {provider}")


def get_embeddings(provider: str = "gemini"):
    """LangChain Embeddings 반환."""
    if provider == "gemini":
        require_key("GOOGLE_API_KEY")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # gemini-embedding-001 기본 출력은 3072차원. 실습 저장·속도를 위해 768로 고정.
        # (768/1536/3072 중 선택 가능)
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL, output_dimensionality=768)
    elif provider == "openai":
        require_key("OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    raise ValueError(f"알 수 없는 provider: {provider}")


if __name__ == "__main__":
    print("ROOT :", ROOT)
    print("DATA :", DATA, "(존재:", DATA.exists(), ")")
    print("GEMINI_MODEL :", GEMINI_MODEL)
    print("키 로드 상태 — GOOGLE_API_KEY:", bool(os.getenv("GOOGLE_API_KEY")),
          "/ OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
