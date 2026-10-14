import os
import re
import unicodedata
from typing import TypeVar

# from google.genai import types
# 검증용 groq API 모델
from groq import AsyncGroq
from pydantic import BaseModel

from schemas.quiz import GeneratedQuizResponse, QuizType
from schemas.study import StudyMaterial
from schemas.verification import (
    GroqAnswerVerificationResponse,
    ExplanationVerificationResponse,
)
from prompts.verification_prompt import (
    GROQ_ANSWER_VERIFICATION_SYSTEM_PROMPT,
    GROQ_EXPLANATION_VERIFICATION_SYSTEM_PROMPT,
    build_answer_verification_prompt,
    build_explanation_verification_prompt,
)
import logging
logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def _normalize_answer(value: str) -> str:
    """
    공백과 유니코드만 정규화합니다.
    대소문자·숫자·부호·단위는 보존합니다.
    """
    value = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", value).strip()


async def _request_verification(
    system_prompt: str,
    user_prompt: str,
    response_schema: type[ResponseModel],
) -> ResponseModel:
    """Groq 텍스트 기반 교차검증. JSON/Pydantic 검증 실패 시 승인하지 않습니다."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY 환경변수가 설정되지 않았습니다.")
    client = AsyncGroq(api_key=key, timeout=90.0, max_retries=1)
    response = await client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Groq 검증 응답이 비어 있습니다.")
    return response_schema.model_validate_json(content, strict=True)


def _index_results(results: list, count: int) -> dict:
    """문항별 검증 결과의 누락·중복·범위를 확인합니다."""
    if len(results) != count:
        raise ValueError("문항 수와 검증 결과 수가 다릅니다.")

    indexed = {}

    for result in results:
        index = result.quizIndex

        if index < 0 or index >= count:
            raise ValueError("검증 결과의 문항 번호가 잘못되었습니다.")

        if index in indexed:
            raise ValueError("검증 결과의 문항 번호가 중복되었습니다.")

        if not result.reason.strip():
            raise ValueError("검증 판정 이유가 비어 있습니다.")

        indexed[index] = result

    if set(indexed) != set(range(count)):
        raise ValueError("검증 결과에서 누락된 문항이 있습니다.")

    return indexed


async def verify_answers(
    ai,
    model,
    data,
    mime,
    material: StudyMaterial,
    generated: GeneratedQuizResponse,
    quiz_type: QuizType,
) -> None:
    """
    생성 정답과 해설을 숨긴 채 독립 풀이를 요청합니다.
    모든 문항이 통과해야 정상 종료합니다.
    """
    if quiz_type not in {
        "MULTIPLE_CHOICE",
        "OX",
        "SHORT_ANSWER",
    }:
        raise ValueError("지원하지 않는 문제 유형입니다.")

    if not generated.quizzes:
        raise ValueError("검증할 문제가 없습니다.")

    response = await _request_verification(
        system_prompt=GROQ_ANSWER_VERIFICATION_SYSTEM_PROMPT,
        user_prompt=build_answer_verification_prompt(
            material,
            generated,
            quiz_type,
        ),
        response_schema=GroqAnswerVerificationResponse,
    )

    results = _index_results(
        response.results,
        len(generated.quizzes),
    )

    for index, quiz in enumerate(generated.quizzes):
        result = results[index]
        label = f"{index + 1}번 문제"

        if not result.chunkSupportsAnswer:
            raise ValueError(
                f"{label}: 연결된 청크의 정답 근거가 부족합니다."
            )

        if not result.evidence.strip():
            raise ValueError(
                f"{label}: 검증 근거가 비어 있습니다."
            )

        if quiz_type == "SHORT_ANSWER":
            if result.correctOptionIndexes:
                raise ValueError(
                    f"{label}: 단답형 검증 결과에 보기 번호가 있습니다."
                )

            verified_answer = _normalize_answer(result.shortAnswer)
            generated_answer = _normalize_answer(
                quiz.quizCorrectAnswer
            )

            if not verified_answer:
                raise ValueError(
                    f"{label}: 단답형 정답을 확정하지 못했습니다."
                )

            if verified_answer != generated_answer:
                raise ValueError(
                    f"{label}: 생성 정답과 독립 풀이 정답이 다릅니다."
                )

        else:
            # 객관식과 OX는 보기 번호로 비교합니다.
            if result.shortAnswer.strip():
                raise ValueError(
                    f"{label}: 객관식/OX 검증 형식이 잘못되었습니다."
                )

            indexes = result.correctOptionIndexes

            # 정답 없음, 복수 정답, 중복 번호 모두 거부
            if len(indexes) != 1:
                raise ValueError(
                    f"{label}: 정답을 하나로 확정하지 못했습니다."
                )

            answer_index = indexes[0]

            if not 0 <= answer_index < len(quiz.quizSelections):
                raise ValueError(
                    f"{label}: 정답 보기 번호가 범위를 벗어났습니다."
                )

            verified_answer = quiz.quizSelections[answer_index]

            if verified_answer != quiz.quizCorrectAnswer:
                raise ValueError(
                    f"{label}: 생성 정답과 독립 풀이 정답이 다릅니다."
                )


async def verify_explanations(
    ai,
    model,
    data,
    mime,
    material: StudyMaterial,
    generated: GeneratedQuizResponse,
    quiz_type: QuizType,
) -> None:
    """정답 검증을 통과한 문제의 해설을 검사합니다."""
    response = await _request_verification(
        system_prompt=GROQ_EXPLANATION_VERIFICATION_SYSTEM_PROMPT,
        user_prompt=build_explanation_verification_prompt(
            material,
            generated,
            quiz_type,
        ),
        response_schema=ExplanationVerificationResponse,
    )

    results = _index_results(
        response.results,
        len(generated.quizzes),
    )

    for index in range(len(generated.quizzes)):
        result = results[index]
        label = f"{index + 1}번 문제"

        if not result.explanationSupported:
            raise ValueError(
                f"{label}: 해설의 근거나 사실·계산을 확인하지 못했습니다."
            )

        if not result.explanationConsistentWithAnswer:
            raise ValueError(
                f"{label}: 해설이 문제 조건 또는 정답과 일치하지 않습니다."
            )


async def verify_quiz_content(
    ai,
    model,
    data,
    mime,
    material,
    generated,
    quiz_type,
):
    logger.info(
        "AI 검증 시작: 문항 수=%d, 유형=%s",
        len(generated.quizzes),
        quiz_type,
    )

    await verify_answers(
        ai=ai,
        model=model,
        data=data,
        mime=mime,
        material=material,
        generated=generated,
        quiz_type=quiz_type,
    )

    logger.info("AI 정답 검증 통과")

    await verify_explanations(
        ai=ai,
        model=model,
        data=data,
        mime=mime,
        material=material,
        generated=generated,
        quiz_type=quiz_type,
    )

    logger.info("AI 해설 검증 통과")
    logger.info("AI 검증 완료: 모든 문항 승인")

    return generated