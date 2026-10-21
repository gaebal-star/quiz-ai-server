import os
import re
import unicodedata
import logging
from typing import TypeVar

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

logger = logging.getLogger("uvicorn.error")

ResponseModel = TypeVar(
    "ResponseModel",
    bound=BaseModel,
)

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
    """
    Groq 텍스트 기반 교차검증.
    JSON/Pydantic 검증 실패 시 승인하지 않습니다.
    """

    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY 환경변수가 설정되지 않았습니다."
        )

    client = AsyncGroq(
        api_key=key,
        timeout=90.0,
        max_retries=1,
    )

    response = await client.chat.completions.create(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={
            "type": "json_object"
        },
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise ValueError(
            "Groq 검증 응답이 비어 있습니다."
        )

    return response_schema.model_validate_json(
        content,
        strict=True,
    )


def _index_results(
    results: list,
    count: int,
) -> dict:
    """
    문항별 검증 결과의 누락·중복·범위를 확인합니다.
    결과 구조 자체가 잘못된 경우에는
    문항별 재생성이 아닌 전체 검증 오류로 처리합니다.
    """

    if len(results) != count:
        raise ValueError(
            "문항 수와 검증 결과 수가 다릅니다." )

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
) -> dict[int, str]:
    """
    생성 정답과 해설을 숨긴 채 Groq 독립 풀이를 요청합니다.
    반환값: {실패한 문항 인덱스: 실패 이유}
    검증 통과 문항은 반환 딕셔너리에 포함하지 않습니다.
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
        system_prompt=(GROQ_ANSWER_VERIFICATION_SYSTEM_PROMPT),
        user_prompt=build_answer_verification_prompt(
            material,
            generated,
            quiz_type,
            ),
        response_schema=GroqAnswerVerificationResponse,
    )

    results = _index_results(response.results, len(generated.quizzes),)

    failures: dict[int, str] = {}

    for index, quiz in enumerate(generated.quizzes):
        result = results[index]
        label = f"{index + 1}번 문제"
        reason = None

        if not result.chunkSupportsAnswer:
            reason = ( "연결된 청크의 정답 근거가 부족합니다.")

        elif not result.evidence.strip():
            reason = ("검증 근거가 비어 있습니다.")
        elif quiz_type == "SHORT_ANSWER":
            if result.correctOptionIndexes:
                reason = ( "단답형 검증 결과에 보기 번호가 있습니다.")
            else:
                verified_answer = _normalize_answer( result.shortAnswer)
                generated_answer = _normalize_answer(quiz.quizCorrectAnswer)

                if not verified_answer:
                    reason = ("단답형 정답을 확정하지 못했습니다.")

                elif verified_answer != generated_answer:
                    reason = ("생성 정답과 독립 풀이 정답이 다릅니다.")

        else:
            # 객관식 및 OX 검증
            if result.shortAnswer.strip():
                reason = ("객관식/OX 검증 형식이 잘못되었습니다.")

            else:
                indexes = result.correctOptionIndexes
                if len(indexes) != 1:
                    reason = ("정답을 하나로 확정하지 못했습니다.")

                else:
                    answer_index = indexes[0]
                    if not (
                        0 <= answer_index <
                        len(quiz.quizSelections)
                    ):
                        reason = ("정답 보기 번호가 범위를 벗어났습니다.")

                    else:
                        verified_answer = (quiz.quizSelections[answer_index])

                        if (verified_answer!= quiz.quizCorrectAnswer):
                            reason = ("생성 정답과 독립 풀이 정답이 다릅니다.")

        if reason is not None:
            failures[index] = reason
            logger.warning(
                "AI 정답 검증 실패: %s - %s",
                label,
                reason,
            )

        else:
            logger.info(
                "AI 정답 검증 통과: %s",
                label,
            )

    return failures


async def verify_explanations(
    ai,
    model,
    data,
    mime,
    material: StudyMaterial,
    generated: GeneratedQuizResponse,
    quiz_type: QuizType,
) -> dict[int, str]:
    """
    정답 검증을 통과한 문제의 해설을 검사합니다.
    반환값:
        {실패한 문항 인덱스: 실패 이유}
    """

    if not generated.quizzes:
        raise ValueError("해설을 검증할 문제가 없습니다.")

    response = await _request_verification(
        system_prompt=(
            GROQ_EXPLANATION_VERIFICATION_SYSTEM_PROMPT
        ),
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

    failures: dict[int, str] = {}

    for index in range(len(generated.quizzes)):
        result = results[index]
        label = f"{index + 1}번 문제"
        reason = None

        if not result.explanationSupported:
            reason = ("해설의 근거나 사실·계산을 ""확인하지 못했습니다.")

        elif not result.explanationConsistentWithAnswer:
            reason = ("해설이 문제 조건 또는 정답과 ""일치하지 않습니다.")

        if reason is not None:
            failures[index] = reason
            logger.warning(
                "AI 해설 검증 실패: %s - %s",
                label,
                reason,
            )
        else:
            logger.info(
                "AI 해설 검증 통과: %s",
                label,
            )

    return failures

async def verify_quiz_content(
    ai,
    model,
    data,
    mime,
    material,
    generated,
    quiz_type,
):
    """
    Groq 독립 AI 정답·해설 교차검증.
    반환:
    {
        "passed_indexes": [...],
        "failed_indexes": [...],
        "failure_reasons": {...}
    }

    정답 검증 실패 문항은 해설 검증에서 제외합니다.
    """

    count = len(generated.quizzes)

    if count == 0:
        raise ValueError("검증할 문제가 없습니다.")

    logger.info(
        "Groq AI 검증 시작: 문항 수=%d, 유형=%s",
        count,
        quiz_type,
    )

    # 1. 정답 교차검증
    answer_failures = await verify_answers(
        ai=ai,
        model=model,
        data=data,
        mime=mime,
        material=material,
        generated=generated,
        quiz_type=quiz_type,
    )

    # 2. 정답 검증을 통과한 문항만 추출
    answer_passed_indexes = [
        index
        for index in range(count)
        if index not in answer_failures
    ]

    failure_reasons = dict(answer_failures)

    # 3. 정답 통과 문항에 대해서만 해설 검증
    if answer_passed_indexes:

        explanation_quizzes = [
            generated.quizzes[index]
            for index in answer_passed_indexes
        ]

        explanation_generated = generated.model_copy(
            update={
                "quizzes": explanation_quizzes
            }
        )

        explanation_failures = await verify_explanations(
            ai=ai,
            model=model,
            data=data,
            mime=mime,
            material=material,
            generated=explanation_generated,
            quiz_type=quiz_type,
        )

        # 해설 검증의 인덱스를 원래 문항 인덱스로 변환
        for local_index, reason in explanation_failures.items():
            original_index = answer_passed_indexes[
                local_index
            ]

            failure_reasons[original_index] = reason

    # 4. 최종 통과/실패 분류
    passed_indexes = [
        index
        for index in range(count)
        if index not in failure_reasons
    ]

    failed_indexes = [
        index
        for index in range(count)
        if index in failure_reasons
    ]

    logger.info(
        "Groq AI 검증 완료: 전체=%d, 통과=%d, 실패=%d",
        count,
        len(passed_indexes),
        len(failed_indexes),
    )

    for index in failed_indexes:
        logger.warning(
            "검증 실패 문항=%d, 이유=%s",
            index + 1,
            failure_reasons[index],
        )

    return {
        "passed_indexes": passed_indexes,
        "failed_indexes": failed_indexes,
        "failure_reasons": failure_reasons,
    }