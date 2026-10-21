
import logging

from google.genai import types

from prompts.quiz_prompt import (
    QUIZ_SYSTEM_PROMPT,
    build_quiz_prompt,
)
from schemas.quiz import GeneratedQuizResponse
from services.validation_service import validate_quizzes
from services.similarity_service import validate_similarity

logger = logging.getLogger("uvicorn.error")

MAX_ATTEMPTS = 1


async def generate_quizzes(
    ai,
    model,
    material,
    subject_id,
    quiz_type,
    difficulty,
    count,
    extra,
    output_language,
):
    feedback = ""

    for attempt in range(MAX_ATTEMPTS):

        logger.info(
            "AI 문제 생성 시도: %d/%d",
            attempt + 1,
            MAX_ATTEMPTS,
        )

        result = await ai.models.generate_content(
            model=model,
            contents=build_quiz_prompt(
                material,
                subject_id,
                quiz_type,
                difficulty,
                count,
                extra,
                output_language,
                feedback,
            ),
            config=types.GenerateContentConfig(
                system_instruction=QUIZ_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=GeneratedQuizResponse,
                temperature=0.3,
            ),
        )

        try:
            if not result.text:
                raise ValueError(
                    "문제 생성 응답이 비어 있습니다."
                )

            response = (
                GeneratedQuizResponse.model_validate_json(
                    result.text
                )
            )

            # 1. 기존 형식 및 출처 검증
            validated = validate_quizzes(
                response,
                quiz_type,
                count,
                material,
                output_language,
            )

            # 2. 원본 이미지 내용과 유사도 검증
            passed, failed = validate_similarity(
                material,
                validated.quizzes,
            )

            if failed:
                logger.warning(
                    "유사도 검증 실패: %d개",
                    len(failed),
                )

                failed_info = "\n".join(
                    f"- {item['question']} "
                    f"(유사도: {item['similarity']:.4f})"
                    for item in failed
                )

                feedback = (
                    "이전 문제 중 원본 학습 자료와 "
                    "유사도가 0.5 미만인 문제가 있습니다.\n"
                    "반드시 해당 학습 내용에 근거하여 "
                    "모든 문제를 다시 생성하세요.\n"
                    f"{failed_info}"
                )[:1500]

                continue

            if len(passed) != count:
                feedback = (
                    f"요청 문제 수: {count}개, "
                    f"유사도 통과 문제 수: {len(passed)}개. "
                    "정확한 개수로 다시 생성하세요."
                )
                continue

            logger.info(
                "유사도 검증 통과: %d개",
                len(passed),
            )

            return validated

        except ValueError as exc:
            feedback = str(exc)[:1500]

            logger.warning(
                "문제 검증 실패: 시도 %d, 원인: %s",
                attempt + 1,
                feedback,
            )

    raise ValueError(
        "문제 생성 또는 유사도 검증에 "
        "3회 연속 실패했습니다."
    )
