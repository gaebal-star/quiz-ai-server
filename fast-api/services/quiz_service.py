
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

MAX_REPAIR_ATTEMPTS = 1


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
    approved_quizzes = []

    for repair_attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        pending_count = count - len(approved_quizzes)
        if pending_count <= 0:
            break

        if repair_attempt == 0:
            logger.info("AI 최초 문제 생성 시작: 요청=%d", pending_count)
        else:
            logger.info(
                "유사도 탈락 문항 보충 생성: 시도=%d/%d, 요청=%d",
                repair_attempt,
                MAX_REPAIR_ATTEMPTS,
                pending_count,
            )

        result = await ai.models.generate_content(
            model=model,
            contents=build_quiz_prompt(
                material,
                subject_id,
                quiz_type,
                difficulty,
                pending_count,
                extra,
                output_language,
                feedback,
            ),
            config=types.GenerateContentConfig(
                system_instruction=QUIZ_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=GeneratedQuizResponse,
                max_output_tokens=min(
                    24576,
                    max(4096, pending_count * 2048),
                ),
                temperature=0.3,
            ),
        )

        try:
            if not result.text:
                raise ValueError("문제 생성 응답이 비어 있습니다.")

            response = GeneratedQuizResponse.model_validate_json(
                result.text
            )
            validated = validate_quizzes(
                response,
                quiz_type,
                pending_count,
                material,
                output_language,
            )
            passed, failed = validate_similarity(
                material,
                validated.quizzes,
            )

            approved_quizzes.extend(passed)

            if not failed and len(approved_quizzes) == count:
                logger.info(
                    "유사도 검증 통과: 전체=%d개",
                    len(approved_quizzes),
                )
                return response.model_copy(
                    update={"quizzes": approved_quizzes}
                )

            if failed:
                logger.warning(
                    "유사도 검증 실패: %d개, 통과: %d개",
                    len(failed),
                    len(passed),
                )
                feedback = (
                    "이전 생성에서 유사도 검증에 실패한 문항만 "
                    "다시 생성하세요. 학습 자료의 핵심 개념과 "
                    "직접 연결된 구체적인 질문을 작성하세요.\n"
                    + "\n".join(
                        f"- 실패 문항: {item['question']} "
                        f"(유사도: {item.get('similarity', 0):.4f})"
                        for item in failed
                    )
                )[:1500]

                if repair_attempt < MAX_REPAIR_ATTEMPTS:
                    continue

                raise ValueError(
                    f"유사도 검증 탈락 문항 {len(failed)}개를 "
                    "보충 생성한 뒤에도 통과하지 못했습니다."
                )

            raise ValueError(
                f"요청한 {count}개 중 검증 통과 문항은 "
                f"{len(approved_quizzes)}개입니다."
            )

        except ValueError as exc:
            feedback = str(exc)[:1500]
            logger.warning(
                "문제 검증 실패: 시도=%d, 원인=%s",
                repair_attempt + 1,
                feedback,
            )
            if repair_attempt >= MAX_REPAIR_ATTEMPTS:
                raise ValueError(
                    f"문제 생성 또는 검증 보충에 실패했습니다: {exc}"
                ) from exc

    raise ValueError("요청한 문항 수를 모두 검증하지 못했습니다.")
