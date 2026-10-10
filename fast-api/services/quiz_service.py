import logging

from google.genai import types

from prompts.quiz_prompt import QUIZ_SYSTEM_PROMPT, build_quiz_prompt
from schemas.quiz import GeneratedQuizResponse
from services.validation_service import validate_quizzes


logger = logging.getLogger(__name__)


async def generate_quizzes(
    ai,
    model,
    material,
    subject_id,
    quiz_type,
    difficulty,
    count,
    extra,
):
    feedback = ""

    for attempt in range(2):
        result = await ai.models.generate_content(
            model=model,
            contents=build_quiz_prompt(
                material,
                subject_id,
                quiz_type,
                difficulty,
                count,
                extra,
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
                raise ValueError("문제 생성 응답이 비어 있습니다.")

            response = GeneratedQuizResponse.model_validate_json(
                result.text
            )
            return validate_quizzes(
                response,
                quiz_type,
                count,
                material,
            )

        except ValueError as exc:
            feedback = str(exc)[:1500]
            logger.warning(
                "문제 형식 검증 실패: 시도 %d, 원인: %s",
                attempt + 1,
                feedback,
            )

    raise ValueError("문제 생성 형식 검증에 반복 실패했습니다.")