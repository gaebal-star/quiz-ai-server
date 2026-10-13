
import logging

from schemas.quiz import Quiz, QuizResponse
from services.study_service import extract_study_material
from services.quiz_service import generate_quizzes
from services.verification_service import verify_quiz_content

logger = logging.getLogger("uvicorn.error")


async def run_pipeline(
    ai,
    model,
    data,
    mime,
    subject_id,
    quiz_type,
    difficulty,
    count,
    extra,
):
    # 1. 이미지에서 학습 자료 추출
    material = await extract_study_material(
        ai,
        model,
        data,
        mime,
    )

    if not isinstance(material, str) or not material.strip():
        raise ValueError(
            "이미지에서 학습 내용을 추출하지 못했습니다."
        )

    logger.info("이미지 학습 자료 추출 완료")

    # 2. AI 문제 생성 + 유사도 0.5 이상 검증
    # quiz_service.py 내부에서 검증
    generated = await generate_quizzes(
        ai,
        model,
        material,
        subject_id,
        quiz_type,
        difficulty,
        count,
        extra,
    )

    logger.info(
        "문제 생성 및 유사도 검증 통과: 문항 수=%d",
        len(generated.quizzes),
    )

    # 3. 독립 AI 정답·해설 검증
    logger.info(
        "정답·해설 검증 시작: 문항 수=%d",
        len(generated.quizzes),
    )

    verified = await verify_quiz_content(
        ai=ai,
        model=model,
        data=data,
        mime=mime,
        material=material,
        generated=generated,
        quiz_type=quiz_type,
    )

    logger.info(
        "정답·해설 검증 통과: 문항 수=%d",
        len(verified.quizzes),
    )

    # 4. 최종 문제 반환
    return QuizResponse(
        quizzes=[
            Quiz.model_validate(
                quiz.model_dump(
                    exclude={"sourceChunkId"}
                )
            )
            for quiz in verified.quizzes
        ]
    )
