from schemas.quiz import Quiz, QuizResponse
from services.study_service import extract_study_material
from services.quiz_service import generate_quizzes
from services.verification_service import verify_quiz_content


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

    # 2. 문제 생성 및 기존 형식 검증
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

    # 3. 독립 정답 검증 → 해설 검증
    # 실패하면 예외가 발생하므로 아래 반환 코드는 실행되지 않습니다.
    verified = await verify_quiz_content(
        ai=ai,
        model=model,
        data=data,
        mime=mime,
        material=material,
        generated=generated,
        quiz_type=quiz_type,
    )

    # 4. 검증을 통과한 문제만 Spring에 반환
    # sourceChunkId는 Python 내부용이므로 제외합니다.
    return QuizResponse(
        quizzes=[
            Quiz.model_validate(
                quiz.model_dump(exclude={"sourceChunkId"})
            )
            for quiz in verified.quizzes
        ]
    )