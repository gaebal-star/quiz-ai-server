from schemas.quiz import Quiz, QuizResponse
from services.study_service import extract_study_material
from services.quiz_service import generate_quizzes

async def run_pipeline(ai, model, data, mime, subject_id, quiz_type, difficulty, count, extra):
    material = await extract_study_material(ai, model, data, mime)
    generated = await generate_quizzes(ai, model, material, subject_id, quiz_type, difficulty, count, extra)
    # Java AiQuizResponseDto와 정확히 일치하는 4개 필드만 반환한다.
    return QuizResponse(quizzes=[Quiz.model_validate(q.model_dump(exclude={'sourceChunkId'}))
                                 for q in generated.quizzes])
