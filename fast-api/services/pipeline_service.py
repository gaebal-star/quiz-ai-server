import logging
import re
import unicodedata

from schemas.quiz import Quiz, QuizResponse
from services.study_service import extract_study_material
from services.language_service import detect_output_language
from services.rag_service import (
    RagIndex,
    build_rag_query,
    configured_top_k,
)
from starlette.concurrency import run_in_threadpool
from services.quiz_service import generate_quizzes
from services.verification_service import verify_quiz_content
from services.taxonomy_service import classify_study_material

logger = logging.getLogger("uvicorn.error")

# 최초 생성 이후 추가 재생성 최대 횟수
MAX_REGENERATION_ATTEMPTS = 1


def _question_key(question: str) -> str:
    """
    생성 문제의 중복 확인용 정규화.
    공백과 유니코드를 정리합니다.
    """
    normalized = unicodedata.normalize(
        "NFC",
        question,
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip().casefold()


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
    catalog,
):
    # =====================================================
    # 1. 이미지에서 학습 자료 추출
    # =====================================================

    material = await extract_study_material(
        ai,
        model,
        data,
        mime,
    )

    if not material.chunks:
        raise ValueError(
            "이미지에서 학습 내용을 추출하지 못했습니다."
        )

    logger.info(
        "이미지 학습 자료 추출 완료: 청크 수=%d",
        len(material.chunks),
    )

    output_language = detect_output_language(material)
    logger.info(
        "원문 언어 판별: %s, 문제 출력 언어 선택: %s",
        material.sourceLanguage,
        output_language,
    )

    taxonomy = await classify_study_material(
        ai,
        model,
        material,
        catalog,
        subject_id,
    )
    logger.info(
    "추출된 원문 내용: %s",
    material.chunks[0].content[:500],
    )
    # =====================================================
    # 2. 이미지 청크 임베딩 및 RAG 검색
    # =====================================================

    index = await run_in_threadpool(
        RagIndex.build,
        material,
    )

    retrieved_material = await run_in_threadpool(
        index.retrieve,
        build_rag_query(
            material,
            f"{taxonomy.categoryName} {taxonomy.subjectName} {extra}",
            subject_id,
        ),
        configured_top_k(),
    )

    if not retrieved_material.chunks:
        raise ValueError(
            "RAG 검색 결과가 없습니다."
        )

    logger.info(
        "RAG 컨텍스트 검색 완료: 전체=%d, 검색=%d",
        len(material.chunks),
        len(retrieved_material.chunks),
    )

    # =====================================================
    # 3. 문제 생성 및 Groq 검증
    # =====================================================

    # 각 위치에 최종 승인된 문제를 보관합니다.
    # 예: [1번 문제, 2번 문제, None]
    approved_quizzes = [None] * count

    # 통과한 문제의 중복을 방지하기 위한 집합
    approved_question_keys = set()

    # 0: 최초 생성
    # 1~3: 실패 문항 재생성
    for attempt in range(
        MAX_REGENERATION_ATTEMPTS + 1
    ):

        # 아직 통과하지 못한 문제 위치
        pending_indexes = [
            index
            for index, quiz in enumerate(
                approved_quizzes
            )
            if quiz is None
        ]

        # 모든 문제가 검증을 통과하면 종료
        if not pending_indexes:
            break

        pending_count = len(pending_indexes)

        if attempt == 0:
            logger.info(
                "AI 최초 문제 생성 시작: 요청=%d",
                pending_count,
            )
        else:
            logger.info(
                "AI 실패 문항 재생성 시작: "
                "시도=%d/%d, 대상=%d문항",
                attempt,
                MAX_REGENERATION_ATTEMPTS,
                pending_count,
            )

        # -------------------------------------------------
        # 3-1. 아직 필요한 문항만 생성
        # generate_quizzes 내부의 유사도 검사 재사용
        # -------------------------------------------------

        generated = await generate_quizzes(
            ai,
            model,
            retrieved_material,
            subject_id,
            quiz_type,
            difficulty,
            pending_count,
            extra,
            output_language,
        )

        if len(generated.quizzes) != pending_count:
            raise ValueError(
                "AI 생성 문항 수가 요청 수와 다릅니다."
            )

        logger.info(
            "문제 생성 및 유사도 검증 통과: "
            "문항 수=%d, 생성 시도=%d",
            len(generated.quizzes),
            attempt + 1,
        )

        # -------------------------------------------------
        # 3-2. Groq 독립 AI 정답·해설 검증
        # -------------------------------------------------

        verification = await verify_quiz_content(
            ai=ai,
            model=model,
            data=data,
            mime=mime,
            material=retrieved_material,
            generated=generated,
            quiz_type=quiz_type,
        )

        passed_indexes = verification[
            "passed_indexes"
        ]

        failed_indexes = verification[
            "failed_indexes"
        ]

        failure_reasons = verification[
            "failure_reasons"
        ]

        logger.info(
            "AI 검증 결과: 생성=%d, 통과=%d, 실패=%d",
            len(generated.quizzes),
            len(passed_indexes),
            len(failed_indexes),
        )

        # -------------------------------------------------
        # 3-3. 검증 통과 문항만 저장
        # -------------------------------------------------

        # 이번 생성 묶음 안에서도 중복을 검사
        batch_question_keys = set()

        # pending_indexes의 순서대로
        # 생성된 문제의 원래 위치를 복원
        for local_index in passed_indexes:

            quiz = generated.quizzes[
                local_index
            ]

            question_key = _question_key(
                quiz.quizQuestion
            )

            # 빈 문제 방지
            if not question_key:
                logger.warning(
                    "빈 문제 제외: 생성 인덱스=%d",
                    local_index,
                )
                continue

            # 기존 통과 문제와 중복 방지
            if (
                question_key in approved_question_keys
                or question_key in batch_question_keys
            ):
                logger.warning(
                    "중복 문제 제외: %s",
                    quiz.quizQuestion,
                )
                continue

            # 생성 순서에 대응하는 최종 위치
            original_index = pending_indexes[
                local_index
            ]

            approved_quizzes[
                original_index
            ] = quiz

            approved_question_keys.add(
                question_key
            )

            batch_question_keys.add(
                question_key
            )

            logger.info(
                "AI 문제 최종 승인: "
                "원래 문항=%d",
                original_index + 1,
            )

        # -------------------------------------------------
        # 3-4. 실패 문항 로그
        # -------------------------------------------------

        for local_index in failed_indexes:

            original_index = pending_indexes[
                local_index
            ]

            reason = failure_reasons.get(
                local_index,
                "검증 실패",
            )

            logger.warning(
                "AI 문제 재생성 대상: "
                "원래 문항=%d, 이유=%s",
                original_index + 1,
                reason,
            )

        remaining = sum(
            quiz is None
            for quiz in approved_quizzes
        )

        logger.info(
            "AI 문제 승인 현황: 전체=%d, 승인=%d, 남음=%d",
            count,
            count - remaining,
            remaining,
        )

    # =====================================================
    # 4. 최종 검증 결과 확인
    # =====================================================

    remaining_indexes = [
        index
        for index, quiz in enumerate(
            approved_quizzes
        )
        if quiz is None
    ]

    if remaining_indexes:
        failed_numbers = [
            index + 1
            for index in remaining_indexes
        ]

        logger.error(
            "AI 문제 최종 생성 실패: "
            "미통과 문항=%s, 최대 재생성=%d회",
            failed_numbers,
            MAX_REGENERATION_ATTEMPTS,
        )

        raise ValueError(
            "AI 정답·해설 검증을 통과한 문제를 "
            "요청 수만큼 확보하지 못했습니다. "
            f"미통과 문항: {failed_numbers}"
        )

    # None이 남지 않았음을 확인한 뒤 반환
    verified_quizzes = [
        quiz
        for quiz in approved_quizzes
        if quiz is not None
    ]

    if len(verified_quizzes) != count:
        raise ValueError(
            "최종 검증 문항 수가 일치하지 않습니다."
        )

    logger.info(
        "정답·해설 검증 최종 통과: 문항 수=%d",
        len(verified_quizzes),
    )

    # =====================================================
    # 5. Spring Boot 반환
    # sourceChunkId는 Python 내부에서만 사용
    # =====================================================

    return QuizResponse(
        categoryName=taxonomy.categoryName,
        subjectName=taxonomy.subjectName,
        quizzes=[
            Quiz.model_validate(
                quiz.model_dump(
                    exclude={
                        "sourceChunkId"
                    }
                )
            )
            for quiz in verified_quizzes
        ]
    )
