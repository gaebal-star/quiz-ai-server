
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

logger = logging.getLogger("uvicorn.error")

SIMILARITY_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def get_similarity_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def validate_similarity(material, quizzes):
    """
    각 문제의 sourceChunkId를 사용하여
    해당 학습 내용과 문제의 유사도를 검사한다.
    """

    model = get_similarity_model()

    # 원본 학습 청크 조회
    chunk_map = {
        chunk.chunkId: chunk
        for chunk in material.chunks
    }

    passed = []
    failed = []

    for quiz in quizzes:

        chunk = chunk_map.get(quiz.sourceChunkId)

        if chunk is None:
            failed.append({
                "question": quiz.quizQuestion,
                "reason": "출처 청크 없음",
                "similarity": 0.0,
            })
            continue

        # 학습 내용 구성
        source_text = " ".join([
            chunk.topic,
            chunk.content,
            *chunk.keyPoints,
            *chunk.evidence,
        ])

        # 생성 문제
        question_text = quiz.quizQuestion

        # 임베딩 생성
        embeddings = model.encode(
            [source_text, question_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # 코사인 유사도 계산
        score = float(
            cos_sim(
                embeddings[0],
                embeddings[1],
            ).item()
        )

        logger.info(
            "문제 유사도 검사: chunk=%d, score=%.4f",
            quiz.sourceChunkId,
            score,
        )

        if score >= SIMILARITY_THRESHOLD:
            passed.append(quiz)
        else:
            failed.append({
                "question": quiz.quizQuestion,
                "sourceChunkId": quiz.sourceChunkId,
                "similarity": round(score, 4),
            })

    return passed, failed
