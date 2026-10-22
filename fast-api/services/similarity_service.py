
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

    # 원본 학습 청크 조회
    chunk_map = {
        chunk.chunkId: chunk
        for chunk in material.chunks
    }

    passed = []
    failed = []
    candidates = []
    source_text_by_chunk_id = {}

    for quiz in quizzes:
        chunk = chunk_map.get(quiz.sourceChunkId)

        if chunk is None:
            failed.append({
                "question": quiz.quizQuestion,
                "reason": "출처 청크 없음",
                "similarity": 0.0,
            })
            continue

        if quiz.sourceChunkId not in source_text_by_chunk_id:
            source_text_by_chunk_id[quiz.sourceChunkId] = " ".join([
                chunk.topic,
                chunk.content,
                *chunk.keyPoints,
                *chunk.evidence,
            ])

        candidates.append((quiz, quiz.sourceChunkId))

    if not candidates:
        return passed, failed

    model = get_similarity_model()

    source_chunk_ids = list(source_text_by_chunk_id)
    source_texts = [
        source_text_by_chunk_id[chunk_id]
        for chunk_id in source_chunk_ids
    ]
    question_texts = [quiz.quizQuestion for quiz, _ in candidates]
    source_indexes = {
        chunk_id: index
        for index, chunk_id in enumerate(source_chunk_ids)
    }

    # 모든 고유 청크와 문제를 한 번에 임베딩한다.
    embeddings = model.encode(
        source_texts + question_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    source_count = len(source_texts)
    similarity_matrix = cos_sim(
        embeddings[source_count:],
        embeddings[:source_count],
    )

    for index, (quiz, chunk_id) in enumerate(candidates):
        score = float(
            similarity_matrix[index, source_indexes[chunk_id]].item()
        )

        logger.info(
            "문제 유사도 검사: chunk=%d, score=%.4f",
            chunk_id,
            score,
        )

        if score >= SIMILARITY_THRESHOLD:
            passed.append(quiz)
        else:
            failed.append({
                "question": quiz.quizQuestion,
                "sourceChunkId": chunk_id,
                "similarity": round(score, 4),
            })

    return passed, failed
