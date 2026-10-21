import logging
import os
from dataclasses import dataclass

import numpy as np
from schemas.study import StudyMaterial
from services.similarity_service import get_similarity_model

"""요청별 메모리 벡터 검색 레그"""

logger = logging.getLogger('uvicorn.error')


def chunk_text(chunk) -> str:
    return '\n'.join(filter(None, [
        f'주제: {chunk.topic}',
        f'내용: {chunk.content}',
        '핵심: ' + '; '.join(chunk.keyPoints),
        '근거: ' + '; '.join(chunk.evidence),
    ]))


@dataclass
class RagIndex:
    material: StudyMaterial
    vectors: np.ndarray

    @classmethod
    def build(cls, material: StudyMaterial):
        if not material.chunks:
            raise ValueError('벡터화할 학습 청크가 없습니다.')
        texts = [chunk_text(chunk) for chunk in material.chunks]
        vectors = get_similarity_model().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cls(material=material, vectors=np.asarray(vectors, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 12) -> StudyMaterial:
        """유사도 순위로 청크 검색 """
        top_k = max(1, min(top_k, len(self.material.chunks)))
        if not query.strip():
            raise ValueError('RAG 검색어가 비어 있습니다.')
        query_vector = get_similarity_model().encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        selected = ranked[:top_k]
        logger.info('RAG 검색 완료: 전체=%d, 선택=%d, 최고 유사도=%.4f',
                    len(scores), len(selected), float(scores[ranked[0]]))
        return StudyMaterial(title=self.material.title,
                             sourceLanguage=self.material.sourceLanguage,
                             chunks=[self.material.chunks[i] for i in selected])


def build_rag_query(material: StudyMaterial, extra: str, subject_id: int) -> str:
    # 사용자 요청이 있다면 우선 반영. 없으면 이미지에서 추출한 주제/핵심어로 검색.
    topics = ' '.join(chunk.topic for chunk in material.chunks[:15])
    subject_context = f'과목 ID: {subject_id}' if subject_id else ''
    return f'{material.title}\n{extra.strip()}\n{topics}\n{subject_context}'.strip()


def configured_top_k() -> int:
    try:
        return max(1, min(30, int(os.getenv('RAG_TOP_K', '12'))))
    except ValueError as exc:
        raise ValueError('RAG_TOP_K는 1~30 사이 정수여야 합니다.') from exc
