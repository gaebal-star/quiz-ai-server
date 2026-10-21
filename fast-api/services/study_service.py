from google.genai import types

from prompts.study_prompt import STUDY_EXTRACTION_PROMPT
from schemas.study import StudyMaterial
from services.validation_service import normalize_study_material


async def extract_study_material(ai, model, data, mime):
    result = await ai.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            (   "원본 이미지의 언어를 그대로 유지하세요. "
                "영어 지문은 영어로 추출하고 한국어로 번역하지 마세요. "
                "sourceLanguage에는 실제 설명 문장의 주된 언어를 적으세요. "
                "수식, 도형 기호, 변수, 숫자, 단독 전문용어만 보이면 "
                "sourceLanguage는 UNKNOWN으로 적으세요. "
                "설명 문장이 분명한 외국어면 해당 언어 이름을 영어로 적으세요. "
                "이미지의 학습 내용만 추출하세요. JSON 객체로 답하세요. "
                "최상위 필드는 title, sourceLanguage, chunks입니다. "
                "기하학/수학의 읽을 수 있는 도형 표기, 숫자, 공식, 조건은 "
                "자연어 설명이 없어도 학습 내용이므로 chunks에 넣으세요. "
                "각 chunk에는 chunkId(1부터 시작하는 정수), topic, content, "
                "keyPoints(문자열 배열), evidence(문자열 배열), "
                "importance(LOW, MEDIUM, HIGH 중 하나)를 넣으세요. "
                "읽을 수 있는 글자, 도형 표기, 숫자, 공식이 모두 없을 때만 "
                "chunks를 빈 배열로 반환하세요."
            ),
        ],
        config=types.GenerateContentConfig(
            system_instruction=STUDY_EXTRACTION_PROMPT,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    if not result.text:
        raise ValueError("학습 내용 추출 응답이 비어 있습니다.")

    material = StudyMaterial.model_validate_json(result.text)
    return normalize_study_material(material)
