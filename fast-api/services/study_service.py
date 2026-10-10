from google.genai import types

from prompts.study_prompt import STUDY_EXTRACTION_PROMPT
from schemas.study import StudyMaterial
from services.validation_service import normalize_study_material


async def extract_study_material(ai, model, data, mime):
    result = await ai.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            (
                "이미지의 학습 내용만 추출하세요. JSON 객체로 답하세요. "
                "최상위 필드는 title과 chunks입니다. "
                "각 chunk에는 chunkId(1부터 시작하는 정수), topic, content, "
                "keyPoints(문자열 배열), evidence(문자열 배열), "
                "importance(LOW, MEDIUM, HIGH 중 하나)를 넣으세요. "
                "읽을 수 있는 학습 내용이 없다면 chunks를 빈 배열로 반환하세요."
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