from google.genai import types
from schemas.study import StudyMaterial
from prompts.study_prompt import STUDY_EXTRACTION_PROMPT
from services.validation_service import normalize_study_material

async def extract_study_material(ai, model, data, mime):
    result = await ai.models.generate_content(
        model=model, contents=[types.Part.from_bytes(data=data, mime_type=mime)],
        config=types.GenerateContentConfig(
            system_instruction=STUDY_EXTRACTION_PROMPT,
            response_mime_type='application/json', response_schema=StudyMaterial,
            temperature=0.1))
    if not result.text:
        raise ValueError('학습 내용 추출 응답이 비어 있습니다.')
    return normalize_study_material(StudyMaterial.model_validate_json(result.text))
