from pydantic import BaseModel, Field


class StudyTaxonomy(BaseModel):
    categoryName: str = Field(min_length=1, max_length=100)
    subjectName: str = Field(min_length=1, max_length=100)


async def classify_study_material(ai, model, material, catalog, subject_id=None):
    """학습 자료와 기존 분류 목록을 바탕으로 카테고리·과목을 정한다."""
    if subject_id is not None:
        for category in catalog:
            for subject in category.get("subjects", []):
                if subject.get("subjectId") == subject_id:
                    return StudyTaxonomy(
                        categoryName=category["categoryName"],
                        subjectName=subject["subjectName"],
                    )
        raise ValueError("선택한 카테고리와 과목을 찾을 수 없습니다.")

    from google.genai import types

    response = await ai.models.generate_content(
        model=model,
        contents=[
            "학습 자료에 가장 적합한 카테고리와 과목을 정하세요. "
            "기존 목록에서 의미상 맞는 항목이 있으면 목록의 이름을 "
            "정확히 그대로 반환하고, 없을 때만 새 이름을 제안하세요. "
            "카테고리는 넓은 분야, 과목은 구체적인 주제입니다. "
            "이름은 한국어로 간결하게 작성하고 자료에 없는 분류를 "
            "추측하지 마세요. JSON 필드는 categoryName, subjectName입니다.\n"
            f"기존 분류 목록: {catalog}\n"
            f"학습 자료: {material.model_dump_json()}"
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StudyTaxonomy,
            temperature=0.1,
        ),
    )

    if not response.text:
        raise ValueError("카테고리·과목 분류 응답이 비어 있습니다.")

    taxonomy = StudyTaxonomy.model_validate_json(response.text)
    return StudyTaxonomy(
        categoryName=taxonomy.categoryName.strip(),
        subjectName=taxonomy.subjectName.strip(),
    )
