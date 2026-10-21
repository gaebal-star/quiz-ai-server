from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator

# 정답을 숨기고 다시 푼 결과
class AnswerVerification(BaseModel):
    quizIndex: int = Field(ge=0)

    imageReadable: bool
    chunkFaithfulToImage: bool
    chunkSupportsAnswer: bool

    # 객관식/OX: 정답인 보기 번호. 0부터 시작.
    # 단답형: 빈 배열.
    correctOptionIndexes: list[int] = Field(max_length=4)

    # 단답형 검증 정답. 객관식/OX는 빈 문자열.
    shortAnswer: str = Field(max_length=1000)

    evidence: str = Field(max_length=3000)
    reason: str = Field(min_length=1, max_length=2000)


class AnswerVerificationResponse(BaseModel):
    results: list[AnswerVerification] = Field(
        min_length=1,
        max_length=10,
    )


# 정답 확인 후 해설을 검토한 결과
class ExplanationVerification(BaseModel):
    quizIndex: int = Field(ge=0)

    explanationSupported: bool
    explanationConsistentWithAnswer: bool

    reason: str = Field(min_length=1, max_length=2000)


class ExplanationVerificationResponse(BaseModel):
    results: list[ExplanationVerification] = Field(
        min_length=1,
        max_length=10,
    )

# Groq 텍스트 검증: 이미지 판독 여부는 판단하지 않습니다.
class GroqAnswerVerification(BaseModel):
    quizIndex: int = Field(ge=0, strict=True)
    chunkSupportsAnswer: bool
    correctOptionIndexes: list[int] = Field(max_length=4)
    shortAnswer: str = Field(max_length=1000)
    evidence: str | list[str] = Field(max_length=3000)
    reason: str = Field(min_length=1, max_length=2000)

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> str:
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            return "\n".join(
                item.strip() for item in value if item.strip()
            )

        raise ValueError(
            "evidence는 문자열 또는 문자열 배열이어야 합니다."
        )


class GroqAnswerVerificationResponse(BaseModel):
    results: list[GroqAnswerVerification] = Field(min_length=1, max_length=10)
