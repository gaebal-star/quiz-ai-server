from common import get_genai_client

client = get_genai_client()

TEST_MODEL = "gemini-3.5-flash-lite"

print("테스트 모델:", TEST_MODEL)

try:
    response = client.models.generate_content(
        model=TEST_MODEL,
        contents="테스트입니다. 연결 성공이라고 짧게 답해주세요.",
    )

    print("===== Gemini 응답 =====")
    print(response.text)

except Exception as e:
    print("===== Gemini 오류 =====")
    print(type(e).__name__)
    print(e)