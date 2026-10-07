from common import get_genai_client, GEMINI_MODEL

client = get_genai_client()

print("현재 모델:", GEMINI_MODEL)

try:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents="안녕하세요. 테스트입니다. '연결 성공'이라고 짧게 답해주세요.",
    )

    print("===== Gemini 응답 =====")
    print(response.text)

except Exception as e:
    print("===== Gemini 오류 =====")
    print(type(e).__name__)
    print(e)