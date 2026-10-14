import os
import asyncio

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY")
)

async def main():

    response = await client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {
                "role": "system",
                "content": "당신은 학습 문제의 정답을 검증하는 AI입니다."
            },
            {
                "role": "user",
                "content": """
문제: Python에서 리스트의 길이를 구하는 함수는?

제시된 정답: len()

제시된 해설:
len() 함수는 리스트에 포함된 요소의 개수를 반환한다.

정답과 해설이 정확한지 검증하세요.
"""
            }
        ],
        temperature=0
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())