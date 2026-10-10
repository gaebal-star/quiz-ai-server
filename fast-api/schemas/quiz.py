from typing import Literal
from pydantic import BaseModel, Field

QuizType = Literal['MULTIPLE_CHOICE', 'SHORT_ANSWER', 'OX']
Difficulty = Literal['EASY', 'MEDIUM', 'HARD']

class Quiz(BaseModel):
    quizQuestion: str = Field(min_length=1, max_length=3000)
    quizSelections: list[str] = Field(max_length=4)
    quizCorrectAnswer: str = Field(min_length=1, max_length=1000)
    quizExplanation: str = Field(min_length=1, max_length=3000)

class GeneratedQuiz(Quiz):
    # Python 내부에서만 사용. Java DTO로 보내지 않는다.
    sourceChunkId: int = Field(ge=1)

class GeneratedQuizResponse(BaseModel):
    quizzes: list[GeneratedQuiz] = Field(min_length=1, max_length=10)

class QuizResponse(BaseModel):
    quizzes: list[Quiz] = Field(min_length=1, max_length=10)
