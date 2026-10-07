from typing import Literal
from pydantic import BaseModel, Field

class StudyChunk(BaseModel):
    chunkId: int = Field(gt=0)
    topic: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=5000)
    keyPoints: list[str] = Field(max_length=20)
    evidence: list[str] = Field(min_length=1, max_length=20)
    importance: Literal['LOW', 'MEDIUM', 'HIGH']

class StudyMaterial(BaseModel):
    title: str = Field(max_length=300)
    chunks: list[StudyChunk] = Field(min_length=1, max_length=30)
