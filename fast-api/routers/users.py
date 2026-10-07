from fastapi import APIRouter, status
from pydantic import BaseModel, EmailStr, Field

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

class UserCreate(BaseModel):
    user_id: str = Field(..., min_length=4, max_length=20)
    password: str = Field(..., min_length=4, max_length=30)
    name: str = Field(..., min_length=2, max_length=20)
    age: int = Field(..., ge=1, le=120)
    email: EmailStr

@router.get("")
def get_users():
    return [
        {"id": 1, "name": "tess"},
        {"id": 2, "name": "xman"}
    ]

# 문자열 파라미터 경로는 중복 방지를 위해 라우팅 순서를 상단으로 이동하거나 경로 명시
@router.get("/hello/{name}")
def hello_name(name: str):
    return {
        "message": f"{name}님 안녕하세요"
    }

@router.get("/{user_id}")
def get_user(user_id: int):
    return {
        "user_id": user_id,
        "name": "tess bro"
    }

@router.get("/{user_id}/orders/{order_id}")
def get_order(user_id: int, order_id: int):
    return {
        "user_id": user_id,
        "order_id": order_id
    }

@router.post("", status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate):
    return {
        "message": "회원 등록 완료",
        "user": user
    }

@router.put("/{user_id}")
def update_user(user_id: int, user: UserCreate):
    return {
        "message": "회원 수정 완료",
        "user_id": user_id,
        "user": user
    }

@router.delete("/{user_id}")
def delete_user_by_id(user_id: int):
    return {
        "user_id": user_id,
        "message": "회원 삭제 완료"
    }

@router.delete("")
def delete_all_users():
    return {
        "message": "전체 회원 삭제 완료"
    }