import asyncio
import hmac
import logging
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from common import MAX_IMAGE_BYTES
from schemas.quiz import QuizResponse, QuizType, Difficulty
from services.image_service import normalize_image
from services.pipeline_service import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/ai', tags=['quiz'])

@router.post('/generate-quiz', response_model=QuizResponse)
async def generate_quiz(
    request: Request,
    file: UploadFile = File(...),
    subject_id: int = Form(..., gt=0),
    quiz_type: QuizType = Form(...),
    quiz_difficulty: Difficulty = Form(...),
    quiz_count: int = Form(..., ge=1, le=10),
    quiz_prompt: str = Form('', max_length=1000),
    x_ai_token: str | None = Header(default=None),
):
    settings = request.app.state.settings
    try:
        if settings.token and not hmac.compare_digest(settings.token.encode(), (x_ai_token or '').encode()):
            raise HTTPException(401, 'AI 서버 인증에 실패했습니다.')
        data = await file.read(MAX_IMAGE_BYTES + 1)
        declared_mime = file.content_type or ''
    finally:
        await file.close()
    if not data:
        raise HTTPException(400, '이미지가 비어 있습니다.')
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, '10MB 이하 이미지를 선택해주세요.')
    try:
        data, mime = await run_in_threadpool(normalize_image, data, declared_mime)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        # Spring 기본 read timeout 180초보다 짧은 전체 AI 처리 제한.
        async with asyncio.timeout(160):
            return await run_pipeline(request.app.state.ai, settings.model, data, mime,
                                      subject_id, quiz_type, quiz_difficulty, quiz_count, quiz_prompt)
    except TimeoutError as exc:
        raise HTTPException(504, 'AI 생성 시간이 초과되었습니다. 다시 시도해주세요.') from exc
    except ValueError as exc:
        logger.warning('AI 응답 검증 실패')
        raise HTTPException(502, '자료가 부족하거나 AI 응답이 올바르지 않습니다. 다른 이미지로 시도해주세요.') from exc
    except Exception as exc:
        logger.exception('AI 서버 요청 실패')
        raise HTTPException(502, 'AI 연결, 모델, API 키 또는 사용 한도를 확인해주세요.') from exc
