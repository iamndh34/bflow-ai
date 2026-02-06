"""
Unified API Endpoint - Entry point cho tất cả modules
"""
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


router = APIRouter(tags=["BFLOW AI - Ask"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Câu hỏi")
    session_id: str | None = Field(None, description="Session ID")
    chat_type: str = Field("thinking", description="Chế độ: 'thinking' hoặc 'free'")
    item_group: str = Field("GOODS", description="Nhóm sản phẩm")
    partner_group: str = Field("CUSTOMER", description="Nhóm đối tác")


@router.post("/ask")
async def ask(
    request: AskRequest,
    x_user_id: str = Header(..., alias="X-User-Id", description="User ID từ header")
):
    from fastapi import HTTPException
    from app.pipeline import get_module_router

    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header is required")

    router = get_module_router()

    return StreamingResponse(
        router.route_and_process(
            question=request.question,
            user_id=x_user_id,
            session_id=request.session_id,
            chat_type=request.chat_type,
            item_group=request.item_group,
            partner_group=request.partner_group,
        ),
        media_type="text/plain; charset=utf-8"
    )
