from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.services.rag_service import RagRouter


router = APIRouter(
    prefix="/accounting",
    tags=["Accounting AI"]
)


# =============================================================================
# UNIFIED ENDPOINT - Tự động phân loại: Posting Engine hoặc COA
# =============================================================================

@router.get("/posting-engine/ask")
async def ask_accounting(
    question: str = Query(..., min_length=2, description="Câu hỏi về kế toán"),
    chat_type: str = Query("thinking", description="Chế độ chat: 'thinking' (phân loại thông minh) hoặc 'free' (tự do)")
):
    """
    Hỏi đáp kế toán (streaming response)

    chat_type:
    - 'thinking': Tự động phân loại (COA, Posting Engine, Compare, etc.)
    - 'free': Chế độ tự do - SLM trả lời trực tiếp không qua phân loại
    """
    return StreamingResponse(
        RagRouter.ask(question, chat_type=chat_type),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/posting-engine/get_history")
async def get_history(
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'")
):
    """Lấy lịch sử hội thoại (reload từ file JSON, 10 câu gần nhất)"""
    history = RagRouter.reload_history(chat_type=chat_type)
    return {
        "count": len(history),
        "history": history
    }


@router.post("/posting-engine/reset_history")
async def reset_history(
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'")
):
    """Xóa 10 câu hỏi gần nhất, giữ lại phần còn lại"""
    RagRouter.reset_history(chat_type=chat_type)
    return {"message": "10 recent items cleared"}
