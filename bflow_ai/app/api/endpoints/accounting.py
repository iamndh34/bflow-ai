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
    question: str = Query(..., min_length=2, description="Câu hỏi về kế toán")
):
    """
    Hỏi đáp kế toán (streaming response)

    Tự động phân loại:
    - Nghiệp vụ/Bút toán -> Posting Engine
    - Tra cứu tài khoản -> COA
    """
    return StreamingResponse(
        RagRouter.ask(question),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/posting-engine/get_history")
async def get_history():
    """Lấy lịch sử hội thoại (reload từ file JSON, 10 câu gần nhất)"""
    history = RagRouter.reload_history()
    return {
        "count": len(history),
        "history": history
    }


@router.post("/posting-engine/reset_history")
async def reset_history():
    """Xóa lịch sử hội thoại (xóa file JSON)"""
    RagRouter.reset_history()
    return {"message": "History cleared"}
