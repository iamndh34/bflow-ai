from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.services.rag_service import RagAccounting


router = APIRouter(
    prefix="/accounting",
    tags=["Accounting AI"]
)


@router.get("/posting-engine/ask")
async def ask_accounting(
    question: str = Query(..., min_length=3)
):
    """Hoi dap ke toan (streaming response)"""
    return StreamingResponse(
        RagAccounting.ask(question),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/posting-engine/get_history")
async def get_history():
    """Lay lich su hoi thoai (5 cau hoi gan nhat)"""
    history = RagAccounting.get_context()
    return {
        "count": len(history),
        "history": history
    }


@router.post("/posting-engine/reset_history")
async def reset_history():
    """Xoa lich su hoi thoai"""
    RagAccounting.reset_context()
    return {"message": "History cleared"}
