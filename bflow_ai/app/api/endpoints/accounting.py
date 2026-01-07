from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.services.rag_service import RagRouter
from app.services.session_manager import get_session_manager


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
    chat_type: str = Query("thinking", description="Chế độ chat: 'thinking' hoặc 'free'"),
    session_id: str = Query(None, description="Session ID (tự tạo mới nếu không có)")
):
    """
    Hỏi đáp kế toán (streaming response)

    chat_type:
    - 'thinking': Tự động phân loại (COA, Posting Engine, Compare, etc.)
    - 'free': Chế độ tự do - SLM trả lời trực tiếp không qua phân loại

    session_id: ID của session, nếu không truyền sẽ tạo session mới
    """
    return StreamingResponse(
        RagRouter.ask(question, chat_type=chat_type, session_id=session_id),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/posting-engine/get_history")
async def get_history(
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'"),
    session_id: str = Query(None, description="Session ID")
):
    """Lấy lịch sử hội thoại của session."""
    if session_id:
        sm = get_session_manager(chat_type)
        history = sm.get_history(session_id)
        return {
            "session_id": session_id,
            "count": len(history),
            "history": history
        }
    # Fallback: dùng cách cũ nếu không có session_id
    history = RagRouter.reload_history(chat_type=chat_type)
    return {
        "count": len(history),
        "history": history
    }


@router.post("/posting-engine/reset_history")
async def reset_history(
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'"),
    session_id: str = Query(None, description="Session ID")
):
    """Xóa history của session hoặc history cũ."""
    if session_id:
        sm = get_session_manager(chat_type)
        sm.clear_session(session_id)
        return {"message": f"Session {session_id} cleared"}
    # Fallback: dùng cách cũ
    RagRouter.reset_history(chat_type=chat_type)
    return {"message": "10 recent items cleared"}


# =============================================================================
# SESSION MANAGEMENT APIs
# =============================================================================

@router.get("/sessions")
async def list_sessions(
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'")
):
    """Lấy danh sách tất cả sessions."""
    sm = get_session_manager(chat_type)
    sessions = sm.list_sessions()
    return {
        "chat_type": chat_type,
        "count": len(sessions),
        "sessions": sessions
    }


@router.post("/sessions/create")
async def create_session(
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'")
):
    """Tạo session mới."""
    sm = get_session_manager(chat_type)
    session_id = sm.create_session()
    return {
        "session_id": session_id,
        "chat_type": chat_type
    }


@router.delete("/sessions/delete")
async def delete_session(
    session_id: str = Query(..., description="Session ID cần xóa"),
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'")
):
    """Xóa session."""
    sm = get_session_manager(chat_type)
    success = sm.delete_session(session_id)
    if success:
        return {"message": f"Session {session_id} deleted"}
    return {"message": f"Session {session_id} not found", "success": False}


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'")
):
    """Lấy thông tin chi tiết của session."""
    sm = get_session_manager(chat_type)
    session = sm.get_session(session_id)
    if session:
        return session
    return {"error": "Session not found"}
