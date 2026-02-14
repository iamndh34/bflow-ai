"""
Session Management API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field


router = APIRouter(tags=["BFLOW AI - Sessions"])


class SessionInfo(BaseModel):
    id: str
    user_id: str
    title: str
    chat_type: str
    created_at: str
    updated_at: str
    message_count: int


class SessionDetail(BaseModel):
    id: str
    user_id: str
    title: str
    chat_type: str
    created_at: str
    updated_at: str
    history: list


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]
    total: int


class MessageResponse(BaseModel):
    message: str
    session_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    chat_type: str = Field("thinking", description="Chat type: 'thinking' hoặc 'free'")


def _get_manager(chat_type: str = "thinking"):
    from app.services.session_manager import get_session_manager
    return get_session_manager(chat_type)


@router.post("/users/{user_id}/sessions", response_model=MessageResponse)
async def create_session(
    user_id: str,
    request: CreateSessionRequest = None,
):
    if request is None:
        request = CreateSessionRequest()

    manager = _get_manager(request.chat_type)
    session_id = manager.create_session(user_id=user_id)

    return MessageResponse(
        message=f"Session created successfully",
        session_id=session_id
    )


@router.get("/users/{user_id}/sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: str,
    chat_type: str = Query("thinking"),
):
    manager = _get_manager(chat_type)
    sessions = manager.list_sessions(user_id=user_id)

    return SessionListResponse(
        sessions=[SessionInfo(**s) for s in sessions],
        total=len(sessions)
    )


@router.get("/users/{user_id}/sessions/{session_id}", response_model=SessionDetail)
async def get_session(
    user_id: str,
    session_id: str,
    chat_type: str = Query("thinking"),
):
    manager = _get_manager(chat_type)
    data = manager.get_session(session_id)

    if not data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return SessionDetail(**data)


@router.delete("/users/{user_id}/sessions/{session_id}", response_model=MessageResponse)
async def delete_session(
    user_id: str,
    session_id: str,
    chat_type: str = Query("thinking"),
):
    manager = _get_manager(chat_type)
    data = manager.get_session(session_id)

    if not data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    manager.delete_session(session_id)

    return MessageResponse(
        message=f"Session {session_id} deleted successfully",
        session_id=session_id
    )


@router.post("/users/{user_id}/sessions/{session_id}/clear", response_model=MessageResponse)
async def clear_session(
    user_id: str,
    session_id: str,
    chat_type: str = Query("thinking"),
):
    manager = _get_manager(chat_type)
    data = manager.get_session(session_id)

    if not data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    manager.clear_session(session_id)

    return MessageResponse(
        message=f"Session {session_id} history cleared successfully",
        session_id=session_id
    )


@router.post("/users/{user_id}/sessions/{session_id}/reload", response_model=SessionDetail)
async def reload_session(
    user_id: str,
    session_id: str,
    chat_type: str = Query("thinking"),
):
    manager = _get_manager(chat_type)
    data = manager.get_session(session_id)

    if not data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    manager.clear_session(session_id)
    data = manager.get_session(session_id)

    return SessionDetail(**data)
