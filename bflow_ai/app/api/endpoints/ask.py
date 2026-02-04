"""
Unified API Endpoint - Entry point cho tất cả modules

UI chỉ cần gọi: /api/ai-bflow/ask

Backend sẽ tự động route đến module phù hợp (Accounting, HR, CRM, etc)
"""
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse


router = APIRouter(
    tags=["BFLOW AI - Unified Endpoint"]
)


@router.get("/ask")
async def unified_ask(
    question: str = Query(..., min_length=2, description="Câu hỏi"),
    session_id: str = Query(None, description="Session ID"),
    chat_type: str = Query("thinking", description="Chế độ: 'thinking' hoặc 'free'"),
    item_group: str = Query("GOODS", description="Nhóm sản phẩm"),
    partner_group: str = Query("CUSTOMER", description="Nhóm đối tác"),

    # Development flags (bypass các bước)
    turn_off_routing: bool = Query(False, description="Bỏ routing"),
    turn_off_history: bool = Query(False, description="Bỏ history check"),
    turn_off_cache: bool = Query(False, description="Bỏ cache check"),
    turn_off_llm: bool = Query(False, description="Bỏ LLM (mock response)"),
):
    """
    Unified Entry Point cho tất cả modules.

    Flow:
        1. Module Router phân loại câu hỏi (Accounting, HR, CRM, etc)
        2. Route đến pipeline của module tương ứng
        3. Trả về streaming response

    UI chỉ cần gọi endpoint này, không cần biết module nào xử lý.

    Returns:
        Streaming response
    """
    from app.pipeline import get_module_router

    router = get_module_router()

    return StreamingResponse(
        router.route_and_process(
            question=question,
            session_id=session_id,
            chat_type=chat_type,
            item_group=item_group,
            partner_group=partner_group,
            # Flags
            turn_off_routing=turn_off_routing,
            turn_off_history_check=turn_off_history,
            turn_off_cache=turn_off_cache,
            turn_off_llm=turn_off_llm,
        ),
        media_type="text/plain; charset=utf-8"
    )


