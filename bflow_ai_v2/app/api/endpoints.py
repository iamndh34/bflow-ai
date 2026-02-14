"""
API Endpoints for bflow_ai_v2 - COA Agent with Corrective RAG

Compatible với bflow_ai UI: /api/ai-bflow/ask

Implement Corrective RAG workflow:
1. retrieve → 2. generate_draft → 3. grade_answer → END
   ↑                                      │
   └──────────────── rewrite_query ←───────┘
"""
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel

from ..agents.coa_langgraph import create_coa_app, CorrectiveRAGState

router = APIRouter(tags=["COA"])


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    chat_type: Optional[str] = "thinking"
    item_group: Optional[str] = "GOODS"
    partner_group: Optional[str] = "CUSTOMER"


@router.post("/ai-bflow/ask")
async def ai_bflow_ask(request: AskRequest):
    """
    Truy vấn tài khoản kế toán với Corrective RAG streaming.

    Flow:
    1. 🔍 retrieve: Lấy tài khoản từ COA index
    2. 🤖 generate_draft: Sinh câu trả lời từ documents
    3. 📊 grade_answer: Đánh giá chất lượng
    4. ✍️ rewrite_query (nếu cần): Viết lại query và loop
    """
    graph = create_coa_app()

    # Initial state theo CorrectiveRAGState
    initial_state: CorrectiveRAGState = {
        "messages": [],
        "query": request.question,
        "rewritten_query": "",
        "documents": [],
        "answer": "",
        "confidence": 0.0,
        "retry_count": 0,
        "needs_rewrite": False,
    }

    async def generate():
        """Generator để stream token-by-token"""
        try:
            # Run graph với streaming
            config = {"configurable": {"thread_id": request.session_id or "default"}}

            final_state = None
            async for chunk in graph.astream(initial_state, config):
                for node_name, node_state in chunk.items():
                    final_state = node_state

                    # Stream intermediate steps character-by-character
                    if node_name == "retrieve":
                        msg = "\n🔍 Đang tìm tài khoản...\n"
                        for char in msg:
                            yield char
                            await asyncio.sleep(0.01)
                    elif node_name == "generate_draft":
                        msg = "\n🤖 Đang sinh câu trả lời...\n"
                        for char in msg:
                            yield char
                            await asyncio.sleep(0.01)
                    elif node_name == "grade_answer":
                        confidence = node_state.get("confidence", 0)
                        msg = f"\n📊 Confidence: {confidence:.2f}\n"
                        for char in msg:
                            yield char
                            await asyncio.sleep(0.01)
                    elif node_name == "rewrite_query":
                        retry = node_state.get("retry_count", 0)
                        msg = f"\n✍️  Đã rewrite query (lần {retry})...\n"
                        for char in msg:
                            yield char
                            await asyncio.sleep(0.01)

            # Stream final answer token-by-token
            if final_state and final_state.get("answer"):
                answer = final_state["answer"]

                # Simple tokenization by character (Việt Unicode)
                for i, char in enumerate(answer):
                    yield char
                    if i % 3 == 0:  # Small delay every few chars
                        await asyncio.sleep(0.005)
            else:
                yield "Không tìm thấy tài khoản phù hợp."

        except Exception as e:
            yield f"\n\n❌ **Lỗi:** {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/ai-bflow/health")
async def ai_bflow_health():
    """Health check endpoint"""
    return {"status": "ok", "service": "bflow_ai_v2", "version": "2.0.0", "architecture": "Corrective RAG"}
