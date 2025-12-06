from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from app.services.rag_service import RagAccounting

app = FastAPI(title="Accounting AI Assistant")
router = APIRouter()

@router.get("/accounting-ask")
async def accounting_ask(question: str):
    """
    Endpoint RAG Streaming.
    - Input: ?question=...
    - Output: Dòng dữ liệu text (text/plain) trả về liên tục.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Mình có thể hỗ trợ gì cho bạn? Vui lòng nhập câu hỏi nhé!")

    print(f"📩 Nhận câu hỏi: {question}")

    # 1. Gọi hàm generator
    response_generator = RagAccounting.rag_accounting(question)

    # 2. Trả về StreamingResponse
    # Sử dụng 'text/plain' để client (console, postman, simple js) hiển thị text thô ngay lập tức.
    return StreamingResponse(response_generator, media_type="text/plain")
