from fastapi import APIRouter, HTTPException
from app.services.rag_service import RagAccounting

router = APIRouter()


@router.get("/ask")
async def ask_ai(question: str):
    """
    Endpoint RAG: Trả lời câu hỏi kế toán dựa trên Vector Search + LLM.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    answer = RagAccounting.rag_accounting(question)

    return {"message": answer}