from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, AIResponse, DocumentSchema
from app.services.rag_service import RAGService
from app.db.mongodb import get_database
from app.services.rag_service_test import RagAccounting

router = APIRouter()
rag_service = RAGService()


@router.get("/ask")
async def ask_ai(question: str):
    request = question
    # return await rag_service.generate_answer(request)
    answer_json = RagAccounting.rag_accounting(user_input=request)
    answer_text = RagAccounting.synthesize_answer(request, answer_json)
    return answer_text