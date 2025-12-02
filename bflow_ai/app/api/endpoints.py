from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, AIResponse, DocumentSchema
from app.services.rag_service import RAGService
from app.db.mongodb import get_database

router = APIRouter()
rag_service = RAGService()


@router.get("/ask")
async def ask_ai(question: str, top_k: int = 3):
    request = QueryRequest(question=question, top_k=top_k)
    return await rag_service.generate_answer(request)