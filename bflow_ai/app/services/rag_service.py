from app.db.mongodb import get_database
from app.models.schemas import QueryRequest


class RAGService:
    async def retrieve_context(self, query: str, limit: int):
        db = await get_database()
        collection = db["documents"]

        # --- LOGIC TÌM KIẾM ---
        # Ở đây dùng text search cơ bản của Mongo.
        # Nếu dùng Vector Search, bạn sẽ convert query -> embedding vector tại đây
        cursor = collection.find(
            {"$text": {"$search": query}}
        ).limit(limit)

        results = await cursor.to_list(length=limit)
        return results

    async def generate_answer(self, query_request: QueryRequest):
        # 1. Retrieve (Truy vấn dữ liệu liên quan từ Mongo)
        context_docs = await self.retrieve_context(query_request.question, query_request.top_k)

        if not context_docs:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy dữ liệu liên quan trong database.",
                "sources": []
            }

        # Gom nội dung context
        context_text = "\n".join([doc['content'] for doc in context_docs])
        sources = [doc['title'] for doc in context_docs]

        # 2. Generate (Gửi context + câu hỏi cho LLM - Ví dụ giả lập)
        # Tại đây bạn sẽ gọi OpenAI / Gemini API / Local LLM
        # prompt = f"Dựa vào thông tin: {context_text}, hãy trả lời: {query_request.question}"

        # Giả lập câu trả lời AI
        ai_answer = f"Dựa trên dữ liệu về '{sources[0]}', câu trả lời cho '{query_request.question}' là: [Nội dung được tóm tắt từ DB...]"

        return {
            "answer": ai_answer,
            "sources": sources
        }