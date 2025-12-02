from pydantic import BaseModel, Field
from typing import List, Optional

# Model dữ liệu trong DB (Document)
class DocumentSchema(BaseModel):
    title: str
    content: str
    tags: List[str] = []
    # embedding: List[float]  <-- Sau này sẽ thêm field này cho Vector Search

# Model cho Request khi người dùng hỏi
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3 # Lấy bao nhiêu kết quả liên quan nhất

# Model cho Response trả về
class AIResponse(BaseModel):
    answer: str
    sources: List[str]