import json
import re
import ollama
from app.core.config import settings

# Import 2 module con
from .rag_coa import RagCOA
from .rag_posting_engine import RagPostingEngine, HISTORY_MANAGER

# =============================================================================
# RAG ROUTER
# =============================================================================

CLASSIFICATION_PROMPT = """Phân loại câu hỏi kế toán.

POSTING_ENGINE: Hỏi về nghiệp vụ, cách hạch toán, định khoản, bút toán, quy trình (Ví dụ: "Bán hàng hạch toán sao?", "Nhập kho định khoản thế nào?")
COA: Hỏi về thông tin tài khoản, tra cứu số TK, tên TK, bản chất tài khoản (Ví dụ: "TK 156 là gì?", "Tài khoản tiền mặt số mấy?")
COMPARE: So sánh sự khác biệt giữa TT200 và TT99 (Ví dụ: "So sánh TK 112 giữa TT200 và TT99", "TK 156 khác gì giữa 2 thông tư?")

Câu hỏi: {question}"""

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["COA", "POSTING_ENGINE", "COMPARE"]
        }
    },
    "required": ["label"]
}

class RagRouter:
    """
    Router trung tâm: Phân loại intent -> Điều hướng đến module xử lý.
    """

    @classmethod
    def classify(cls, question: str) -> str:
        """
        Phân loại câu hỏi bằng SLM (structured output) hoặc Fallback Rule
        """
        print(f"[Router] Analyzing: {question}")

        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            response = client.chat(
                model="qwen2.5:3b",
                messages=[
                    {"role": "user", "content": CLASSIFICATION_PROMPT.format(question=question)}
                ],
                format=CLASSIFICATION_SCHEMA,
                stream=False
            )
            content = response.get("message", {}).get("content", "")
            result = json.loads(content)
            label = result.get("label", "POSTING_ENGINE")
            print(f"[Router] SLM Label: {label}")
            return label

        except Exception as e:
            print(f"[Router] SLM Error: {e}. Using fallback rules.")
            return cls._fallback_classify(question)

    @classmethod
    def _fallback_classify(cls, question: str) -> str:
        """Fallback classification bằng Regular Expression"""
        question_lower = question.lower()

        # Rule 0: So sánh TT200 vs TT99 -> COMPARE
        compare_keywords = ["so sánh", "khác gì", "khác nhau", "tt200", "tt99", "thông tư 200", "thông tư 99"]
        if any(kw in question_lower for kw in compare_keywords):
            # Cần có số TK để so sánh
            if re.search(r'\b\d{3,5}\b', question_lower):
                return "COMPARE"

        # Rule 1: Hỏi đích danh số tài khoản -> COA
        if re.search(r'\b(tk|tài khoản)\s*\d{3,5}\b', question_lower):
            return "COA"
        # Rule 2: Hỏi định nghĩa "là gì" kèm con số -> COA
        if re.search(r'\b\d{3,5}\b.*(là gì|là gì\?)', question_lower):
            return "COA"
        # Rule 3: Hỏi danh sách/nhóm -> COA
        if any(kw in question_lower for kw in ["danh sách", "các loại", "nhóm tài khoản", "hệ thống tài khoản"]):
            return "COA"

        # Default: Các câu hỏi khác (hạch toán, bút toán, nghiệp vụ...) -> POSTING_ENGINE
        return "POSTING_ENGINE"

    @classmethod
    def ask(cls, question: str, item_group: str = "GOODS", partner_group: str = "CUSTOMER"):
        """
        Unified endpoint - API sẽ gọi hàm này.
        """
        # 1. Phân loại
        category = cls.classify(question)
        print(f"[Router] Routing to -> {category}")

        # 2. Điều hướng và thu thập response
        full_response = ""

        if category == "COMPARE":
            for chunk in RagCOA.compare(question):
                full_response += chunk
                yield chunk
            # Lưu history cho COMPARE
            HISTORY_MANAGER.add(question, full_response, "COMPARE")

        elif category == "COA":
            for chunk in RagCOA.ask(question):
                full_response += chunk
                yield chunk
            # Lưu history cho COA
            HISTORY_MANAGER.add(question, full_response, "COA")

        else:
            # POSTING_ENGINE tự lưu history trong method của nó
            for chunk in RagPostingEngine.ask(question, item_group, partner_group):
                yield chunk

    @staticmethod
    def reset_history():
        """Reset history (xóa file JSON)"""
        HISTORY_MANAGER.clear()

    @staticmethod
    def get_history():
        """Get current history"""
        return HISTORY_MANAGER.history

    @staticmethod
    def reload_history():
        """Reload history từ file JSON"""
        HISTORY_MANAGER.reload()
        return HISTORY_MANAGER.history