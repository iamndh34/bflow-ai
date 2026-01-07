#   Question → SLM classify
#                 ↓
#       ┌─────────┴─────────┬──────────┬────────────────┬──────────────────┬───────────────┐
#       ↓                   ↓          ↓                ↓                  ↓               ↓
#      COA           POSTING_ENGINE  COMPARE    COMPARE_CIRCULAR   GENERAL_ACCOUNTING  GENERAL_FREE
#       ↓                   ↓          ↓                ↓                  ↓               ↓
#    RagCOA.ask()    RagPE.ask()   compare()   compare_circular()    SLM (kế toán)    SLM (tự do)

import json
import ollama
from app.core.config import settings

# Import modules
from .rag_coa import RagCOA
from .rag_posting_engine import RagPostingEngine
from .history_manager import HISTORY_MANAGER

# =============================================================================
# RAG ROUTER
# =============================================================================

CLASSIFICATION_PROMPT = """Phân loại câu hỏi.

POSTING_ENGINE: Hỏi về nghiệp vụ, cách hạch toán, định khoản, bút toán, quy trình (Ví dụ: "Bán hàng hạch toán sao?", "Nhập kho định khoản thế nào?")
COA: Hỏi về thông tin tài khoản, tra cứu số TK, tên TK, bản chất tài khoản (Ví dụ: "TK 156 là gì?", "Tài khoản tiền mặt số mấy?")
COMPARE: So sánh một tài khoản cụ thể giữa TT200 và TT99 - CẦN CÓ SỐ TÀI KHOẢN (Ví dụ: "So sánh TK 112 giữa TT200 và TT99", "TK 156 khác gì giữa 2 thông tư?")
COMPARE_CIRCULAR: So sánh TỔNG QUAN giữa các thông tư, KHÔNG CÓ số tài khoản cụ thể (Ví dụ: "Điểm khác biệt giữa TT99 và TT200", "TT99 có gì mới so với TT200?")
GENERAL_ACCOUNTING: Câu hỏi về kế toán nhưng KHÔNG thuộc các loại trên (Ví dụ: "Nguyên tắc kế toán là gì?", "Báo cáo tài chính gồm những gì?", "Khấu hao là gì?")
GENERAL_FREE: Câu hỏi KHÔNG liên quan đến kế toán (Ví dụ: "Thời tiết hôm nay?", "Thủ đô Việt Nam là gì?", "Hello")

Câu hỏi: {question}"""

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["COA", "POSTING_ENGINE", "COMPARE", "COMPARE_CIRCULAR", "GENERAL_ACCOUNTING", "GENERAL_FREE"]
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
        """Fallback khi SLM classify fail - để SLM tự trả lời về kế toán"""
        return "GENERAL_ACCOUNTING"

    @classmethod
    def _general_accounting_answer(cls, question: str):
        """SLM trả lời câu hỏi kế toán tổng quát (dựa trên kiến thức SLM)"""
        print(f"[GENERAL_ACCOUNTING] Question: {question}")

        system_prompt = """Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT.
Trả lời các câu hỏi về kế toán dựa trên kiến thức chuyên môn của bạn.
Trình bày rõ ràng, có cấu trúc, dễ hiểu."""

        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            stream = client.chat(
                model="qwen2.5:3b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                stream=True
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            print(f"[GENERAL_ACCOUNTING Error] {e}")
            yield "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."

    @classmethod
    def _general_free_answer(cls, question: str):
        """SLM trả lời câu hỏi tự do (không liên quan kế toán)"""
        print(f"[GENERAL_FREE] Question: {question}")

        system_prompt = """Bạn là trợ lý AI thông minh. LUÔN trả lời bằng TIẾNG VIỆT.
Trả lời ngắn gọn, chính xác theo kiến thức của bạn."""

        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            stream = client.chat(
                model="qwen2.5:3b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                stream=True
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            print(f"[GENERAL_FREE Error] {e}")
            yield "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."

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

        if category == "COMPARE_CIRCULAR":
            for chunk in RagCOA.compare_circular(question):
                full_response += chunk
                yield chunk
            HISTORY_MANAGER.add(question, full_response, "COMPARE_CIRCULAR")

        elif category == "COMPARE":
            for chunk in RagCOA.compare(question):
                full_response += chunk
                yield chunk
            HISTORY_MANAGER.add(question, full_response, "COMPARE")

        elif category == "COA":
            for chunk in RagCOA.ask(question):
                full_response += chunk
                yield chunk
            HISTORY_MANAGER.add(question, full_response, "COA")

        elif category == "POSTING_ENGINE":
            for chunk in RagPostingEngine.ask(question, item_group, partner_group):
                yield chunk

        elif category == "GENERAL_ACCOUNTING":
            for chunk in cls._general_accounting_answer(question):
                full_response += chunk
                yield chunk
            HISTORY_MANAGER.add(question, full_response, "GENERAL_ACCOUNTING")

        else:
            # GENERAL_FREE: SLM trả lời tự do
            for chunk in cls._general_free_answer(question):
                full_response += chunk
                yield chunk
            HISTORY_MANAGER.add(question, full_response, "GENERAL_FREE")

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