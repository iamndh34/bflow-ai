#                       ┌─────────────────┐
#                       │   chat_type?    │
#                       └────────┬────────┘
#                 ┌──────────────┼──────────────┐
#                 ↓                             ↓
#            "thinking"                       "free"
#                 ↓                             ↓
#       ┌─────────────────┐           ┌─────────────────┐
#       │ Chain-of-Thought│           │  Skip classify  │
#       │  Classification │           │  Direct SLM     │
#       └────────┬────────┘           └────────┬────────┘
#                ↓                             ↓
#       COA/POSTING/COMPARE/...         GENERAL_FREE

import json
import ollama
from app.core.config import settings

# Import modules
from .rag_coa import RagCOA
from .rag_posting_engine import RagPostingEngine
from .history_manager import HISTORY_MANAGER, FREE_HISTORY_MANAGER
from .stream_utils import stream_by_sentence

# =============================================================================
# RAG ROUTER
# =============================================================================

CLASSIFICATION_PROMPT = """Phân loại câu hỏi kế toán. Hãy suy luận từng bước:

BƯỚC 1: Câu hỏi có chứa số tài khoản (3-5 chữ số như 111, 112, 156, 331, 6421) không?
BƯỚC 2: Câu hỏi có từ "so sánh", "khác gì", "khác nhau" không?
BƯỚC 3: Dựa vào kết quả trên, phân loại:

- Nếu CÓ số TK + CÓ từ so sánh → COMPARE (so sánh 1 TK cụ thể giữa TT200 và TT99)
- Nếu KHÔNG có số TK + CÓ từ so sánh + nhắc đến thông tư → COMPARE_CIRCULAR (so sánh tổng quan 2 thông tư)
- Nếu CÓ số TK + KHÔNG có từ so sánh → COA (tra cứu thông tin tài khoản)
- Nếu hỏi về hạch toán, định khoản, bút toán, nghiệp vụ → POSTING_ENGINE
- Nếu hỏi về kế toán chung (nguyên tắc, báo cáo, khái niệm) → GENERAL_ACCOUNTING
- Nếu KHÔNG liên quan kế toán → GENERAL_FREE

Ví dụ:
- "So sánh TK 112" → Có số 112 + có "so sánh" → COMPARE
- "TT99 khác gì TT200" → Không có số TK + có "khác gì" → COMPARE_CIRCULAR
- "TK 156 là gì" → Có số 156 + không có so sánh → COA
- "Bán hàng hạch toán sao" → Có "hạch toán" → POSTING_ENGINE

Câu hỏi: {question}"""

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Suy luận từng bước"
        },
        "has_account_number": {
            "type": "boolean",
            "description": "Câu hỏi có chứa số tài khoản không?"
        },
        "has_compare_keyword": {
            "type": "boolean",
            "description": "Câu hỏi có từ so sánh không?"
        },
        "label": {
            "type": "string",
            "enum": ["COA", "POSTING_ENGINE", "COMPARE", "COMPARE_CIRCULAR", "GENERAL_ACCOUNTING", "GENERAL_FREE"]
        }
    },
    "required": ["reasoning", "has_account_number", "has_compare_keyword", "label"]
}

class RagRouter:
    """
    Router trung tâm: Phân loại intent -> Điều hướng đến module xử lý.
    """

    @classmethod
    def classify(cls, question: str) -> str:
        """
        Phân loại câu hỏi bằng SLM với Chain-of-Thought reasoning
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

            # Log chain-of-thought reasoning
            reasoning = result.get("reasoning", "")
            has_account = result.get("has_account_number", False)
            has_compare = result.get("has_compare_keyword", False)
            label = result.get("label", "GENERAL_ACCOUNTING")

            print(f"[Router] Reasoning: {reasoning}")
            print(f"[Router] Has Account Number: {has_account}, Has Compare Keyword: {has_compare}")
            print(f"[Router] Label: {label}")

            return label

        except Exception as e:
            print(f"[Router] SLM Error: {e}. Using fallback.")
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
            # Xây dựng messages với history để chat liền mạch
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(HISTORY_MANAGER.get_messages_format())
            messages.append({"role": "user", "content": question})

            stream = client.chat(
                model="qwen2.5:3b",
                messages=messages,
                stream=True
            )
            # Yield theo câu để tránh lỗi encoding tiếng Việt
            for sentence in stream_by_sentence(stream):
                yield sentence
        except Exception as e:
            print(f"[GENERAL_ACCOUNTING Error] {e}")
            yield "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."

    @classmethod
    def _general_free_answer(cls, question: str, history_manager=None):
        """SLM trả lời câu hỏi tự do - BUỘC trả lời bằng Tiếng Việt"""
        print(f"[GENERAL_FREE] Question: {question}")

        # Sử dụng history manager được chỉ định, mặc định là HISTORY_MANAGER
        hm = history_manager if history_manager else HISTORY_MANAGER

        system_prompt = """Bạn là trợ lý AI thông minh.

QUY TẮC BẮT BUỘC:
- LUÔN LUÔN trả lời bằng TIẾNG VIỆT, không được dùng ngôn ngữ khác.
- Dù người dùng hỏi bằng tiếng Anh hay ngôn ngữ khác, vẫn phải trả lời bằng Tiếng Việt.
- Trả lời ngắn gọn, chính xác, dễ hiểu."""

        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            # Xây dựng messages với history để chat liền mạch
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(hm.get_messages_format())
            messages.append({"role": "user", "content": question})

            stream = client.chat(
                model="qwen2.5:3b",
                messages=messages,
                stream=True
            )
            # Yield theo câu để tránh lỗi encoding tiếng Việt
            for sentence in stream_by_sentence(stream):
                yield sentence
        except Exception as e:
            print(f"[GENERAL_FREE Error] {e}")
            yield "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."

    @classmethod
    def ask(cls, question: str, item_group: str = "GOODS", partner_group: str = "CUSTOMER", chat_type: str = "thinking"):
        """
        Unified endpoint - API sẽ gọi hàm này.

        chat_type:
        - 'thinking': Phân loại thông minh (COA, POSTING_ENGINE, COMPARE, etc.)
        - 'free': Chế độ tự do - SLM trả lời trực tiếp không qua phân loại
        """
        full_response = ""

        # Chế độ FREE: Bỏ qua phân loại, SLM trả lời tự do (dùng history riêng)
        if chat_type == "free":
            print(f"[Router] Mode: FREE - Direct SLM answer")
            for chunk in cls._general_free_answer(question, history_manager=FREE_HISTORY_MANAGER):
                full_response += chunk
                yield chunk
            FREE_HISTORY_MANAGER.add(question, full_response, "FREE")
            return

        # Chế độ THINKING: Phân loại thông minh
        print(f"[Router] Mode: THINKING - Smart classification")
        category = cls.classify(question)
        print(f"[Router] Routing to -> {category}")

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
    def reset_history(chat_type: str = "thinking"):
        """Reset history (xóa 10 câu gần nhất)"""
        if chat_type == "free":
            FREE_HISTORY_MANAGER.clear()
        else:
            HISTORY_MANAGER.clear()

    @staticmethod
    def get_history(chat_type: str = "thinking"):
        """Get 10 câu hỏi gần nhất"""
        if chat_type == "free":
            return FREE_HISTORY_MANAGER.get_recent()
        return HISTORY_MANAGER.get_recent()

    @staticmethod
    def reload_history(chat_type: str = "thinking"):
        """Reload history từ file JSON và trả về 10 câu gần nhất"""
        if chat_type == "free":
            FREE_HISTORY_MANAGER.reload()
            return FREE_HISTORY_MANAGER.get_recent()
        HISTORY_MANAGER.reload()
        return HISTORY_MANAGER.get_recent()