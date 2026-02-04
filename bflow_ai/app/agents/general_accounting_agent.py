"""
General Accounting Agent - Chuyên gia về kế toán tổng quát

Chuyên gia về:
- Nguyên tắc kế toán
- Báo cáo tài chính
- Chuẩn mực kế toán
- Câu hỏi kế toán chung không cần tra cứu tài khoản cụ thể
"""

from .base import BaseAgent, AgentRole, AgentResult, AgentContext
from ..core.config import settings
from ..core.ollama_client import get_ollama_client
from ..services.stream_utils import stream_by_char


class GeneralAccountingAgent(BaseAgent):
    """
    General Accounting Agent - Chuyên gia về kế toán tổng quát

    Xử lý:
    - Câu hỏi về nguyên tắc, khái niệm kế toán
    - Báo cáo tài chính
    - Chuẩn mực kế toán
    - Các câu hỏi kế toán không cần tra cứu cụ thể
    """

    def __init__(self):
        super().__init__()
        self._init_tools()

    @property
    def name(self) -> str:
        return "GENERAL_ACCOUNTING"

    @property
    def role(self) -> AgentRole:
        return AgentRole.GENERALIST

    @property
    def description(self) -> str:
        return "Chuyên gia về kế toán tổng quát. Trả lời các câu hỏi về nguyên tắc, báo cáo tài chính, chuẩn mực kế toán."

    def _init_tools(self):
        """Đăng ký tools - agent này không có RAG tools"""
        # Agent này dùng SLM knowledge nên không có tools phức tạp
        pass

    def can_handle(self, context: AgentContext) -> tuple[bool, float]:
        """
        Kiểm tra agent có thể xử lý query không.

        Đây là fallback agent - luôn return True với confidence thấp.
        Các chuyên gia khác sẽ có confidence cao hơn nếu match domain.
        """
        return True, 0.3

    def execute(self, context: AgentContext) -> AgentResult:
        """Thực thi query"""
        system_prompt = """Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT.
Trả lời các câu hỏi về kế toán dựa trên kiến thức chuyên môn của bạn.
Trình bày rõ ràng, có cấu trúc, dễ hiểu."""

        try:
            client = get_ollama_client()

            # Build messages with history
            messages = [{"role": "system", "content": system_prompt}]
            if context.history:
                messages.extend(context.history)
            messages.append({"role": "user", "content": context.question})

            response = client.chat(
                model=settings.GENERATION_MODEL,
                messages=messages,
                options=settings.OLLAMA_OPTIONS,
                stream=False
            )
            content = response.get("message", {}).get("content", "")

            return AgentResult(
                agent_name=self.name,
                content=content,
                confidence=0.6,
                sources=["SLM Knowledge"]
            )
        except Exception as e:
            print(f"[GeneralAccountingAgent Error] {e}")
            return AgentResult(
                agent_name=self.name,
                content="Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                confidence=0.3
            )

    def stream_execute(self, context: AgentContext):
        """Execute với streaming response"""
        system_prompt = """Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT.
Trả lời các câu hỏi về kế toán dựa trên kiến thức chuyên môn của bạn.
Trình bày rõ ràng, có cấu trúc, dễ hiểu."""

        try:
            client = get_ollama_client()

            # Build messages with history
            messages = [{"role": "system", "content": system_prompt}]
            if context.history:
                messages.extend(context.history)
            messages.append({"role": "user", "content": context.question})

            stream = client.chat(
                model=settings.GENERATION_MODEL,
                messages=messages,
                options=settings.OLLAMA_OPTIONS,
                stream=True
            )

            for char in stream_by_char(stream):
                yield char

        except Exception as e:
            print(f"[GeneralAccountingAgent Stream Error] {e}")
            yield "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."


class GeneralFreeAgent(BaseAgent):
    """
    General Free Agent - Trợ lý AI tổng quát

    Xử lý:
    - Câu hỏi không liên quan kế toán
    - Chat tự do
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "GENERAL_FREE"

    @property
    def role(self) -> AgentRole:
        return AgentRole.GENERALIST

    @property
    def description(self) -> str:
        return "Trợ lý AI thông minh. Trả lời các câu hỏi tự do, không liên quan kế toán."

    def can_handle(self, context: AgentContext) -> tuple[bool, float]:
        """Luôn có thể xử lý (fallback cuối cùng)"""
        return True, 0.1

    def execute(self, context: AgentContext) -> AgentResult:
        """Thực thi query"""
        system_prompt = """Bạn là trợ lý AI thông minh.

QUY TẮC BẮT BUỘC:
- LUÔN LUÔN trả lời bằng TIẾNG VIỆT, không được dùng ngôn ngữ khác.
- Dù người dùng hỏi bằng tiếng Anh hay ngôn ngữ khác, vẫn phải trả lời bằng Tiếng Việt.
- Trả lời ngắn gọn, chính xác, dễ hiểu."""

        try:
            client = get_ollama_client()

            messages = [{"role": "system", "content": system_prompt}]
            if context.history:
                messages.extend(context.history)
            messages.append({"role": "user", "content": context.question})

            response = client.chat(
                model=settings.GENERATION_MODEL,
                messages=messages,
                options=settings.OLLAMA_OPTIONS,
                stream=False
            )
            content = response.get("message", {}).get("content", "")

            return AgentResult(
                agent_name=self.name,
                content=content,
                confidence=0.5,
                sources=["SLM General Knowledge"]
            )
        except Exception as e:
            print(f"[GeneralFreeAgent Error] {e}")
            return AgentResult(
                agent_name=self.name,
                content="Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                confidence=0.2
            )

    def stream_execute(self, context: AgentContext):
        """Execute với streaming response"""
        system_prompt = """Bạn là trợ lý AI thông minh.

QUY TẮC BẮT BUỘC:
- LUÔN LUÔN trả lời bằng TIẾNG VIỆT, không được dùng ngôn ngữ khác.
- Dù người dùng hỏi bằng tiếng Anh hay ngôn ngữ khác, vẫn phải trả lời bằng Tiếng Việt.
- Trả lời ngắn gọn, chính xác, dễ hiểu."""

        try:
            client = get_ollama_client()

            messages = [{"role": "system", "content": system_prompt}]
            if context.history:
                messages.extend(context.history)
            messages.append({"role": "user", "content": context.question})

            stream = client.chat(
                model=settings.GENERATION_MODEL,
                messages=messages,
                options=settings.OLLAMA_OPTIONS,
                stream=True
            )

            for char in stream_by_char(stream):
                yield char

        except Exception as e:
            print(f"[GeneralFreeAgent Stream Error] {e}")
            yield "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
