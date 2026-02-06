"""
Module Router - Phân loại câu hỏi đến đúng module

Architecture:
    /api/ai-bflow/ask (Unified Entry Point)
                ↓
        Module Router (SLM Classification)
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
Accounting    General    Future Modules
Pipeline      Pipeline    (HR, CRM, ...)
"""
import json
import re
from typing import Optional, Generator, Dict, Any

from .ask import AccountingPipeline
from ..core.config import settings
from ..services.llm_service import get_llm_service


# =============================================================================
# MODULE DEFINITIONS
# =============================================================================

AVAILABLE_MODULES = {
    "ACCOUNTING": {
        "name": "Kế toán",
        "description": "Câu hỏi về kế toán, tài khoản, hạch toán, bút toán, báo cáo tài chính",
        "keywords": ["tài khoản", "hạch toán", "bút toán", "tk", "thuế", "gtgt", "khấu hao",
                     "doanh thu", "chi phí", "lợi nhuận", "báo cáo", "nhập kho", "xuất kho"],
        "pipeline_class": AccountingPipeline
    },
    "GENERAL": {
        "name": "Tổng quát",
        "description": "Câu hỏi chung, xã giao, không liên quan đến chuyên môn",
        "keywords": ["hello", "xin chào", "cảm ơn", "bạn", "thời tiết", "help"],
        "pipeline_class": None  # Sẽ xử lý trực tiếp
    }
}


# =============================================================================
# MODULE CLASSIFICATION
# =============================================================================

MODULE_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "module": {
            "type": "string",
            "enum": list(AVAILABLE_MODULES.keys())
        },
        "reasoning": {
            "type": "string",
            "description": "Lý do chọn module này"
        }
    },
    "required": ["module", "reasoning"]
}


def build_module_classification_prompt(question: str) -> str:
    """Build prompt để phân loại module"""
    module_descriptions = "\n".join([
        f"- {code}: {info['name']} - {info['description']}"
        for code, info in AVAILABLE_MODULES.items()
    ])

    return f"""Bạn là classifier chuyên nghiệp. Hãy phân loại câu hỏi sau vào MODULE phù hợp.

CÁC MODULES:
{module_descriptions}

QUY TẮC:
1. Có từ khóa kế toán/tài khoản/hạch toán → ACCOUNTING
2. Câu hỏi chung chung, xã giao → GENERAL
3. Chọn 1 module PHÙ HỢP NHẤT

Câu hỏi: {question}

Hãy phân loại và trả về JSON:
{{
    "module": "ACCOUNTING",
    "reasoning": "Có từ khóa 'tài khoản'"
}}
"""


def classify_module_with_slm(question: str) -> Optional[str]:
    """
    Phân loại module bằng SLM.

    Args:
        question: Câu hỏi

    Returns:
        Module code (ACCOUNTING, GENERAL, etc) hoặc None
    """
    try:
        llm_service = get_llm_service()

        prompt = build_module_classification_prompt(question)

        response = llm_service.chat(
            model=settings.GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=MODULE_CLASSIFICATION_SCHEMA,
            stream=False,
            use_cache=True
        )

        content = response.get("message", {}).get("content", "")
        result = json.loads(content)

        module_code = result.get("module")
        reasoning = result.get("reasoning", "")

        print(f"[ModuleRouter] SLM classified to: {module_code}")
        print(f"[ModuleRouter] Reasoning: {reasoning[:100]}...")

        return module_code

    except Exception as e:
        print(f"[ModuleRouter] SLM Error: {e}")
        return None


def classify_module_with_keywords(question: str) -> str:
    """
    Phân loại module bằng keyword matching (Fast fallback).

    Args:
        question: Câu hỏi

    Returns:
        Module code
    """
    question_lower = question.lower()

    # Check each module's keywords
    for module_code, module_info in AVAILABLE_MODULES.items():
        keywords = module_info.get("keywords", [])
        if any(kw in question_lower for kw in keywords):
            print(f"[ModuleRouter] Keyword matched: {module_code}")
            return module_code

    # Default fallback
    print("[ModuleRouter] No keyword match, using default: GENERAL")
    return "GENERAL"


# =============================================================================
# MODULE ROUTER
# =============================================================================

class ModuleRouter:
    """
    Router phân loại câu hỏi đến module phù hợp.

    Usage:
        router = ModuleRouter()

        for chunk in router.route_and_process(
            question="TK 156 là gì?",
            session_id="abc123",
            chat_type="thinking"
        ):
            yield chunk
    """

    def __init__(self):
        """Initialize module router."""
        self._pipelines: Dict[str, Any] = {}

    def _get_pipeline(self, module_code: str):
        """Get hoặc tạo pipeline instance cho module."""
        if module_code not in self._pipelines:
            module_info = AVAILABLE_MODULES.get(module_code)

            if module_code == "GENERAL" or module_info is None:
                # General không có pipeline riêng
                return None

            pipeline_class = module_info.get("pipeline_class")
            if pipeline_class:
                self._pipelines[module_code] = pipeline_class()

        return self._pipelines.get(module_code)

    def classify_module(self, question: str, use_slm: bool = True) -> str:
        """
        Phân loại câu hỏi đến module phù hợp.

        Args:
            question: Câu hỏi
            use_slm: Có dùng SLM classification hay không

        Returns:
            Module code (ACCOUNTING, GENERAL, etc)
        """
        # Step 1: Fast keyword matching
        module = classify_module_with_keywords(question)

        # Step 2: SLM classification (nếu enabled)
        if use_slm:
            slm_module = classify_module_with_slm(question)
            if slm_module:
                module = slm_module

        return module

    def route_and_process(
        self,
        question: str,
        user_id: str = None,
        session_id: str = None,
        chat_type: str = "thinking",
        item_group: str = "GOODS",
        partner_group: str = "CUSTOMER",
    ) -> Generator[str, None, None]:
        """
        Route câu hỏi đến module phù hợp và xử lý.

        Args:
            question: Câu hỏi
            user_id: User ID (phân quyền)
            session_id: Session ID
            chat_type: Loại chat
            item_group: Nhóm sản phẩm
            partner_group: Nhóm đối tác
            **kwargs: Tham số thêm

        Yields:
            Response chunks
        """
        print(f"\n{'='*60}")
        print(f"[ModuleRouter] Processing: {question}")
        print(f"[ModuleRouter] User ID: {user_id}")
        print(f"{'='*60}\n")

        # Step 1: Phân loại module
        module_code = self.classify_module(question, use_slm=True)
        print(f"[ModuleRouter] Routed to: {module_code} ({AVAILABLE_MODULES.get(module_code, {}).get('name', 'Unknown')})")

        # Step 2: Get pipeline
        pipeline = self._get_pipeline(module_code)

        if pipeline is None:
            # Module GENERAL - gọi GeneralFreeAgent
            print("[ModuleRouter] General mode - calling GeneralFreeAgent")
            from ..agents.orchestrator import get_orchestrator

            orchestrator = get_orchestrator()
            agent = orchestrator.get_agent("GENERAL_FREE")

            if agent:
                from ..agents.base import AgentContext
                from ..services.session_manager import get_session_manager
                import uuid

                # Auto-generate session_id nếu không có
                if not session_id:
                    session_id = str(uuid.uuid4())[:8]
                    print(f"[ModuleRouter] Generated new session_id: {session_id}")

                # Debug session_id
                print(f"[ModuleRouter] GENERAL mode - session_id='{session_id}', chat_type='{chat_type}'")

                # Load history từ session - chỉ lấy 10 tin nhắn gần nhất
                sm = get_session_manager(chat_type or "thinking")
                history_data = sm.get_history(session_id or "", max_count=10) if session_id else []

                # Summary nếu quá nhiều tin nhắn (tiết kiệm token)
                # Early trigger: > 3 tin nhắn (người dùng nói summary sớm)
                if history_data and len(history_data) > 3:
                    original_count = len(history_data)
                    history_data = self._summarize_history(history_data)
                    print(f"[ModuleRouter] Summary: {original_count} messages → {len(history_data)} (saved ~{original_count - len(history_data)} messages)")

                print(f"[ModuleRouter] GENERAL mode - loaded {len(history_data)} messages from history")

                # Convert history sang format cho AgentContext
                history = []
                if history_data:
                    for item in history_data:
                        if item.get("role") == "user":
                            history.append({"role": "user", "content": item.get("content", "")})
                        elif item.get("role") == "assistant":
                            history.append({"role": "assistant", "content": item.get("content", "")})

                context = AgentContext(
                    question=question,
                    session_id=session_id or "",
                    chat_type=chat_type or "thinking",
                    item_group=item_group or "GOODS",
                    partner_group=partner_group or "CUSTOMER",
                    history=history
                )

                # Collect full response để lưu vào session
                full_response = ""
                for chunk in agent.stream_execute(context):
                    full_response += chunk
                    yield chunk

                # Lưu response vào session/history sau khi stream xong
                if session_id:
                    sm.add_message(session_id, question, full_response, "GENERAL_FREE", user_id=user_id)
                    print(f"[ModuleRouter] Saved to session {session_id[:8]}...")

                return
            else:
                # Fallback nếu không có agent
                response = self._get_general_response(question)
                for char in response:
                    yield char
                return

        # Step 3: Process qua pipeline
        for chunk in pipeline.process(
            question=question,
            user_id=user_id,
            session_id=session_id,
            chat_type=chat_type,
            item_group=item_group,
            partner_group=partner_group,
        ):
            yield chunk

    def _get_general_response(self, question: str) -> str:
        """Get response cho mode GENERAL."""
        # Có thể gọi LLM đơn giản hoặc trả về câu chào mặc định
        question_lower = question.lower()

        greetings = ["hello", "hi", "xin chào", "chào"]
        if any(g in question_lower for g in greetings):
            return "Xin chào! Tôi là BFLOW AI, trợ lý thông minh của bạn. Tôi có thể giúp gì cho bạn?"

        thanks = ["cảm ơn", "thanks", "thank"]
        if any(t in question_lower for t in thanks):
            return "Rất vui được giúp đỡ bạn! Cần hỗ trợ thêm gì không?"

        return f"Hiểu câu hỏi: {question}. Tôi có thể giúp bạn tìm thông tin về kế toán, tài khoản, hạch toán..."

    def _summarize_history(self, history_data: list) -> list:
        """
        Summary lịch sử trò chuyện nếu quá dài để tiết kiệm token.

        Strategy - Early trigger, nhiều context hơn:
        - Trigger khi > 3 tin nhắn (thay vì > 5)
        - Giữ 5 tin nhắn gần nhất
        - Summary các tin nhắn cũ với đầy đủ context

        Args:
            history_data: List of history items (mới nhất → cũ nhất)

        Returns:
            List summary messages (giữ format {role, content})
        """
        if not history_data:
            return history_data

        # Early trigger: > 3 messages thay vì > 5
        if len(history_data) <= 3:
            return history_data

        # Giữ 5 tin nhắn gần nhất (context mới nhất)
        recent = history_data[:5]

        # Summary các tin nhắn cũ hơn (từ thứ 6 trở đi)
        older = history_data[5:]

        # Tạo summary chi tiết hơn
        summary_parts = []
        user_msgs = [h.get("content", "") for h in older if h.get("role") == "user"]
        asst_msgs = [h.get("content", "") for h in older if h.get("role") == "assistant"]

        # Summary các chủ đề chính
        if user_msgs:
            # Lấy keywords chính từ user messages
            keywords = []
            for msg in user_msgs:
                # Trích xuất keywords (các từ quan trọng)
                words = msg.split()[:5]  # 5 từ đầu tiên làm keywords
                keywords.extend([w[:20] + "..." if len(w) > 20 else w for w in words])

            # Ghép keywords thành câu
            if keywords:
                topics = ", ".join(keywords[:8])  # Max 8 keywords
                summary_parts.append(f"User đã nói về: {topics}")

        # Summary các response chính của AI
        if asst_msgs:
            summary_parts.append(f"AI đã trả lời {len(asst_msgs)} tin")

        # Tạo summary text
        if summary_parts:
            summary_text = f"[Hội thoại trước: {' | '.join(summary_parts)}]"
        else:
            summary_text = "[Hội thoại trước: các tin nhắn cũ]"

        summary_msg = {"role": "system", "content": summary_text}

        return recent + [summary_msg]


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_module_router_instance = None


def get_module_router() -> ModuleRouter:
    """Get singleton module router instance."""
    global _module_router_instance
    if _module_router_instance is None:
        _module_router_instance = ModuleRouter()
    return _module_router_instance
