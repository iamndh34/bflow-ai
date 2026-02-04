"""
Agentic RAG Orchestrator - Điều phối các Agents

Orchestrator:
1. Nhận query từ user
2. Phân loại thông minh (SLM + Embeddings)
3. Route đến agent phù hợp
4. Support multi-agent collaboration
5. Stream response về user
"""
import json
import re
from typing import Optional, List
import numpy as np

from .base import BaseAgent, AgentOrchestrator, AgentContext, AgentResult
from .coa_agent import COAAgent
from .posting_engine_agent import PostingEngineAgent
from .general_accounting_agent import GeneralAccountingAgent, GeneralFreeAgent
from ..core.config import settings
from ..core.ollama_client import get_ollama_client
from ..core.embeddings import get_embed_model
from ..services.llm_service import get_llm_service
from ..services.streaming_cache import cached_stream, _simulate_streaming
from ..services.history_search import find_in_history_before_llm
from ..services.session_manager import get_session_manager


# =============================================================================
# SMART CLASSIFICATION - Few-shot Examples
# =============================================================================

CLASSIFICATION_EXAMPLES = """
VÍ DỤ HUẤN LUYỆN:

=== COA (Tra cứu thông tin tài khoản) ===
Query: "TK 156 là gì?"
→ Agent: COA
→ Lý do: Có số tài khoản 156 cụ thể, cần tra cứu thông tin TK

Query: "So sánh TK 112 giữa TT200 và TT99"
→ Agent: COA
→ Lý do: So sánh tài khoản cụ thể giữa 2 thông tư

Query: "TT99 khác gì TT200?"
→ Agent: COA
→ Lý do: So sánh tổng quan giữa 2 thông tư về hệ thống TK

=== POSTING_ENGINE (Hạch toán, định khoản) ===
Query: "Bán hàng hạch toán sao?"
→ Agent: POSTING_ENGINE
→ Lý do: Hỏi về hạch toán nghiệp vụ kinh doanh

Query: "Xuất hóa đơn thì định khoản thế nào?"
→ Agent: POSTING_ENGINE
→ Lý do: Hỏi về bút toán, định khoản nghiệp vụ

Query: "Phiếu thu ghi nhận thế nào?"
→ Agent: POSTING_ENGINE
→ Lý do: Hỏi về bút toán phiếu thu

=== GENERAL_ACCOUNTING (Lý thuyết kế toán) ===
Query: "Nguyên tắc cơ sở dồn tích là gì?"
→ Agent: GENERAL_ACCOUNTING
→ Lý do: Câu hỏi lý thuyết kế toán, không có số TK cụ thể

Query: "Báo cáo tài chính gồm những gì?"
→ Agent: GENERAL_ACCOUNTING
→ Lý do: Câu hỏi về báo cáo kế toán tổng quát

=== GENERAL_FREE (Trợ lý tổng quát) ===
Query: "hello"
→ Agent: GENERAL_FREE
→ Lý do: Lời chào thông thường

Query: "thời tiết hôm nay thế nào?"
→ Agent: GENERAL_FREE
→ Lý do: Câu hỏi về thời tiết

Query: "bóng đá hôm nay có ai thắng?"
→ Agent: GENERAL_FREE
→ Lý do: Câu hỏi về thể thao


Note: Khi thêm agent mới (ví dụ: LEGAL, HR, SALES...),
chỉ cần thêm ví dụ vào đây và mô tả agent trong phần AGENTS.
"""

# Dynamic agent descriptions - tự động cập nhật khi thêm agent mới
DEFAULT_AGENT_DESCRIPTIONS = """
1. COA: Tra cứu THÔNG TIN TÀI KHOẢN (có số TK như 111, 156, 331...) hoặc SO SÁNH THÔNG TƯ (TT99 vs TT200)
2. POSTING_ENGINE: Hạch toán, định khoản, bút toán cho NGHIỆP VỤ (bán hàng, mua hàng, xuất hóa đơn, phiếu thu/chi...)
3. GENERAL_ACCOUNTING: Câu hỏi LÝ THUYẾT kế toán (nguyên tắc, chuẩn mực, báo cáo...) - KHÔNG có số TK cụ thể
4. GENERAL_FREE: Trợ lý tổng quát cho các câu hỏi chung chung, xã giao
"""

def get_agent_descriptions(orchestrator) -> str:
    """
    Dynamic agent descriptions - Tự động cập nhật khi thêm agent mới
    """
    descriptions = []
    for agent in orchestrator.agents:
        descriptions.append(f"- {agent.name}: {agent.description}")
    return "\n".join(descriptions)

ORCHESTRATOR_CLASSIFICATION_PROMPT = """Bạn là classifier chuyên nghiệp cho hệ thống AI Multi-Agent.
Nhiệm vụ: Phân loại câu hỏi vào đúng AGENT phù hợp.

{examples}

CÁC AGENTS HIỆN CÓ:
{agent_descriptions}

QUY TRÌNH PHÂN LOẠI:
1. Đọc câu hỏi
2. Xem xét mô tả của từng agent
3. Chọn agent PHÙ HỢP NHẤT có khả năng trả lời câu hỏi
4. Nếu nhiều agent có thể trả lời, chọn agent CHUYÊN SÂU NHẤT (domain specialist > generalist)

Câu hỏi: {question}

Hãy phân loại và suy luận từng bước."""

def get_classification_schema(orchestrator) -> dict:
    """
    Dynamic classification schema - Tự động cập nhật agent enum khi thêm agent mới
    """
    agent_names = [agent.name for agent in orchestrator.agents]
    return {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": agent_names
            },
            "reasoning": {
                "type": "string",
                "description": "Suy luận từng bước"
            }
        },
        "required": ["agent", "reasoning"]
    }


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class AccountingOrchestrator(AgentOrchestrator):
    """
    Orchestrator cho hệ thống Accounting AI

    Điều phối các agents:
    - COAAgent: Chuyên gia tài khoản
    - PostingEngineAgent: Chuyên gia hạch toán
    - GeneralAccountingAgent: Chuyên gia kế toán tổng quát
    - GeneralFreeAgent: Trợ lý tổng quát
    """

    # Embedding model for semantic classification
    _embed_model = None
    _agent_embeddings = None

    def __init__(self):
        super().__init__()
        # Đăng ký tất cả agents
        self.register_agent(COAAgent())
        self.register_agent(PostingEngineAgent())
        self.register_agent(GeneralAccountingAgent())
        self.register_agent(GeneralFreeAgent())

        # Init embeddings for semantic fallback
        self._init_embeddings()

    @classmethod
    def _init_embeddings(cls):
        """Initialize embeddings cho semantic classification"""
        if cls._embed_model is None:
            cls._embed_model = get_embed_model()

            # Example queries cho từng agent type
            agent_examples = {
                "COA": [
                    "TK 156 là gì?", "tài khoản 111", "số hiệu 331",
                    "so sánh TT99 và TT200", "thông tư 99 khác gì thông tư 200",
                    "TK 112 trong TT99", "danh mục tài khoản"
                ],
                "POSTING_ENGINE": [
                    "bán hàng hạch toán sao", "xuất hóa đơn định khoản thế nào",
                    "phiếu thu ghi nhận gì", "nhập kho thì bút toán ra sao",
                    "mua hàng hạch toán", "thu tiền từ khách"
                ],
                "GENERAL_ACCOUNTING": [
                    "nguyên tắc kế toán là gì", "báo cáo tài chính gồm những gì",
                    "chuẩn mực kế toán", "cơ sở dồn tích", "nguyên tắc phù hợp"
                ],
                "GENERAL_FREE": [
                    "hello", "xin chào", "cảm ơn", "bạn khỏe không",
                    "thời tiết hôm nay", "what is your name"
                ],
            }

            # Create embeddings cho mỗi agent type
            cls._agent_embeddings = {}
            for agent_name, examples in agent_examples.items():
                embeddings = cls._embed_model.encode(examples, normalize_embeddings=True)
                cls._agent_embeddings[agent_name] = {
                    "embeddings": embeddings,
                    "centroid": np.mean(embeddings, axis=0)
                }

    def _semantic_classify(self, question: str) -> Optional[str]:
        """
        Fallback: Sử dụng semantic similarity để classify
        """
        if self._agent_embeddings is None:
            self._init_embeddings()

        query_emb = self._embed_model.encode(question, normalize_embeddings=True)

        scores = {}
        for agent_name, data in self._agent_embeddings.items():
            # Tính similarity với centroid của mỗi agent
            sim = float(np.dot(query_emb, data["centroid"]))
            scores[agent_name] = sim

        best_agent = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_agent]

        print(f"[Orchestrator] Semantic scores: {scores}")
        print(f"[Orchestrator] Semantic best: {best_agent} (score: {best_score:.3f})")

        # Only use if confidence is high enough
        if best_score > 0.3:
            return best_agent

        return None

    def route(self, context: AgentContext) -> BaseAgent:
        """
        Route query đến agent phù hợp nhất.

        Strategy (Simplified - Faster):
        1. Fast rule-based routing cho common patterns (TK + number, keywords)
        2. SLM classification với few-shot examples
        3. Semantic fallback: Embedding-based classification
        4. Final fallback: GENERAL_FREE

        Optimized: Bỏ confidence scoring step để giảm latency.
        """
        question_lower = context.question.lower()

        # === FAST RULE-BASED ROUTING (O(1) lookup) ===
        # Pattern 1: Có số tài khoản -> COA
        if re.search(r'\b\d{3,5}\b', context.question):
            # Có 3-5 chữ số liên tiếp -> có thể là tài khoản
            # Check thêm keywords để phân loại COA vs COMPARE
            if any(kw in question_lower for kw in ["so sánh", "khác gì", "khác nhau", "giữa"]):
                print("[Orchestrator] Fast route: COMPARE (account number + compare keyword)")
                return self.get_agent("COA")  # COA agent handles compare
            else:
                print("[Orchestrator] Fast route: COA (account number)")
                return self.get_agent("COA")

        # Pattern 2: Keywords mạnh -> POSTING_ENGINE
        posting_keywords = [
            "hạch toán", "định khoản", "bút toán", "ghi nhận",
            "nợ", "có", "phiếu thu", "phiếu chi", "xuất hóa đơn",
            "nhập kho", "xuất kho", "bán hàng", "mua hàng"
        ]
        if any(kw in question_lower for kw in posting_keywords):
            print("[Orchestrator] Fast route: POSTING_ENGINE (posting keyword)")
            return self.get_agent("POSTING_ENGINE")

        # Pattern 3: So sánh thông tư -> COA (compare circular)
        if any(kw in question_lower for kw in ["tt99", "tt200", "thông tư 99", "thông tư 200"]):
            if "so sánh" in question_lower or "khác" in question_lower:
                print("[Orchestrator] Fast route: COA (circular compare)")
                return self.get_agent("COA")

        # === SLM CLASSIFICATION ===
        slm_agent = self._classify_with_slm(context.question)
        if slm_agent:
            print(f"[Orchestrator] SLM classified to: {slm_agent.name}")
            return slm_agent

        # === SEMANTIC FALLBACK ===
        semantic_agent_name = self._semantic_classify(context.question)
        if semantic_agent_name:
            agent = self.get_agent(semantic_agent_name)
            if agent:
                print(f"[Orchestrator] Semantic classified to: {semantic_agent_name}")
                return agent

        # === FINAL FALLBACK ===
        print("[Orchestrator] Using default: GENERAL_ACCOUNTING")
        return self.get_agent("GENERAL_ACCOUNTING")

    def _classify_with_slm(self, question: str) -> Optional[BaseAgent]:
        """
        Sử dụng SLM để classify query với few-shot examples.

        Dynamic: Tự động cập nhật agent descriptions và schema khi thêm agent mới.
        CÓ CACHE cho classification responses (non-streaming).
        """
        try:
            # Dùng CachedLLMService để enable cache
            llm_service = get_llm_service()

            # Dynamic agent descriptions - tự động lấy từ agents đã đăng ký
            agent_descriptions = get_agent_descriptions(self)

            # Dynamic schema - tự động cập nhật enum
            schema = get_classification_schema(self)

            # Build prompt with examples và dynamic agent descriptions
            full_prompt = ORCHESTRATOR_CLASSIFICATION_PROMPT.format(
                examples=CLASSIFICATION_EXAMPLES,
                agent_descriptions=agent_descriptions,
                question=question
            )

            response = llm_service.chat(
                model=settings.GENERATION_MODEL,  # Use 3b for better accuracy
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                format=schema,
                stream=False,
                use_cache=True  # Enable cache!
            )
            content = response.get("message", {}).get("content", "")
            result = json.loads(content)

            agent_name = result.get("agent")
            reasoning = result.get("reasoning", "")

            print(f"[Orchestrator] SLM Reasoning: {reasoning[:200]}...")  # Truncate for cleaner log

            # Get agent by name
            agent = self.get_agent(agent_name)
            if agent:
                return agent

        except Exception as e:
            print(f"[Orchestrator] SLM Classification Error: {e}")

        return None

    def ask(self, question: str, item_group: str = "GOODS", partner_group: str = "CUSTOMER",
            chat_type: str = "thinking", session_id: str = None):
        """
        Main entry point - xử lý query và stream response.

        Args:
            question: Câu hỏi của user
            item_group: Nhóm sản phẩm (cho posting engine)
            partner_group: Nhóm đối tác (cho posting engine)
            chat_type: Loại chat ('thinking' hoặc 'free')
            session_id: Session ID

        Yields:
            str: Streaming response
        """
        full_response = ""

        # Lấy session manager
        sm = get_session_manager(chat_type)

        # Tạo session mới nếu chưa có
        if not session_id:
            session_id = sm.create_session()
            print(f"[Orchestrator] Created new session: {session_id}")

        # Yield session_id đầu tiên
        yield f"__SESSION_ID__:{session_id}\n"

        # Build context
        history_messages = sm.get_messages_format(session_id) if session_id else []
        context = AgentContext(
            question=question,
            session_id=session_id,
            chat_type=chat_type,
            item_group=item_group,
            partner_group=partner_group,
            history=history_messages
        )

        # Chế độ FREE: Bỏ qua orchestration, dùng GeneralFreeAgent trực tiếp
        if chat_type == "free":
            print(f"[Orchestrator] Mode: FREE - Direct general response")
            agent = self.get_agent("GENERAL_FREE")
            for chunk in agent.stream_execute(context):
                full_response += chunk
                yield chunk
            sm.add_message(session_id, question, full_response, "GENERAL_FREE")
            return

        # Chế độ THINKING: Orchestration thông minh
        print(f"[Orchestrator] Mode: THINKING - Smart routing")

        # Route đến agent phù hợp
        agent = self.route(context)
        agent_name = agent.name

        # === STEP 1: SEMANTIC HISTORY MATCH ===
        # Tìm trong history bằng độ tương đồng (similarity)
        # Câu hỏi giống → trả lời ngay (không cần gọi LLM)
        if session_id and settings.ENABLE_SEMANTIC_HISTORY:
            print(f"[Orchestrator] Checking history for similar question...")
            history_response = find_in_history_before_llm(
                question=question,
                session_id=session_id,
                agent_name=agent_name,
                chat_type=chat_type
            )

            if history_response:
                # Tìm thấy trong history → simulate streaming
                print(f"[Orchestrator] Found in history! Simulating streaming...")
                for chunk in _simulate_streaming(
                    history_response,
                    chars_per_chunk=settings.CACHE_CHARS_PER_CHUNK,
                    delay=settings.CACHE_SIMULATE_DELAY
                ):
                    full_response += chunk
                    yield chunk

                # Lưu lại vào history (cùng câu hỏi, cùng response)
                sm.add_message(session_id, question, full_response, agent_name)
                return

        # === STEP 2: STREAMING CACHE ===
        # Check cache trước khi gọi agent. Cache hit → trả lời ngay (rất nhanh!)
        def agent_stream_func():
            for chunk in agent.stream_execute(context):
                yield chunk

        # Sử dụng cached_stream wrapper với simulate settings từ config
        cache_context = {
            "item_group": item_group,
            "partner_group": partner_group,
            "chat_type": chat_type
        }

        for chunk in cached_stream(
            question,
            agent_name,
            agent_stream_func,
            cache_context,
            simulate_delay=settings.CACHE_SIMULATE_DELAY
        ):
            full_response += chunk
            yield chunk

        # Lưu vào history
        sm.add_message(session_id, question, full_response, agent_name)

    # =========================================================================
    # MULTI-AGENT COLLABORATION (Future Enhancement)
    # =========================================================================

    def collaborative_ask(self, question: str, session_id: str = None, max_agents: int = 2):
        """
        Multi-agent collaboration - Nhiều agents cùng xử lý query.

        Use case:
        - Query phức tạp cần nhiều domain experts
        - Agent này không tự tin, gọi agent khác hỗ trợ

        Args:
            question: Câu hỏi
            session_id: Session ID
            max_agents: Số lượng agents tối đa tham gia

        Yields:
            str: Streaming response
        """
        context = AgentContext(
            question=question,
            session_id=session_id,
            chat_type="thinking"
        )

        # Tìm tất cả agents có thể xử lý
        candidates = self.find_agents_for_query(context)

        # Lấy top N agents
        top_agents = [agent for agent, _ in candidates[:max_agents]]

        if len(top_agents) == 1:
            # Chỉ có 1 agent, behavior như bình thường
            for chunk in top_agents[0].stream_execute(context):
                yield chunk
            return

        # Multi-agent: Gọi từng agent và aggregate results
        results = []

        yield f"[Collaboration] Đang phối hợp {len(top_agents)} chuyên gia...\n\n"

        for i, agent in enumerate(top_agents, 1):
            yield f"### {i}. {agent.name}:\n\n"

            agent_result = ""
            for chunk in agent.stream_execute(context):
                agent_result += chunk
                yield chunk

            results.append({
                "agent": agent.name,
                "content": agent_result
            })
            yield "\n\n---\n\n"

        # Summary
        yield "### TỔNG HỢP:\n\n"
        yield f"Đã tham vấn {len(top_agents)} chuyên gia. Vui lòng tham khảo các ý kiến trên."


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global orchestrator instance
_orchestrator_instance = None


def get_orchestrator() -> AccountingOrchestrator:
    """Get global orchestrator instance (singleton)"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AccountingOrchestrator()
    return _orchestrator_instance
