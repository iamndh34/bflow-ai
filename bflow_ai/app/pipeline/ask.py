"""
BFLOW AI - Accounting Assistant System

Architecture: Pipeline-based với các bước xử lý riêng biệt.

Flow:
    User Request → Session Manager → Router → History Check → Cache Check →
    Agent Executor → LLM Service → Stream Processor → Response

Mỗi bước là 1 class riêng với các methods rõ ràng.
"""

# =============================================================================
# STEP 1: SESSION MANAGEMENT
# =============================================================================

class SessionManagerStep:
    """
    STEP 1: Quản lý session người dùng.

    Tasks:
    - Tạo session mới nếu chưa có
    - Lấy history của session
    - Format history cho LLM
    - Lưu message vào history
    """

    def __init__(self, chat_type: str = "thinking"):
        """Initialize session manager."""
        from ..services.session_manager import get_session_manager
        self.sm = get_session_manager(chat_type)
        self.chat_type = chat_type

    def create_session_if_needed(self, session_id: str = None, turn_off: bool = False) -> str:
        """
        Tạo session mới nếu chưa có.

        Args:
            session_id: Session ID hiện tại (None nếu chưa có)
            turn_off: Bypass bước này nếu True

        Returns:
            session_id: Session ID (mới hoặc cũ)
        """
        if turn_off:
            return session_id or "dummy_session"

        if not session_id:
            session_id = self.sm.create_session()
            print(f"[SessionStep] Created new session: {session_id}")
        else:
            print(f"[SessionStep] Using existing session: {session_id}")

        return session_id

    def get_session_history(self, session_id: str, max_count: int = 10, turn_off: bool = False) -> list:
        """
        Lấy history của session.

        Args:
            session_id: Session ID
            max_count: Số message tối đa lấy
            turn_off: Bypass, trả về list rỗng

        Returns:
            List of message dicts
        """
        if turn_off:
            return []

        return self.sm.get_history(session_id, max_count=max_count)

    def format_history_for_llm(self, session_id: str, max_count: int = 10, turn_off: bool = False) -> list:
        """
        Format history thành dạng messages cho LLM.

        Args:
            session_id: Session ID
            max_count: Số message tối đa
            turn_off: Bypass, trả về list rỗng

        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        if turn_off:
            return []

        return self.sm.get_messages_format(session_id, max_count=max_count)

    def save_message_to_history(
        self,
        session_id: str,
        question: str,
        response: str,
        agent_name: str,
        turn_off: bool = False
    ) -> None:
        """
        Lưu message vào history.

        Args:
            session_id: Session ID
            question: Câu hỏi
            response: Trả lời
            agent_name: Tên agent đã xử lý
            turn_off: Bypass, không lưu
        """
        if turn_off:
            return

        self.sm.add_message(session_id, question, response, agent_name)
        print(f"[SessionStep] Saved message to session {session_id}")


# =============================================================================
# STEP 2: CONTEXT BUILDER
# =============================================================================

class ContextBuilderStep:
    """
    STEP 2: Xây dựng context cho request.

    Tasks:
    - Parse các tham số từ request
    - Build AgentContext object
    - Prepare metadata cho agent
    """

    def build_context(
        self,
        question: str,
        session_id: str,
        chat_type: str,
        item_group: str = "GOODS",
        partner_group: str = "CUSTOMER",
        history: list = None
    ):
        """
        Xây dựng AgentContext từ request parameters.

        Args:
            question: Câu hỏi
            session_id: Session ID
            chat_type: Loại chat (thinking/free)
            item_group: Nhóm sản phẩm
            partner_group: Nhóm đối tác
            history: History messages từ session

        Returns:
            AgentContext object
        """
        from ..agents.base import AgentContext

        # === XÁC ĐỊNH CHẾ ĐỘ CHAT ===
        if chat_type == "free":
            mode_desc = "FREE - Trả lời tổng quát"
        else:
            mode_desc = "THINKING - Phân loại thông minh"

        print(f"[ContextStep] Building context for: {mode_desc}")

        # === BUILD CONTEXT OBJECT ===
        context = AgentContext(
            question=question,
            session_id=session_id,
            chat_type=chat_type,
            item_group=item_group,
            partner_group=partner_group,
            history=history or []
        )

        return context


# =============================================================================
# STEP 3: ROUTER
# =============================================================================

class AgentRouterStep:
    """
    STEP 3: Route câu hỏi đến agent phù hợp.

    Tasks:
    - Fast rule-based routing (O(1))
    - SLM classification với few-shot learning
    - Semantic fallback với embeddings
    - Return agent phù hợp nhất
    """

    def __init__(self):
        """Initialize router."""
        from ..agents import get_orchestrator
        self.orchestrator = get_orchestrator()

    def route_to_agent(
        self,
        context,
        use_fast_rules: bool = True,
        use_slm_classification: bool = True,
        use_semantic_fallback: bool = True,
        turn_off_routing: bool = False
    ):
        """
        Route câu hỏi đến agent phù hợp.

        Args:
            context: AgentContext object
            use_fast_rules: Dùng rule-based nhanh
            use_slm_classification: Dùng SLM để classify
            use_semantic_fallback: Fallback với semantic similarity
            turn_off_routing: Bypass, trả về GENERAL_FREE

        Returns:
            BaseAgent object
        """
        from ..agents.base import BaseAgent

        # === CHẾ ĐỘ FREE: BỎ QUA ROUTING ===
        if context.chat_type == "free" or turn_off_routing:
            print("[RouterStep] Mode: FREE - Using GENERAL_FREE agent")
            return self.orchestrator.get_agent("GENERAL_FREE")

        # === STEP 3.1: FAST RULE-BASED ROUTING (O(1)) ===
        if use_fast_rules:
            agent = self._fast_rule_based_routing(context)
            if agent:
                return agent

        # === STEP 3.2: SLM CLASSIFICATION ===
        if use_slm_classification:
            agent = self._slm_classification(context)
            if agent:
                return agent

        # === STEP 3.3: SEMANTIC FALLBACK ===
        if use_semantic_fallback:
            agent = self._semantic_fallback(context)
            if agent:
                return agent

        # === STEP 3.4: FINAL FALLBACK ===
        print("[RouterStep] Using fallback: GENERAL_ACCOUNTING")
        return self.orchestrator.get_agent("GENERAL_ACCOUNTING")

    def _fast_rule_based_routing(self, context):
        """
        Fast routing với rule-based matching.

        Rules:
        - Có số tài khoản (3-5 chữ số) → COA
        - Có từ "so sánh" → COA (compare)
        - Có keywords hạch toán → POSTING_ENGINE

        Args:
            context: AgentContext

        Returns:
            Agent hoặc None
        """
        import re

        question_lower = context.question.lower()

        # === RULE 1: CÓ SỐ TÀI KHOẢN ===
        code_match = re.search(r'\b\d{3,5}\b', context.question)
        if code_match:
            if any(kw in question_lower for kw in ["so sánh", "khác gì", "khác nhau"]):
                print("[RouterStep] Rule: COA (account + compare keyword)")
                return self.orchestrator.get_agent("COA")
            else:
                print("[RouterStep] Rule: COA (account number)")
                return self.orchestrator.get_agent("COA")

        # === RULE 2: KEYWORDS HẠCH TOÁN ===
        posting_keywords = [
            "hạch toán", "định khoản", "bút toán", "ghi nhận",
            "phiếu thu", "phiếu chi", "xuất hóa đơn"
        ]
        if any(kw in question_lower for kw in posting_keywords):
            print("[RouterStep] Rule: POSTING_ENGINE (posting keyword)")
            return self.orchestrator.get_agent("POSTING_ENGINE")

        # === RULE 3: SO SÁNH THÔNG TƯ ===
        if any(kw in question_lower for kw in ["tt99", "tt200", "thông tư"]):
            if "so sánh" in question_lower or "khác" in question_lower:
                print("[RouterStep] Rule: COA (circular compare)")
                return self.orchestrator.get_agent("COA")

        return None

    def _slm_classification(self, context):
        """
        Classification bằng SLM với few-shot learning.

        Args:
            context: AgentContext

        Returns:
            Agent hoặc None
        """
        from ..services.llm_service import get_llm_service
        from ..core.config import settings
        import json

        try:
            llm_service = get_llm_service()

            # Build classification prompt
            prompt = self._build_classification_prompt(context.question)

            # Build schema
            schema = self._get_classification_schema()

            # Call LLM
            response = llm_service.chat(
                model=settings.GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                use_cache=True  # Enable cache cho classification
            )

            # Parse result
            content = response.get("message", {}).get("content", "")
            result = json.loads(content)

            agent_name = result.get("agent")
            reasoning = result.get("reasoning", "")

            print(f"[RouterStep] SLM classified to: {agent_name}")
            print(f"[RouterStep] Reasoning: {reasoning[:100]}...")

            return self.orchestrator.get_agent(agent_name)

        except Exception as e:
            print(f"[RouterStep] SLM Error: {e}")
            return None

    def _semantic_fallback(self, context):
        """
        Semantic classification với embeddings.

        Args:
            context: AgentContext

        Returns:
            Agent hoặc None
        """
        import numpy as np

        try:
            from ..core.embeddings import get_embed_model

            model = get_embed_model()

            # Agent examples
            agent_examples = {
                "COA": ["TK 156 là gì?", "số hiệu 331", "danh mục tài khoản"],
                "POSTING_ENGINE": ["bán hàng hạch toán", "định khoản thế nào"],
                "GENERAL_ACCOUNTING": ["nguyên tắc kế toán", "báo cáo tài chính"],
                "GENERAL_FREE": ["hello", "xin chào", "cảm ơn"]
            }

            # Get agent embeddings
            if not hasattr(self, '_agent_embeddings'):
                self._init_agent_embeddings(model)

            # Encode query
            query_emb = model.encode(context.question, normalize_embeddings=True)

            # Find best match
            scores = {}
            for agent_name, data in self._agent_embeddings.items():
                sim = float(np.dot(query_emb, data["centroid"]))
                scores[agent_name] = sim

            best_agent = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_agent]

            if best_score > 0.3:
                print(f"[RouterStep] Semantic: {best_agent} (score: {best_score:.3f})")
                return self.orchestrator.get_agent(best_agent)

        except Exception as e:
            print(f"[RouterStep] Semantic Error: {e}")

        return None

    def _init_agent_embeddings(self, model):
        """Initialize agent embeddings."""
        from ..core.embeddings import encode_batch

        agent_examples = {
            "COA": ["TK 156 là gì?", "tài khoản 111", "số hiệu 331"],
            "POSTING_ENGINE": ["bán hàng hạch toán", "định khoản thế nào"],
            "GENERAL_ACCOUNTING": ["nguyên tắc kế toán", "báo cáo tài chính"],
            "GENERAL_FREE": ["hello", "xin chào"]
        }

        self._agent_embeddings = {}
        for agent_name, examples in agent_examples.items():
            embs = encode_batch(examples, normalize=True)
            self._agent_embeddings[agent_name] = {
                "embeddings": embs,
                "centroid": np.mean(embs, axis=0)
            }

    def _build_classification_prompt(self, question: str) -> str:
        """Build classification prompt cho SLM."""
        # Simplified prompt
        return f"""Phân loại câu hỏi kế toán sau vào agent phù hợp:

CÁC AGENTS:
1. COA: Tra cứu thông tin tài khoản (có số TK như 111, 156, 331)
2. POSTING_ENGINE: Hạch toán, định khoản nghiệp vụ
3. GENERAL_ACCOUNTING: Lý thuyết kế toán (nguyên tắc, báo cáo)
4. GENERAL_FREE: Câu hỏi chung, xã giao

Câu hỏi: {question}

Hãy phân loại và trả về JSON:
{{
    "agent": "COA",
    "reasoning": "Có số tài khoản 156"
}}
"""

    def _get_classification_schema(self) -> dict:
        """Get schema cho classification."""
        return {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": ["COA", "POSTING_ENGINE", "GENERAL_ACCOUNTING", "GENERAL_FREE"]
                },
                "reasoning": {
                    "type": "string",
                    "description": "Suy luận"
                }
            },
            "required": ["agent", "reasoning"]
        }


# =============================================================================
# STEP 4: SEMANTIC HISTORY CHECK
# =============================================================================

class HistorySearchStep:
    """
    STEP 4: Kiểm tra history bằng semantic similarity.

    Tasks:
    - Lấy history từ session
    - Extract keywords từ câu hỏi
    - Compute hybrid similarity (sentence + keywords)
    - Return response nếu match
    """

    def __init__(self):
        """Initialize semantic history checker."""
        from ..services.history_search import get_semantic_history_cache
        from ..core.config import settings

        self.cache = get_semantic_history_cache()
        self.mode = settings.SEMANTIC_MODE
        self.alpha = settings.SEMANTIC_ALPHA
        self.threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD

    def check_history(
        self,
        question: str,
        session_id: str,
        agent_name: str,
        chat_type: str = "thinking",
        turn_off: bool = False
    ):
        """
        Kiểm tra history bằng semantic similarity.

        Args:
            question: Câu hỏi hiện tại
            session_id: Session ID
            agent_name: Tên agent
            chat_type: Loại chat
            turn_off: Bypass kiểm tra history

        Returns:
            Response từ history nếu có match, None nếu không
        """
        if turn_off:
            return None

        if not session_id:
            return None

        print(f"[HistoryStep] Checking history (mode: {self.mode}, threshold: {self.threshold})")

        # === GET HISTORY ===
        from ..services.session_manager import get_session_manager
        sm = get_session_manager(chat_type)
        history = sm.get_history(session_id, max_count=50)

        if not history:
            print("[HistoryStep] No history found")
            return None

        # === FILTER BY AGENT ===
        agent_history = [
            item for item in history
            if item.get("category") == agent_name
        ]

        if not agent_history:
            print("[HistoryStep] No history for this agent")
            return None

        # === CHECK SIMILARITY ===
        response = self.cache.find_with_agent_hint(
            question=question,
            session_id=session_id,
            agent_name=agent_name,
            chat_type=chat_type
        )

        return response

    def extract_keywords(self, text: str) -> list:
        """
        Trích xuất keywords từ text.

        Args:
            text: Câu hỏi

        Returns:
            List of keywords
        """
        import re

        keywords = set()

        # Số tài khoản
        account_numbers = re.findall(r'\b\d{3,5}\b', text)
        keywords.update(account_numbers)

        # Từ khóa kế toán
        important_terms = [
            'hàng hóa', 'tiền mặt', 'phải thu', 'phải trả',
            'hạch toán', 'định khoản', 'bút toán', 'ghi nhận',
            'doanh thu', 'chi phí', 'lợi nhuận', 'nguyên vật liệu',
            'tài sản', 'nợ phải trả', 'vốn chủ sở hữu',
            'thuế', 'gtgt', 'khấu hao'
        ]

        text_lower = text.lower()
        for term in important_terms:
            if term in text_lower:
                keywords.add(term)

        return list(keywords)


# =============================================================================
# STEP 5: STREAMING CACHE CHECK
# =============================================================================

class StreamingCacheStep:
    """
    STEP 5: Kiểm tra streaming cache.

    Tasks:
    - Generate cache key từ request
    - Check cache storage
    - Return cached response nếu có
    - Simulate streaming từ cache
    """

    def __init__(self):
        """Initialize streaming cache checker."""
        from ..services.streaming_cache import get_streaming_cache
        from ..core.config import settings

        self.cache = get_streaming_cache()
        self.ttl = 3600  # 1 hour
        self.simulate_delay = settings.CACHE_SIMULATE_DELAY
        self.chars_per_chunk = settings.CACHE_CHARS_PER_CHUNK

    def check_cache(
        self,
        question: str,
        agent_name: str,
        cache_context: dict,
        turn_off: bool = False
    ):
        """
        Kiểm tra cache cho question.

        Khi cache hit: regenerate phần VÍ DỤ (part 4) với số mới.

        Args:
            question: Câu hỏi
            agent_name: Tên agent
            cache_context: Context dict (item_group, partner_group, etc.)
            turn_off: Bypass cache check

        Returns:
            Generator yielding chunks từ cache + example mới (hoặc None nếu cache miss)
        """
        if turn_off:
            return None

        # === GENERATE CACHE KEY ===
        cache_key = self._generate_cache_key(question, agent_name, cache_context)

        # === CHECK CACHE ===
        cached_response = self.cache.get(cache_key)

        if cached_response is not None:
            print(f"[CacheStep] ✓ CACHE HIT! Regenerating example with new numbers...")

            # Cache hit: regenerate example (part 4)
            full_response = self._regenerate_example(cached_response, agent_name)

            # Return generator simulate streaming
            return self._simulate_streaming_from_cache(full_response)

        print(f"[CacheStep] ✗ Cache miss (key: {cache_key[:12]}...)")
        return None

    def _regenerate_example(self, cached_response: str, agent_name: str) -> str:
        """
        Regenerate phần 4 (VÍ DỤ) với số mới, dùng LLM để phân loại.

        Preserve phần "Ghi chú" và "Lưu ý" từ cached response.

        Args:
            cached_response: Response từ cache (không có phần 4, có thể có Ghi chú/Lưu ý)
            agent_name: Tên agent

        Returns:
            Full response với example mới + Ghi chú/Lưu ý (nếu có)
        """
        import re

        # Tách cached response thành: main_content + footer (Ghi chú, Lưu ý)
        main_lines = []
        footer_lines = []
        in_footer = False

        for line in cached_response.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('Ghi chú:') or line_stripped.startswith('Lưu ý:'):
                in_footer = True

            if in_footer:
                footer_lines.append(line)
            else:
                main_lines.append(line)

        main_content = '\n'.join(main_lines).strip()
        footer_content = '\n'.join(footer_lines).strip() if footer_lines else ""

        # Bước 1: Dùng LLM để xác định tx_type
        tx_type = self._classify_tx_type_with_llm(cached_response, agent_name)

        # Bước 2: Generate example với số ngẫu nhiên
        example = self._generate_example_for_tx_type(tx_type, cached_response)

        # Bước 3: Combine: main + example + footer
        result = f"{main_content}\n\n4. VÍ DỤ:\n{example}"
        if footer_content:
            result += f"\n\n{footer_content}"

        return result

    def _classify_tx_type_with_llm(self, cached_response: str, agent_name: str) -> str:
        """
        Dùng LLM để phân loại tx_type từ cached response.

        Thông minh hơn regex/account mapping - LLM hiểu ngữ nghĩa.
        """
        from ..core.ollama_client import get_ollama_client
        from ..core.config import settings

        client = get_ollama_client()

        # Lấy phần đầu của response (chứa tên nghiệp vụ và bút toán)
        first_part = '\n'.join(cached_response.split('\n')[:20])

        prompt = f"""Phân loại loại giao dịch kế toán sau thành MỘT trong các loại sau:

CÁC LOẠI GIAO DỊCH:
- DO_SALE: Xuất kho bán hàng (chưa xuất hóa đơn)
- SALES_INVOICE: Xuất hóa đơn bán hàng
- CASH_IN: Thu tiền từ khách hàng
- GRN_PURCHASE: Nhập kho mua hàng (chưa có hóa đơn)
- PURCHASE_INVOICE: Nhận hóa đơn mua hàng
- CASH_OUT: Chi tiền cho nhà cung cấp

RESPONSE ĐỂ PHÂN LOẠI:
{first_part}

CHỈ TRẢ VỀ MỘT TỪ: tên loại giao dịch (ví dụ: DO_SALE, SALES_INVOICE, CASH_IN, etc.)"""

        try:
            response = client.chat(
                model=settings.CLASSIFIER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options=settings.OLLAMA_OPTIONS,
                stream=False
            )

            result = response.get("message", {}).get("content", "").strip().upper()

            # Clean result - remove common variations
            for valid_type in ['DO_SALE', 'SALES_INVOICE', 'CASH_IN', 'GRN_PURCHASE', 'PURCHASE_INVOICE', 'CASH_OUT']:
                if valid_type in result:
                    print(f"[RegenerateExample] LLM classified as: {valid_type}")
                    return valid_type

            print(f"[RegenerateExample] LLM returned unknown: {result}, using DO_SALE fallback")
            return 'DO_SALE'

        except Exception as e:
            print(f"[RegenerateExample] LLM classification failed: {e}, using DO_SALE fallback")
            return 'DO_SALE'

    def _generate_example_for_tx_type(self, tx_type: str, cached_response: str) -> str:
        """
        Generate example cho tx_type với số ngẫu nhiên.
        """
        import random
        import re

        # Config: tx_type -> description template
        DESC_TEMPLATES = {
            'DO_SALE': "Công ty giao hàng cho khách A, giá trị hàng {amount:,}đ (giá vốn {cost:,}đ), thuế GTGT {tax:,}đ (chưa xuất HĐ).",
            'SALES_INVOICE': "Xuất hóa đơn cho khách A giá trị hàng hóa {amount:,}đ, thuế GTGT {tax:,}đ.",
            'CASH_IN': "Khách hàng thanh toán {amount:,}đ cho công nợ.",
            'GRN_PURCHASE': "Nhập {amount_qty} nguyên liệu giá {unit_price:,}đ/kg (tổng {amount:,}đ), thuế GTGT {tax:,}đ, chưa nhận hóa đơn.",
            'PURCHASE_INVOICE': "Nhận hóa đơn NCC giá trị hàng hóa {amount:,}đ, thuế GTGT {tax:,}đ.",
            'CASH_OUT': "Thanh toán {amount:,}đ cho NCC thanh toán công nợ.",
        }

        # Config: Account -> amount calculation rule
        AMOUNT_RULES = {
            '153': lambda b, t: t,
            '33311': lambda b, t: t,
            '632': lambda b, t: int(b * 0.6),
            '13881': lambda b, t: b + t,
            '33881': lambda b, t: b + t,
        }

        # Extract entries from cached response (flexible regex)
        entries = []
        for line in cached_response.split('\n'):
            match = re.match(r'-?\s*(Nợ|Có)\s+TK\s+(\d+):', line.strip())
            if match:
                entries.append({
                    'side': match.group(1),
                    'account': match.group(2)
                })

        # Deduplicate
        seen = set()
        unique_entries = []
        for e in entries:
            key = f"{e['side']}_{e['account']}"
            if key not in seen:
                seen.add(key)
                unique_entries.append(e)
        entries = unique_entries

        # Generate random amounts
        base_amount = random.randint(1, 900) * 1000000
        tax_amount = base_amount // 10
        cost_amount = int(base_amount * 0.6)

        # Build description
        template = DESC_TEMPLATES.get(tx_type, "Giao dịch có giá trị {amount:,}đ.")

        # Special handling for GRN_PURCHASE with quantity
        if tx_type == 'GRN_PURCHASE':
            unit_price = random.randint(100, 500) * 1000
            amount_qty = random.randint(100, 1000)
            base_amount = unit_price * amount_qty
            tax_amount = base_amount // 10
            description = template.format(amount=base_amount, tax=tax_amount,
                                         amount_qty=amount_qty, unit_price=unit_price)
        else:
            description = template.format(amount=base_amount, tax=tax_amount, cost=cost_amount)

        # Build example lines
        example_lines = [description]

        for entry in entries:
            acc = entry['account']
            amount = AMOUNT_RULES.get(acc, lambda b, t: b)(base_amount, tax_amount)

            if entry['side'] == 'Nợ':
                example_lines.append(f"- Nợ TK {entry['account']}: {amount:,}đ")
            else:
                example_lines.append(f"- Có TK {entry['account']}: {amount:,}đ")

        return '\n'.join(example_lines)

    def _generate_cache_key(self, question: str, agent_name: str, context: dict) -> str:
        """
        Generate cache key từ request parameters.

        Args:
            question: Câu hỏi
            agent_name: Tên agent
            context: Context dict

        Returns:
            MD5 hash key
        """
        import hashlib
        import json

        key_data = {
            "question": question,
            "agent": agent_name,
        }

        # Add relevant context
        relevant_context = {
            k: v for k, v in context.items()
            if k in ["item_group", "partner_group", "chat_type"]
        }
        if relevant_context:
            key_data["context"] = relevant_context

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _simulate_streaming_from_cache(self, response: str):
        """
        Simulate streaming từ cached response.

        Args:
            response: Full cached response

        Yields:
            Chunks để simulate typing effect
        """
        import time

        print(f"[CacheStep] Simulating streaming ({len(response)} chars)")

        for i in range(0, len(response), self.chars_per_chunk):
            chunk = response[i:i + self.chars_per_chunk]
            yield chunk
            # Delay để simulate typing
            time.sleep(self.simulate_delay)

    def save_to_cache(self, question: str, agent_name: str, response: str, cache_context: dict):
        """
        Save response to cache (WITHOUT example - part 4).

        Example sẽ được regenerate mới mỗi cache hit để có số khác.
        Phần "Ghi chú" và "Lưu ý" được giữ lại.

        Args:
            question: Câu hỏi
            agent_name: Tên agent
            response: Full response
            cache_context: Context dict
        """
        import re

        # Strip phần 4 (VÍ DỤ) nhưng GIỮ lại phần "Ghi chú" và "Lưu ý"
        lines = response.split('\n')
        cache_lines = []
        skip_example = False

        for line in lines:
            line_stripped = line.strip()

            # Bắt đầu phần 4 - skip dòng header
            if re.match(r'^4\.\s*VÍ DỤ:', line_stripped):
                skip_example = True
                continue

            # Kết thúc phần 4 - khi gặp "Ghi chú:", "Lưu ý:", hoặc section mới
            if skip_example and (
                line_stripped.startswith('Ghi chú:') or
                line_stripped.startswith('Lưu ý:') or
                re.match(r'^\d+\.', line_stripped)  # Section mới như "5."
            ):
                skip_example = False

            # Skip chỉ các dòng example (có "-" hoặc là mô tả context)
            # Giữ lại các dòng khác như "Ghi chú:", "Lưu ý:"
            if skip_example:
                # Skip nếu là dòng example (bắt đầu bằng "-") hoặc dòng mô tả
                if line_stripped.startswith('-') or (
                    line_stripped and not line_stripped.startswith('Ghi chú') and
                    not line_stripped.startswith('Lưu ý') and
                    not re.match(r'^\d+\.', line_stripped) and
                    ':' not in line_stripped  # Không có dấu ":" -> không phải header
                ):
                    continue

            cache_lines.append(line)

        cached_response = '\n'.join(cache_lines).strip()

        cache_key = self._generate_cache_key(question, agent_name, cache_context)
        self.cache.set(cache_key, cached_response)
        print(f"[CacheStep] Saved to cache WITHOUT example (key: {cache_key[:12]}...)")


# =============================================================================
# STEP 6: AGENT EXECUTOR
# =============================================================================

class AgentExecutorStep:
    """
    STEP 6: Thực thi agent và gọi LLM.

    Tasks:
    - Extract keywords từ câu hỏi
    - Search data (COA, Posting Engine)
    - Build context từ data
    - Build prompt cho LLM
    - Call Ollama streaming
    """

    def __init__(self):
        """Initialize agent executor."""
        from ..core.ollama_client import get_ollama_client
        from ..core.config import settings

        self.client = get_ollama_client()
        self.model = settings.GENERATION_MODEL
        self.options = settings.OLLAMA_OPTIONS

    def execute_agent(
        self,
        agent,
        context,
        turn_off_llm: bool = False
    ):
        """
        Thực thi agent và stream response.

        Args:
            agent: BaseAgent instance
            context: AgentContext
            turn_off_llm: Bypass LLM call (cho testing)

        Yields:
            Response chunks from agent (or LLM)
        """
        print(f"[ExecutorStep] Executing agent: {agent.name}")

        if turn_off_llm:
            # Mock response cho testing
            yield f"[Mock response from {agent.name}]"
            return

        # === AGENT STREAM EXECUTE ===
        for chunk in agent.stream_execute(context):
            yield chunk


# =============================================================================
# STEP 7: STREAM PROCESSOR
# =============================================================================

class StreamProcessorStep:
    """
    STEP 7: Xử lý streaming response.

    Tasks:
    - Buffer streaming chunks
    - Optimize buffer size
    - Yield mượt mà cho frontend
    """

    def __init__(self):
        """Initialize stream processor."""
        from ..core.config import settings
        self.buffer_size_words = settings.CACHE_CHARS_PER_CHUNK  # Reuse as char chunk size
        self.buffer_size_words = 5  # Words per buffer

    def process_stream(
        self,
        ollama_stream,
        buffer_size: int = 5,
        turn_off_processing: bool = False
    ):
        """
        Process ollama stream và yield optimized chunks.

        Args:
            ollama_stream: Iterator từ ollama.chat(stream=True)
            buffer_size: Số từ mỗi buffer
            turn_off_processing: Pass-through, không xử lý

        Yields:
            Optimized chunks
        """
        if turn_off_processing:
            for chunk in ollama_stream:
                yield chunk
            return

        from ..services.stream_utils import stream_by_sentence

        # Use existing stream utils
        for chunk in stream_by_sentence(ollama_stream, buffer_size_words=buffer_size):
            yield chunk


# =============================================================================
# STEP 8: RESPONSE SAVER
# =============================================================================

class ResponseSaverStep:
    """
    STEP 8: Lưu response và cache.

    Tasks:
    - Accumulate full response
    - Save to streaming cache
    - Save to session history
    """

    def save_response(
        self,
        question: str,
        response_chunks: list,
        session_id: str,
        agent_name: str,
        cache_checker,
        item_group: str = "GOODS",
        partner_group: str = "CUSTOMER",
        chat_type: str = "thinking",
        turn_off_saving: bool = False
    ):
        """
        Lưu response vào cache và history.

        Args:
            question: Câu hỏi
            response_chunks: List of chunks
            session_id: Session ID
            agent_name: Tên agent
            cache_checker: StreamingCacheCheckerStep instance
            item_group: Nhóm sản phẩm
            partner_group: Nhóm đối tác
            chat_type: Loại chat
            turn_off_saving: Bypass saving
        """
        if turn_off_saving:
            return

        # === ACCUMULATE FULL RESPONSE ===
        full_response = "".join(response_chunks)

        # === SAVE TO CACHE ===
        cache_context = {
            "item_group": item_group,
            "partner_group": partner_group,
            "chat_type": chat_type
        }
        cache_checker.save_to_cache(question, agent_name, full_response, cache_context)

        # === SAVE TO HISTORY ===
        from ..services.session_manager import get_session_manager
        sm = get_session_manager("thinking")
        sm.add_message(session_id, question, full_response, agent_name)

        print(f"[SaverStep] Saved response ({len(full_response)} chars)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class AccountingPipeline:
    """
    Main Pipeline - Kết hợp tất cả các bước.

    Usage:
        pipeline = AccountingPipeline()

        for chunk in pipeline.process(
            question="TK 156 là gì?",
            session_id=None,
            chat_type="thinking"
        ):
            yield chunk  # Stream to user
    """

    def __init__(self):
        """Initialize pipeline với tất cả các steps."""
        print("[Pipeline] Initializing Accounting Pipeline...")

        # Initialize các steps
        self.session_step = SessionManagerStep()
        self.context_step = ContextBuilderStep()
        self.router_step = AgentRouterStep()
        self.history_checker = HistorySearchStep()
        self.cache_checker = StreamingCacheStep()
        self.executor = AgentExecutorStep()
        self.stream_processor = StreamProcessorStep()
        self.saver = ResponseSaverStep()

        print("[Pipeline] Pipeline initialized successfully!")

    def process(
        self,
        question: str,
        session_id: str = None,
        chat_type: str = "thinking",
        item_group: str = "GOODS",
        partner_group: str = "CUSTOMER",

        # Optional flags để turn off các steps
        turn_off_session: bool = False,
        turn_off_routing: bool = False,
        turn_off_history_check: bool = False,
        turn_off_cache: bool = False,
        turn_off_llm: bool = False,
        turn_off_stream_processing: bool = False,
        turn_off_saving: bool = False
    ):
        """
        Xử lý question và stream response.

        Args:
            question: Câu hỏi
            session_id: Session ID
            chat_type: Loại chat
            item_group: Nhóm sản phẩm
            partner_group: Nhóm đối tác
            turn_off_*: Flags để bypass các bước

        Yields:
            str: Response chunks (từng chữ một)
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] Processing: {question}")
        print(f"{'='*60}\n")

        # =========================================================================
        # STEP 1: SESSION MANAGEMENT
        # =========================================================================
        print("[Pipeline] STEP 1: Session Management")
        session_id = self.session_step.create_session_if_needed(session_id, turn_off=turn_off_session)
        yield f"__SESSION_ID__:{session_id}\n"

        # =========================================================================
        # STEP 2: BUILD CONTEXT
        # =========================================================================
        print("[Pipeline] STEP 2: Building Context")
        history_messages = self.session_step.format_history_for_llm(
            session_id, max_count=10, turn_off=turn_off_session
        )
        context = self.context_step.build_context(
            question=question,
            session_id=session_id,
            chat_type=chat_type,
            item_group=item_group,
            partner_group=partner_group,
            history=history_messages
        )

        # =========================================================================
        # FREE MODE: Skip routing, go directly to agent
        # =========================================================================
        if chat_type == "free":
            print("[Pipeline] FREE MODE - Skipping routing")
            agent = self.router_step.orchestrator.get_agent("GENERAL_FREE")

            response_chunks = []
            for chunk in self.executor.execute_agent(agent, context, turn_off_llm):
                response_chunks.append(chunk)
                yield chunk

            self.saver.save_response(
                question, response_chunks, session_id, "GENERAL_FREE",
                self.cache_checker,
                item_group=item_group,
                partner_group=partner_group,
                chat_type=chat_type,
                turn_off_saving=turn_off_saving
            )
            return

        # =========================================================================
        # STEP 3: ROUTE TO AGENT
        # =========================================================================
        print("[Pipeline] STEP 3: Routing to Agent")
        agent = self.router_step.route_to_agent(context, turn_off_routing=turn_off_routing)
        agent_name = agent.name

        # =========================================================================
        # STEP 4: STREAMING CACHE CHECK (History disabled)
        # =========================================================================
        print("[Pipeline] STEP 4: Streaming Cache Check")
        cached_stream_gen = self.cache_checker.check_cache(
            question=question,
            agent_name=agent_name,
            cache_context={
                "item_group": item_group,
                "partner_group": partner_group,
                "chat_type": chat_type
            },
            turn_off=turn_off_cache
        )

        if cached_stream_gen is not None:
            # Cache hit! Stream từ cache
            print("[Pipeline] ✓ Cache hit - Streaming from cache...")

            response_chunks = []
            for chunk in cached_stream_gen:
                response_chunks.append(chunk)
                yield chunk

            # Response đã được lưu trong cache_checker
            return

        # =========================================================================
        # STEP 5: AGENT EXECUTION (LLM Call)
        # =========================================================================
        print("[Pipeline] STEP 5: Agent Execution (Calling LLM...)")
        response_chunks = []

        # Execute agent và stream response
        llm_stream = self.executor.execute_agent(agent, context, turn_off_llm=turn_off_llm)

        # =========================================================================
        # STEP 6: STREAM PROCESSING
        # =========================================================================
        print("[Pipeline] STEP 6: Stream Processing")
        # Pass-through: agent đã handle streaming với stream_by_char
        for chunk in self.stream_processor.process_stream(
            llm_stream, buffer_size=5, turn_off_processing=True  # Agent đã xử lý streaming
        ):
            response_chunks.append(chunk)
            yield chunk

        # =========================================================================
        # STEP 7: SAVE RESPONSE
        # =========================================================================
        print("[Pipeline] STEP 7: Saving Response")
        self.saver.save_response(
            question, response_chunks, session_id, agent_name,
            self.cache_checker,
            item_group=item_group,
            partner_group=partner_group,
            chat_type=chat_type,
            turn_off_saving=turn_off_saving
        )

        print(f"[Pipeline] ✓ Completed. Total response: {len(''.join(response_chunks))} chars")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

# Global pipeline instance
_pipeline_instance = None


def get_pipeline() -> AccountingPipeline:
    """Get singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AccountingPipeline()
    return _pipeline_instance
