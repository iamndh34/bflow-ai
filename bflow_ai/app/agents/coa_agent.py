"""
COA Agent - Chart of Accounts Specialist

Chuyên gia về:
- Tra cứu thông tin tài khoản
- So sánh tài khoản giữa TT200 và TT99
- So sánh tổng quan giữa 2 thông tư
"""
import json
import os
import re
from collections import defaultdict
from typing import Optional, List, Dict, Any

import numpy as np

from .base import BaseAgent, AgentRole, AgentResult, AgentContext, Tool
from ..core.config import settings
from ..core.ollama_client import get_ollama_client
from ..core.embeddings import get_embed_model, batch_cosine_similarity, encode_batch
from ..services.stream_utils import stream_by_char
from .templates import get_lookup_template, get_compare_template, get_compare_circular_template


# =============================================================================
# CONFIG & DATA LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COA_99_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_99.json")
COA_200_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_200.json")
COA_COMPARE_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_compare_99_vs_200.json")

# Load TT99
if not os.path.exists(COA_99_FILE):
    print(f"[WARN] {COA_99_FILE} not found.")
    COA_99_DATA = []
else:
    with open(COA_99_FILE, "r", encoding="utf-8") as f:
        COA_99_DATA = json.load(f)

# Load TT200
if not os.path.exists(COA_200_FILE):
    print(f"[WARN] {COA_200_FILE} not found.")
    COA_200_DATA = []
else:
    with open(COA_200_FILE, "r", encoding="utf-8") as f:
        COA_200_DATA = json.load(f)

# Load Compare Data
if not os.path.exists(COA_COMPARE_FILE):
    print(f"[WARN] {COA_COMPARE_FILE} not found.")
    COA_COMPARE_DATA = []
else:
    with open(COA_COMPARE_FILE, "r", encoding="utf-8") as f:
        COA_COMPARE_DATA = json.load(f)

# Build indexes
COA_DATA = COA_99_DATA
COA_BY_CODE = {acc["code"]: acc for acc in COA_99_DATA}
COA_BY_TYPE = {}
for acc in COA_99_DATA:
    type_name = acc["type_name"]
    if type_name not in COA_BY_TYPE:
        COA_BY_TYPE[type_name] = []
    COA_BY_TYPE[type_name].append(acc)

COA_200_BY_CODE = {acc["code"]: acc for acc in COA_200_DATA}

COA_COMPARE_BY_TYPE = defaultdict(list)
for item in COA_COMPARE_DATA:
    COA_COMPARE_BY_TYPE[item["change_type"]].append(item)


# =============================================================================
# COA AGENT
# =============================================================================

class COAAgent(BaseAgent):
    """
    Chart of Accounts Agent - Chuyên gia về tài khoản kế toán

    Xử lý:
    - Tra cứu thông tin tài khoản (COA)
    - So sánh tài khoản giữa TT200 và TT99 (COMPARE)
    - So sánh tổng quan 2 thông tư (COMPARE_CIRCULAR)
    """

    _embed_model = None
    _coa_embeddings = None

    def __init__(self):
        super().__init__()
        self._init_tools()

    @property
    def name(self) -> str:
        return "COA"

    @property
    def role(self) -> AgentRole:
        return AgentRole.DOMAIN_SPECIALIST

    @property
    def description(self) -> str:
        return "Chuyên gia về hệ thống tài khoản kế toán Việt Nam (TT99, TT200). Tra cứu thông tin tài khoản, so sánh giữa các thông tư."

    def _init_tools(self):
        """Đăng ký tools cho agent"""
        self.add_tool(
            name="lookup_by_code",
            description="Tra cứu tài khoản theo số hiệu",
            func=self._tool_lookup_by_code
        )
        self.add_tool(
            name="lookup_by_type",
            description="Tra cứu danh sách tài khoản theo loại",
            func=self._tool_lookup_by_type
        )
        self.add_tool(
            name="search_by_name",
            description="Tìm kiếm tài khoản theo tên",
            func=self._tool_search_by_name
        )
        self.add_tool(
            name="search_by_embedding",
            description="Tìm kiếm tài khoản bằng semantic search",
            func=self._tool_search_by_embedding
        )
        self.add_tool(
            name="compare_accounts",
            description="So sánh tài khoản giữa TT200 và TT99",
            func=self._tool_compare_accounts
        )
        self.add_tool(
            name="compare_circular",
            description="So sánh tổng quan giữa TT200 và TT99",
            func=self._tool_compare_circular
        )

    def can_handle(self, context: AgentContext) -> tuple[bool, float]:
        """
        Kiểm tra agent có thể xử lý query không.

        Criteria:
        - Có chứa số tài khoản (3-5 chữ số)
        - Có từ khóa so sánh + nhắc đến thông tư
        - Hỏi về tài khoản, hệ thống tài khoản
        """
        question = context.question.lower()

        # Check keywords liên quan đến COA
        coa_keywords = [
            "tài khoản", "tk", "số hiệu", "thông tư", "tt99", "tt200",
            "hệ thống tài khoản", "danh mục tài khoản", "tk ", " tk"
        ]

        has_coa_keyword = any(kw in question for kw in coa_keywords)

        # Check account number pattern
        has_account_number = bool(re.search(r'\b(\d{3,5})\b', context.question))

        # Check comparison keywords
        compare_keywords = ["so sánh", "khác gì", "khác nhau", "giữa", "và"]
        has_compare_keyword = any(kw in question for kw in compare_keywords)

        # Determine confidence
        confidence = 0.0

        if has_account_number:
            # Có số TK -> rất chắc chắn là COA
            confidence = 0.95
        elif has_coa_keyword and has_compare_keyword:
            # So sánh thông tư -> chắc chắn
            confidence = 0.90
        elif has_coa_keyword:
            # Có keyword COA -> có thể
            confidence = 0.70

        return confidence > 0.5, confidence

    def execute(self, context: AgentContext) -> AgentResult:
        """Thực thi query"""
        question = context.question
        question_lower = question.lower()

        # Determine query type
        code_match = re.search(r'\b(\d{3,5})\b', question)
        has_compare = any(kw in question_lower for kw in ["so sánh", "khác gì", "khác nhau"])

        if has_compare:
            if code_match:
                # So sánh 1 TK cụ thể
                return self._execute_compare(context)
            else:
                # So sánh tổng quan
                return self._execute_compare_circular(context)
        else:
            # Tra cứu thông tin TK
            return self._execute_lookup(context)

    def _execute_lookup(self, context: AgentContext) -> AgentResult:
        """Tra cứu thông tin tài khoản"""
        import ollama

        question = context.question
        question_lower = question.lower()

        # Find accounts
        accounts = self._find_accounts(question, question_lower)

        if not accounts:
            return AgentResult(
                agent_name=self.name,
                content="Không tìm thấy tài khoản phù hợp trong danh mục Thông tư 99.",
                confidence=0.5
            )

        # Build context
        if len(accounts) == 1:
            acc = accounts[0]
            acc_info = f"TK {acc['code']} - {acc['name']} ({acc.get('name_en', 'N/A')}), Loại: {acc['type_name']}"
            sources = [f"TT99 - TK {acc['code']}"]
        else:
            acc_info = "\n".join([f"- TK {a['code']}: {a['name']} ({a['type_name']})" for a in accounts[:5]])
            sources = [f"TT99 - {len(accounts)} TKs"]

        # Generate response using SLM
        system_prompt = "Bạn là trợ lý kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT. Trả lời đầy đủ, cấu trúc rõ ràng."
        slm_prompt = f"""Câu hỏi: {question}

Thông tin tài khoản tìm được:
{acc_info}

Hãy trả lời theo định dạng sau:
{get_lookup_template()}"""

        try:
            client = get_ollama_client()
            response = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": slm_prompt}
                ],
                options=settings.OLLAMA_OPTIONS,
                stream=False
            )
            content = response.get("message", {}).get("content", "")

            if not content or "1." not in content:
                content = self._generate_fallback(accounts)

            content += "\n\n(Căn cứ: Phụ lục II - Thông tư 99/2025/TT-BTC)"

            return AgentResult(
                agent_name=self.name,
                content=content,
                confidence=0.9,
                sources=sources
            )
        except Exception as e:
            print(f"[COAAgent Error] {e}")
            return AgentResult(
                agent_name=self.name,
                content=self._generate_fallback(accounts),
                confidence=0.7,
                sources=sources
            )

    def _execute_compare(self, context: AgentContext) -> AgentResult:
        """So sánh tài khoản giữa TT200 và TT99"""
        import ollama

        question = context.question

        # Extract account code
        code_match = re.search(r'\b(\d{3,5})\b', question)
        if not code_match:
            return AgentResult(
                agent_name=self.name,
                content="Vui lòng chỉ định số tài khoản cần so sánh (ví dụ: 'So sánh TK 156 giữa TT200 và TT99').",
                confidence=0.3
            )

        code = code_match.group(1)
        acc_99 = COA_BY_CODE.get(code)
        acc_200 = COA_200_BY_CODE.get(code)

        if not acc_99 and not acc_200:
            return AgentResult(
                agent_name=self.name,
                content=f"Không tìm thấy tài khoản {code} trong cả TT200 và TT99.",
                confidence=0.5
            )

        compare_info = self._build_compare_context(code, acc_99, acc_200)

        system_prompt = "Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT. So sánh sự khác biệt giữa 2 thông tư một cách rõ ràng."
        slm_prompt = f"""Câu hỏi: {question}

Thông tin so sánh:
{compare_info}

Hãy trả lời theo định dạng sau:
{get_compare_template()}"""

        try:
            client = get_ollama_client()
            response = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": slm_prompt}
                ],
                options=settings.OLLAMA_OPTIONS,
                stream=False
            )
            content = response.get("message", {}).get("content", "")

            if not content or "SO SÁNH" not in content.upper():
                content = self._generate_compare_fallback(code, acc_99, acc_200)

            sources = []
            if acc_200:
                sources.append(f"TT200 - TK {code}")
            if acc_99:
                sources.append(f"TT99 - TK {code}")

            return AgentResult(
                agent_name=self.name,
                content=content,
                confidence=0.95,
                sources=sources
            )
        except Exception as e:
            print(f"[COAAgent Compare Error] {e}")
            return AgentResult(
                agent_name=self.name,
                content=self._generate_compare_fallback(code, acc_99, acc_200),
                confidence=0.7
            )

    def _execute_compare_circular(self, context: AgentContext) -> AgentResult:
        """So sánh tổng quan giữa TT200 và TT99"""
        import ollama

        question = context.question
        diff = self._analyze_circular_diff()
        ctx = self._build_circular_context(diff)

        system_prompt = """Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT.
Dựa trên dữ liệu so sánh được cung cấp, hãy tóm tắt những điểm khác biệt chính giữa Thông tư 200/2014/TT-BTC và Thông tư 99/2025/TT-BTC về hệ thống tài khoản kế toán.
Trình bày rõ ràng, có cấu trúc, dễ hiểu."""

        slm_prompt = f"""Câu hỏi: {question}

Dữ liệu so sánh giữa TT200 và TT99:
{ctx}

YÊU CẦU: Trả lời ĐẦY ĐỦ theo format sau:
{get_compare_circular_template()}"""

        try:
            client = get_ollama_client()
            response = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": slm_prompt}
                ],
                options=settings.OLLAMA_OPTIONS,
                stream=False
            )
            content = response.get("message", {}).get("content", "")

            if not content or len(content) < 50:
                content = self._generate_circular_fallback(diff)

            return AgentResult(
                agent_name=self.name,
                content=content,
                confidence=0.95,
                sources=["TT200", "TT99"]
            )
        except Exception as e:
            print(f"[COAAgent CompareCircular Error] {e}")
            return AgentResult(
                agent_name=self.name,
                content=self._generate_circular_fallback(diff),
                confidence=0.7
            )

    # =========================================================================
    # STREAMING VERSION
    # =========================================================================

    def stream_execute(self, context: AgentContext):
        """Execute với streaming response"""
        import ollama

        question = context.question
        question_lower = question.lower()

        # Determine query type
        code_match = re.search(r'\b(\d{3,5})\b', question)
        has_compare = any(kw in question_lower for kw in ["so sánh", "khác gì", "khác nhau"])

        if has_compare:
            if code_match:
                # So sánh 1 TK cụ thể
                yield from self._stream_compare(context)
            else:
                # So sánh tổng quan
                yield from self._stream_compare_circular(context)
        else:
            # Tra cứu thông tin TK
            yield from self._stream_lookup(context)

    def _stream_lookup(self, context: AgentContext):
        """Tra cứu với streaming"""
        import ollama

        question = context.question
        question_lower = question.lower()

        # Find accounts
        accounts = self._find_accounts(question, question_lower)

        if not accounts:
            yield "Không tìm thấy tài khoản phù hợp trong danh mục Thông tư 99."
            return

        # Build context
        if len(accounts) == 1:
            acc = accounts[0]
            acc_info = f"TK {acc['code']} - {acc['name']} ({acc.get('name_en', 'N/A')}), Loại: {acc['type_name']}"
        else:
            acc_info = "\n".join([f"- TK {a['code']}: {a['name']} ({a['type_name']})" for a in accounts[:5]])

        # Build prompt
        system_prompt = "Bạn là trợ lý kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT. Trả lời đầy đủ, cấu trúc rõ ràng."
        slm_prompt = f"""Câu hỏi: {question}

Thông tin tài khoản tìm được:
{acc_info}

Hãy trả lời theo định dạng sau:
{get_lookup_template()}"""

        try:
            client = get_ollama_client()
            stream = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": slm_prompt}
                ],
                options=settings.OLLAMA_OPTIONS,
                stream=True
            )
            slm_output = ""
            for char in stream_by_char(stream):
                slm_output += char
                yield char

            # Fallback if needed
            if not slm_output or "1." not in slm_output:
                yield self._generate_fallback(accounts)
            else:
                yield "\n\n(Căn cứ: Phụ lục II - Thông tư 99/2025/TT-BTC)"

        except Exception as e:
            print(f"[COAAgent Stream Error] {e}")
            yield self._generate_fallback(accounts)

    def _stream_compare(self, context: AgentContext):
        """So sánh với streaming"""
        import ollama

        question = context.question

        code_match = re.search(r'\b(\d{3,5})\b', question)
        if not code_match:
            yield "Vui lòng chỉ định số tài khoản cần so sánh (ví dụ: 'So sánh TK 156 giữa TT200 và TT99')."
            return

        code = code_match.group(1)
        acc_99 = COA_BY_CODE.get(code)
        acc_200 = COA_200_BY_CODE.get(code)

        if not acc_99 and not acc_200:
            yield f"Không tìm thấy tài khoản {code} trong cả TT200 và TT99."
            return

        compare_info = self._build_compare_context(code, acc_99, acc_200)

        system_prompt = "Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT. So sánh sự khác biệt giữa 2 thông tư một cách rõ ràng."
        slm_prompt = f"""Câu hỏi: {question}

Thông tin so sánh:
{compare_info}

Hãy trả lời theo định dạng sau:
{get_compare_template()}"""

        try:
            client = get_ollama_client()
            stream = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": slm_prompt}
                ],
                options=settings.OLLAMA_OPTIONS,
                stream=True
            )
            slm_output = ""
            for char in stream_by_char(stream):
                slm_output += char
                yield char

            if not slm_output or "SO SÁNH" not in slm_output.upper():
                yield self._generate_compare_fallback(code, acc_99, acc_200)

        except Exception as e:
            print(f"[COAAgent Compare Stream Error] {e}")
            yield self._generate_compare_fallback(code, acc_99, acc_200)

    def _stream_compare_circular(self, context: AgentContext):
        """So sánh tổng quan với streaming"""
        import ollama

        question = context.question
        diff = self._analyze_circular_diff()
        ctx = self._build_circular_context(diff)

        system_prompt = """Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT.
Dựa trên dữ liệu so sánh được cung cấp, hãy tóm tắt những điểm khác biệt chính giữa Thông tư 200/2014/TT-BTC và Thông tư 99/2025/TT-BTC về hệ thống tài khoản kế toán.
Trình bày rõ ràng, có cấu trúc, dễ hiểu."""

        slm_prompt = f"""Câu hỏi: {question}

Dữ liệu so sánh giữa TT200 và TT99:
{ctx}

YÊU CẦU: Trả lời ĐẦY ĐỦ theo format sau:
{get_compare_circular_template()}"""

        try:
            client = get_ollama_client()
            stream = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": slm_prompt}
                ],
                options=settings.OLLAMA_OPTIONS,
                stream=True
            )
            slm_output = ""
            for char in stream_by_char(stream):
                slm_output += char
                yield char

            if not slm_output or len(slm_output) < 50:
                yield self._generate_circular_fallback(diff)

        except Exception as e:
            print(f"[COAAgent CompareCircular Stream Error] {e}")
            yield self._generate_circular_fallback(diff)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @classmethod
    def _init_embeddings(cls):
        """
        Lazy init embeddings với batch processing.

        Optimized: Encode tất cả accounts trong 1 batch thay vì loop.
        """
        if cls._embed_model is None:
            cls._embed_model = get_embed_model()

            # Batch encode tất cả accounts - nhanh hơn loop rất nhiều
            if COA_DATA:
                texts = []
                codes = []
                for acc in COA_DATA:
                    text = f"{acc['code']} {acc['name']} {acc.get('name_en', '')} {acc['type_name']}"
                    texts.append(text)
                    codes.append(acc["code"])

                # Batch encode - nhanh hơn 10-20x so với loop
                from ..core.embeddings import encode_batch
                embeddings = encode_batch(texts, normalize=True)

                cls._coa_embeddings = {
                    code: emb for code, emb in zip(codes, embeddings)
                }
                print(f"[COAAgent] Batch encoded {len(codes)} accounts")

    def _find_accounts(self, question: str, question_lower: str) -> list:
        """Tìm tài khoản phù hợp"""
        # 1. Tìm theo code
        code_match = re.search(r'\b(\d{3,5})\b', question)
        if code_match:
            code = code_match.group(1)
            acc = COA_BY_CODE.get(code)
            if acc: return [acc]

        # 2. Tìm theo loại
        type_keywords = {
            "tài sản": "Tài sản", "nợ phải trả": "Nợ phải trả", "vốn chủ sở hữu": "Vốn chủ sở hữu",
            "doanh thu": "Doanh thu", "chi phí": "Chi phí SXKD",
            "thu nhập khác": "Thu nhập khác", "chi phí khác": "Chi phí khác",
        }
        for kw, type_name in type_keywords.items():
            if kw in question_lower and any(w in question_lower for w in ["nhóm", "loại", "các", "những", "danh sách"]):
                accounts = COA_BY_TYPE.get(type_name, [])
                if accounts: return accounts

        # 3. Tìm theo tên (keyword)
        search_terms = ["hàng hóa", "tiền mặt", "phải thu", "phải trả", "doanh thu", "giá vốn",
                        "nguyên vật liệu", "công cụ", "tài sản cố định", "vốn", "lợi nhuận",
                        "thuế", "sinh học", "tiền gửi"]
        for term in search_terms:
            if term in question_lower:
                results = []
                for acc in COA_DATA:
                    if term in acc["name"].lower() or term in acc.get("name_en", "").lower():
                        results.append(acc)
                if results: return results

        # 4. Embedding search (optimized with batch operations)
        self._init_embeddings()

        # Build embeddings matrix for batch similarity
        codes = list(self._coa_embeddings.keys())
        embeddings_matrix = np.array([self._coa_embeddings[c] for c in codes])

        # Encode query
        query_emb = self._embed_model.encode(question, normalize_embeddings=True)

        # Batch compute similarities (vectorized)
        scores = batch_cosine_similarity(query_emb, embeddings_matrix)

        # Get top 3
        top_3_indices = np.argpartition(scores, -3)[-3:]
        top_3_indices = top_3_indices[np.argsort(-scores[top_3_indices])]

        return [COA_BY_CODE[codes[i]] for i in top_3_indices if scores[i] > 0.3]

    def _generate_fallback(self, accounts: list) -> str:
        """Fallback khi SLM không hoạt động"""
        if not accounts:
            return "Không tìm thấy thông tin tài khoản phù hợp trong danh mục Thông tư 99."

        if len(accounts) == 1:
            acc = accounts[0]
            type_id = acc.get('type_id', 0)
            balance_side = 'Nợ' if type_id in [1, 5] else 'Có' if type_id in [2, 3, 4] else 'Không có số dư cuối kỳ'

            lines = [
                "1. THÔNG TIN CƠ BẢN:",
                f"- Số hiệu: {acc['code']}",
                f"- Tên tài khoản: {acc['name']}",
                f"- Tên tiếng Anh: {acc.get('name_en', 'N/A')}",
                f"- Phân loại: {acc['type_name']}",
                "",
                "2. NỘI DUNG PHẢN ÁNH:",
                f"Tài khoản {acc['code']} được sử dụng để theo dõi và phản ánh tình hình biến động của {acc['name'].lower()} tại doanh nghiệp.",
                "",
                "3. KẾT CẤU:",
                "- Bên Nợ: Ghi nhận phát sinh tăng.",
                "- Bên Có: Ghi nhận phát sinh giảm.",
                f"- Số dư: Thường nằm bên {balance_side}.",
                "",
                "4. LƯU Ý:",
                "- Kế toán cần hạch toán chi tiết theo từng đối tượng quản lý nếu cần thiết.",
                "",
                "(Căn cứ: Phụ lục II - Thông tư 99/2025/TT-BTC)"
            ]
        else:
            lines = ["Dưới đây là danh sách các tài khoản liên quan phù hợp với từ khóa của bạn:", ""]
            grouped = defaultdict(list)
            for acc in accounts:
                grouped[acc['type_name']].append(acc)

            for type_name, accs in grouped.items():
                lines.append(f"--- {type_name.upper()} ---")
                for acc in accs:
                    lines.append(f"• TK {acc['code']}: {acc['name']}")
                lines.append("")
            lines.append("(Căn cứ: Phụ lục II - Thông tư 99/2025/TT-BTC)")

        return "\n".join(lines)

    def _build_compare_context(self, code: str, acc_99: dict, acc_200: dict) -> str:
        """Build context string cho so sánh"""
        lines = [f"Tài khoản: {code}\n"]

        if acc_200:
            lines.append("TT200 (2014):")
            lines.append(f"  - Tên: {acc_200['name']}")
            lines.append(f"  - Tên EN: {acc_200.get('name_en', 'N/A')}")
            lines.append(f"  - Loại: {acc_200['type_name']}")
        else:
            lines.append("TT200 (2014): Không có tài khoản này")

        lines.append("")

        if acc_99:
            lines.append("TT99 (2025):")
            lines.append(f"  - Tên: {acc_99['name']}")
            lines.append(f"  - Tên EN: {acc_99.get('name_en', 'N/A')}")
            lines.append(f"  - Loại: {acc_99['type_name']}")
        else:
            lines.append("TT99 (2025): Không có tài khoản này")

        return "\n".join(lines)

    def _generate_compare_fallback(self, code: str, acc_99: dict, acc_200: dict) -> str:
        """Fallback khi so sánh"""
        name_99 = acc_99['name'] if acc_99 else "Không có"
        name_200 = acc_200['name'] if acc_200 else "Không có"
        name_en_99 = acc_99.get('name_en', 'N/A') if acc_99 else "N/A"
        name_en_200 = acc_200.get('name_en', 'N/A') if acc_200 else "N/A"
        type_99 = acc_99['type_name'] if acc_99 else "N/A"
        type_200 = acc_200['type_name'] if acc_200 else "N/A"

        lines = [
            f"SO SÁNH TÀI KHOẢN {code}",
            "",
            f"| Tiêu chí         | TT200 (2014)              | TT99 (2025)               |",
            f"|------------------|---------------------------|---------------------------|",
            f"| Số hiệu          | {code:<25} | {code:<25} |",
            f"| Tên tài khoản    | {name_200:<25} | {name_99:<25} |",
            f"| Tên tiếng Anh    | {name_en_200:<25} | {name_en_99:<25} |",
            f"| Loại tài khoản   | {type_200:<25} | {type_99:<25} |",
            "",
            "NHẬN XÉT:",
        ]

        if not acc_200 and acc_99:
            lines.append(f"- TK {code} là tài khoản MỚI được bổ sung trong TT99.")
        elif acc_200 and not acc_99:
            lines.append(f"- TK {code} đã bị LOẠI BỎ trong TT99.")
        elif name_200 != name_99:
            lines.append(f"- TT99 đổi tên tài khoản từ \"{name_200}\" thành \"{name_99}\".")
        else:
            lines.append("- Không có thay đổi đáng kể về tên và phân loại.")

        return "\n".join(lines)

    def _analyze_circular_diff(self) -> dict:
        """Lấy dữ liệu so sánh từ file"""
        return {
            'added': COA_COMPARE_BY_TYPE.get('ADDED', []),
            'removed': COA_COMPARE_BY_TYPE.get('REMOVED', []),
            'renamed': COA_COMPARE_BY_TYPE.get('RENAMED', []),
            'renamed_and_details_removed': COA_COMPARE_BY_TYPE.get('RENAMED_AND_DETAILS_REMOVED', []),
            'details_removed': COA_COMPARE_BY_TYPE.get('DETAILS_REMOVED', []),
            'details_added': COA_COMPARE_BY_TYPE.get('DETAILS_ADDED', []),
            'total_changes': len(COA_COMPARE_DATA)
        }

    def _build_circular_context(self, diff: dict) -> str:
        """Build context cho so sánh tổng quan"""
        lines = [
            "THỐNG KÊ THAY ĐỔI:",
            f"- Tổng số thay đổi: {diff['total_changes']}",
            f"- TK mới (ADDED): {len(diff['added'])}",
            f"- TK bị xóa (REMOVED): {len(diff['removed'])}",
            f"- TK đổi tên (RENAMED): {len(diff['renamed'])}",
            f"- TK đổi tên + bỏ chi tiết: {len(diff['renamed_and_details_removed'])}",
            f"- TK bỏ chi tiết cấp 2: {len(diff['details_removed'])}",
            f"- TK thêm chi tiết: {len(diff['details_added'])}",
            ""
        ]

        if diff['added']:
            lines.append("TÀI KHOẢN MỚI (ADDED):")
            for item in diff['added'][:5]:
                lines.append(f"  - {item['content']}")
            lines.append("")

        if diff['removed']:
            lines.append("TÀI KHOẢN BỊ XÓA (REMOVED):")
            for item in diff['removed'][:5]:
                lines.append(f"  - {item['content']}")
            lines.append("")

        if diff['renamed']:
            lines.append("TÀI KHOẢN ĐỔI TÊN (RENAMED):")
            for item in diff['renamed'][:5]:
                lines.append(f"  - {item['content']}")
            lines.append("")

        return "\n".join(lines)

    def _generate_circular_fallback(self, diff: dict) -> str:
        """Fallback khi so sánh tổng quan"""
        lines = [
            "# SO SÁNH TỔNG QUAN: THÔNG TƯ 200/2014 vs THÔNG TƯ 99/2025",
            "",
            "## 1. THỐNG KÊ THAY ĐỔI",
            f"| Loại thay đổi | Số lượng |",
            f"|---------------|----------|",
            f"| TK mới (ADDED) | {len(diff['added'])} |",
            f"| TK bị xóa (REMOVED) | {len(diff['removed'])} |",
            f"| TK đổi tên (RENAMED) | {len(diff['renamed'])} |",
            f"| TK đổi tên + bỏ chi tiết | {len(diff['renamed_and_details_removed'])} |",
            f"| TK bỏ chi tiết cấp 2 | {len(diff['details_removed'])} |",
            f"| TK thêm chi tiết | {len(diff['details_added'])} |",
            ""
        ]

        if diff['added']:
            lines.append("## 2. TÀI KHOẢN MỚI TRONG TT99")
            for item in diff['added'][:3]:
                lines.append(f"- **TK {item['account_number']}**: {item['name_tt99']}")
            lines.append("")

        if diff['removed']:
            lines.append("## 3. TÀI KHOẢN BỊ XÓA")
            for item in diff['removed'][:3]:
                lines.append(f"- **TK {item['account_number']}**: {item['name_tt200']}")
            lines.append("")

        if diff['renamed']:
            lines.append("## 4. TÀI KHOẢN ĐỔI TÊN")
            for item in diff['renamed'][:3]:
                lines.append(f"- **TK {item['account_number']}**: {item['name_tt200']} → {item['name_tt99']}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    def _tool_lookup_by_code(self, code: str) -> Optional[dict]:
        """Tool: Tra cứu tài khoản theo số hiệu"""
        return COA_BY_CODE.get(code.strip())

    def _tool_lookup_by_type(self, type_name: str) -> list:
        """Tool: Tra cứu danh sách tài khoản theo loại"""
        return COA_BY_TYPE.get(type_name, [])

    def _tool_search_by_name(self, keyword: str) -> list:
        """Tool: Tìm kiếm tài khoản theo tên"""
        keyword_lower = keyword.lower()
        results = []
        for acc in COA_DATA:
            if keyword_lower in acc["name"].lower() or keyword_lower in acc.get("name_en", "").lower():
                results.append(acc)
        return results

    def _tool_search_by_embedding(self, query: str, top_k: int = 5) -> list:
        """
        Tool: Tìm kiếm bằng semantic search (optimized).

        Uses batch vectorized operations instead of loop.
        """
        self._init_embeddings()

        # Build embeddings matrix
        codes = list(self._coa_embeddings.keys())
        embeddings_matrix = np.array([self._coa_embeddings[c] for c in codes])

        # Encode query
        query_emb = self._embed_model.encode(query, normalize_embeddings=True)

        # Batch compute similarities (vectorized - nhanh hơn loop rất nhiều)
        scores = batch_cosine_similarity(query_emb, embeddings_matrix)

        # Get top k
        top_k = min(top_k, len(scores))
        top_k_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

        return [COA_BY_CODE[codes[i]] for i in top_k_indices if scores[i] > 0.3]

    def _tool_compare_accounts(self, code: str) -> dict:
        """Tool: So sánh tài khoản giữa TT200 và TT99"""
        return {
            "tt200": COA_200_BY_CODE.get(code),
            "tt99": COA_BY_CODE.get(code)
        }

    def _tool_compare_circular(self) -> dict:
        """Tool: So sánh tổng quan giữa TT200 và TT99"""
        return self._analyze_circular_diff()
