"""
Posting Engine Agent - Chuyên gia về hạch toán, định khoản, bút toán

Chuyên gia về:
- Phân loại nghiệp vụ kế toán
- Định khoản, hạch toán các nghiệp vụ kinh tế
- Tư vấn bút toán kế toán
"""
import json
import os
from collections import defaultdict

import numpy as np
import networkx as nx

from .base import BaseAgent, AgentRole, AgentResult, AgentContext, Tool
from ..core.config import settings
from ..core.ollama_client import get_ollama_client
from ..core.embeddings import get_embed_model
from ..services.stream_utils import stream_by_char
from .templates import get_response_template


# =============================================================================
# CONFIG LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTING_CONFIG_FILE = os.path.join(BASE_DIR, "services", "rag_json", "posting_engine.json")
COA_99_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_99.json")

if not os.path.exists(POSTING_CONFIG_FILE):
    raise FileNotFoundError(f"posting_engine.json not found at {POSTING_CONFIG_FILE}")

with open(POSTING_CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DOCUMENT_TYPES = {d["transaction_key"]: d for d in CONFIG["document_types"]}
POSTING_RULES = {r["je_doc_type"]: r["rules"] for r in CONFIG["posting_rules"]}
GL_MAPPING = CONFIG["gl_mapping"]
POSTING_GROUPS = {g["code"]: g for g in CONFIG["posting_groups"]}
ROLE_KEYS = CONFIG["role_keys"]

# Load ACCOUNT_NAMES
ACCOUNT_NAMES = {}
if os.path.exists(COA_99_FILE):
    with open(COA_99_FILE, "r", encoding="utf-8") as f:
        coa_data = json.load(f)
        ACCOUNT_NAMES = {item["code"]: item["name"] for item in coa_data}

ACCOUNT_NAMES.update({
    "1331": "Thuế GTGT được khấu trừ",
    "1388": "Phải thu khác",
    "13881": "Phải thu tạm (Giao hàng chưa xuất HĐ)",
    "33311": "Thuế GTGT đầu ra",
    "3388": "Phải trả, phải nộp khác",
    "33881": "Phải trả tạm (Nhập hàng chưa có HĐ)",
})

# Load TRANSACTION_NAMES
TRANSACTION_NAMES = {}
for doc in CONFIG.get("document_types", []):
    tx_key = doc.get("transaction_key")
    desc = doc.get("description", "")
    name = desc.split(" - ")[0] if " - " in desc else desc
    TRANSACTION_NAMES[tx_key] = name


# =============================================================================
# SLM CLASSIFICATION PROMPTS
# =============================================================================

TX_CLASSIFICATION_PROMPT = """Phân loại nghiệp vụ kế toán.

DO_SALE: Xuất kho bán hàng, giao hàng cho khách, giảm tồn kho, ghi nhận giá vốn
SALES_INVOICE: Xuất hóa đơn bán hàng, ghi nhận công nợ phải thu, thuế GTGT đầu ra
CASH_IN: Thu tiền từ khách hàng, phiếu thu, nhận thanh toán
GRN_PURCHASE: Nhập kho mua hàng, nhận hàng từ NCC, tăng tồn kho
PURCHASE_INVOICE: Nhận hóa đơn mua hàng, ghi nhận công nợ phải trả, thuế GTGT đầu vào
CASH_OUT: Chi tiền cho NCC, phiếu chi, thanh toán công nợ

Câu hỏi: {question}"""

TX_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction": {
            "type": "string",
            "enum": ["DO_SALE", "SALES_INVOICE", "CASH_IN", "GRN_PURCHASE", "PURCHASE_INVOICE", "CASH_OUT"]
        }
    },
    "required": ["transaction"]
}


# =============================================================================
# MINI-RAG CLASSES
# =============================================================================

class MiniRAGGraph:
    """Knowledge graph cho transaction types"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.keyword_to_tx = defaultdict(set)
        self.account_to_tx = defaultdict(list)
        self._build_graph()

    def _build_graph(self):
        for tx, doc in DOCUMENT_TYPES.items():
            self.graph.add_node(f"TX:{tx}", node_type="TRANSACTION", name=TRANSACTION_NAMES.get(tx, tx))
            for kw in doc.get("keywords", []):
                self.keyword_to_tx[kw.lower()].add(tx)

        for tx, rules in POSTING_RULES.items():
            for rule in rules:
                if rule.get("account_source_type") == "FIXED":
                    acc = rule.get("fixed_account_code", "")
                    if acc:
                        self.account_to_tx[acc].append((tx, rule["side"]))

    def get_neighbors(self, node_id, edge_type=None):
        if not self.graph.has_node(node_id): return []
        return [target for _, target, data in self.graph.edges(node_id, data=True)
                if edge_type is None or data.get("edge_type") == edge_type]


class MiniRAGRetriever:
    """Retrieval system cho transaction classification"""
    def __init__(self, graph: MiniRAGGraph):
        self.graph = graph
        self.embed_model = get_embed_model()
        self.tx_embeddings = {}
        for tx, doc in DOCUMENT_TYPES.items():
            text = f"{tx} {TRANSACTION_NAMES.get(tx, '')} {doc.get('description', '')}"
            self.tx_embeddings[tx] = self.embed_model.encode(text, normalize_embeddings=True)

    def retrieve(self, query: str) -> dict:
        """Retrieve transaction type using SLM + fallback"""
        slm_result = self._classify_with_slm(query)
        if slm_result:
            return {
                "transaction": slm_result,
                "score": 1.0,
                "method": "SLM"
            }

        return self._fallback_retrieve(query)

    def _classify_with_slm(self, query: str) -> str:
        """Use SLM to classify transaction type"""
        try:
            client = get_ollama_client()
            response = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "user", "content": TX_CLASSIFICATION_PROMPT.format(question=query)}
                ],
                format=TX_CLASSIFICATION_SCHEMA,
                options=settings.OLLAMA_OPTIONS,
                stream=False
            )
            content = response.get("message", {}).get("content", "")
            result = json.loads(content)
            tx = result.get("transaction")
            if tx in DOCUMENT_TYPES:
                return tx
        except Exception as e:
            print(f"[PostingEngineAgent SLM Error] {e}")
        return None

    def _fallback_retrieve(self, query: str) -> dict:
        """Fallback: keyword + embedding matching"""
        query_lower = query.lower()
        scores = defaultdict(float)
        matched_keywords = []

        for kw, tx_set in self.graph.keyword_to_tx.items():
            if kw in query_lower:
                matched_keywords.append(kw)
                for tx in tx_set:
                    scores[tx] += 3.0

        if not matched_keywords:
            query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
            for tx, tx_emb in self.tx_embeddings.items():
                scores[tx] += float(np.dot(query_embedding, tx_emb))

        best_tx = max(scores.keys(), key=lambda k: scores[k]) if scores else "DO_SALE"
        return {
            "transaction": best_tx,
            "score": scores[best_tx] if scores else 0.0,
            "matched_keywords": matched_keywords,
            "method": "FALLBACK"
        }


class PostingEngineResolver:
    """Resolve journal entries based on rules"""
    @staticmethod
    def resolve(tx, item_group, partner_group):
        rules = POSTING_RULES.get(tx, [])
        entries = []
        for r in sorted(rules, key=lambda x: x["priority"]):
            role_key = r["role_key"]
            acc_type = r.get("account_source_type", "FIXED")
            acc = ""
            if acc_type == "FIXED":
                acc = r.get("fixed_account_code", "")
            elif acc_type == "LOOKUP":
                item_group_info = POSTING_GROUPS.get(item_group, {})
                if item_group_info.get("posting_group_type") == "ITEM_GROUP":
                    acc = GL_MAPPING.get(item_group, {}).get(role_key, "")
                else:
                    acc = GL_MAPPING.get(partner_group, {}).get(role_key, "")

            entries.append({
                "side": r["side"],
                "account": acc,
                "priority": r["priority"],
                "description": r.get("description", ""),
                "is_lookup": acc_type == "LOOKUP"
            })
        return entries


# =============================================================================
# POSTING ENGINE AGENT
# =============================================================================

class PostingEngineAgent(BaseAgent):
    """
    Posting Engine Agent - Chuyên gia về hạch toán, định khoản

    Xử lý:
    - Phân loại nghiệp vụ kế toán
    - Định khoản các nghiệp vụ
    - Tư vấn bút toán
    """

    def __init__(self):
        super().__init__()
        self._graph = MiniRAGGraph()
        self._retriever = MiniRAGRetriever(self._graph)
        self._init_tools()

    @property
    def name(self) -> str:
        return "POSTING_ENGINE"

    @property
    def role(self) -> AgentRole:
        return AgentRole.DOMAIN_SPECIALIST

    @property
    def description(self) -> str:
        return "Chuyên gia về hạch toán, định khoản, bút toán kế toán. Phân loại nghiệp vụ và tư vấn bút toán chi tiết."

    def _init_tools(self):
        """Đăng ký tools"""
        self.add_tool(
            name="classify_transaction",
            description="Phân loại nghiệp vụ kế toán",
            func=self._tool_classify_transaction
        )
        self.add_tool(
            name="resolve_journal_entries",
            description="Giải quyết bút toán cho nghiệp vụ",
            func=self._tool_resolve_journal_entries
        )
        self.add_tool(
            name="get_account_name",
            description="Lấy tên tài khoản theo số hiệu",
            func=self._tool_get_account_name
        )
        self.add_tool(
            name="get_transaction_info",
            description="Lấy thông tin chi tiết về nghiệp vụ",
            func=self._tool_get_transaction_info
        )

    def can_handle(self, context: AgentContext) -> tuple[bool, float]:
        """
        Kiểm tra agent có thể xử lý query không.

        Keywords:
        - hạch toán, định khoản, bút toán
        - xuất hóa đơn, nhập kho, phiếu thu, phiếu chi
        - bán hàng, mua hàng
        - thu tiền, chi tiền
        """
        question = context.question.lower()

        posting_keywords = [
            "hạch toán", "định khoản", "bút toán", "nghiệp vụ",
            "xuất hóa đơn", "nhập kho", "phiếu thu", "phiếu chi",
            "bán hàng", "mua hàng", "thu tiền", "chi tiền",
            "xuất kho", "nhập hàng", "giao hàng", "nhận hàng",
            "thanh toán", "công nợ", "phải thu", "phải trả",
            "gtgt", "thuế", "giá vốn"
        ]

        matches = sum(1 for kw in posting_keywords if kw in question)

        confidence = 0.0
        if matches >= 2:
            confidence = 0.95
        elif matches == 1:
            confidence = 0.80
        elif any(kw in question for kw in ["tk ", " tk", "tài khoản"]):
            # Might be COA, but could be posting
            confidence = 0.40

        return confidence > 0.5, confidence

    def execute(self, context: AgentContext) -> AgentResult:
        """Thực thi query"""
        question = context.question
        item_group = context.item_group
        partner_group = context.partner_group

        # 1. Retrieval
        result = self._retriever.retrieve(question)
        tx = result["transaction"]

        # 2. Resolve
        entries = PostingEngineResolver.resolve(tx, item_group, partner_group)

        # 3. Build entries text
        entries_list = []
        for e in sorted(entries, key=lambda x: x["priority"]):
            acc = e["account"]
            acc_name = ACCOUNT_NAMES.get(acc, acc)
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            marker = " (*)" if e.get("is_lookup") else ""
            entries_list.append(f"- {side} TK {acc}: {acc_name}{marker}")

        entries_text = "\n".join(entries_list)
        tx_name = TRANSACTION_NAMES.get(tx, tx)
        # Dùng template cụ thể cho từng nghiệp vụ
        response_template = get_response_template(tx)

        # 4. Generate response
        system_prompt = """Bạn là trợ lý kế toán Việt Nam. QUY TẮC BẮT BUỘC:
- LUÔN trả lời bằng TIẾNG VIỆT
- CHỈ sử dụng ĐÚNG các bút toán trong phần "BÚT TOÁN TỪ HỆ THỐNG"
- KHÔNG được thêm, bớt, thay đổi hoặc tự bịa bút toán
- BẮT BUỘC hoàn thành ĐẦY ĐỦ 4 phần theo đúng thứ tự:
  1. TÊN NGHIỆP VỤ
  2. BẢNG BÚT TOÁN (liệt kê, KHÔNG có số tiền)
  3. GIẢI THÍCH (giải thích từng dòng)
  4. VÍ DỤ (với số tiền cụ thể)
- QUAN TRỌNG: KHÔNG được dừng giữa chừng, phải viết đến hết phần 4"""

        slm_prompt = f"""Câu hỏi: {question}

NGHIỆP VỤ: {tx_name}

BÚT TOÁN TỪ HỆ THỐNG (chỉ sử dụng các bút toán này):
{entries_text}

YÊU CẦU: Trả lời ĐẦY ĐỦ 4 phần theo đúng format sau:
{response_template}"""

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

            # Add notes
            content += self._generate_notes(entries)

            return AgentResult(
                agent_name=self.name,
                content=content,
                confidence=0.95,
                metadata={"transaction": tx, "method": result.get("method")},
                sources=[f"Posting Engine - {tx_name}"]
            )
        except Exception as e:
            print(f"[PostingEngineAgent Error] {e}")
            return AgentResult(
                agent_name=self.name,
                content=self._generate_fallback(tx_name, entries),
                confidence=0.7,
                metadata={"transaction": tx}
            )

    def stream_execute(self, context: AgentContext):
        """Execute với streaming response"""
        question = context.question
        item_group = context.item_group
        partner_group = context.partner_group

        # 1. Retrieval
        result = self._retriever.retrieve(question)
        tx = result["transaction"]

        # 2. Resolve
        entries = PostingEngineResolver.resolve(tx, item_group, partner_group)

        # 3. Build entries text
        entries_list = []
        for e in sorted(entries, key=lambda x: x["priority"]):
            acc = e["account"]
            acc_name = ACCOUNT_NAMES.get(acc, acc)
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            marker = " (*)" if e.get("is_lookup") else ""
            entries_list.append(f"- {side} TK {acc}: {acc_name}{marker}")

        entries_text = "\n".join(entries_list)
        tx_name = TRANSACTION_NAMES.get(tx, tx)
        # Dùng template cụ thể cho từng nghiệp vụ
        response_template = get_response_template(tx)

        # 4. Generate response
        system_prompt = """Bạn là trợ lý kế toán Việt Nam. QUY TẮC BẮT BUỘC:
- LUÔN trả lời bằng TIẾNG VIỆT
- CHỈ sử dụng ĐÚNG các bút toán trong phần "BÚT TOÁN TỪ HỆ THỐNG"
- KHÔNG được thêm, bớt, thay đổi hoặc tự bịa bút toán
- BẮT BUỘC hoàn thành ĐẦY ĐỦ 4 phần theo đúng thứ tự:
  1. TÊN NGHIỆP VỤ
  2. BẢNG BÚT TOÁN (liệt kê, KHÔNG có số tiền)
  3. GIẢI THÍCH (giải thích từng dòng)
  4. VÍ DỤ (với số tiền cụ thể)
- QUAN TRỌNG: KHÔNG được dừng giữa chừng, phải viết đến hết phần 4"""

        slm_prompt = f"""Câu hỏi: {question}

NGHIỆP VỤ: {tx_name}

BÚT TOÁN TỪ HỆ THỐNG (chỉ sử dụng các bút toán này):
{entries_text}

YÊU CẦU: Trả lời ĐẦY ĐỦ 4 phần theo đúng format sau:
{response_template}"""

        full_response = ""
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

            # Debug: collect first few chunks
            chunk_count = 0
            for char in stream_by_char(stream):
                if chunk_count == 0:
                    print(f"[PostingEngineAgent Stream DEBUG] First char received: '{char}'")
                chunk_count += 1
                full_response += char
                yield char

            print(f"[PostingEngineAgent Stream DEBUG] Total chunks received: {chunk_count}")
            print(f"[PostingEngineAgent Stream DEBUG] Full response length: {len(full_response)}")

        except Exception as e:
            print(f"[PostingEngineAgent Stream Error] {e}")

        # Add notes
        notes = self._generate_notes(entries)
        full_response += notes
        yield notes

    def _generate_fallback(self, tx_name, entries):
        """Fallback khi SLM không hoạt động"""
        lines = [f"1. TÊN NGHIỆP VỤ:\n{tx_name}", "", "2. BẢNG BÚT TOÁN:"]
        for e in entries:
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            lines.append(f"- {side} TK {e['account']}: {ACCOUNT_NAMES.get(e['account'], e['account'])}")
        return "\n".join(lines)

    def _generate_notes(self, entries):
        """Generate notes for entries"""
        notes = []

        has_lookup = any(e.get("is_lookup") for e in entries)
        if has_lookup:
            notes.append("\n\nGhi chú: Các dòng có dấu (*) là các dòng được cấu hình `account_source_type` = `LOOKUP`. Hệ thống sẽ dựa vào nhóm sản phẩm `(Item Group)` hoặc nhóm đối tác `(Partner Group)` để xác định tài khoản cụ thể.")

        has_clearing = any(e["account"] in ["13881", "33881"] for e in entries)
        if has_clearing:
            notes.append("\n\nLưu ý: Tài khoản `13881` và `33881` là các tài khoản trung gian (Clearing Accounts) được định nghĩa trong Posting Engine để xử lý độ trễ giữa thời điểm giao/nhận hàng và thời điểm xuất/nhận hóa đơn.")

        return "".join(notes)

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    def _tool_classify_transaction(self, query: str) -> dict:
        """Tool: Phân loại nghiệp vụ"""
        return self._retriever.retrieve(query)

    def _tool_resolve_journal_entries(self, tx: str, item_group: str = "GOODS", partner_group: str = "CUSTOMER") -> list:
        """Tool: Giải quyết bút toán"""
        return PostingEngineResolver.resolve(tx, item_group, partner_group)

    def _tool_get_account_name(self, account_code: str) -> str:
        """Tool: Lấy tên tài khoản"""
        return ACCOUNT_NAMES.get(account_code, account_code)

    def _tool_get_transaction_info(self, tx: str) -> dict:
        """Tool: Lấy thông tin nghiệp vụ"""
        doc = DOCUMENT_TYPES.get(tx, {})
        return {
            "key": tx,
            "name": TRANSACTION_NAMES.get(tx, tx),
            "description": doc.get("description", ""),
            "keywords": doc.get("keywords", [])
        }
