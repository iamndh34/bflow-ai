import json
import os
from datetime import datetime
from typing import Optional
from collections import defaultdict

import numpy as np
import networkx as nx
import ollama
from sentence_transformers import SentenceTransformer

from app.core.config import settings

# =============================================================================
# CONFIG LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTING_CONFIG_FILE = os.path.join(BASE_DIR, "services", "rag_json", "posting_engine.json")

if not os.path.exists(POSTING_CONFIG_FILE):
    raise FileNotFoundError(f"posting_engine.json not found at {POSTING_CONFIG_FILE}")

with open(POSTING_CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DOCUMENT_TYPES = {d["transaction_key"]: d for d in CONFIG["document_types"]}
POSTING_RULES = {r["je_doc_type"]: r["rules"] for r in CONFIG["posting_rules"]}
GL_MAPPING = CONFIG["gl_mapping"]
POSTING_GROUPS = {g["code"]: g for g in CONFIG["posting_groups"]}
ROLE_KEYS = CONFIG["role_keys"]

# =============================================================================
# CONSTANTS & DEFINITIONS
# =============================================================================

ACCOUNT_NAMES = {
    "111": "Tiền mặt",
    "112": "Tiền gửi không kỳ hạn",
    "131": "Phải thu của khách hàng",
    "1331": "Thuế GTGT được khấu trừ",
    "1388": "Phải thu khác",
    "13881": "Phải thu tạm (Giao hàng chưa xuất HĐ)",
    "152": "Nguyên liệu, vật liệu",
    "153": "Công cụ, dụng cụ",
    "154": "Chi phí SXKD dở dang",
    "155": "Sản phẩm",
    "156": "Hàng hóa",
    "157": "Hàng gửi đi bán",
    "211": "Tài sản cố định hữu hình",
    "215": "Tài sản sinh học",
    "331": "Phải trả cho người bán",
    "33311": "Thuế GTGT đầu ra",
    "3388": "Phải trả, phải nộp khác",
    "33881": "Phải trả tạm (Nhập hàng chưa có HĐ)",
    "511": "Doanh thu bán hàng và cung cấp dịch vụ",
    "521": "Các khoản giảm trừ doanh thu",
    "632": "Giá vốn hàng bán",
    "641": "Chi phí bán hàng",
    "642": "Chi phí quản lý doanh nghiệp",
    "811": "Chi phí khác",
    "82112": "Chi phí thuế TNDN bổ sung (Thuế tối thiểu toàn cầu)",
}

TRANSACTION_NAMES = {
    "DO_SALE": "Phiếu xuất kho bán hàng",
    "SALES_INVOICE": "Hóa đơn phải thu",
    "CASH_IN": "Phiếu thu tiền",
    "GRN_PURCHASE": "Phiếu nhập kho mua hàng",
    "PURCHASE_INVOICE": "Hóa đơn phải chi",
    "CASH_OUT": "Phiếu chi tiền",
}

# =============================================================================
# SLM TRANSACTION CLASSIFICATION
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

FEW_SHOT_QA = """
1. TÊN NGHIỆP VỤ:
Phiếu xuất kho bán hàng

2. BẢNG BÚT TOÁN:
- Nợ TK 632: Giá vốn hàng bán (*)
- Có TK 156: Hàng hóa (*)
- Nợ TK 13881: Phải thu tạm
- Có TK 511: Doanh thu bán hàng (*)

3. GIẢI THÍCH:
- Nợ TK 632: Ghi nhận chi phí giá vốn hàng bán khi xuất kho
- Có TK 156: Giảm giá trị hàng tồn kho khi xuất bán
- Nợ TK 13881: Ghi nhận khoản phải thu tạm (đã giao hàng nhưng chưa xuất hóa đơn)
- Có TK 511: Ghi nhận doanh thu bán hàng

4. VÍ DỤ:
Xuất kho bán 100 sản phẩm A, giá vốn 50.000đ/sp, giá bán 80.000đ/sp:
- Nợ TK 632: 5.000.000đ
- Có TK 156: 5.000.000đ
- Nợ TK 13881: 8.000.000đ
- Có TK 511: 8.000.000đ
"""


# =============================================================================
# MINI-RAG CLASSES
# =============================================================================

class MiniRAGGraph:
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
    def __init__(self, graph: MiniRAGGraph):
        self.graph = graph
        self.embed_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
        self.tx_embeddings = {}
        for tx, doc in DOCUMENT_TYPES.items():
            text = f"{tx} {TRANSACTION_NAMES.get(tx, '')} {doc.get('description', '')}"
            self.tx_embeddings[tx] = self.embed_model.encode(text, normalize_embeddings=True)

    def retrieve(self, query: str) -> dict:
        """
        Retrieve transaction type using SLM classification with fallback to keyword matching.
        """
        # 1. Try SLM classification first
        slm_result = self._classify_with_slm(query)
        if slm_result:
            print(f"[Retriever] SLM classified: {slm_result}")
            return {
                "transaction": slm_result,
                "score": 1.0,
                "matched_keywords": [],
                "matched_concepts": [],
                "method": "SLM",
                "debug": {}
            }

        # 2. Fallback to keyword/embedding matching
        print("[Retriever] SLM failed, using fallback...")
        return self._fallback_retrieve(query)

    def _classify_with_slm(self, query: str) -> str:
        """Use SLM to classify transaction type."""
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            response = client.chat(
                model="qwen2.5:3b",
                messages=[
                    {"role": "user", "content": TX_CLASSIFICATION_PROMPT.format(question=query)}
                ],
                format=TX_CLASSIFICATION_SCHEMA,
                stream=False
            )
            content = response.get("message", {}).get("content", "")
            result = json.loads(content)
            tx = result.get("transaction")
            if tx in DOCUMENT_TYPES:
                return tx
        except Exception as e:
            print(f"[Retriever SLM Error] {e}")
        return None

    def _fallback_retrieve(self, query: str) -> dict:
        """Fallback: keyword + embedding matching."""
        query_lower = query.lower()
        scores = defaultdict(float)
        matched_keywords = []

        # 1. Keyword matching (weight: 3.0)
        for kw, tx_set in self.graph.keyword_to_tx.items():
            if kw in query_lower:
                matched_keywords.append(kw)
                for tx in tx_set:
                    scores[tx] += 3.0

        # 2. Embedding fallback - chỉ dùng khi chưa có keyword match
        if not matched_keywords:
            query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
            for tx, tx_emb in self.tx_embeddings.items():
                scores[tx] += float(np.dot(query_embedding, tx_emb))

        best_tx = max(scores.keys(), key=lambda k: scores[k]) if scores else "DO_SALE"
        return {
            "transaction": best_tx,
            "score": scores[best_tx] if scores else 0.0,
            "matched_keywords": matched_keywords,
            "method": "FALLBACK",
            "debug": dict(scores)
        }


# =============================================================================
# POSTING ENGINE RESOLVER
# =============================================================================

class PostingEngineResolver:
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
# CONTEXT & MAIN SERVICE
# =============================================================================

MINI_RAG_GRAPH = MiniRAGGraph()
MINI_RAG_RETRIEVER = MiniRAGRetriever(MINI_RAG_GRAPH)


class HistoryManager:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history_file = os.path.join(BASE_DIR, "services", "rag_json", "conversation_history.json")
        self.history = self._load_from_file()

    def _load_from_file(self) -> list:
        """Load lịch sử từ file JSON"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Lấy 10 cái gần nhất
                    return data[-self.max_history:] if len(data) > self.max_history else data
        except Exception as e:
            print(f"[Context] Error loading history: {e}")
        return []

    def _save_to_file(self):
        """Lưu lịch sử ra file JSON"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Context] Error saving history: {e}")

    def add(self, question: str, response: str, category: str = "POSTING_ENGINE"):
        """Thêm câu hỏi và câu trả lời vào lịch sử"""
        self.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "response": response,
            "category": category
        })
        # Giữ tối đa max_history items
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        # Lưu ra file
        self._save_to_file()

    def get_last_category(self) -> str:
        """Lấy category của câu hỏi trước"""
        return self.history[-1]["category"] if self.history else None

    def reload(self):
        """Reload lịch sử từ file"""
        self.history = self._load_from_file()
        print(f"[Context] Reloaded {len(self.history)} items")

    def clear(self):
        """Xóa lịch sử hội thoại"""
        self.history = []
        self._save_to_file()
        print("[Context] Cleared")


HISTORY_MANAGER = HistoryManager()


class RagPostingEngine:
    @staticmethod
    def ask(question, item_group="GOODS", partner_group="CUSTOMER"):
        # 1. Retrieval
        result = MINI_RAG_RETRIEVER.retrieve(question)
        tx = result["transaction"]

        # 2. Resolve
        entries = PostingEngineResolver.resolve(tx, item_group, partner_group)

        # 3. Generate Prompt
        entries_list = []
        for e in sorted(entries, key=lambda x: x["priority"]):
            acc = e["account"]
            acc_name = ACCOUNT_NAMES.get(acc, acc)
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            marker = " (*)" if e.get("is_lookup") else ""
            entries_list.append(f"- {side} TK {acc}: {acc_name}{marker}")
        entries_text = "\n".join(entries_list)
        tx_name = TRANSACTION_NAMES.get(tx, tx)

        system_prompt = """Bạn là trợ lý kế toán Việt Nam. QUY TẮC BẮT BUỘC:
- LUÔN trả lời bằng TIẾNG VIỆT
- CHỈ sử dụng ĐÚNG các bút toán trong phần "BÚT TOÁN TỪ HỆ THỐNG"
- KHÔNG được thêm, bớt, thay đổi hoặc tự bịa bút toán
- Trả lời ĐÚNG 4 phần theo thứ tự:
  1. TÊN NGHIỆP VỤ: Tên nghiệp vụ
  2. BẢNG BÚT TOÁN: Liệt kê bút toán (KHÔNG có số tiền)
  3. GIẢI THÍCH: Giải thích ngắn gọn cho từng dòng bút toán
  4. VÍ DỤ: Ví dụ cụ thể với số tiền"""

        slm_prompt = f"""Câu hỏi: {question}

NGHIỆP VỤ: {tx_name}

BÚT TOÁN TỪ HỆ THỐNG (chỉ sử dụng các bút toán này):
{entries_text}

Trả lời theo định dạng mẫu sau:
{FEW_SHOT_QA}"""

        # 4. Call SLM
        print(f"[Posting] Asking SLM about {tx} with entries_text:\n{entries_text}")
        full_response = ""
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            stream = client.chat(
                model="qwen2.5:3b",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": slm_prompt}],
                stream=True
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_response += content
                    yield content
        except Exception as e:
            print(f"[Posting Error] {e}")

        # 5. Fallback nếu SLM không trả về đúng format
        if not full_response or "1." not in full_response:
            fallback = RagPostingEngine._generate_fallback(tx_name, entries)
            full_response = fallback
            yield fallback

        # 6. Ghi chú
        note = "\n\nGhi chú: Các dòng có dấu (*) là tài khoản có `account_source_type` = 'LOOKUP'."
        full_response += note
        yield note

        # 7. Ghi chú về tài khoản trung gian (nếu có 13881 hoặc 33881)
        has_clearing_account = any(e["account"] in ["13881", "33881"] for e in entries)
        if has_clearing_account:
            clearing_note = "\n\nLưu ý: Tài khoản 13881 và 33881 là các tài khoản trung gian (Clearing Accounts) được định nghĩa trong Posting Engine để xử lý độ trễ giữa thời điểm giao/nhận hàng và thời điểm xuất/nhận hóa đơn."
            full_response += clearing_note
            yield clearing_note

        # 8. Lưu vào history
        HISTORY_MANAGER.add(question, full_response, "POSTING_ENGINE")

    @staticmethod
    def _generate_fallback(tx_name, entries):
        lines = [f"1. TÊN NGHIỆP VỤ:\n{tx_name}", "", "2. BẢNG BÚT TOÁN:"]
        for e in entries:
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            lines.append(f"- {side} TK {e['account']}: {ACCOUNT_NAMES.get(e['account'], e['account'])}")
        return "\n".join(lines)