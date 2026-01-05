import json
import os
from typing import Optional
from collections import defaultdict

import numpy as np
import networkx as nx
import ollama
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTING_CONFIG_FILE = os.path.join(BASE_DIR, "services", "account_deter_json", "posting_engine.json")

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
# ACCOUNT & TRANSACTION NAMES
# =============================================================================

ACCOUNT_NAMES = {
    "131": "Phải thu khách hàng",
    "1331": "Thuế GTGT được khấu trừ",
    "1388": "Phải thu khác",
    "13881": "Phải thu tạm (Xuất kho bán hàng chưa có Hóa đơn phải thu)",
    "1111": "Tiền mặt",
    "1121": "Tiền gửi ngân hàng",
    "152": "Nguyên vật liệu",
    "153": "Công cụ dụng cụ",
    "154": "Chi phí SXKD dở dang",
    "155": "Thành phẩm",
    "1561": "Hàng hóa",
    "157": "Hàng gửi đi bán",
    "331": "Phải trả người bán",
    "3388": "Phải trả khác",
    "33881": "Phải trả tạm (Nhập kho mua hàng chưa có Hóa đơn phải chi)",
    "33311": "Thuế GTGT phải nộp",
    "5111": "Doanh thu bán hàng hóa",
    "5112": "Doanh thu bán thành phẩm",
    "5113": "Doanh thu cung cấp dịch vụ",
    "5211": "Chiết khấu thương mại",
    "632": "Giá vốn hàng bán",
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
# FEW-SHOT EXAMPLES - Ví dụ Q&A hoàn chỉnh để SLM học theo
# =============================================================================

FEW_SHOT_QA = """
1. TÊN NGHIỆP VỤ:
Phiếu xuất kho bán hàng

2. BẢNG BÚT TOÁN:
- Nợ TK 632: Giá vốn hàng bán
- Có TK 1561: Hàng hóa
- Nợ TK 13881: Phải thu tạm
- Có TK 5111: Doanh thu bán hàng hóa

3. GIẢI THÍCH:
- Nợ TK 632: Khi xuất kho bán hàng, doanh nghiệp ghi nhận chi phí giá vốn hàng bán
- Có TK 1561: Hàng hóa xuất kho làm giảm giá trị hàng tồn kho
- Nợ TK 13881: Ghi nhận khoản phải thu tạm vì đã giao hàng nhưng chưa xuất hóa đơn
- Có TK 5111: Ghi nhận doanh thu bán hàng khi giao hàng cho khách

4. VÍ DỤ:
Xuất kho bán 100 sản phẩm A, giá vốn 50.000đ/sp, giá bán 80.000đ/sp:
- Nợ TK 632: 5.000.000đ
- Có TK 1561: 5.000.000đ
- Nợ TK 13881: 8.000.000đ
- Có TK 5111: 8.000.000đ
"""

# =============================================================================
# QUERY EXPANSION - Từ đồng nghĩa để mở rộng câu hỏi
# =============================================================================

SYNONYMS = {
    # Bán hàng
    "xuất kho": ["xuất hàng", "giao hàng", "bán hàng", "ship hàng", "delivery"],
    "bán": ["bán hàng", "bán ra", "tiêu thụ"],
    "giao": ["giao hàng", "giao cho khách", "ship"],
    "khách hàng": ["khách", "customer", "KH"],

    # Mua hàng
    "nhập kho": ["nhập hàng", "nhận hàng", "hàng về", "receive"],
    "mua": ["mua hàng", "mua vào", "purchase"],
    "nhà cung cấp": ["NCC", "supplier", "vendor", "nhà cc"],

    # Hóa đơn
    "hóa đơn": ["hoá đơn", "HĐ", "invoice", "bill"],
    "phải thu": ["công nợ thu", "AR", "receivable"],
    "phải trả": ["công nợ trả", "AP", "payable"],
    "phải chi": ["phải trả", "công nợ chi", "AP"],

    # Tiền
    "thu tiền": ["nhận tiền", "tiền về", "cash in", "receipt"],
    "chi tiền": ["trả tiền", "thanh toán", "cash out", "payment"],
    "tiền mặt": ["cash", "TM"],
    "chuyển khoản": ["CK", "bank transfer", "tiền gửi"],

    # Thuế
    "thuế": ["VAT", "GTGT", "tax"],
    "thuế đầu ra": ["VAT out", "output tax"],
    "thuế đầu vào": ["VAT in", "input tax"],

    # Kế toán
    "hạch toán": ["định khoản", "ghi sổ", "bút toán", "journal entry"],
    "tài khoản": ["TK", "account", "acc"],
}

# =============================================================================
# NEGATIVE KEYWORDS - Từ khóa loại trừ để phân biệt nghiệp vụ tương tự
# =============================================================================

NEGATIVE_KEYWORDS = {
    # DO_SALE: Xuất kho bán - KHÔNG phải hóa đơn, KHÔNG phải thu tiền
    "DO_SALE": ["hóa đơn", "hoá đơn", "invoice", "thu tiền", "nhận tiền", "thanh toán"],

    # SALES_INVOICE: Hóa đơn bán - KHÔNG phải xuất kho, KHÔNG phải thu tiền
    "SALES_INVOICE": ["xuất kho", "giao hàng", "thu tiền", "nhận tiền", "tiền về"],

    # CASH_IN: Thu tiền - KHÔNG phải xuất kho, KHÔNG phải hóa đơn
    "CASH_IN": ["xuất kho", "hóa đơn", "hoá đơn", "giao hàng"],

    # GRN_PURCHASE: Nhập kho - KHÔNG phải hóa đơn, KHÔNG phải chi tiền
    "GRN_PURCHASE": ["hóa đơn", "hoá đơn", "invoice", "chi tiền", "trả tiền", "thanh toán"],

    # PURCHASE_INVOICE: Hóa đơn mua - KHÔNG phải nhập kho, KHÔNG phải chi tiền
    "PURCHASE_INVOICE": ["nhập kho", "hàng về", "chi tiền", "trả tiền"],

    # CASH_OUT: Chi tiền - KHÔNG phải nhập kho, KHÔNG phải hóa đơn
    "CASH_OUT": ["nhập kho", "hóa đơn", "hoá đơn", "hàng về"],
}

# =============================================================================
# DISAMBIGUATION RULES - Quy tắc phân biệt nghiệp vụ
# =============================================================================

DISAMBIGUATION_RULES = [
    # Bán hàng flow
    {
        "if_contains": ["xuất kho", "giao hàng"],
        "and_not_contains": ["hóa đơn", "thu tiền"],
        "then": "DO_SALE",
        "boost": 5.0
    },
    {
        "if_contains": ["hóa đơn", "phải thu"],
        "and_not_contains": ["xuất kho", "thu tiền", "nhập", "mua"],
        "then": "SALES_INVOICE",
        "boost": 5.0
    },
    {
        "if_contains": ["thu tiền", "khách trả", "khách thanh toán"],
        "and_not_contains": ["xuất kho", "hóa đơn"],
        "then": "CASH_IN",
        "boost": 5.0
    },

    # Mua hàng flow
    {
        "if_contains": ["nhập kho", "hàng về", "nhận hàng"],
        "and_not_contains": ["hóa đơn", "chi tiền"],
        "then": "GRN_PURCHASE",
        "boost": 5.0
    },
    {
        "if_contains": ["hóa đơn", "phải trả", "phải chi"],
        "and_not_contains": ["nhập kho", "chi tiền", "xuất", "bán"],
        "then": "PURCHASE_INVOICE",
        "boost": 5.0
    },
    {
        "if_contains": ["chi tiền", "trả tiền", "thanh toán ncc", "thanh toán nhà cung cấp"],
        "and_not_contains": ["nhập kho", "hóa đơn"],
        "then": "CASH_OUT",
        "boost": 5.0
    },
]

# Concepts - khớp với keywords trong posting_engine.json + biến thể ngắn
CONCEPTS = {
    "DO_SALE": [
        # Khớp với keywords JSON
        "xuất kho bán", "phiếu xuất kho", "giao hàng cho khách",
        "giảm tồn kho", "kết chuyển giá vốn", "phiếu giao hàng",
        "xuất hàng đi bán", "hàng xuất bán", "delivery order",
        # Biến thể ngắn
        "xuất kho", "xuất hàng", "giao hàng", "giá vốn", "phiếu xuất",
    ],
    "SALES_INVOICE": [
        # Khớp với keywords JSON
        "hóa đơn bán hàng", "hóa đơn đầu ra", "hóa đơn phải thu",
        "xuất hóa đơn cho khách", "lập hóa đơn bán", "công nợ phải thu",
        "thuế đầu ra", "VAT đầu ra", "hóa đơn GTGT bán ra",
        "AR invoice", "ghi nợ khách hàng",
        # Biến thể ngắn
        "hóa đơn bán", "xuất hóa đơn", "lập hóa đơn", "phải thu khách",
    ],
    "CASH_IN": [
        # Khớp với keywords JSON
        "thu tiền khách hàng", "khách hàng trả tiền", "phiếu thu tiền",
        "thu hồi công nợ", "nhận thanh toán từ khách", "thu nợ khách",
        "khách thanh toán", "cash receipt", "tiền về từ khách",
        # Biến thể ngắn
        "thu tiền", "phiếu thu", "tiền về", "khách trả",
    ],
    "GRN_PURCHASE": [
        # Khớp với keywords JSON
        "nhập kho mua", "phiếu nhập kho", "nhận hàng từ NCC",
        "tăng tồn kho", "hàng về kho", "nhập hàng mua",
        "hàng nhập kho", "goods receipt", "GRN",
        # Biến thể ngắn
        "nhập kho", "nhập hàng", "hàng về", "phiếu nhập", "nhận hàng",
    ],
    "PURCHASE_INVOICE": [
        # Khớp với keywords JSON
        "hóa đơn mua hàng", "hóa đơn đầu vào", "hóa đơn phải chi",
        "nhận hóa đơn từ NCC", "hóa đơn nhà cung cấp", "công nợ phải trả",
        "thuế đầu vào", "VAT đầu vào", "hóa đơn GTGT mua vào",
        "AP invoice", "ghi nợ nhà cung cấp",
        # Biến thể ngắn
        "hóa đơn mua", "nhận hóa đơn", "phải trả NCC", "nợ NCC",
    ],
    "CASH_OUT": [
        # Khớp với keywords JSON
        "chi tiền cho NCC", "trả tiền nhà cung cấp", "phiếu chi tiền",
        "thanh toán công nợ NCC", "trả nợ nhà cung cấp", "chi trả NCC",
        "thanh toán cho NCC", "cash payment", "tiền ra cho NCC",
        # Biến thể ngắn
        "chi tiền", "phiếu chi", "trả tiền", "thanh toán NCC",
    ],
}


# =============================================================================
# MINI-RAG: HETEROGENEOUS GRAPH BUILDER
# =============================================================================

class MiniRAGGraph:
    """
    Heterogeneous Graph với các loại node:
    - TRANSACTION: DO_SALE, SALES_INVOICE, etc.
    - ACCOUNT: 131, 632, 1561, etc.
    - KEYWORD: các từ khóa từ config
    - CONCEPT: các khái niệm kế toán

    Các loại edge:
    - TRANSACTION -[HAS_KEYWORD]-> KEYWORD
    - TRANSACTION -[HAS_CONCEPT]-> CONCEPT
    - TRANSACTION -[DEBIT]-> ACCOUNT
    - TRANSACTION -[CREDIT]-> ACCOUNT
    - CONCEPT -[RELATED]-> CONCEPT (cross-transaction)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.keyword_to_tx = defaultdict(set)  # keyword -> set of transactions
        self.concept_to_tx = defaultdict(set)  # concept -> set of transactions
        self.account_to_tx = defaultdict(list)  # account -> [(tx, side)]
        self._build_graph()

    def _build_graph(self):
        """Xây dựng heterogeneous graph từ config"""

        # 1. Thêm TRANSACTION nodes
        for tx, doc in DOCUMENT_TYPES.items():
            self.graph.add_node(
                f"TX:{tx}",
                node_type="TRANSACTION",
                name=TRANSACTION_NAMES.get(tx, tx),
                description=doc.get("description", ""),
                module_id=doc.get("module_id", 0)
            )

            # 2. Thêm KEYWORD nodes và edges
            for kw in doc.get("keywords", []):
                kw_node = f"KW:{kw}"
                if not self.graph.has_node(kw_node):
                    self.graph.add_node(kw_node, node_type="KEYWORD", text=kw)
                self.graph.add_edge(f"TX:{tx}", kw_node, edge_type="HAS_KEYWORD")
                self.keyword_to_tx[kw.lower()].add(tx)

            # 3. Thêm CONCEPT nodes và edges
            for concept in CONCEPTS.get(tx, []):
                concept_node = f"CONCEPT:{concept}"
                if not self.graph.has_node(concept_node):
                    self.graph.add_node(concept_node, node_type="CONCEPT", text=concept)
                self.graph.add_edge(f"TX:{tx}", concept_node, edge_type="HAS_CONCEPT")
                self.concept_to_tx[concept.lower()].add(tx)

        # 4. Thêm ACCOUNT nodes và edges từ posting rules
        for tx, rules in POSTING_RULES.items():
            for rule in rules:
                account_source_type = rule.get("account_source_type", "FIXED")
                if account_source_type == "FIXED":
                    acc = rule.get("fixed_account_code", "")
                    if acc:
                        acc_node = f"ACC:{acc}"
                        if not self.graph.has_node(acc_node):
                            self.graph.add_node(
                                acc_node,
                                node_type="ACCOUNT",
                                code=acc,
                                name=ACCOUNT_NAMES.get(acc, acc)
                            )
                        edge_type = "DEBIT" if rule["side"] == "DEBIT" else "CREDIT"
                        self.graph.add_edge(
                            f"TX:{tx}",
                            acc_node,
                            edge_type=edge_type,
                            role=rule["role_key"],
                            description=rule.get("description", "")
                        )
                        self.account_to_tx[acc].append((tx, rule["side"]))

        # 5. Thêm RELATED edges giữa các concepts tương tự
        self._add_concept_relations()

    def _add_concept_relations(self):
        """Thêm edges giữa các concepts liên quan (cùng domain)"""
        # Bán hàng domain
        sale_concepts = ["xuất kho", "bán hàng", "giao hàng", "doanh thu"]
        self._connect_concepts(sale_concepts)

        # Mua hàng domain
        purchase_concepts = ["nhập kho", "mua hàng", "nhận hàng", "hàng về"]
        self._connect_concepts(purchase_concepts)

        # Công nợ domain
        debt_concepts = ["công nợ phải thu", "công nợ phải trả", "khách hàng nợ", "nợ NCC"]
        self._connect_concepts(debt_concepts)

        # Tiền domain
        cash_concepts = ["thu tiền", "chi tiền", "tiền về", "tiền ra"]
        self._connect_concepts(cash_concepts)

    def _connect_concepts(self, concepts):
        """Kết nối các concepts trong cùng domain"""
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                n1, n2 = f"CONCEPT:{c1}", f"CONCEPT:{c2}"
                if self.graph.has_node(n1) and self.graph.has_node(n2):
                    self.graph.add_edge(n1, n2, edge_type="RELATED", weight=0.5)
                    self.graph.add_edge(n2, n1, edge_type="RELATED", weight=0.5)

    def get_node_info(self, node_id):
        """Lấy thông tin của node"""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None

    def get_neighbors(self, node_id, edge_type=None):
        """Lấy các node láng giềng"""
        if not self.graph.has_node(node_id):
            return []

        neighbors = []
        for _, target, data in self.graph.edges(node_id, data=True):
            if edge_type is None or data.get("edge_type") == edge_type:
                neighbors.append((target, data))
        return neighbors

    def get_transactions_by_account(self, account_code):
        """Tìm transactions sử dụng account này"""
        return self.account_to_tx.get(account_code, [])


# =============================================================================
# MINI-RAG: GRAPH-BASED RETRIEVER
# =============================================================================

class MiniRAGRetriever:
    """
    Graph-based retrieval kết hợp:
    1. Query Expansion (mở rộng từ đồng nghĩa)
    2. Keyword matching trên graph
    3. Concept matching trên graph
    4. Negative keyword penalty
    5. Disambiguation rules
    6. Graph traversal để tìm related nodes
    7. Embedding similarity (fallback)
    """

    def __init__(self, graph: MiniRAGGraph):
        self.graph = graph
        self.embed_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

        # Pre-compute embeddings cho transactions
        self.tx_embeddings = {}
        for tx, doc in DOCUMENT_TYPES.items():
            text = f"{tx} {TRANSACTION_NAMES.get(tx, '')} {doc.get('description', '')} {' '.join(doc.get('keywords', []))}"
            self.tx_embeddings[tx] = self.embed_model.encode(text, normalize_embeddings=True)

    @staticmethod
    def expand_query(query: str) -> str:
        """Mở rộng query bằng từ đồng nghĩa"""
        query_lower = query.lower()
        expanded_terms = [query_lower]

        for term, synonyms in SYNONYMS.items():
            # Nếu query chứa synonym, thêm term gốc
            for syn in synonyms:
                if syn.lower() in query_lower:
                    expanded_terms.append(term)
                    break
            # Nếu query chứa term gốc, thêm các synonyms
            if term in query_lower:
                expanded_terms.extend(synonyms)

        return " ".join(expanded_terms)

    @staticmethod
    def apply_negative_penalty(query_lower: str, scores: dict) -> dict:
        """Áp dụng penalty cho negative keywords"""
        penalties = {}
        for tx, neg_keywords in NEGATIVE_KEYWORDS.items():
            for neg_kw in neg_keywords:
                if neg_kw in query_lower:
                    if tx not in penalties:
                        penalties[tx] = 0
                    penalties[tx] += 2.0  # Penalty weight

        for tx, penalty in penalties.items():
            if tx in scores:
                scores[tx] -= penalty

        return penalties

    @staticmethod
    def apply_disambiguation(query_lower: str, scores: dict) -> Optional[dict]:
        """Áp dụng disambiguation rules"""
        matched_rule = None
        for rule in DISAMBIGUATION_RULES:
            # Check if_contains
            has_positive = any(kw in query_lower for kw in rule["if_contains"])
            # Check and_not_contains
            has_negative = any(kw in query_lower for kw in rule["and_not_contains"])

            if has_positive and not has_negative:
                tx = rule["then"]
                scores[tx] += rule["boost"]
                matched_rule = rule
                break  # Chỉ áp dụng rule đầu tiên match

        return matched_rule

    def retrieve(self, query: str) -> dict:
        """
        Retrieve nghiệp vụ phù hợp nhất với query

        Returns:
            {
                "transaction": str,
                "score": float,
                "method": str,
                "matched_keywords": list,
                "matched_concepts": list,
                "debug": dict
            }
        """
        query_lower = query.lower()
        scores = defaultdict(float)
        debug = {
            "original_query": query,
            "expanded_query": "",
            "keyword_matches": {},
            "concept_matches": {},
            "negative_penalties": {},
            "disambiguation_rule": None,
            "graph_traversal": [],
            "embedding_scores": {}
        }

        # 1. KEYWORD MATCHING (weight: 3.0) - dùng query gốc trước
        matched_keywords = {}
        for kw, txs in self.graph.keyword_to_tx.items():
            if kw in query_lower:
                for tx in txs:
                    scores[tx] += 3.0
                    if tx not in matched_keywords:
                        matched_keywords[tx] = []
                    matched_keywords[tx].append(kw)
        debug["keyword_matches"] = matched_keywords

        # 2. CONCEPT MATCHING (weight: 2.0) - dùng query gốc
        matched_concepts = {}
        for concept, txs in self.graph.concept_to_tx.items():
            if concept in query_lower:
                for tx in txs:
                    scores[tx] += 2.0
                    if tx not in matched_concepts:
                        matched_concepts[tx] = []
                    matched_concepts[tx].append(concept)
        debug["concept_matches"] = matched_concepts

        # 2.5 QUERY EXPANSION - Chỉ dùng khi KHÔNG có direct match
        if not matched_keywords and not matched_concepts:
            expanded_query = self.expand_query(query)
            debug["expanded_query"] = expanded_query
            # Thử lại với expanded query
            for kw, txs in self.graph.keyword_to_tx.items():
                if kw in expanded_query:
                    for tx in txs:
                        scores[tx] += 2.0  # Weight thấp hơn direct match
                        if tx not in matched_keywords:
                            matched_keywords[tx] = []
                        matched_keywords[tx].append(f"{kw} (expanded)")
            debug["keyword_matches"] = matched_keywords

        # 3. NEGATIVE KEYWORD PENALTY - Phạt nếu chứa từ loại trừ
        penalties = self.apply_negative_penalty(query_lower, scores)
        debug["negative_penalties"] = penalties

        # 4. DISAMBIGUATION RULES - Áp dụng quy tắc phân biệt
        matched_rule = self.apply_disambiguation(query_lower, scores)
        if matched_rule:
            debug["disambiguation_rule"] = matched_rule.get("then")

        # 5. GRAPH TRAVERSAL - Tìm related transactions qua concepts (weight: 0.5)
        traversed = set()
        for tx in list(scores.keys()):
            if scores[tx] > 0:
                tx_node = f"TX:{tx}"
                for neighbor, edge_data in self.graph.get_neighbors(tx_node, "HAS_CONCEPT"):
                    for related, rel_data in self.graph.get_neighbors(neighbor, "RELATED"):
                        if related not in traversed:
                            traversed.add(related)
                            for back_edge in self.graph.graph.in_edges(related, data=True):
                                source, _, data = back_edge
                                if source.startswith("TX:") and data.get("edge_type") == "HAS_CONCEPT":
                                    related_tx = source.replace("TX:", "")
                                    if related_tx != tx:
                                        scores[related_tx] += 0.5 * rel_data.get("weight", 0.5)
        debug["graph_traversal"] = list(traversed)

        # 6. EMBEDDING SIMILARITY (weight: 1.5) - Fallback
        query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
        for tx, tx_emb in self.tx_embeddings.items():
            sim = float(np.dot(query_embedding, tx_emb))
            debug["embedding_scores"][tx] = round(sim, 4)
            if tx not in matched_keywords and tx not in matched_concepts:
                scores[tx] += sim * 1.5
            else:
                scores[tx] += sim * 0.5

        # 7. Chọn transaction tốt nhất
        if not scores:
            best_tx = max(self.tx_embeddings.keys(),
                         key=lambda t: np.dot(query_embedding, self.tx_embeddings[t]))
            method = "embedding_only"
        else:
            best_tx = max(scores.keys(), key=lambda k: scores[k])
            if debug["disambiguation_rule"]:
                method = "disambiguation"
            elif matched_keywords.get(best_tx):
                method = "keyword"
            elif matched_concepts.get(best_tx):
                method = "concept"
            else:
                method = "graph_traversal"

        return {
            "transaction": best_tx,
            "score": round(scores.get(best_tx, 0), 4),
            "method": method,
            "matched_keywords": matched_keywords.get(best_tx, []),
            "matched_concepts": matched_concepts.get(best_tx, []),
            "all_scores": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
            "debug": debug
        }


# =============================================================================
# POSTING ENGINE (unchanged)
# =============================================================================

class PostingEngineResolver:
    @staticmethod
    def resolve(tx, item_group, partner_group):
        rules = POSTING_RULES.get(tx, [])
        entries = []

        for r in sorted(rules, key=lambda x: x["priority"]):
            role_key = r["role_key"]
            account_source_type = r.get("account_source_type", "FIXED")

            if account_source_type == "FIXED":
                acc = r.get("fixed_account_code", "")
            elif account_source_type == "LOOKUP":
                item_group_info = POSTING_GROUPS.get(item_group, {})
                if item_group_info.get("posting_group_type") == "ITEM_GROUP":
                    acc = GL_MAPPING.get(item_group, {}).get(role_key, "")
                else:
                    acc = GL_MAPPING.get(partner_group, {}).get(role_key, "")
            else:
                raise ValueError(f"Invalid account_source_type: {account_source_type}")

            entries.append({
                "side": r["side"],
                "account": acc,
                "priority": r["priority"],
                "description": r.get("description", "")
            })

        return entries


# =============================================================================
# INIT MINI-RAG
# =============================================================================

MINI_RAG_GRAPH = MiniRAGGraph()
MINI_RAG_RETRIEVER = MiniRAGRetriever(MINI_RAG_GRAPH)


# =============================================================================
# CONTEXT WINDOW - Lưu lịch sử hội thoại
# =============================================================================

class ConversationContext:
    """Lưu trữ context của 5 câu hỏi gần nhất"""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history = []  # [{question, transaction, transaction_name}]

    def add(self, question: str, transaction: str):
        """Thêm câu hỏi vào lịch sử"""
        self.history.append({
            "question": question,
            "transaction": transaction,
            "transaction_name": TRANSACTION_NAMES.get(transaction, transaction)
        })
        # Giữ tối đa max_history
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_last_transaction(self) -> Optional[str]:
        """Lấy nghiệp vụ của câu hỏi trước"""
        if self.history:
            return self.history[-1]["transaction"]
        return None

    def get_context_summary(self) -> str:
        """Tóm tắt context cho prompt"""
        if not self.history:
            return ""
        lines = ["Context truoc do:"]
        for i, h in enumerate(self.history[-3:], 1):  # Chỉ lấy 3 gần nhất
            lines.append(f"  Q{i}: {h['question'][:50]}... -> {h['transaction_name']}")
        return "\n".join(lines)

    def clear(self):
        """Xóa lịch sử"""
        self.history = []

    def load(self, history: list):
        """Load lịch sử từ bên ngoài (API, database...)"""
        self.history = []
        for item in history[-self.max_history:]:  # Chỉ lấy max_history gần nhất
            if "question" in item and "transaction" in item:
                self.history.append({
                    "question": item["question"],
                    "transaction": item["transaction"],
                    "transaction_name": TRANSACTION_NAMES.get(item["transaction"], item["transaction"])
                })

    def is_followup_question(self, question: str) -> bool:
        """Kiểm tra xem câu hỏi có phải follow-up không"""
        if not self.history:
            return False

        question_lower = question.lower()

        # Các từ khóa chỉ follow-up question
        followup_indicators = [
            "còn", "thế còn", "vậy còn", "còn gì", "còn nữa",
            "thế", "vậy", "sao", "thế nào",
            "nó", "cái đó", "cái này", "nghiệp vụ đó", "nghiệp vụ này",
            "tiếp", "tiếp theo", "bước tiếp",
            "thuế", "chiết khấu",  # Thường hỏi thêm về thuế/CK của nghiệp vụ trước
            "ví dụ", "cho ví dụ",
            "giải thích", "giải thích thêm", "chi tiết hơn",
        ]

        # Câu hỏi ngắn (< 30 ký tự) thường là follow-up
        is_short = len(question) < 30

        # Chứa từ khóa follow-up
        has_indicator = any(ind in question_lower for ind in followup_indicators)

        # Không chứa keywords/concepts của bất kỳ nghiệp vụ nào
        result = MINI_RAG_RETRIEVER.retrieve(question)
        no_direct_match = not result['matched_keywords'] and not result['matched_concepts']

        return (is_short and has_indicator) or (has_indicator and no_direct_match)


# Global context instance
CONVERSATION_CONTEXT = ConversationContext()


# =============================================================================
# RAG SERVICE (Updated to use MiniRAG)
# =============================================================================

class RagAccounting:

    @staticmethod
    def reset_context():
        """Reset conversation context"""
        CONVERSATION_CONTEXT.clear()
        print("[Context] Cleared")

    @staticmethod
    def get_context():
        """Get current context history"""
        return CONVERSATION_CONTEXT.history

    @staticmethod
    def load_context(history: list):
        """Load context history from external source"""
        CONVERSATION_CONTEXT.load(history)
        print(f"[Context] Loaded {len(CONVERSATION_CONTEXT.history)} items")

    @staticmethod
    def ask(question, item_group="GOODS", partner_group="CUSTOMER"):
        # ========== STEP 0: CHECK CONTEXT ==========
        is_followup = CONVERSATION_CONTEXT.is_followup_question(question)
        context_tx = CONVERSATION_CONTEXT.get_last_transaction()

        if is_followup and context_tx:
            print(f"\n[0] CONTEXT")
            print(f"    Follow-up detected, using: {context_tx} ({TRANSACTION_NAMES.get(context_tx, '')})")

        # ========== STEP 1: RETRIEVAL ==========
        print(f"\n[1] RETRIEVAL")
        print(f"    Query: {question}")

        result = MINI_RAG_RETRIEVER.retrieve(question)
        tx = result["transaction"]
        debug = result['debug']

        # Nếu là follow-up và không có direct match -> dùng context
        if is_followup and context_tx:
            if not result['matched_keywords'] and not result['matched_concepts']:
                tx = context_tx
                print(f"    No direct match, using context: {tx}")

        print(f"    Keywords: {result['matched_keywords'] or '(none)'}")
        print(f"    Concepts: {result['matched_concepts'] or '(none)'}")
        if debug.get('disambiguation_rule'):
            print(f"    Disambiguation: {debug['disambiguation_rule']}")
        print(f"    -> {tx} ({TRANSACTION_NAMES.get(tx, tx)})")

        # Lưu vào context
        CONVERSATION_CONTEXT.add(question, tx)

        # ========== STEP 2: POSTING ENGINE ==========
        print(f"\n[2] POSTING ENGINE")
        entries = PostingEngineResolver.resolve(tx, item_group, partner_group)
        print(f"    Item Group: {item_group}")

        for e in sorted(entries, key=lambda x: x["priority"]):
            acc = e["account"]
            acc_name = ACCOUNT_NAMES.get(acc, acc)
            side = "Dr" if e["side"] == "DEBIT" else "Cr"
            print(f"    {side} {acc} ({acc_name})")

        # ========== STEP 3: BUILD PROMPT & CALL SLM ==========
        tx_name = TRANSACTION_NAMES.get(tx, tx)

        print(f"\n[3] PROMPT")
        print(f"    Question: {question}")
        print(f"    Transaction: {tx_name}")

        # Build entries list for prompt
        entries_list = []
        for e in sorted(entries, key=lambda x: x["priority"]):
            acc = e["account"]
            acc_name = ACCOUNT_NAMES.get(acc, acc)
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            entries_list.append(f"{side} TK {acc} - {acc_name}")
        entries_text = "\n".join(entries_list)

        # System prompt
        system_prompt = "Bạn là trợ lý kế toán. Trả lời ngắn gọn, chính xác, chỉ dùng text thuần (không markdown)."

        # User prompt
        slm_prompt = f"""Câu hỏi: {question}

Bút toán:
{entries_text}

Trả lời theo định dạng:
{FEW_SHOT_QA}"""

        print(f"\n[4] SLM Generating (streaming)...")

        # Stream response from SLM
        slm_output = ""
        try:
            stream = ollama.chat(
                model="qwen2.5:3b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": slm_prompt}
                ],
                stream=True
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    slm_output += content
                    yield content
        except Exception as e:
            print(f"[SLM Error] {e}")

        # Fallback nếu SLM trả rỗng hoặc thiếu format
        if not slm_output or "1." not in slm_output:
            print("[Fallback] SLM output invalid, using fallback...")
            fallback = RagAccounting._generate_fallback(tx_name, entries)
            yield fallback

        print(f"\n    {'-'*40}")

    @staticmethod
    def _generate_fallback(tx_name: str, entries: list) -> str:
        """Fallback khi SLM trả rỗng hoặc sai format"""
        lines = [
            f"1. TÊN NGHIỆP VỤ:\n{tx_name}",
            "",
            "2. BẢNG BÚT TOÁN:"
        ]
        for e in sorted(entries, key=lambda x: x["priority"]):
            acc = e["account"]
            acc_name = ACCOUNT_NAMES.get(acc, acc)
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            lines.append(f"- {side} TK {acc}: {acc_name}")
        lines.append("")

        # Section 3: GIẢI THÍCH
        lines.append("3. GIẢI THÍCH:")
        for e in sorted(entries, key=lambda x: x["priority"]):
            acc = e["account"]
            side = "Nợ" if e["side"] == "DEBIT" else "Có"
            desc = e.get("description", "")
            lines.append(f"- {side} TK {acc}: {desc}")
        lines.append("")

        # Section 4: VÍ DỤ
        lines.append("4. VÍ DỤ:")
        lines.append("(Vui lòng tham khảo ví dụ cụ thể từ kế toán)")

        return "\n".join(lines)
