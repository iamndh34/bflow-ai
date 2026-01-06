import json
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from app.core.config import settings

# =============================================================================
# CONFIG & DATA LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COA_99_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_99.json")
COA_200_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_200.json")

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

# Build indexes for TT99 (default)
COA_DATA = COA_99_DATA
COA_BY_CODE = {acc["code"]: acc for acc in COA_99_DATA}
COA_BY_TYPE = {}
for acc in COA_99_DATA:
    type_name = acc["type_name"]
    if type_name not in COA_BY_TYPE:
        COA_BY_TYPE[type_name] = []
    COA_BY_TYPE[type_name].append(acc)

# Build indexes for TT200
COA_200_BY_CODE = {acc["code"]: acc for acc in COA_200_DATA}

# =============================================================================
# FEW-SHOT PROMPTS (UPDATED PROFESSIONAL VERSION)
# =============================================================================

FEW_SHOT_COMPARE = """
SO SÁNH TÀI KHOẢN 112 - TIỀN GỬI NGÂN HÀNG

| Tiêu chí         | TT200 (2014)              | TT99 (2025)                   |
|------------------|---------------------------|-------------------------------|
| Số hiệu          | 112                       | 112                           |
| Tên tài khoản    | Tiền gửi ngân hàng        | Tiền gửi không kỳ hạn         |
| Tên tiếng Anh    | Cash in bank              | Cash in bank (Demand deposits)|
| Loại tài khoản   | Tài sản                   | Tài sản                       |
| TK con           | 1121, 1122, 1123          | Không có TK con               |

NHẬN XÉT:
- TT99 đổi tên từ "Tiền gửi ngân hàng" thành "Tiền gửi không kỳ hạn" để phân biệt rõ với tiền gửi có kỳ hạn.
- TT99 đơn giản hóa cấu trúc, không chia TK con theo loại tiền (VND, ngoại tệ, vàng).
"""

FEW_SHOT_COA = """
1. THÔNG TIN CƠ BẢN:
- Số hiệu: 156
- Tên tài khoản: Hàng hóa (Merchandise inventory)
- Loại tài khoản: Tài sản (Loại 1)

2. NỘI DUNG PHẢN ÁNH:
- Tài khoản này dùng để phản ánh trị giá hiện có và tình hình biến động tăng, giảm của các loại hàng hóa của doanh nghiệp. Hàng hóa bao gồm: hàng mua về để bán, hàng gửi đi bán, hàng hóa bất động sản.

3. KẾT CẤU VÀ NỘI DUNG:
- Bên Nợ:
  + Trị giá mua vào của hàng hóa nhập kho.
  + Chi phí thu mua hàng hóa.
  + Trị giá hàng hóa bị trả lại.
- Bên Có:
  + Trị giá vốn của hàng hóa xuất kho để bán, gửi đi bán.
  + Chiết khấu thương mại, giảm giá hàng mua được hưởng.
- Số dư bên Nợ: Trị giá thực tế của hàng hóa tồn kho cuối kỳ.
"""


# =============================================================================
# CLASS RAG COA
# =============================================================================

class RagCOA:
    """
    Hỏi đáp về hệ thống tài khoản theo TT99
    """

    _embed_model = None
    _coa_embeddings = None

    @classmethod
    def _init_embeddings(cls):
        """Lazy init embeddings"""
        if cls._embed_model is None:
            cls._embed_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
            cls._coa_embeddings = {}
            for acc in COA_DATA:
                text = f"{acc['code']} {acc['name']} {acc.get('name_en', '')} {acc['type_name']}"
                cls._coa_embeddings[acc["code"]] = cls._embed_model.encode(text, normalize_embeddings=True)

    @staticmethod
    def lookup_by_code(code: str) -> Optional[dict]:
        code = code.strip()
        return COA_BY_CODE.get(code)

    @staticmethod
    def lookup_by_type(type_name: str) -> list:
        return COA_BY_TYPE.get(type_name, [])

    @staticmethod
    def search_by_name(keyword: str) -> list:
        keyword_lower = keyword.lower()
        results = []
        for acc in COA_DATA:
            if keyword_lower in acc["name"].lower() or keyword_lower in acc.get("name_en", "").lower():
                results.append(acc)
        return results

    @classmethod
    def search_by_embedding(cls, query: str, top_k: int = 5) -> list:
        cls._init_embeddings()
        query_emb = cls._embed_model.encode(query, normalize_embeddings=True)

        scores = []
        for code, emb in cls._coa_embeddings.items():
            sim = float(np.dot(query_emb, emb))
            scores.append((code, sim))

        scores.sort(key=lambda x: -x[1])
        return [COA_BY_CODE[code] for code, _ in scores[:top_k]]

    @classmethod
    def ask(cls, question: str):
        """
        Hỏi đáp về tài khoản (streaming response)
        """
        question_lower = question.lower()
        print(f"\n[COA] Question: {question}")

        # 1. Tìm tài khoản phù hợp
        accounts = cls._find_accounts(question, question_lower)

        if not accounts:
            yield "Không tìm thấy tài khoản phù hợp trong danh mục Thông tư 99."
            return

        # 2. Build context
        if len(accounts) == 1:
            acc = accounts[0]
            acc_info = f"TK {acc['code']} - {acc['name']} ({acc.get('name_en', 'N/A')}), Loại: {acc['type_name']}"
        else:
            acc_info = "\n".join([f"- TK {a['code']}: {a['name']} ({a['type_name']})" for a in accounts[:5]])

        print(f"[COA] Found {len(accounts)} account(s)")

        # 3. Build prompt
        system_prompt = "Bạn là trợ lý kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT. Trả lời đầy đủ, cấu trúc rõ ràng."
        slm_prompt = f"""Câu hỏi: {question}

Thông tin tài khoản tìm được:
{acc_info}

Hãy trả lời theo định dạng mẫu sau:
{FEW_SHOT_COA}"""

        print(f"[COA] SLM Generating (streaming)...")

        # 4. Stream response
        slm_output = ""
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            stream = client.chat(
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
            print(f"[COA SLM Error] {e}")

        # 5. Fallback
        if not slm_output or "1." not in slm_output:
            print("[COA Fallback] SLM output invalid, using fallback...")
            fallback = cls._generate_fallback(accounts)
            yield fallback
        else:
            yield "\n\n(Căn cứ: Phụ lục II - Thông tư 99/2025/TT-BTC)"

        print(f"\n    {'-' * 40}")

    @classmethod
    def _find_accounts(cls, question: str, question_lower: str) -> list:
        # 1. Tìm theo code
        code_match = re.search(r'\b(\d{3,5})\b', question)
        if code_match:
            code = code_match.group(1)
            acc = cls.lookup_by_code(code)
            if acc: return [acc]

        # 2. Tìm theo loại
        type_keywords = {
            "tài sản": "Tài sản", "nợ phải trả": "Nợ phải trả", "vốn chủ sở hữu": "Vốn chủ sở hữu",
            "doanh thu": "Doanh thu", "chi phí": "Chi phí SXKD",
            "thu nhập khác": "Thu nhập khác", "chi phí khác": "Chi phí khác",
        }
        for kw, type_name in type_keywords.items():
            if kw in question_lower and any(w in question_lower for w in ["nhóm", "loại", "các", "những", "danh sách"]):
                accounts = cls.lookup_by_type(type_name)
                if accounts: return accounts

        # 3. Tìm theo tên (keyword)
        search_terms = ["hàng hóa", "tiền mặt", "phải thu", "phải trả", "doanh thu", "giá vốn",
                        "nguyên vật liệu", "công cụ", "tài sản cố định", "vốn", "lợi nhuận",
                        "thuế", "sinh học", "tiền gửi"]
        for term in search_terms:
            if term in question_lower:
                results = cls.search_by_name(term)
                if results: return results

        # 4. Embedding search
        return cls.search_by_embedding(question, top_k=3)

    @staticmethod
    def _generate_fallback(accounts: list) -> str:
        """Fallback chuyên nghiệp khi SLM trả rỗng"""
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

    # =========================================================================
    # COMPARE TT200 vs TT99
    # =========================================================================

    @classmethod
    def compare(cls, question: str):
        """
        So sánh tài khoản giữa TT200 và TT99 (streaming response)
        """
        print(f"\n[COMPARE] Question: {question}")

        # 1. Tìm mã tài khoản trong câu hỏi
        code_match = re.search(r'\b(\d{3,5})\b', question)
        if not code_match:
            yield "Vui lòng chỉ định số tài khoản cần so sánh (ví dụ: 'So sánh TK 156 giữa TT200 và TT99')."
            return

        code = code_match.group(1)
        acc_99 = COA_BY_CODE.get(code)
        acc_200 = COA_200_BY_CODE.get(code)

        print(f"[COMPARE] Code: {code}, TT99: {acc_99 is not None}, TT200: {acc_200 is not None}")

        # 2. Build comparison context
        if not acc_99 and not acc_200:
            yield f"Không tìm thấy tài khoản {code} trong cả TT200 và TT99."
            return

        compare_info = cls._build_compare_context(code, acc_99, acc_200)

        # 3. Build prompt
        system_prompt = "Bạn là chuyên gia kế toán Việt Nam. LUÔN trả lời bằng TIẾNG VIỆT. So sánh sự khác biệt giữa 2 thông tư một cách rõ ràng."
        slm_prompt = f"""Câu hỏi: {question}

Thông tin so sánh:
{compare_info}

Hãy trả lời theo định dạng mẫu sau:
{FEW_SHOT_COMPARE}"""

        print(f"[COMPARE] SLM Generating...")

        # 4. Stream response
        slm_output = ""
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            stream = client.chat(
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
            print(f"[COMPARE SLM Error] {e}")

        # 5. Fallback
        if not slm_output or "SO SÁNH" not in slm_output.upper():
            print("[COMPARE Fallback] Using fallback...")
            yield cls._generate_compare_fallback(code, acc_99, acc_200)

        print(f"\n    {'-' * 40}")

    @staticmethod
    def _build_compare_context(code: str, acc_99: dict, acc_200: dict) -> str:
        """Build context string for comparison"""
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

    @staticmethod
    def _generate_compare_fallback(code: str, acc_99: dict, acc_200: dict) -> str:
        """Fallback khi SLM không hoạt động"""
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