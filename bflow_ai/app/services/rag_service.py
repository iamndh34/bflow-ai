import json
import os
import faiss
import ollama
import numpy as np
import re
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_NAME = 'account_deter_json/80785ce8-f138-48b8-b7fa-5fb1971fe204.json'

class HandleJsonFile:
    @staticmethod
    def read(file_name):
        file_path = os.path.join(BASE_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ Không tìm thấy file dữ liệu tại: {file_path}")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Lỗi đọc file JSON: {e}")
            return []

print("⏳ Đang load model SentenceTransformer...")
_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
print("✅ Model loaded thành công!")

accounting_data = HandleJsonFile.read(file_name=DATA_FILE_NAME)

accounting_texts = []
try:
    if accounting_data:
        # Tạo text để embedding
        accounting_texts = [
            f"{item.get('nghiep_vu', '')} - {item.get('mo_ta_chi_tiet', '')} - {json.dumps(item.get('loai_nghiep_vu', []), ensure_ascii=False)}"
            for item in accounting_data
        ]
except Exception as e:
    print(f"⚠️ Cảnh báo cấu trúc JSON: {e}")

if accounting_texts:
    print("⏳ Đang tạo vector database...")
    account_embedding = _model.encode(accounting_texts, convert_to_numpy=True, show_progress_bar=True)
    account_dimension = account_embedding.shape[1]
    account_index = faiss.IndexFlatL2(account_dimension)
    account_index.add(account_embedding)
    print("✅ FAISS index đã sẵn sàng!")
else:
    account_index = None
    print("❌ Không có dữ liệu để tạo index.")


class RagAccounting:
    @staticmethod
    def rag_accounting(user_input: str, top_k: int = 1):
        if not account_index:
            return "Hệ thống chưa có dữ liệu."

        try:
            # 1. Tìm kiếm context (Vector Search)
            user_embedding = _model.encode([user_input], convert_to_numpy=True, show_progress_bar=False)
            D, I = account_index.search(np.array(user_embedding).astype('float32'), k=top_k)

            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < 0: continue
                item = accounting_data[idx]
                results.append(item)

            if not results:
                return "Xin lỗi, tôi không tìm thấy nghiệp vụ kế toán phù hợp trong dữ liệu."

            # 2. Tổng hợp câu trả lời
            return RagAccounting.synthesize_answer(user_input, results)

        except Exception as e:
            print(f"❌ Lỗi RAG: {e}")
            return "Có lỗi xảy ra trong quá trình xử lý."

    @staticmethod
    def synthesize_answer(user_query, retrieved_data):
        context_str = json.dumps(retrieved_data, indent=2, ensure_ascii=False)

        # --- Prompt RAG chi tiết ---
        prompt = f"""
Bạn là Kế toán trưởng am hiểu Thông tư 200. Nhiệm vụ của bạn là hướng dẫn hạch toán dựa trên dữ liệu JSON được cung cấp.

[DỮ LIỆU KẾ TOÁN TÌM ĐƯỢC]:
{context_str}

[CÂU HỎI CỦA NGƯỜI DÙNG]:
"{user_query}"

[YÊU CẦU TRẢ LỜI]:
1. Chỉ dùng thông tin trong [DỮ LIỆU KẾ TOÁN TÌM ĐƯỢC]. Chỉ diễn giải, không bịa thêm.
2. Trả lời bằng định dạng **HTML**.
3. Cấu trúc câu trả lời:
   - <b>Nghiệp vụ xác định:</b> [Tên nghiệp vụ từ JSON]
   - <b>Hướng dẫn hạch toán:</b>
     - <ul>
       <li><b>Nợ TK:</b> [Liệt kê số TK và TÊN TK đầy đủ. Ví dụ: 111 (Tiền mặt)]</li>
       <li><b>Có TK:</b> [Liệt kê số TK và TÊN TK đầy đủ. Ví dụ: 112 (Tiền gửi ngân hàng)]</li>
     </ul>
   - <b>Giải thích số hiệu TK:</b> [Giải thích ngắn gọn ý nghĩa của từng tài khoản Nợ/Có xuất hiện ở trên. Ví dụ: TK 156 dùng để theo dõi giá trị hàng hóa...]
   - <b>Lưu ý/Căn cứ:</b> [Lấy từ trường ghi_chu và tham_chieu]
4. Giọng văn chuyên nghiệp, rõ ràng.

Bắt đầu trả lời ngay code HTML:
"""

        model = "qwen2.5"
        print(f"⏳ Đang gọi Ollama (model: {model})...")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'temperature': 0.1}
        )

        raw_text = response["response"].strip()

        # Clean HTML (Xóa markdown code block nếu có)
        clean_html = re.sub(r'^```html\s*', '', raw_text, flags=re.IGNORECASE)
        clean_html = re.sub(r'^```\s*', '', clean_html)
        clean_html = re.sub(r'\s*```$', '', clean_html)

        return clean_html.strip()
