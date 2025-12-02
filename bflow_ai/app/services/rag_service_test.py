import json
import os
import faiss
import ollama
from uuid import UUID

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /app/services

APP_DIR = os.path.dirname(BASE_DIR)  # /app

class HandleJsonFile:
    @staticmethod
    def read(file_name='account_determination_export.json'):
        file_path = os.path.join(APP_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# RAG
print("Đang load model SentenceTransformer...")
_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
print("Model loaded thành công!")

account_determination = HandleJsonFile.read(file_name='account_determination_export.json')
account_determination_text = [f"{ac['transaction_key']} {ac['foreign_title']} {ac['description']} {ac['type']}" for ac in account_determination]
account_embedding = _model.encode(account_determination_text, convert_to_numpy=True, show_progress_bar=True)
account_dimension = account_embedding.shape[1]
account_index = faiss.IndexFlatL2(account_dimension)
account_index.add(account_embedding)
print("FAISS index đã sẵn sàng cho accounting!")

class RagAccounting:
    @staticmethod
    def rag_accounting(user_input: str, top_k: int = 1):
        try:
            user_embedding = _model.encode([user_input], convert_to_numpy=True, show_progress_bar=True)
            D, I = account_index.search(np.array(user_embedding).astype('float32'), k=top_k)
            print(user_input)
            results = []
            for idx, dist in zip(I[0], D[0]):
                func = account_determination[idx]
                results.append({
                    "transaction_key": func['transaction_key'],
                    "foreign_title": func['foreign_title'],
                    "description": func['description'],
                    "type": func['type'],
                    "sub_items": func['sub_items'],
                    "distance": float(dist),
                })
            for r in results:
                print(r)
            return results[0] if results else None
        except Exception as e:
            print("Lỗi RAG:", e)
            return None

    @staticmethod
    def synthesize_answer(user_query, retrieved_texts):
        """
        Tổng hợp câu trả lời bằng Qwen 2.5 (1.5B) chạy qua Ollama.
        """

        # CHUYỂN DICT → JSON FORMAT (RẤT QUAN TRỌNG)
        context = json.dumps(retrieved_texts, indent=2, ensure_ascii=False)
        print("\n===== CONTEXT DÙNG CHO LLM =====")
        print(context)
        print("================================\n")
        prompt = f"""
    Bạn là một hệ thống trả lời dựa trên tài liệu kế toán (RAG).  
**TUYỆT ĐỐI chỉ dùng thông tin có trong JSON dưới đây. Không được bịa, không suy luận ngoài dữ liệu.**

Câu hỏi của người dùng:
{user_query}

Dữ liệu JSON liên quan (đã flatten nếu cần):
{context}

Yêu cầu:
1. Trả lời bằng **tiếng Việt**, rõ ràng, dưới dạng **html đẹp, xuống dòng nếu cần thiết**.
2. Phải sử dụng **tất cả thông tin trong JSON**, bao gồm:
   - transaction_key
   - foreign_title
   - description
   - type
   - distance
   - TẤT CẢ sub_items, mỗi sub_item phải bao gồm: transaction_key_sub, description, account_mapped_code, account_mapped_name
3. Nếu user hỏi “ghi vào đâu”, hãy liệt kê **từng sub_item** dưới dạng danh sách, ví dụ:
   - Nội dung → Tài khoản → Mã tài khoản → Mã sub_item
4. **Không bỏ sót bất kỳ sub_item nào.**
5. Nếu không có thông tin trong JSON, trả: "Tôi không tìm thấy thông tin chính xác trong tài liệu."

Bắt đầu trả lời:
    """
        response = ollama.generate(
            model="gemma2:2b",
            prompt=prompt,
            options={'temperature': 0.1}
        )
        return response["response"].strip().replace("*", "")