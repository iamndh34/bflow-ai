import json
import os
import faiss
import ollama
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Lưu ý: Cần đảm bảo file JSON nằm đúng vị trí này so với file chạy
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_NAME = os.path.join(BASE_DIR, 'services', 'account_deter_json', '80785ce8-f138-48b8-b7fa-5fb1971fe204.json')


class HandleJsonFile:
    @staticmethod
    def read(file_path):
        if not os.path.exists(file_path):
            print(f"⚠️ Không tìm thấy file dữ liệu tại: {file_path}")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Lỗi đọc file JSON: {e}")
            return []


# --- KHỞI TẠO MODEL & VECTOR DB ---
# (Phần này sẽ chạy 1 lần khi server start)
print("⏳ Đang load model SentenceTransformer...")
try:
    _model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    print("✅ Model loaded thành công!")

    # Load dữ liệu
    # Điều chỉnh đường dẫn tương đối tùy theo cấu trúc thư mục thực tế của bạn
    # Ở đây giả định file json nằm ở ../data/...
    accounting_data = HandleJsonFile.read(DATA_FILE_NAME)

    accounting_texts = []
    if accounting_data:
        for item in accounting_data:
            nv = item.get('nghiep_vu', '')
            mt = item.get('mo_ta_chi_tiet', '')
            bct_raw = item.get('bo_chung_tu', [])
            bct_str = ", ".join(bct_raw) if isinstance(bct_raw, list) else str(bct_raw)
            text_embed = f"Nghiệp vụ: {nv}. Mô tả: {mt}. Chứng từ bao gồm: {bct_str}"
            accounting_texts.append(text_embed)

    if accounting_texts:
        print(f"⏳ Đang tạo vector database cho {len(accounting_texts)} nghiệp vụ...")
        account_embedding = _model.encode(accounting_texts, convert_to_numpy=True, show_progress_bar=False)
        account_dimension = account_embedding.shape[1]
        account_index = faiss.IndexFlatL2(account_dimension)
        account_index.add(account_embedding)
        print("✅ FAISS index đã sẵn sàng!")
    else:
        account_index = None
        print("❌ Không có dữ liệu để tạo index.")

except Exception as e:
    print(f"❌ Lỗi khởi tạo Model/Index: {e}")
    _model = None
    account_index = None
    accounting_data = []


class RagAccounting:
    @staticmethod
    def rag_accounting(user_input: str, top_k: int = 1):
        """
        Hàm Generator trả về từng token văn bản.
        """
        if not account_index or not _model:
            yield "Hệ thống đang khởi động hoặc chưa có dữ liệu. Vui lòng thử lại sau."
            return

        try:
            # 1. Tìm kiếm context
            print('Đang tìm context')
            user_embedding = _model.encode([user_input], convert_to_numpy=True, show_progress_bar=False)
            D, I = account_index.search(np.array(user_embedding).astype('float32'), k=top_k)

            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < 0: continue
                # Kiểm tra biên an toàn
                if idx < len(accounting_data):
                    item = accounting_data[idx]
                    results.append(item)

            if not results:
                yield "Xin lỗi, tôi không tìm thấy nghiệp vụ kế toán phù hợp trong cơ sở dữ liệu."
                return

            # 2. Gọi LLM Streaming
            yield from RagAccounting.synthesize_answer(user_input, results)

        except Exception as e:
            print(f"❌ Lỗi RAG: {e}")
            yield f"Có lỗi xảy ra: {str(e)}"

    @staticmethod
    def synthesize_answer(user_query, retrieved_data):
        context_str = json.dumps(retrieved_data, indent=2, ensure_ascii=False)

        # Prompt định dạng Markdown/Text (Không dùng HTML)
        prompt = f"""
            Bạn là Kế toán trưởng chuyên nghiệp. Dựa vào dữ liệu được cung cấp dưới đây để hướng dẫn hạch toán.
            
            [DỮ LIỆU TÌM ĐƯỢC]:
            {context_str}
            
            [CÂU HỎI]: "{user_query}"
            
            [YÊU CẦU TRẢ LỜI]:
            1. KHÔNG dùng thẻ HTML.
            2. Trình bày bằng văn bản (Markdown) rõ ràng, chuyên nghiệp.
            3. Sử dụng các ký tự như (-, +, *, >) hoặc Emoji để phân tách ý.
            4. Cấu trúc câu trả lời bắt buộc:
            
               🎯 NGHIỆP VỤ: [Tên nghiệp vụ]
            
               📄 MÔ TẢ: [Mô tả chi tiết]
            
               📂 BỘ CHỨNG TỪ BẮT BUỘC:
                 - [Liệt kê các chứng từ...]
            
               💰 ĐỊNH KHOẢN:
                 * Nợ TK [Số TK] - [Tên TK]
                 * Có TK [Số TK] - [Tên TK]
            
               💡 LƯU Ý & GIẢI THÍCH:
                 > [Nội dung ghi chú/tham chiếu]
            
            Bắt đầu trả lời ngay:
        """

        model = "qwen2.5:1.5b"  # Hoặc model bạn đang dùng

        # stream=True để nhận từng token
        client = ollama.Client(host='http://mis_ollama:11434')
        stream = client.generate(
            model=model,
            prompt=prompt,
            options={'temperature': 0.2},
            stream=True
        )

        for chunk in stream:
            content = chunk.get('response', '')
            if content:
                yield content

        print('Done')
        