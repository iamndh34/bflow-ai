"""
Template Module cho Posting Engine - Ngắn gọn, tiết kiệm tokens

Thay thế FEW_SHOT_EXAMPLES (~120 dòng) bằng template (~25 dòng)
Giảm ~80% tokens cho phần example trong prompt
"""


def get_response_template() -> str:
    """
    Trả về template format cho response.

    Template này:
    - Ngắn gọn (~25 dòng vs 120 dòng của few-shot)
    - Chỉ hướng dẫn format, không có ví dụ cụ thể
    - Giúp SLM hiểu rõ cấu trúc 4 phần cần hoàn thành

    Returns:
        Template string
    """
    return """1. TÊN NGHIỆP VỤ:
[Tên nghiệp vụ từ hệ thống]

2. BẢNG BÚT TOÁN:
- Nợ/Có TK [Mã TK]: [Tên tài khoản] (* nếu là LOOKUP)
- ... (liệt kê đủ các bút toán từ hệ thống)

3. GIẢI THÍCH:
- Nợ/Có TK [Mã TK]: [Ý nghĩa kế toán - tại sao ghi nhận]
- ... (giải thích từng dòng bút toán)

4. VÍ DỤ:
[Mô tả tình huống cụ thể với số tiền giả định]:
- Nợ/Có TK [Mã TK]: [Số tiền]đ
- ... (đầy đủ số liệu minh họa)

QUAN TRỌNG: BẮT BUỘC viết đầy đủ đến hết phần 4, không được dừng giữa chừng."""
