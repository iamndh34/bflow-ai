"""
Templates cho COA Agent - Chart of Accounts Specialist

Chứa các template format cho:
- Lookup: Tra cứu thông tin tài khoản
- Compare: So sánh tài khoản giữa TT200 và TT99
- Compare Circular: So sánh tổng quan 2 thông tư
"""


def get_lookup_template() -> str:
    """
    Template cho tra cứu thông tin tài khoản.

    Returns:
        Template string
    """
    return """1. THÔNG TIN CƠ BẢN:
- Số hiệu TK: [Mã tài khoản]
- Tên tài khoản: [Tên tiếng Việt]
- Tên tiếng Anh: [Tên tiếng Anh]
- Loại tài khoản: [Tài sản/Nợ phải trả/Vốn/Doanh thu/Chi phí...]

2. NỘI DUNG PHẢN ÁNH:
- Mô tả chức năng chính của tài khoản
- Các giao dịch thường xuyên liên quan

3. KẾT CẤU VÀ SỐ DƯ:
- Bên Nợ: Ghi nhận những gì phát sinh tăng
- Bên Có: Ghi nhận những gì phát sinh giảm
- Số dư: Thường nằm bên nào (Nợ/Có/Không có)

4. LIÊN KẾT VỚI CÁC TK KHÁC:
- Các tài khoản thường đối ứng
- Các tài khoản liên quan trong chuỗi giá trị"""



def get_compare_template() -> str:
    """
    Template cho so sánh tài khoản giữa TT200 và TT99.

    Returns:
        Template string
    """
    return """SO SÁNH TÀI KHOẢN [Số hiệu TK]

| Tiêu chí | TT200 (2014) | TT99 (2025) |
|----------|--------------|-------------|
| Số hiệu | [Số hiệu] | [Số hiệu] |
| Tên TK | [Tên TT200] | [Tên TT99] |
| Tên EN | [Tên EN TT200] | [Tên EN TT99] |
| Loại TK | [Loại TT200] | [Loại TT99] |
| TK con | [Có/Không, liệt kê] | [Có/Không] |

NHẬN XÉT:
- [Nếu TK mới trong TT99] TK này được bổ sung để phản ánh...
- [Nếu TK bị xóa] TK này bị loại bỏ vì...
- [Nếu đổi tên] TT99 đổi tên từ "..." thành "..." để rõ nghĩa hơn về...
- [Nếu khác biệt khác] Các thay đổi khác về..."""



def get_compare_circular_template() -> str:
    """
    Template cho so sánh tổng quan TT200 vs TT99.

    Returns:
        Template string
    """
    return """SO SÁNH TỔNG QUAN: THÔNG TƯ 200/2014 vs THÔNG TƯ 99/2025

1. TỔNG QUAN VỀ CÁC THAY ĐỔI
- Tổng số thay đổi: [Số lượng]
- Phân loại: [Số lượng từng loại: thêm, xóa, đổi tên...]

2. CÁC TÀI KHOẢN MỚI ĐƯỢC BỔ SUNG TRONG TT99
- [Liệt kê các TK mới]
- Giải thích ý nghĩa của việc bổ sung

3. CÁC TÀI KHOẢN BỊ LOẠI BỎ SO VỚI TT200
- [Liệt kê các TK bị xóa]
- Lý do hoặc tác động của việc loại bỏ

4. CÁC TÀI KHOẢN ĐƯỢC ĐỔI TÊN
- [Liệt kê các TK đổi tên, format: TK xxx: "Tên cũ" → "Tên mới"]
- Giải thích lý do đổi tên

5. CÁC TÀI KHOẢN BỎ CHI TIẾT CẤP 2
- [Liệt kê các TK bị bỏ cấp 2]
- Tác động đến việc hạch toán chi tiết

6. NHẬN XÉT TỔNG QUAN VỀ XU HƯỚNG THAY ĐỔI
- Đánh giá xu hướng: Đơn giản hóa / Phức tạp hóa / Hiện đại hóa?
- Ý nghĩa thực tiễn cho kế toán doanh nghiệp"""
