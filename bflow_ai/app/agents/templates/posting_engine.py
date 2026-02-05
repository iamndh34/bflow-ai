"""
Template Module cho Posting Engine - Ví dụ cụ thể cho từng nghiệp vụ

Mỗi nghiệp vụ có example riêng để template phù hợp, tiết kiệm tokens.

NOTE: Dấu (*) đánh dấu các tài khoản LOOKUP (phụ thuộc item_group/partner_group)
"""

# =============================================================================
# TEMPLATES CHO TỪNG NGHIỆP VỤ
# =============================================================================

_TEMPLATES = {
    "DO_SALE": """1. TÊN NGHIỆP VỤ:
Xuất kho bán hàng hóa (chưa xuất hóa đơn)

2. BẢNG BÚT TOÁN:
- Nợ TK 632 (*): Giá vốn hàng bán
- Có TK 156 (*): Hàng hóa
- Nợ TK 13881: Phải thu tạm (Giao hàng chưa xuất HĐ)
- Có TK 511 (*): Doanh thu bán hàng

3. GIẢI THÍCH:
- Nợ TK 632: Ghi nhận giá vốn hàng bán
- Có TK 156: Giảm hàng hóa trong kho
- Nợ TK 13881: Ghi nhận công nợ tạm phải thu từ khách hàng (đợi xuất HĐ)
- Có TK 511: Ghi nhận doanh thu bán hàng

4. VÍ DỤ:
Công ty giao hàng cho khách A, giá trị hàng 500.000.000đ, giá vốn 300.000.000đ, thuế GTGT 50.000.000đ (chưa xuất HĐ).
- Nợ TK 632: 300.000.000đ
- Có TK 156: 500.000.000đ
- Nợ TK 13881: 550.000.000đ
- Có TK 511: 500.000.000đ""",

    "SALES_INVOICE": """1. TÊN NGHIỆP VỤ:
Xuất hóa đơn bán hàng (đối chiếu giao hàng chưa xuất HĐ)

2. BẢNG BÚT TOÁN:
- Nợ TK 131: Phải thu khách hàng
- Có TK 13881: Phải thu tạm (Giao hàng chưa xuất HĐ)
- Có TK 33311: Thuế GTGT đầu ra
- Nợ TK 511: Doanh thu

3. GIẢI THÍCH:
- Nợ TK 131: Ghi nhận công nợ phải thu từ khách hàng (theo HĐ)
- Có TK 13881: Đối chiếu bút toán tạm khi giao hàng
- Có TK 33311: Ghi nhận thuế GTGT trên hóa đơn
- Nợ TK 511: Ghi nhận doanh thu

4. VÍ DỤ:
Xuất hóa đơn cho khách A giá trị hàng hóa 100.000.000đ, thuế GTGT 10.000.000đ (đã giao hàng trước đó).
- Nợ TK 131: 110.000.000đ
- Có TK 13881: 110.000.000đ (đối chiếu)
- Có TK 33311: 10.000.000đ
- Nợ TK 511: 100.000.000đ""",

    "CASH_IN": """1. TÊN NGHIỆP VỤ:
Thu tiền từ khách hàng

2. BẢNG BÚT TOÁN:
- Nợ TK 111: Tiền mặt
- Có TK 131: Phải thu khách hàng

3. GIẢI THÍCH:
- Nợ TK 111: Ghi nhận tiền mặt thu từ khách hàng
- Có TK 131: Giảm công nợ phải thu khách hàng

4. VÍ DỤ:
Khách hàng A thanh toán 50.000.000đ cho công nợ.
- Nợ TK 111: 50.000.000đ
- Có TK 131: -50.000.000đ (giảm công nợ)""",

    "GRN_PURCHASE": """1. TÊN NGHIỆP VỤ:
Nhập kho mua hàng (chưa nhận hóa đơn)

2. BẢNG BÚT TOÁN:
- Nợ TK 152 (*): Nguyên vật liệu
- Có TK 33881: Phải trả tạm (Nhập hàng chưa có HĐ)

3. GIẢI THÍCH:
- Nợ TK 152: Ghi nhận hàng hóa nhập kho
- Có TK 33881: Ghi nhận công nợ tạm phải trả NCC (đợi nhận HĐ)

4. VÍ DỤ:
Nhập 500kg nguyên liệu giá 200.000đ/kg (tổng 100.000.000đ), chưa nhận hóa đơn.
- Nợ TK 152: 100.000.000đ
- Có TK 33881: 100.000.000đ""",

    "PURCHASE_INVOICE": """1. TÊN NGHIỆP VỤ:
Nhận hóa đơn mua hàng (đối chiếu nhập hàng chưa có HĐ)

2. BẢNG BÚT TOÁN:
- Nợ TK 153: Thuế GTGT đầu vào
- Nợ TK 331: Phải trả nhà cung cấp
- Có TK 33881: Phải trả tạm (Nhập hàng chưa có HĐ)

3. GIẢI THÍCH:
- Nợ TK 153: Ghi nhận thuế GTGT trên hóa đơn (đối chiếu/bổ sung)
- Nợ TK 331: Ghi nhận công nợ phải trả NCC theo hóa đơn
- Có TK 33881: Đối chiếu bút toán tạm khi nhập hàng

4. VÍ DỤ:
Nhận hóa đơn NCC giá trị hàng hóa 100.000.000đ, thuế GTGT 10.000.000đ (đã nhập kho trước đó).
- Nợ TK 153: 10.000.000đ (đối chiếu/bổ sung)
- Nợ TK 331: 110.000.000đ
- Có TK 33881: 110.000.000đ (đối chiếu)""",

    "CASH_OUT": """1. TÊN NGHIỆP VỤ:
Chi tiền cho nhà cung cấp

2. BẢNG BÚT TOÁN:
- Nợ TK 331: Phải trả nhà cung cấp
- Có TK 112: Tiền gửi ngân hàng

3. GIẢI THÍCH:
- Nợ TK 331: Giảm công nợ phải trả NCC
- Có TK 112: Giảm tiền gửi ngân hàng

4. VÍ DỤ:
Thanh toán 80.000.000đ cho NCC thanh toán công nợ.
- Nợ TK 331: -80.000.000đ
- Có TK 112: -80.000.000đ""",
}


# =============================================================================
# INSTRUCTIONS ONLY (khi không tìm thấy template)
# =============================================================================

_INSTRUCTIONS_ONLY = """

---
YÊU CẦU: Trả lời ĐẦY ĐỦ 4 phần theo đúng thứ tự:
1. Tên nghiệp vụ
2. Bảng bút toán (liệt kê Nợ/Có với số TK và tên TK)
3. Giải thích từng dòng bút toán
4. Ví dụ với số tiền minh họa
"""


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def get_response_template(tx_type: str) -> str:
    """
    Lấy template với ví dụ cụ thể cho từng nghiệp vụ.

    Args:
        tx_type: Loại nghiệp vụ (DO_SALE, SALES_INVOICE, CASH_IN, GRN_PURCHASE, PURCHASE_INVOICE, CASH_OUT)

    Returns:
        Template string với ví dụ phù hợp, hoặc chỉ instructions nếu không tìm thấy
    """
    if tx_type not in _TEMPLATES:
        print(f"[PostingEngine Template] Warning: Template not found for tx_type='{tx_type}'")
        print(f"[PostingEngine Template] Available tx_types: {list(_TEMPLATES.keys())}")
        print(f"[PostingEngine Template] Returning instructions only (no example)")
        return _INSTRUCTIONS_ONLY

    template = _TEMPLATES[tx_type]

    instructions = """

---
YÊU CẦU: Trả lời ĐẦY ĐỦ 4 phần theo đúng format VÍ DỤ trên:
- Phần 1: Tên nghiệp vụ
- Phần 2: Bảng bút toán (liệt kê Nợ/Có với số TK và tên TK)
- Phần 3: Giải thích từng dòng bút toán
- Phần 4: Ví dụ với số tiền minh họa"""

    return template + instructions
