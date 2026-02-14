# BFLOW AI - Hướng Dẫn Phát Triển & Mở Rộng

Tài liệu này dành cho developer muốn:
- Thêm module/agent mới
- Mở rộng Posting Engine với giao dịch mới
- Tùy chỉnh hiệu năng hệ thống
- Hiểu rõ kiến trúc và cách hoạt động

---

## 📋 Mục Lục

- [API Reference](#api-reference)
- [Tổng Quan Hệ Thống](#tổng-quan-hệ-thống)
- [Kiến Trúc & Key Concepts](#kiến-trúc--key-concepts)
- [Thêm Transaction Mới (Posting Engine)](#thêm-transaction-mới-posting-engine)
- [GeneralFreeAgent - Free Chat & Xã Giao](#generalfreeagent---free-chat--xã-giao)
- [Thêm Agent Mới](#thêm-agent-mới)
- [Thêm Module Mới](#thêm-module-mới)
- [Cache Management](#cache-management)
- [Configuration Files](#configuration-files)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Authentication

**Header:** `X-User-Id` (bắt buộc cho tất cả requests)

```bash
curl -H "X-User-Id: user123" ...
```

### Endpoints

#### 1. Ask - Gửi câu hỏi

**POST** `/api/ai-bflow/ask`

| Type | Field | Required | Default |
|------|-------|----------|---------|
| Header | `X-User-Id` | ✅ | - |
| Body | `question` | ✅ | - |
| Body | `session_id` | ❌ | null |
| Body | `chat_type` | ❌ | "thinking" |
| Body | `item_group` | ❌ | "GOODS" |
| Body | `partner_group` | ❌ | "CUSTOMER" |

```bash
curl -X POST "http://localhost:8000/api/ai-bflow/ask" \
  -H "X-User-Id: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "TK 156 là gì?",
    "chat_type": "thinking"
  }'
```

#### 2. Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ai-bflow/users/{user_id}/sessions` | List sessions |
| POST | `/api/ai-bflow/users/{user_id}/sessions` | Create session |
| GET | `/api/ai-bflow/users/{user_id}/sessions/{session_id}` | Get session detail |
| DELETE | `/api/ai-bflow/users/{user_id}/sessions/{session_id}` | Delete session |
| POST | `/api/ai-bflow/users/{user_id}/sessions/{session_id}/clear` | Clear history |
| POST | `/api/ai-bflow/users/{user_id}/sessions/{session_id}/reload` | Reload session |

```bash
# List sessions
curl "http://localhost:8000/api/ai-bflow/users/user123/sessions"

# Create session
curl -X POST "http://localhost:8000/api/ai-bflow/users/user123/sessions" \
  -H "Content-Type: application/json" \
  -d '{"chat_type": "thinking"}'

# Delete session
curl -X DELETE "http://localhost:8000/api/ai-bflow/users/user123/sessions/sess_abc123"
```

#### 3. Root

**GET** `/` - API information

### Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 401 | Missing X-User-Id header |
| 403 | Access denied (wrong user) |
| 404 | Resource not found |

---

## Tổng Quan Hệ Thống

**BFLOW AI** là trợ lý AI thông minh với kiến trúc pipeline-based, hỗ trợ mở rộng nhiều chuyên ngành.

### Modules Hiện Có

| Module | Mô tả | Agents |
|--------|-------|---------|
| **ACCOUNTING** | Kế toán, tài khoản, hạch toán | COA, POSTING_ENGINE, GENERAL_ACCOUNTING |
| **GENERAL** | Câu hỏi chung, xã giao | GENERAL_FREE |

### Tech Stack

**Core:**
- FastAPI, Python 3.11+, Pydantic

**AI/ML:**
- Ollama (qwen2.5:3b cho generation, qwen2.5:0.5b cho classification)
- Sentence-Transformers (Vietnamese embeddings)

**Cache/Storage:**
- Redis, In-Memory Fallback, MongoDB

---

## Kiến Trúc & Key Concepts

### Pipeline Architecture

```
1. Session Management      → Quản lý session/history
2. Context Builder         → Xây dựng context từ request
3. Agent Router            → Phân loại query đến agent phù hợp
4. Streaming Cache Check  → Kiểm tra cache, regenerate example nếu hit
5. Agent Execution         → Thực thi query với LLM
6. Stream Processing       → Xử lý streaming response
7. Response Saver          → Lưu response vào cache/session
```

### Key Concepts

**1. Multi-Agent System**
- Mỗi agent chuyên biệt cho một domain
- Agent có `can_handle()` để xác định confidence score
- Orchestrator chọn agent có confidence cao nhất

**2. LLM-based Classification**
- Dùng qwen2.5:0.5b (nhẹ, nhanh) để phân loại
- Không phụ thuộc format cứng nhắc
- Hiểu ngữ nghĩa, chịu được biến thể

**3. Smart Cache with Example Regeneration**
- Cache chỉ lưu phần 1-3 (không có VÍ DỤ)
- Khi cache hit: regenerate phần 4 với số ngẫu nhiên
- Preserve footer (Ghi chú, Lưu ý) từ response gốc

**4. Data-Driven Configuration**
- Posting Engine: transaction types, posting rules, GL mapping
- Không cần sửa code khi thêm giao dịch mới
- Templates cho từng transaction type

---

## Thêm Transaction Mới (Posting Engine)

### Cách Tiếp Cận Data-Driven

Hệ thống Posting Engine được thiết kế **data-driven** - không cần sửa code logic khi thêm giao dịch mới.

### Bước 1: Cập Nhật Posting Engine Config

**File:** `app/services/rag_json/posting_engine.json`

```json
{
  "document_types": [
    {
      "transaction_key": "TEN_GIAODICH",
      "description": "Mô tả giao dịch - Tên nghiệp vụ",
      "keywords": ["từ_khoa1", "từ_khoa2", "từ_khoa3"]
    }
  ],
  "posting_rules": [
    {
      "je_doc_type": "TEN_GIAODICH",
      "rules": [
        {
          "role_key": "ten_role_key",
          "priority": 1,
          "side": "DEBIT",
          "account_source_type": "FIXED",
          "fixed_account_code": "111"
        }
      ]
    }
  ],
  "gl_mapping": {
    "NHOM_HANG": {
      "ten_role_key": "tk_ghi_nghia"
    }
  }
}
```

**Các `side` khả dụng:**
- `DEBIT`: Nợ
- `CREDIT`: Có

**Các `account_source_type`:**
- `FIXED`: Account cố định
- `LOOKUP`: Account lookup từ `gl_mapping`

### Bước 2: Thêm Template (Khuyến Nghị)

**File:** `app/agents/templates/posting_engine.py`

```python
_TEMPLATES = {
    "TEN_GIAODICH": """1. TÊN NGHIỆP VỤ:
Tên giao dịch đầy đủ

2. BẢNG BÚT TOÁN:
- Nợ TK XXX: Tên tài khoản
- Có TK YYY: Tên tài khoản

3. GIẢI THÍCH:
- Nợ TK XXX: Giải thích
- Có TK YYY: Giải thích

4. VÍ DỤ:
Mô tả ngữ cảnh cụ thể""",
}
```

### Bước 3: Cập Nhật Example Generation (Bắt Buộc)

**File:** `app/pipeline/ask.py`

**3a. Thêm tx_type vào LLM classification prompt:**

```python
prompt = f"""Phân loại loại giao dịch kế toán sau...

CÁC LOẠI GIAO DỊCH:
- DO_SALE: Xuất kho bán hàng
- SALES_INVOICE: Xuất hóa đơn bán hàng
- CASH_IN: Thu tiền từ khách hàng
- GRN_PURCHASE: Nhập kho mua hàng
- PURCHASE_INVOICE: Nhận hóa đơn mua hàng
- CASH_OUT: Chi tiền cho nhà cung cấp
- TEN_GIAODICH: Mô tả ngắn gọn
...
"""
```

**3b. Thêm description template:**

```python
DESC_TEMPLATES = {
    'TEN_GIAODICH': "Mô tả ngữ cảnh với {amount:,}đ, thuế GTGT {tax:,}đ.",
}
```

---

## GeneralFreeAgent - Free Chat & Xã Giao

### Tổng Quan

`GeneralFreeAgent` xử lý các câu hỏi không liên quan kế toán: chat xã giao, hỏi thăm, cảm ơn, v.v.

**Đặc điểm:**
- Dùng LLM classification để phát hiện general chat
- Temperature cao hơn (0.6) để response tự nhiên
- Fallback cuối cùng khi không có agent nào match

### Temperature Settings

**Accounting agents** (chính xác):
```python
OLLAMA_OPTIONS = {"temperature": 0.3}
```

**GeneralFreeAgent** (tự nhiên):
```python
GENERAL_FREE_OPTIONS = {"temperature": 0.6}
```

---

## Thêm Agent Mới

### Bước 1: Tạo Agent Class

**File:** `app/agents/your_agent.py`

```python
from .base import BaseAgent, AgentRole, AgentResult, AgentContext

class YourAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "YOUR_AGENT"

    @property
    def role(self) -> AgentRole:
        return AgentRole.DOMAIN_SPECIALIST

    @property
    def description(self) -> str:
        return "Chuyên gia về..."

    def can_handle(self, context: AgentContext) -> tuple[bool, float]:
        question = context.question.lower()
        keywords = ["từ_khoa1", "từ_khoa2"]
        matches = sum(1 for kw in keywords if kw in question)

        if matches >= 2:
            return True, 0.95
        elif matches == 1:
            return True, 0.80
        else:
            return False, 0.0

    def execute(self, context: AgentContext) -> AgentResult:
        # Implementation
        pass

    def stream_execute(self, context: AgentContext):
        # Implementation
        yield response
```

### Bước 2: Đăng ký Agent

**File:** `app/agents/orchestrator.py`

```python
self._register_agent(YourAgent())
```

---

## Thêm Module Mới

### Bước 1: Tạo Pipeline Class

**File:** `app/pipeline/your_module_pipeline.py`

```python
class YourModulePipeline:
    def __init__(self):
        self.session_step = SessionManagerStep()
        self.context_builder = ContextBuilderStep()
        self.router_step = YourRouterStep()
        self.cache_checker = StreamingCacheStep()
        self.executor = AgentExecutorStep()
        self.stream_processor = StreamProcessorStep()
        self.saver = ResponseSaverStep()

    def process(self, question, user_id=None, session_id=None, **kwargs):
        # Implementation
        pass
```

### Bước 2: Đăng ký Module

**File:** `app/pipeline/router.py` → `AVAILABLE_MODULES`

```python
AVAILABLE_MODULES = {
    "YOUR_MODULE": {
        "name": "Tên Module",
        "description": "Mô tả module",
        "keywords": ["từ_khoa1"],
        "pipeline_class": YourModulePipeline
    }
}
```

---

## Cache Management

### Cache Keys

Cache key được generate từ:
- `question`
- `agent_name`
- `item_group`
- `partner_group`
- `chat_type`

### Xóa Cache Khi Development

```python
from app.services.streaming_cache import get_streaming_cache

cache = get_streaming_cache()
cache.clear()
```

---

## Configuration Files

### posting_engine.json

**Location:** `app/services/rag_json/posting_engine.json`

**Cấu trúc:**
- `document_types`: Danh sách các loại giao dịch
- `posting_rules`: Rules hạch toán
- `gl_mapping`: Mapping nhóm hàng/đối tác → account

### coa_99.json & coa_200.json

**Location:** `app/services/rag_json/`

**Dùng cho:** COA Agent tra cứu thông tin tài khoản

---

## Troubleshooting

### Agent Không Được Gọi

**Kiểm tra:**
1. Agent đã đăng ký trong orchestrator?
2. `can_handle()` trả về `True`?
3. Keywords có đủ không?

### Bút Toán Sai

**Kiểm tra:**
1. `posting_engine.json` có đúng không?
2. `priority` có đúng thứ tự không?
3. `side` (Nợ/Có) có đúng không?

### Cache Không Hoạt Động

**Kiểm tra:**
1. `item_group`, `partner_group`, `chat_type` có khớp không?
2. Cache key generation có nhất quán không?

---

## Best Practices

1. **Data-Driven:** Luôn ưu tiên config/data thay vì hard-code
2. **Template First:** Tạo template trước khi implement logic
3. **Test Thorough:** Test với các edge cases
4. **Document:** Cập nhật tài liệu khi thêm feature mới
5. **Monitor Logs:** Theo dõi log để debug

---

**Last Updated:** 2026-02-06
**Version:** 3.0.0

## Changelog

### v3.0.0 (2026-02-06)
- **RESTful API redesign**
  - POST /api/ai-bflow/ask với JSON body
  - User authentication via X-User-Id header
  - Session endpoints với user_id trong path
  - Xóa các development flags (turn_off_*)
- **User-based session isolation**
  - Mỗi user chỉ thấy sessions của mình
  - Access control 403 khi truy cập session của người khác
- **Standardized API patterns**
  - GET: Query parameters
  - POST: JSON body
  - DELETE: Query parameters

### v2.2.0 (2026-02-05)
- Thêm dấu `(*)` đánh dấu LOOKUP accounts
- Tự động thêm "Lưu ý" khi có tài khoản 13881 hoặc 33881

### v2.1.0 (2026-02-05)
- LLM-based tx_type classification
- Smart example regeneration
- Footer preservation (Ghi chú, Lưu ý)
