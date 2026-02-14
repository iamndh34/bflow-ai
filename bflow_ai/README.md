# BFLOW AI - Trợ Lý AI Đa Chức Năng

Trợ lý AI thông minh với kiến trúc Pipeline-based, hỗ trợ mở rộng nhiều chuyên ngành.

## 🎯 Tổng Quan

**BFLOW AI** là trợ lý AI thông minh với kiến trúc pipeline-based:

- ✅ **Multi-Module Routing**: Tự động phân loại câu hỏi
- ✅ **Hybrid Caching**: Kết hợp semantic search + exact match cache
- ✅ **Redis-Backed Cache**: Persistent cache với in-memory fallback
- ✅ **Multi-Agent System**: Các agents chuyên biệt cho từng domain
- ✅ **Streaming Response**: Character-by-character streaming mượt mà
- ✅ **Session Management**: Quản lý lịch sử với semantic history matching

### Modules Hiện Có

| Module | Mô tả | Agents |
|--------|-------|---------|
| **ACCOUNTING** | Kế toán, tài khoản, hạch toán | COA, POSTING_ENGINE, GENERAL_ACCOUNTING |
| **GENERAL** | Câu hỏi chung, xã giao | GENERAL_FREE |

---

## 🏗️ Kiến Trúc

```
UI Client → GET /api/ai-bflow/ask
                ↓
        Module Router (SLM + Keywords)
                ↓
    Accounting Pipeline (8 Steps):
        1. Session Management
        2. Context Builder
        3. Agent Router
        4. Streaming Cache Check
        5. Agent Execution (LLM)
        6. Stream Processing
        7. Response Saver
                ↓
        Multi-Agent Execution (COA, POSTING_ENGINE, GENERAL)
                ↓
        Ollama LLM (qwen2.5:7b)
                ↓
        Hybrid Cache (Redis + Memory)
                ↓
        Stream Response to User
```

---

## 🔧 Tech Stack

**Core:**
- FastAPI, Python 3.11+, Pydantic

**AI/ML:**
- Ollama (qwen2.5:7b, qwen2.5:0.5b)
- Sentence-Transformers (Vietnamese embeddings)

**Cache/Storage:**
- Redis, In-Memory Fallback, File-based Session History

---

## 📁 Cấu Trúc Project

```
bflow_ai/
├── app/
│   ├── main.py                      # FastAPI entry point
│   ├── api/                          # API endpoints
│   ├── core/                         # Config, embeddings, Ollama client
│   ├── pipeline/                     # Processing pipeline (8 steps)
│   ├── agents/                       # Multi-agent system
│   │   ├── templates/                 # Response templates
│   └── services/                     # Business services
├── .env                             # Environment config
├── main.py                          # App entry point
└── requirements.txt                 # Python dependencies
```

---

## 🌐 API Endpoints

### Unified Endpoint

**`GET /api/ai-bflow/ask`**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `question` | string | ✅ | - |
| `session_id` | string | ❌ | null |
| `chat_type` | string | ❌ | thinking |
| `item_group` | string | ❌ | GOODS |
|`partner_group` | string | ❌ | CUSTOMER |

---

## ⚙️ Cấu Hình

### Environment Variables (.env)

```bash
# Service
OLLAMA_HOST=http://localhost:11434
REDIS_HOST=localhost
REDIS_PORT=6379

# Models
CLASSIFIER_MODEL=qwen2.5:0.5b
GENERATION_MODEL=qwen2.5:7b

# Cache
ENABLE_LLM_CACHE=true
CACHE_TTL=3600
MAX_CACHE_SIZE=100

# Semantic History
ENABLE_SEMANTIC_HISTORY=true
SEMANTIC_SIMILARITY_THRESHOLD=0.85
```

---

## 🚀 Cài Đặt

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# Run server
uvicorn main:app --port 8010
```

---

## 📖 Sử Dụ

```python
import requests

API_BASE = "http://localhost:8010"

response = requests.get(
    f"{API_BASE}/api/ai-bflow/ask",
    params={"question": "TK 111 là gì?"}
)

for line in response.iter_lines():
    print(line.decode('utf-8'), end='', flush=True)
```

---

## 🐛 Troubleshooting

| Vấn đề | Giải pháp |
|--------|----------|
| Redis connection refused | `sudo systemctl status redis` |
| Ollama not responding | `ollama list` |
| Slow responses | Bật cache trong `.env` |
| Out of memory | Giảm `MAX_CACHE_SIZE` |

---

**Xem thêm hướng dẫn phát triển và mở rộng:** [README_DEV.md](README_DEV.md)

---

**Version:** 1.0.0
**Last Updated:** 2026-02-05
