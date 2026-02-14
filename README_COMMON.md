# BFlow AI - Common Documentation

Tài liệu chung cho cả **bflow_ai** (v1) và **bflow_ai_v2** (v2).

## 📋 Tổng quan

### bflow_ai (Version 1)
- **Architecture**: Pipeline-based với 8 steps
- **RAG**: Vector search (embeddings) + Hybrid similarity
- **LLM**: Ollama (qwen2.5)
- **Database**: MongoDB + Redis (cache)
- **Agents**: Multi-agent system (COA, Posting Engine, General)

### bflow_ai_v2 (Version 2)
- **Architecture**: FastAPI + LangChain + GraphRAG
- **RAG**: Knowledge graph-based (GraphRAG) + Local/Global search
- **LLM**: Ollama (qwen2.5) + LangChain integration
- **Graph**: NetworkX + Community detection
- **Vector Store**: ChromaDB (fallback)

## 🔄 Cách Switch Qua Lại Backend

### Cách 1: Dùng cùng port 8010 (Khuyến nghị)

```bash
# Stop backend hiện tại (Ctrl+C)

# Chạy bflow_ai (v1)
cd bflow_ai
uvicorn main:app --port 8010

# HOẶC chạy bflow_ai_v2 (v2)
cd bflow_ai_v2
uvicorn app.main:app --port 8010
```

### Cách 2: Chạy song song 2 port

```bash
# Terminal 1: bflow_ai v1
cd bflow_ai
uvicorn main:app --port 8010

# Terminal 2: bflow_ai_v2 v2
cd bflow_ai_v2
uvicorn app.main:app --port 8011
```

Sau đó đổi port trong UI config.

## 📊 So sánh Tính năng

| Tính năng | bflow_ai (v1) | bflow_ai_v2 (v2) |
|-----------|---------------|-------------------|
| **RAG Method** | Vector Search (embeddings) | GraphRAG (Knowledge Graph) |
| **Global Questions** | ❌ Yếu | ✅ Mạnh |
| **Local Questions** | ✅ Tốt | ✅ Tốt |
| **Query Type** | Similarity search | Entity-based + Community-based |
| **Framework** | Custom pipeline | LangChain |
| **Caching** | Redis + Streaming cache | Redis + Vector cache |
| **Data Format** | JSON files | Text (từ JSON convert) |
| **Indexing** | On-the-fly | Pre-built graph |

## 🎯 Khi nào dùng Version nào?

### Dùng bflow_ai (v1) khi:
- Câu hỏi cụ thể: "TK 111 là gì?", "Hạch toán bán hàng thế nào?"
- Cần response nhanh
- Câu hỏi về một account cụ thể

### Dùng bflow_ai_v2 (v2) khi:
- Câu hỏi tổng quan: "So sánh TT99 và TT200?", "Tổng quan các chuẩn mực?"
- Cần context toàn diện
- Câu hỏi liên quan đến nhiều chủ đề
- Muốn tận dụng knowledge graph

## 🔌 API Endpoints (Compatible)

Cả 2 backend đều hỗ trợ endpoint giống nhau để UI có thể switch:

```
GET /api/ai-bflow/ask
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | Câu hỏi |
| `session_id` | string | null | Session ID |
| `chat_type` | string | "thinking" | "thinking" hoặc "free" |
| `item_group` | string | "GOODS" | Nhóm sản phẩm |
| `partner_group` | string | "CUSTOMER" | Nhóm đối tác |

**Response:** `text/plain; charset=utf-8` (streaming)

## 📁 Cấu trúc Project

```
bflow-ai/
├── bflow_ai/              # Version 1 (Pipeline-based)
│   ├── app/
│   │   ├── agents/        # Multi-agent system
│   │   ├── api/           # Endpoints
│   │   ├── core/          # Config, LLM, embeddings
│   │   ├── pipeline/      # 8-step pipeline
│   │   └── services/      # Services (cache, search, etc)
│   ├── main.py
│   └── requirements.txt
│
├── bflow_ai_v2/           # Version 2 (GraphRAG + LangChain)
│   ├── app/
│   │   ├── agents/
│   │   ├── api/           # Endpoints (compatible + v2)
│   │   ├── core/          # Config, LangChain LLM
│   │   ├── models/        # Pydantic schemas
│   │   └── services/      # GraphRAG, Vector store
│   ├── scripts/           # Convert data, build graph
│   ├── ragtest/           # GraphRAG output
│   ├── main.py
│   └── requirements.txt
│
└── README_COMMON.md       # File này
```

## 🚀 Quick Start

### 1. Cài đặt Ollama và Pull Models

```bash
# Cài đặt Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull qwen2.5:7b
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text  # Cho v2
```

### 2. Chạy bflow_ai (v1)

```bash
cd bflow_ai

# Install dependencies (lần đầu)
pip install -r requirements.txt

# Chạy server
uvicorn main:app --port 8010
```

### 3. Chạy bflow_ai_v2 (v2)

```bash
cd bflow_ai_v2

# Install dependencies (lần đầu)
pip install -r requirements.txt

# Convert data từ bflow_ai
python scripts/convert_data.py

# Build knowledge graph (lần đầu)
python scripts/build_graph.py

# Chạy server
uvicorn app.main:app --port 8010
```

## 🧪 Testing

### Test bằng curl

```bash
# Test endpoint chính
curl "http://localhost:8010/api/ai-bflow/ask?question=TK+111+là+gì?"

# Test với câu hỏi global (chỉ v2 trả lời tốt)
curl "http://localhost:8010/api/ai-bflow/ask?question=So+sánh+TT99+và+TT200?"
```

### Test câu hỏi ví dụ

| Loại câu hỏi | Ví dụ | Version khuyến nghị |
|-------------|-------|-------------------|
| **Specific** | "TK 111 là gì?" | v1 hoặc v2 |
| **Specific** | "Hạch toán bán hàng?" | v1 hoặc v2 |
| **Global** | "So sánh TT99 và TT200?" | v2 |
| **Global** | "Tổng quan chuẩn mực kế toán?" | v2 |

## 📊 Monitoring

### bflow_ai (v1)

```bash
# Logs hiển thị pipeline steps:
# [Pipeline] STEP 1: Session Management
# [Pipeline] STEP 2: Building Context
# [Pipeline] STEP 3: Routing to Agent
# ...
```

### bflow_ai_v2 (v2)

```bash
# Health check
curl http://localhost:8010/api/health

# Graph status
curl http://localhost:8010/api/graph/status

# Swagger UI
# Mở trình duyệt: http://localhost:8010/docs
```

## ⚙️ Configuration

### bflow_ai (v1) - `app/core/config.py`

```python
# Ollama
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"

# Cache
ENABLE_LLM_CACHE = True
CACHE_TTL = 3600
```

### bflow_ai_v2 (v2) - `app/core/config.py`

```python
# Ollama
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
GRAPH_RAG_EMBEDDING_MODEL = "nomic-embed-text"

# GraphRAG
GRAPH_RAG_ENABLED = True
GRAPH_RAG_ROOT = "./ragtest"
```

## 🔧 Troubleshooting

### Ollama không chạy

```bash
# Kiểm tra Ollama
ollama list

# Start Ollama
ollama serve
```

### Port 8010 đã được sử dụng

```bash
# Tìm process đang dùng port
lsof -i :8010

# Kill process
kill -9 <PID>
```

### v2: Graph chưa được build

```bash
cd bflow_ai_v2
python scripts/build_graph.py
```

### v2: Không có input files

```bash
cd bflow_ai_v2
python scripts/convert_data.py
```

## 📝 Notes

1. **UI không cần thay đổi** - Cả 2 backend đều dùng endpoint `/api/ai-bflow/ask`
2. **Session ID** - Được giữ nguyên format `__SESSION_ID__:{id}\n`
3. **Streaming** - Cả 2 đều trả về streaming `text/plain`
4. **CORS** - Cả 2 đều cho phép tất cả origins

## 🚧 Development Roadmap

### bflow_ai (v1)
- ✅ Multi-agent system
- ✅ Hybrid semantic search
- ✅ Streaming cache
- ✅ Pipeline architecture

### bflow_ai_v2 (v2)
- ✅ GraphRAG integration
- ✅ LangChain orchestration
- ✅ Local & Global search
- ⏳ Agent integration with LangChain Tools
- ⏳ Advanced graph visualization

## 📧 Support

For issues or questions:
1. Check logs in terminal
2. Check `/api/health` endpoint
3. Review troubleshooting section above
