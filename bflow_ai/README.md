# BFLOW AI - Multi-Module AI Assistant

Unified Multi-Module AI Assistant với Pipeline Architecture, hỗ trợ mở rộng nhiều chuyên domains (Accounting, HR, CRM, Sales, etc.)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Request Flow](#request-flow)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Optimizations](#optimizations)

---

## 🎯 Overview

**BFLOW AI** là một trợ lý AI thông minh với kiến trúc pipeline-based, hỗ trợ:

- ✅ **Multi-Module Routing**: Tự động phân loại câu hỏi đến module phù hợp
- ✅ **Hybrid Semantic Caching**: Kết hợp similarity search + exact match cache
- ✅ **Redis-Backed Cache**: Persistent cache với in-memory fallback
- ✅ **Multi-Agent System**: Mỗi module có các agents chuyên biệt
- ✅ **Streaming Response**: Character-by-character streaming cho natural feel
- ✅ **Session Management**: File-based session với semantic history matching

### Current Modules

| Module | Description | Agents |
|--------|-------------|---------|
| **ACCOUNTING** | Kế toán, tài khoản, hạch toán | COA, POSTING_ENGINE, GENERAL_ACCOUNTING |
| **GENERAL** | Câu hỏi chung, xã giao | GENERAL_FREE |

---

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     UI / Client                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Unified Entry Point                            │
│         GET /api/ai-bflow/ask                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Module Router (SLM + Keywords)                 │
│   Phân loại: ACCOUNTING / GENERAL / HR / CRM / ...         │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │Accounting│  │   HR    │   │   CRM   │  ...
        │ Pipeline │  │ Pipeline │  │ Pipeline │
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
             ▼             ▼             ▼
    ┌────────────────────────────────────────┐
    │       Processing Pipeline (8 Steps)   │
    └────────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────┐
    │         Multi-Agent Execution          │
    │  COA / POSTING_ENGINE / GENERAL       │
    └────────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────┐
    │       Ollama LLM (qwen2.5:3b)         │
    └────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Cache System                             │
│   ┌────────────────┐    ┌────────────────┐                 │
│   │ Semantic       │    │  Streaming     │                 │
│   │ History Cache  │    │  Cache (Exact) │                 │
│   │ (Similarity)   │    │  (Redis+Memory)│                 │
│   └────────────────┘    └────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### Processing Pipeline (8 Steps)

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Session Management                                  │
│ - Tạo session mới nếu chưa có                                │
│ - Get session history                                       │
│ - Format messages cho LLM                                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Context Builder                                     │
│ - Parse request parameters                                  │
│ - Build AgentContext object                                 │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Agent Router                                        │
│ - Fast rule-based routing (O(1))                            │
│ - SLM classification với few-shot learning                  │
│ - Semantic fallback với embeddings                          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Semantic History Check                              │
│ - Search trong session history bằng hybrid similarity       │
│   * Sentence similarity (70%)                               │
│   * Keyword similarity (30%)                                │
│ - Threshold: 0.85                                           │
│ - Return cached response nếu match                          │
└──────────────────────────────────────────────────────────────┘
                              │ (miss)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: Streaming Cache Check                               │
│ - Exact match cache với MD5 hash key                        │
│ - Redis-backed với in-memory fallback                       │
│ - TTL: 3600s (1 hour)                                       │
│ - Return cached response nếu match                          │
└──────────────────────────────────────────────────────────────┘
                              │ (miss)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: Agent Execution (LLM Call)                          │
│ - Extract keywords từ câu hỏi                              │
│ - Search data (COA, Posting Engine)                         │
│ - Build context từ data                                     │
│ - Build prompt cho LLM                                      │
│ - Call Ollama streaming                                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 7: Stream Processing                                   │
│ - Buffer streaming chunks                                   │
│ - Optimize buffer size (5 words/buffer)                     │
│ - Yield mượt mà cho frontend                                │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 8: Response Saver                                      │
│ - Accumulate full response                                  │
│ - Save to Streaming Cache (Redis)                           │
│ - Save to Session History (file-based)                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

### Core Framework
- **FastAPI** - Async web framework với auto OpenAPI docs
- **Python 3.11+** - Primary language
- **Pydantic** - Data validation và settings management

### AI/ML
- **Ollama** - Local LLM serving
  - `qwen2.5:3b` - Generation model (quantized q4_0)
  - `qwen2.5:0.5b` - Classification model (SLM)
- **Sentence-Transformers** - Embeddings cho semantic search
  - `dangvantuan/vietnamese-embedding` - Vietnamese embeddings

### Cache & Storage
- **Redis** - Persistent cache backend
  - Streaming cache
  - LLM response cache
  - Session storage (optional)
- **In-Memory Fallback** - Python dict với LRU eviction
- **File-based Storage** - Session history (JSON files)

### Data Processing
- **NumPy** - Vectorized similarity computation
- **Hashlib** - MD5 cache key generation

---

## 📁 Project Structure

```
bflow_ai/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry point
│   │
│   ├── api/                          # API Endpoints
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       └── ask.py               # Unified endpoint
│   │
│   ├── core/                         # Core Services
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration management
│   │   ├── embeddings.py            # Embedding model with cache
│   │   ├── ollama_client.py         # LLM client pool
│   │   └── redis_client.py          # Cache client wrapper
│   │
│   ├── pipeline/                     # Pipeline Architecture
│   │   ├── __init__.py
│   │   ├── ask.py                   # Processing Pipeline (8 steps)
│   │   └── router.py                # Module Router
│   │
│   ├── agents/                       # Multi-Agent System
│   │   ├── __init__.py
│   │   ├── base.py                  # Base agent classes
│   │   ├── orchestrator.py          # Agent orchestration
│   │   └── [module]_agents.py       # Domain-specific agents
│   │
│   ├── services/                     # Business Services
│   │   ├── [module]_index.py        # Data lookup indexes
│   │   ├── similarity.py            # Similarity computation
│   │   ├── history_search.py        # History search
│   │   ├── session_manager.py       # Session management
│   │   ├── [cache]_service.py       # Cache services
│   │   └── data/                    # RAG Data files
│   │
│   └── [other]/                      # Other modules
│
├── .env                             # Environment configuration (private)
├── main.py                          # Application entry point
├── docker-compose.yaml              # Docker services (optional)
└── requirements.txt                 # Python dependencies
```

---

## 🔄 Request Flow

### Complete Flow Example

```
User: "TK 156 là gì?"
   │
   ▼
UI: GET /api/ai-bflow/ask?question=TK+156+là+gì?
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Module Router Classify                                   │
│    - Keywords: "TK", "156" → ACCOUNTING module             │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Accounting Pipeline → Session Manager                    │
│    - Create/Get session: sess_abc123                        │
│    - Get history (last 10 messages)                         │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Context Builder                                          │
│    - Build AgentContext with question, history, etc.       │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Agent Router                                             │
│    - Fast rule: Has "156" → COA agent                      │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Semantic History Check                                  │
│    - Search history with similarity                         │
│    - Not found → Continue                                  │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Streaming Cache Check                                   │
│    - MD5 hash lookup in Redis                               │
│    - Not found → Continue                                  │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. COA Agent Execution                                     │
│    - Extract keywords: ["156"]                             │
│    - COA Index lookup: "156" → "Hàng hóa"                   │
│    - Build prompt with COA context                         │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Call Ollama LLM                                          │
│    - Model: qwen2.5:3b-q4_0                                │
│    - Stream response character-by-character                │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Stream Processing                                        │
│    - Buffer chunks (5 words)                                │
│    - Yield to frontend                                     │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. Save Response                                           │
│     - Save to Redis Streaming Cache (key: md5)             │
│     - Save to Session History (JSON file)                  │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
UI: Stream response to user
```

---

## 🌐 API Endpoints

### Unified Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/ai-bflow/ask` | **Unified entry point** - Auto route to appropriate module |

#### Request Parameters

```
GET /api/ai-bflow/ask
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | ✅ | - | Câu hỏi |
| `session_id` | string | ❌ | null | Session ID (auto-create if null) |
| `chat_type` | string | ❌ | thinking | Chat mode: `thinking` or `free` |
| `item_group` | string | ❌ | GOODS | Item group (for posting engine) |
| `partner_group` | string | ❌ | CUSTOMER | Partner group (for posting engine) |
| `turn_off_routing` | bool | ❌ | false | Dev: Skip routing |
| `turn_off_history` | bool | ❌ | false | Dev: Skip history check |
| `turn_off_cache` | bool | ❌ | false | Dev: Skip cache check |
| `turn_off_llm` | bool | ❌ | false | Dev: Mock LLM response |

#### Response Format

```
__SESSION_ID__:sess_abc123
TK 156 là tài khoản Hàng hóa...
[streaming character by character]
```

### Info Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information & available endpoints |

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# =============================================================================
# Service Configuration
# =============================================================================
OLLAMA_HOST=<ollama_host>
REDIS_HOST=<redis_host>
REDIS_PORT=<redis_port>
REDIS_DB=<redis_db>
REDIS_PASSWORD=<redis_password>  # Optional

# =============================================================================
# Model Configuration
# =============================================================================
CLASSIFIER_MODEL=<model_name>
GENERATION_MODEL=<model_name>

# =============================================================================
# Cache Configuration
# =============================================================================
ENABLE_LLM_CACHE=true
CACHE_TTL=3600
MAX_CACHE_SIZE=100
CACHE_SIMULATE_DELAY=0.02
CACHE_CHARS_PER_CHUNK=1

# =============================================================================
# Semantic History Configuration
# =============================================================================
ENABLE_SEMANTIC_HISTORY=true
SEMANTIC_MODE=hybrid
SEMANTIC_ALPHA=0.7
SEMANTIC_SIMILARITY_THRESHOLD=0.85
```

> **Note**: Xem `.env.example` để có template đầy đủ.

### Key Settings Explained

| Setting | Description | Recommended Values |
|---------|-------------|-------------------|
| `SEMANTIC_MODE` | Similarity algorithm | `hybrid` (best), `sentence`, `keyword` |
| `SEMANTIC_ALPHA` | Sentence vs keyword weight | `0.7` (70% sentence), `0.5` (balanced), `0.3` (keyword-focused) |
| `SEMANTIC_SIMILARITY_THRESHOLD` | Match threshold | `0.90` (strict), `0.85` (default), `0.80` (loose) |
| `CACHE_SIMULATE_DELAY` | Cached response typing speed | `0.02` (smooth), `0.01` (fast), `0.03` (slow) |
| `CACHE_CHARS_PER_CHUNK` | Characters per chunk from cache | `1` (char-by-char), `3-5` (phrase-by-phrase) |

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Ollama (with qwen2.5 models)
- Redis (optional, recommended for production)
- 8GB RAM minimum (16GB recommended)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd bflow_ai
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (xem .env để biết model cụ thể)
ollama pull <model_name>
```

### Step 4: Install Redis (Optional but Recommended)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Or use Docker
docker run -d -p <redis_port>:6379 redis:alpine
```

### Step 5: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env  # hoặc vi .env
```

### Step 6: Run Application

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 7: Verify Installation

```bash
# Check API
curl <your_api_base_url>/

# Check OpenAPI docs
open <your_api_base_url>/docs

# Test query
curl "<your_api_base_url>/api/ai-bflow/ask?question=Your+question"
```

---

## 📖 Usage

### Basic Usage

```python
import requests

# Simple question
API_BASE = "<your_api_base_url>"  # e.g., "https://api.yourdomain.com"
response = requests.get(
    f"{API_BASE}/api/ai-bflow/ask",
    params={"question": "Your question here"}
)

# Stream response
for line in response.iter_lines():
    print(line.decode('utf-8'), end='', flush=True)
```

### With Session

```python
import requests

API_BASE = "<your_api_base_url>"

# First question - creates session
response1 = requests.get(
    f"{API_BASE}/api/ai-bflow/ask",
    params={"question": "First question"}
)
session_id = response1.text.split('\n')[0].split(':')[1]

# Follow-up question - uses session
response2 = requests.get(
    f"{API_BASE}/api/ai-bflow/ask",
    params={
        "question": "Follow-up question",
        "session_id": session_id
    }
)
```

### Free Mode (No routing)

```python
response = requests.get(
    "<your_api_base_url>/api/ai-bflow/ask",
    params={
        "question": "Hello, how are you?",
        "chat_type": "free"
    }
)
```

---

## 🛠️ Development

### Adding New Module

1. **Create Pipeline Class**

```python
# app/pipeline/hr_pipeline.py
class HRPipeline:
    def __init__(self):
        self.session_step = SessionManagerStep()
        # ... other steps

    def process(self, question, session_id, **kwargs):
        # Implementation
        pass
```

2. **Register in Module Router**

```python
# app/pipeline/router.py
AVAILABLE_MODULES = {
    "ACCOUNTING": {...},
    "HR": {
        "name": "Nhân sự",
        "description": "Câu hỏi về lương, tuyển dụng...",
        "keywords": ["lương", "tuyển dụng", "nhân viên"],
        "pipeline_class": HRPipeline
    }
}
```

### Adding New Agent

```python
# app/agents/new_agent.py
class NewAgent(BaseAgent):
    name = "NEW_AGENT"
    description = "Agent description"

    def stream_execute(self, context: AgentContext):
        # Implementation
        yield response
```

### Cache Management

```python
from app.services.streaming_cache import clear_streaming_cache
from app.services.llm_service import get_llm_service

# Clear streaming cache
clear_streaming_cache()

# Clear LLM cache
llm_service = get_llm_service()
llm_service.clear_cache()

# Get cache statistics
stats = llm_service.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

---

## ⚡ Optimizations

### Performance Optimizations Applied

1. **Connection Pooling** - Singleton Ollama client (~50ms saved per request)
2. **Single Embedding Model** - LRU cache for embeddings (~2s startup saved)
3. **COA Indexing** - O(1) lookup instead of O(n) search
4. **Optimized Streaming** - 5-word buffer reduces yield calls by ~90%
5. **Redis Cache** - Persistent cache with ~1ms latency
6. **Hybrid Similarity** - Vectorized NumPy operations
7. **Batch Operations** - Encode multiple texts at once

### Cache Hierarchy

```
Request → Semantic History (50-150ms)
         → Streaming Cache (1ms)
         → LLM Cache (1ms)
         → LLM Call (500-2000ms)
```

### Performance Tips

1. **Enable Redis** for production - 10-100x faster than file I/O
2. **Use appropriate thresholds** - Higher threshold = fewer false positives
3. **Tune buffer sizes** - 5 words is optimal for Vietnamese
4. **Monitor cache hit rates** - Target >70% hit rate

---

## 📊 Monitoring

### Cache Statistics

```python
from app.services.llm_service import get_llm_service

stats = get_llm_service().get_stats()
# {
#     "cache_backend": "Redis",
#     "total_requests": 1000,
#     "cache_hits": 750,
#     "cache_misses": 250,
#     "hit_rate": 0.75
# }
```

### Redis Statistics

```python
from app.core.redis_client import RedisClient

stats = RedisClient.get_stats()
# {
#     "available": true,
#     "connected_clients": 2,
#     "used_memory_human": "45.2M",
#     "total_keys": 1523
# }
```

---

## 🔒 Security Considerations

1. **API Rate Limiting** - Implement rate limiting for production
2. **Input Validation** - Sanitize user inputs
3. **Redis Authentication** - Use `REDIS_PASSWORD` in production
4. **CORS Configuration** - Restrict `allow_origins` in production
5. **Session Management** - Implement session expiration

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Redis connection refused | Check if Redis is running: `sudo systemctl status redis` |
| Ollama not responding | Check Ollama: `ollama list` |
| Import errors | Run: `pip install -r requirements.txt` |
| Slow responses | Check if cache is enabled in `.env` |
| Out of memory | Reduce `MAX_CACHE_SIZE` or use Redis |

---

## 📝 License

[Your License Here]

---

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Contact

[Your Contact Information]

---

**Last Updated**: 2026-02-04
**Version**: 1.0.0
