# BFlow AI V2 - COA Agent

Version 2 đơn giản hóa: **Chỉ COA Agent - Tra cứu tài khoản kế toán**

## 🎯 Tính năng

- **Tra cứu tài khoản**: TK 156 là gì?
- **Tra cứu theo loại**: Tài sản ngắn hạn có những TK nào?
- **So sánh TT99 vs TT200**: TK 111 trong TT99 khác TT200 thế nào?
- **Tra cứu theo từ khóa**: Tài khoản về hàng hóa
- **Ollama Local LLM**: Không cần API key
- **Streaming Response**: Real-time streaming

## 📋 Yêu cầu

- Python 3.10+
- Ollama đang chạy
- Model: `gemma3:4b`

## 🚀 Cài đặt

```bash
cd bflow_ai_v2

# Install dependencies
pip install -r requirements.txt

# Pull model
ollama pull gemma3:4b

# Run server
python -m app.main
```

## 📖 API Endpoints

### COA Query

```
GET /api/coa/ask
```

Query Parameters:
- `question`: Câu hỏi (required)

Examples:

```bash
# Tra cứu tài khoản
curl "http://localhost:8010/api/coa/ask?question=TK+156+là+gì?"

# So sánh
curl "http://localhost:8010/api/coa/ask?question=So+sánh+TK+111+giữa+TT99+và+TT200"

# Tra cứu theo từ khóa
curl "http://localhost:8010/api/coa/ask?question=Tài+khoản+về+hàng+hóa"
```

### Health Check

```
GET /api/coa/health
```

Docs: `http://localhost:8010/api/docs`

## 📁 Cấu trúc

```
bflow_ai_v2/
├── app/
│   ├── agents/
│   │   ├── base.py          # Base agent
│   │   └── coa_agent.py     # COA specialist
│   ├── api/
│   │   └── endpoints.py      # COA API endpoints
│   ├── core/
│   │   ├── config.py        # Configuration
│   │   └── ollama_client.py # Ollama client
│   ├── services/
│   │   └── coa_index.py      # COA data indexing
│   └── main.py              # Application
├── data/
│   └── coa/
│       ├── coa_99.json
│       ├── coa_200.json
│       └── coa_compare_99_vs_200.json
└── requirements.txt
```

## 🧪 Testing

```bash
# Test health
curl http://localhost:8010/api/coa/health

# Test query
curl "http://localhost:8010/api/coa/ask?question=TK+111+là+gì?"
```

## ⚙️ Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:
```
HOST=0.0.0.0
PORT=8010
OLLAMA_BASE_URL=http://localhost:11434
GENERATION_MODEL=gemma3:4b
```
