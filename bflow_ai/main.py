from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints.ask import router as ask_router
from app.api.endpoints.sessions import router as sessions_router
from fastapi.middleware.cors import CORSMiddleware
from app.db.mongodb import close_mongo_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("[Startup] Clearing all caches...")
    try:
        from app.services.streaming_cache import get_streaming_cache
        cache = get_streaming_cache()
        cache.clear()
        print("[Startup] ✓ In-memory cache cleared")
    except Exception as e:
        print(f"[Startup] ✗ In-memory cache error: {e}")

    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.flushdb()
        print("[Startup] ✓ Redis cache cleared")
    except Exception as e:
        print(f"[Startup] ✗ Redis not available or error: {e}")

    print("[Startup] Cache clearing complete")
    print("=" * 60)

    yield

    await close_mongo_connection()


app = FastAPI(
    title="BFLOW AI",
    description="Unified Multi-Module AI Assistant - RESTful API",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router, prefix="/api/ai-bflow")
app.include_router(sessions_router, prefix="/api/ai-bflow")


@app.get("/")
async def root():
    return {
        "message": "BFLOW AI - RESTful API",
        "version": "2.0",
        "authentication": {
            "method": "X-User-Id header",
            "description": "User ID passed via header for all requests"
        },
        "endpoints": {
            "ask": {
                "method": "POST",
                "path": "/api/ai-bflow/ask",
                "header": "X-User-Id (required)",
                "body": {
                    "question": "string (required)",
                    "session_id": "string (optional)",
                    "chat_type": "thinking|free (default: thinking)",
                    "item_group": "GOODS (default)",
                    "partner_group": "CUSTOMER (default)"
                }
            },
            "sessions": {
                "list": {"method": "GET", "path": "/api/ai-bflow/users/{user_id}/sessions"},
                "create": {"method": "POST", "path": "/api/ai-bflow/users/{user_id}/sessions"},
                "detail": {"method": "GET", "path": "/api/ai-bflow/users/{user_id}/sessions/{session_id}"},
                "delete": {"method": "DELETE", "path": "/api/ai-bflow/users/{user_id}/sessions/{session_id}"},
                "clear": {"method": "POST", "path": "/api/ai-bflow/users/{user_id}/sessions/{session_id}/clear"},
                "reload": {"method": "POST", "path": "/api/ai-bflow/users/{user_id}/sessions/{session_id}/reload"}
            }
        }
    }
