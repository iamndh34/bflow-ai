from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints.ask import router as unified_router
from fastapi.middleware.cors import CORSMiddleware
from app.db.mongodb import close_mongo_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown - đóng MongoDB connection
    await close_mongo_connection()


app = FastAPI(
    title="BFLOW AI",
    description="Unified Multi-Module AI Assistant - Pipeline Architecture",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include unified endpoint
app.include_router(unified_router, prefix="/api/ai-bflow")


@app.get("/")
async def root():
    return {
        "message": "BFLOW AI - Unified Multi-Module Architecture",
        "endpoints": {
            "unified": "/api/ai-bflow/ask",
            "description": "Single endpoint for all modules (Accounting, HR, CRM, etc)"
        },
        "modules": [
            "ACCOUNTING - Kế toán, tài khoản, hạch toán",
            "GENERAL - Câu hỏi chung"
        ],
        "features": [
            "Module Router - Auto route to appropriate module",
            "Hybrid Semantic History Matching",
            "Streaming Cache with Simulation",
            "Multi-Agent System",
            "Optimized Vector Search"
        ]
    }
