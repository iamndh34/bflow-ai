"""
bflow_ai_v2 - Main Application
COA Agent Only - Chart of Accounts Query Service
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.api.endpoints import router as coa_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    print(f"[{settings.APP_NAME}] Starting on {settings.HOST}:{settings.PORT}")
    print(f"[{settings.APP_NAME}] Ollama: {settings.OLLAMA_BASE_URL}")
    print(f"[{settings.APP_NAME}] Model: {settings.GENERATION_MODEL}")
    yield
    # Shutdown (if needed)


def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(coa_router, prefix=settings.API_PREFIX, tags=["COA"])

    @app.get("/")
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running"
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
