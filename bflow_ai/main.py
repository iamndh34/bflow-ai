from fastapi import FastAPI
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.db.mongodb import db
from app.api.endpoints.accounting import router
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Kết nối DB
    db.client = AsyncIOMotorClient(settings.MONGO_URL)
    print("Connected to MongoDB")
    yield
    # Shutdown: Đóng kết nối
    db.client.close()
    print("Disconnected MongoDB")

app = FastAPI(title="BFLOW AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Để trống
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/ai-bflow")

@app.get("/")
async def root():
    return {"message": "BFLOW AI is running"}