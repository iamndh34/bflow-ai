from fastapi import FastAPI
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.db.mongodb import db
from app.api.endpoints import router
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

# Cho phép frontend truy cập
origins = [
    "http://127.0.0.1:8001",
    "http://localhost:8001",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    # Nếu dùng extension, VSCode live server, thêm vào đây
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # hoặc ["*"] để mở full
    allow_credentials=False,
    allow_methods=["*"],           # GET, POST, ...
    allow_headers=["*"],           # Cho phép mọi header
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "BFLOW AI is running"}