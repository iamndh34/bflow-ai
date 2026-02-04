"""
MongoDB Client Module
Hỗ trợ kết nối và thao tác với MongoDB cho các tính năng mở rộng trong tương lai.
Có thể dùng để lưu: conversation history, user sessions, cache, logs...
"""

from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from app.core.config import settings

# Global MongoDB client
_client: Optional[AsyncIOMotorClient] = None


def get_mongo_client() -> AsyncIOMotorClient:
    """
    Lấy MongoDB client instance (singleton pattern).
    Tạo mới nếu chưa tồn tại.
    """
    global _client
    if _client is None:
        # MONGO_URL từ environment variable, fallback về localhost
        mongo_url = getattr(settings, "MONGO_URL", "mongodb://localhost:27017/bflow_db")
        _client = AsyncIOMotorClient(mongo_url)
    return _client


def get_database():
    """
    Lấy database instance. Sử dụng trong FastAPI dependencies.
    """
    client = get_mongo_client()
    # Tên database từ URL, fallback về 'bflow_db'
    db_name = "bflow_db"
    return client[db_name]


async def close_mongo_connection():
    """
    Đóng MongoDB connection khi shutdown app.
    Gọi trong FastAPI lifespan event.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ============================================================================
# Collection Helpers (sử dụng khi cần)
# ============================================================================

async def get_collection(collection_name: str):
    """
    Lấy một collection cụ thể từ database.

    Example:
        users_col = await get_collection("users")
        await users_col.insert_one({"name": "test"})
    """
    db = get_database()
    return db[collection_name]


async def find_one(collection_name: str, filter_dict: dict):
    """Tìm một document trong collection."""
    col = await get_collection(collection_name)
    return await col.find_one(filter_dict)


async def find_many(collection_name: str, filter_dict: dict = None, limit: int = 100):
    """Tìm nhiều documents trong collection."""
    col = await get_collection(collection_name)
    cursor = col.find(filter_dict or {}).limit(limit)
    return await cursor.to_list(length=limit)


async def insert_one(collection_name: str, document: dict):
    """Chèn một document vào collection."""
    col = await get_collection(collection_name)
    result = await col.insert_one(document)
    return result.inserted_id


async def update_one(collection_name: str, filter_dict: dict, update_dict: dict):
    """Cập nhật một document trong collection."""
    col = await get_collection(collection_name)
    return await col.update_one(filter_dict, {"$set": update_dict})


async def delete_one(collection_name: str, filter_dict: dict):
    """Xóa một document khỏi collection."""
    col = await get_collection(collection_name)
    return await col.delete_one(filter_dict)
