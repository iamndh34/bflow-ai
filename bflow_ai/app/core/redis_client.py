"""
Redis Client - Singleton wrapper cho Redis operations

Tích hợp Redis cho:
- Streaming Cache
- LLM Cache
- Session History
"""
import redis
from typing import Optional, Any, Dict, List
import json
import logging

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Singleton Redis client với connection pooling.

    Usage:
        from app.core.redis_client import get_redis_client

        redis_client = get_redis_client()
        redis_client.set("key", "value", ttl=3600)
        value = redis_client.get("key")
    """

    _instance: Optional[redis.Redis] = None
    _initialized: bool = False

    @classmethod
    def get_client(cls) -> Optional[redis.Redis]:
        """Get singleton Redis client instance."""
        if cls._instance is None:
            cls._instance = cls._create_client()
        return cls._instance

    @classmethod
    def _create_client(cls) -> Optional[redis.Redis]:
        """Create Redis client with connection pooling."""
        from app.core.config import settings

        try:
            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            client.ping()
            cls._initialized = True
            logger.info(f"✓ Redis connected: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            return client

        except Exception as e:
            logger.warning(f"✗ Redis unavailable: {e}. Using fallback.")
            cls._initialized = False
            return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Redis is available."""
        return cls._initialized and cls._instance is not None

    @classmethod
    def set(cls, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value với TTL.

        Args:
            key: Redis key
            value: Value (sẽ được JSON serialize)
            ttl: Time-to-live trong seconds

        Returns:
            True nếu thành công, False nếu thất bại
        """
        client = cls.get_client()
        if not client:
            return False

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)

            client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """
        Get value từ Redis.

        Args:
            key: Redis key

        Returns:
            Value hoặc None nếu không tìm thấy
        """
        client = cls.get_client()
        if not client:
            return None

        try:
            value = client.get(key)
            if value is None:
                return None

            # Try to parse JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    @classmethod
    def delete(cls, *keys: str) -> int:
        """
        Delete keys.

        Args:
            *keys: Keys to delete

        Returns:
            Số keys đã xóa
        """
        client = cls.get_client()
        if not client:
            return 0

        try:
            return client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return 0

    @classmethod
    def exists(cls, *keys: str) -> int:
        """
        Check nếu keys tồn tại.

        Args:
            *keys: Keys to check

        Returns:
            Số keys tồn tại
        """
        client = cls.get_client()
        if not client:
            return 0

        try:
            return client.exists(*keys)
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return 0

    @classmethod
    def clear_pattern(cls, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Redis key pattern (ví dụ: "cache:*")

        Returns:
            Số keys đã xóa
        """
        client = cls.get_client()
        if not client:
            return 0

        try:
            keys = client.keys(pattern)
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear_pattern error: {e}")
            return 0

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        Get Redis statistics.

        Returns:
            Dict với stats info
        """
        client = cls.get_client()
        if not client:
            return {"available": False}

        try:
            info = client.info()
            return {
                "available": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_keys": info.get("db0", {}).get("keys", 0),
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"available": False, "error": str(e)}

    @classmethod
    def hset(cls, name: str, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set hash field.

        Args:
            name: Hash name
            key: Field key
            value: Field value
            ttl: Optional TTL cho hash

        Returns:
            True nếu thành công
        """
        client = cls.get_client()
        if not client:
            return False

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)

            client.hset(name, key, value)

            if ttl:
                client.expire(name, ttl)

            return True
        except Exception as e:
            logger.error(f"Redis hset error: {e}")
            return False

    @classmethod
    def hget(cls, name: str, key: str) -> Optional[Any]:
        """
        Get hash field.

        Args:
            name: Hash name
            key: Field key

        Returns:
            Field value hoặc None
        """
        client = cls.get_client()
        if not client:
            return None

        try:
            value = client.hget(name, key)
            if value is None:
                return None

            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        except Exception as e:
            logger.error(f"Redis hget error: {e}")
            return None

    @classmethod
    def hgetall(cls, name: str) -> Dict[str, Any]:
        """
        Get all hash fields.

        Args:
            name: Hash name

        Returns:
            Dict của tất cả fields
        """
        client = cls.get_client()
        if not client:
            return {}

        try:
            data = client.hgetall(name)
            result = {}
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            return result

        except Exception as e:
            logger.error(f"Redis hgetall error: {e}")
            return {}

    @classmethod
    def lpush(cls, name: str, *values: Any) -> int:
        """
        Push values to list head.

        Args:
            name: List name
            *values: Values to push

        Returns:
            List length sau khi push
        """
        client = cls.get_client()
        if not client:
            return 0

        try:
            serialized_values = []
            for v in values:
                if isinstance(v, (dict, list)):
                    serialized_values.append(json.dumps(v, ensure_ascii=False))
                else:
                    serialized_values.append(str(v))

            return client.lpush(name, *serialized_values)
        except Exception as e:
            logger.error(f"Redis lpush error: {e}")
            return 0

    @classmethod
    def lrange(cls, name: str, start: int = 0, end: int = -1) -> List[Any]:
        """
        Get list range.

        Args:
            name: List name
            start: Start index
            end: End index

        Returns:
            List of values
        """
        client = cls.get_client()
        if not client:
            return []

        try:
            values = client.lrange(name, start, end)
            result = []
            for v in values:
                try:
                    result.append(json.loads(v))
                except json.JSONDecodeError:
                    result.append(v)
            return result

        except Exception as e:
            logger.error(f"Redis lrange error: {e}")
            return []

    @classmethod
    def ltrim(cls, name: str, start: int, end: int) -> bool:
        """
        Trim list to range.

        Args:
            name: List name
            start: Start index
            end: End index

        Returns:
            True nếu thành công
        """
        client = cls.get_client()
        if not client:
            return False

        try:
            client.ltrim(name, start, end)
            return True
        except Exception as e:
            logger.error(f"Redis ltrim error: {e}")
            return False


# Convenience functions
def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client instance."""
    return RedisClient.get_client()


def redis_available() -> bool:
    """Check if Redis is available."""
    return RedisClient.is_available()
