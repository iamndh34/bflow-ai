"""
Streaming Response Cache - Redis-backed with in-memory fallback

Vấn đề: Streaming không thể cache trực tiếp vì yield từng phần.
Giải pháp: Cache full response, lần sau yield từ cache với simulated speed.

Storage:
- Redis (nếu available) - Persistent, shareable
- In-memory fallback (nếu Redis không có)
"""
import hashlib
import json
import time
from typing import Generator, Callable, Optional
from functools import lru_cache


class StreamingCache:
    """
    Cache cho streaming responses với Redis backend.

    Lưu trữ full response sau khi stream hoàn thành.
    Lần sau, yield từ cache với simulated streaming speed.
    """

    def __init__(self, ttl: int = 3600, max_size: int = 100):
        """
        Initialize streaming cache.

        Args:
            ttl: Time-to-live cho cache entries (seconds)
            max_size: Số entries tối đa (cho in-memory fallback)
        """
        self._ttl = ttl
        self._max_size = max_size
        self._in_memory_cache = {}  # Fallback khi Redis không available
        self._use_redis = False

        # Try to use Redis
        try:
            from app.core.redis_client import redis_available
            self._use_redis = redis_available()
            if self._use_redis:
                print("[StreamingCache] ✓ Using Redis for streaming cache")
            else:
                print("[StreamingCache] ✗ Redis unavailable, using in-memory fallback")
        except Exception as e:
            print(f"[StreamingCache] Redis check failed: {e}. Using in-memory fallback.")

    def _generate_key(self, question: str, agent_name: str, context: dict = None) -> str:
        """Generate cache key từ input parameters"""
        key_data = {
            "question": question,
            "agent": agent_name,
        }
        if context:
            # Chỉ lấy các fields quan trọng
            relevant_context = {
                k: v for k, v in context.items()
                if k in ["item_group", "partner_group", "chat_type"]
            }
            key_data["context"] = relevant_context

        key_str = json.dumps(key_data, sort_keys=True)
        md5_hash = hashlib.md5(key_str.encode()).hexdigest()

        # Redis key prefix
        return f"streaming:cache:{md5_hash}"

    def get(self, key: str) -> Optional[str]:
        """Get cached response nếu có và chưa expired"""
        if self._use_redis:
            # Try Redis
            from app.core.redis_client import RedisClient
            value = RedisClient.get(key)
            if value is not None:
                print(f"[StreamingCache] Redis cache hit: {key[:40]}...")
                return value

        # Fallback to in-memory
        if key in self._in_memory_cache:
            entry = self._in_memory_cache[key]
            if time.time() - entry["timestamp"] < self._ttl:
                print(f"[StreamingCache] Memory cache hit: {key[:40]}...")
                return entry["response"]
            else:
                # Expired, remove
                del self._in_memory_cache[key]

        return None

    def set(self, key: str, response: str):
        """Cache response"""
        if self._use_redis:
            # Save to Redis
            from app.core.redis_client import RedisClient
            success = RedisClient.set(key, response, ttl=self._ttl)
            if success:
                print(f"[StreamingCache] ✓ Saved to Redis: {key[:40]}...")
                return

        # Fallback to in-memory
        # Evict oldest nếu cache full
        if len(self._in_memory_cache) >= self._max_size:
            oldest_key = min(self._in_memory_cache.keys(), key=lambda k: self._in_memory_cache[k]["timestamp"])
            del self._in_memory_cache[oldest_key]

        self._in_memory_cache[key] = {
            "response": response,
            "timestamp": time.time()
        }

    def clear(self):
        """Clear all cache"""
        if self._use_redis:
            from app.core.redis_client import RedisClient
            count = RedisClient.clear_pattern("streaming:cache:*")
            print(f"[StreamingCache] Cleared {count} Redis entries")
        else:
            self._in_memory_cache.clear()
            print("[StreamingCache] Cleared in-memory cache")

    def stats(self) -> dict:
        """Get cache statistics"""
        if self._use_redis:
            from app.core.redis_client import RedisClient
            redis_stats = RedisClient.get_stats()
            return {
                "backend": "Redis",
                "available": redis_stats.get("available", False),
                "ttl": self._ttl
            }
        else:
            return {
                "backend": "in-memory",
                "size": len(self._in_memory_cache),
                "max_size": self._max_size,
                "ttl": self._ttl
            }


# Global cache instance
_streaming_cache = StreamingCache()


def _simulate_streaming(text: str, chars_per_chunk: int = 1, delay: float = 0.02) -> Generator[str, None, None]:
    """
    Simulate streaming từ cached text.

    Chia text thành small chunks và yield với delay để tạo cảm giác "đang typing".

    Args:
        text: Full cached response
        chars_per_chunk: Số characters mỗi chunk (default: 1 = từng chữ)
        delay: Delay giữa các chunks (default: 0.02s = ~20ms)

    Yields:
        str: Small chunks để simulate streaming
    """
    for i in range(0, len(text), chars_per_chunk):
        chunk = text[i:i + chars_per_chunk]
        yield chunk
        time.sleep(delay)


def cached_stream(
    question: str,
    agent_name: str,
    stream_func: Callable,
    context: dict = None,
    simulate_delay: float = 0.02
) -> Generator[str, None, None]:
    """
    Wrapper cho streaming với cache.

    Args:
        question: Câu hỏi
        agent_name: Tên agent
        stream_func: Function trả về generator (agent.stream_execute)
        context: Optional context dict
        simulate_delay: Delay khi simulate streaming từ cache (seconds)

    Yields:
        str: Streaming response chunks
    """
    cache_key = _streaming_cache._generate_key(question, agent_name, context)

    # Try cache first
    cached_response = _streaming_cache.get(cache_key)
    if cached_response is not None:
        # === CACHE HIT ===
        print(f"[StreamingCache] ✓ Simulating streaming from cache...")
        for chunk in _simulate_streaming(cached_response, delay=simulate_delay):
            yield chunk
        return

    # === CACHE MISS ===
    print(f"[StreamingCache] Cache miss, calling LLM...")

    full_response = ""

    # Stream từ agent
    for chunk in stream_func():
        yield chunk
        full_response += chunk

    # Cache full response cho lần sau
    _streaming_cache.set(cache_key, full_response)


def get_streaming_cache() -> StreamingCache:
    """Get global streaming cache instance"""
    return _streaming_cache


def clear_streaming_cache():
    """Clear all streaming cache"""
    _streaming_cache.clear()


# =============================================================================
# Quick LRU cache cho exact matches (in-memory, very fast)
# =============================================================================

@lru_cache(maxsize=100)
def get_cached_response(question: str, agent_name: str) -> Optional[str]:
    """
    Simple LRU cache cho exact question matches.

    Args:
        question: Câu hỏi (exact string)
        agent_name: Tên agent

    Returns:
        Cached response hoặc None
    """
    cache = get_streaming_cache()
    key = cache._generate_key(question, agent_name)
    return cache.get(key)


def cache_response(question: str, agent_name: str, response: str):
    """
    Cache response cho câu hỏi.

    Args:
        question: Câu hỏi
        agent_name: Tên agent
        response: Full response
    """
    cache = get_streaming_cache()
    key = cache._generate_key(question, agent_name)
    cache.set(key, response)
