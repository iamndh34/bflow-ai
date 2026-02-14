"""
Cached LLM Service - Redis-backed with in-memory fallback

Features:
1. Response caching với TTL (Redis优先, fallback to in-memory)
2. Cache key dựa trên prompt + model + options
3. Thread-safe cache operations
4. Statistics tracking
5. Fallback khi cache miss
"""

import hashlib
import json
import time
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.core.config import settings
import ollama


class CachedLLMService:
    """
    Service wrapper cho Ollama với Redis caching capability.

    Usage:
        service = CachedLLMService()

        # Chat với cache
        response = service.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "Hello"}],
            use_cache=True
        )

        # Lấy statistics
        stats = service.get_stats()
    """

    def __init__(
        self,
        host: Optional[str] = None,
        enable_cache: Optional[bool] = None,
        cache_ttl: Optional[int] = None,
        max_cache_size: Optional[int] = None
    ):
        """Initialize CachedLLMService"""
        self.host = host or settings.OLLAMA_HOST
        self.enable_cache = enable_cache if enable_cache is not None else settings.ENABLE_LLM_CACHE
        self.cache_ttl = cache_ttl or settings.CACHE_TTL
        self.max_cache_size = max_cache_size or settings.MAX_CACHE_SIZE

        self.client = ollama.Client(host=self.host)

        # In-memory fallback cache
        self._memory_cache: Dict[str, Any] = {}

        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "cache_backend": "unknown"
        }

        # Thread lock cho cache operations
        self._lock = threading.RLock()

        # Default options từ config
        self.default_options = settings.OLLAMA_OPTIONS

        # Check Redis availability
        self._use_redis = False
        try:
            from app.core.redis_client import redis_available
            self._use_redis = redis_available()
            self._stats["cache_backend"] = "Redis" if self._use_redis else "in-memory"
        except Exception:
            self._stats["cache_backend"] = "in-memory"

    def _generate_cache_key(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        options: Optional[Dict] = None,
        format_schema: Optional[Dict] = None
    ) -> str:
        """Tạo cache key từ input parameters"""
        key_parts = {
            "model": model,
            "options": options or self.default_options,
        }

        if prompt is not None:
            key_parts["prompt"] = prompt
        elif messages is not None:
            key_parts["messages"] = json.dumps(messages, sort_keys=True)

        if format_schema is not None:
            key_parts["format"] = json.dumps(format_schema, sort_keys=True)

        key_str = json.dumps(key_parts, sort_keys=True)
        md5_hash = hashlib.md5(key_str.encode()).hexdigest()

        # Redis key prefix
        return f"llm:cache:{md5_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Lấy response từ cache (Redis优先, fallback to in-memory)"""
        if not self.enable_cache:
            return None

        with self._lock:
            # Try Redis first
            if self._use_redis:
                try:
                    from app.core.redis_client import RedisClient
                    value = RedisClient.get(cache_key)
                    if value is not None:
                        self._stats["cache_hits"] += 1
                        self._stats["total_requests"] += 1
                        return value
                except Exception as e:
                    print(f"[LLMService] Redis get error: {e}")

            # Fallback to in-memory
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]

                # Check if expired
                if time.time() - entry.get("created_at", 0) < self.cache_ttl:
                    entry["hit_count"] = entry.get("hit_count", 0) + 1
                    self._stats["cache_hits"] += 1
                    self._stats["total_requests"] += 1
                    return entry["response"]
                else:
                    # Expired, remove
                    del self._memory_cache[cache_key]

        return None

    def _set_cache(self, cache_key: str, response: Any, metadata: Optional[Dict] = None):
        """Lưu response vào cache (Redis优先, fallback to in-memory)"""
        if not self.enable_cache:
            return

        with self._lock:
            # Try Redis first
            if self._use_redis:
                try:
                    from app.core.redis_client import RedisClient
                    success = RedisClient.set(cache_key, response, ttl=self.cache_ttl)
                    if success:
                        return  # Saved to Redis, done
                except Exception as e:
                    print(f"[LLMService] Redis set error: {e}")

            # Fallback to in-memory
            # Check cache size limit
            if len(self._memory_cache) >= self.max_cache_size:
                # LRU: Remove entry với hit_count thấp nhất
                lru_key = min(
                    self._memory_cache.keys(),
                    key=lambda k: (self._memory_cache[k].get("hit_count", 0),
                                   self._memory_cache[k].get("created_at", 0))
                )
                del self._memory_cache[lru_key]

            # Store new entry
            self._memory_cache[cache_key] = {
                "response": response,
                "created_at": time.time(),
                "hit_count": 0,
                "metadata": metadata or {}
            }

    def _update_miss_stats(self):
        """Update cache miss statistics"""
        with self._lock:
            self._stats["cache_misses"] += 1
            self._stats["total_requests"] += 1

    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text với caching.

        Args:
            model: Model name
            prompt: Input prompt
            options: Generation options (sẽ merge với default_options)
            use_cache: Có sử dụng cache không
            **kwargs: Additional arguments cho ollama.generate

        Returns:
            Response từ Ollama
        """
        # Merge options với default
        merged_options = {**self.default_options, **(options or {})}

        # Tạo cache key
        cache_key = self._generate_cache_key(
            model=model,
            prompt=prompt,
            options=merged_options
        )

        # Try cache first
        if use_cache:
            cached_response = self._get_from_cache(cache_key)
            if cached_response is not None:
                return cached_response

        # Cache miss - call Ollama
        self._update_miss_stats()

        response = self.client.generate(
            model=model,
            prompt=prompt,
            options=merged_options,
            **kwargs
        )

        # Store in cache
        if use_cache:
            self._set_cache(
                cache_key,
                response,
                metadata={"type": "generate", "model": model}
            )

        return response

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict] = None,
        format: Optional[Dict] = None,
        use_cache: bool = True,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat với caching.

        Args:
            model: Model name
            messages: List of message dicts
            options: Generation options
            format: JSON schema cho structured output
            use_cache: Có sử dụng cache không
            stream: Streaming (cache không hoạt động với stream=True)
            **kwargs: Additional arguments

        Returns:
            Response từ Ollama
        """
        # Streaming không support cache
        if stream:
            use_cache = False

        # Merge options
        merged_options = {**self.default_options, **(options or {})}

        # Tạo cache key
        cache_key = self._generate_cache_key(
            model=model,
            messages=messages,
            options=merged_options,
            format_schema=format
        )

        # Try cache first
        if use_cache:
            cached_response = self._get_from_cache(cache_key)
            if cached_response is not None:
                return cached_response

        # Cache miss - call Ollama
        self._update_miss_stats()

        response = self.client.chat(
            model=model,
            messages=messages,
            options=merged_options,
            format=format,
            **kwargs
        )

        # Store in cache
        if use_cache:
            self._set_cache(
                cache_key,
                response,
                metadata={"type": "chat", "model": model}
            )

        return response

    def clear_cache(self, pattern: Optional[str] = None):
        """
        Xóa cache.

        Args:
            pattern: Nếu cung cấp, chỉ xóa entries với pattern (Redis only)
        """
        with self._lock:
            if self._use_redis:
                try:
                    from app.core.redis_client import RedisClient
                    if pattern:
                        RedisClient.clear_pattern(pattern)
                    else:
                        RedisClient.clear_pattern("llm:cache:*")
                except Exception as e:
                    print(f"[LLMService] Redis clear error: {e}")

            # Always clear in-memory cache
            self._memory_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy cache statistics.

        Returns:
            Dictionary với stats info
        """
        with self._lock:
            total = self._stats["total_requests"]
            hits = self._stats["cache_hits"]

            return {
                "cache_enabled": self.enable_cache,
                "cache_backend": self._stats["cache_backend"],
                "cache_size": len(self._memory_cache),
                "max_cache_size": self.max_cache_size,
                "cache_ttl_seconds": self.cache_ttl,
                "total_requests": total,
                "cache_hits": hits,
                "cache_misses": self._stats["cache_misses"],
                "hit_rate": hits / total if total > 0 else 0.0,
            }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global instance - sử dụng trong toàn app
_llm_service: Optional[CachedLLMService] = None
_service_lock = threading.Lock()


def get_llm_service() -> CachedLLMService:
    """
    Get singleton instance của CachedLLMService.

    Usage:
        from app.services.llm_service import get_llm_service

        llm = get_llm_service()
        response = llm.chat(...)
    """
    global _llm_service

    if _llm_service is None:
        with _service_lock:
            if _llm_service is None:
                _llm_service = CachedLLMService()

    return _llm_service
