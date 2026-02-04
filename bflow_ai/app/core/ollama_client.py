"""
Ollama Client Pool - Singleton connection pooling cho Ollama

Thay vì tạo ollama.Client() mới mỗi request (chậm),
dùng singleton này để reuse connection.

Usage:
    from app.core.ollama_client import get_ollama_client

    client = get_ollama_client()
    response = client.chat(...)
"""
import threading
from typing import Optional
import ollama
from app.core.config import settings


class OllamaClientPool:
    """Singleton Ollama client pool"""

    _instance: Optional[ollama.Client] = None
    _lock = threading.Lock()
    _host: str = settings.OLLAMA_HOST

    @classmethod
    def get_client(cls) -> ollama.Client:
        """Get singleton ollama client instance"""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = ollama.Client(host=cls._host)
                    print(f"[OllamaPool] Created singleton client for {cls._host}")
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset client instance (cho testing hoặc khi đổi config)"""
        with cls._lock:
            cls._instance = None


# Convenience function
def get_ollama_client() -> ollama.Client:
    """Get singleton ollama client"""
    return OllamaClientPool.get_client()
