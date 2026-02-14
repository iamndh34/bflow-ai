"""
Ollama Client - Singleton for bflow_ai_v2
"""
import threading
from typing import Optional
import ollama
from .config import settings


class OllamaClientPool:
    """Singleton Ollama client pool"""

    _instance: Optional[ollama.Client] = None
    _lock = threading.Lock()
    _host: str = settings.OLLAMA_BASE_URL

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
        """Reset client instance"""
        with cls._lock:
            cls._instance = None


def get_ollama_client() -> ollama.Client:
    """Get singleton ollama client"""
    return OllamaClientPool.get_client()
