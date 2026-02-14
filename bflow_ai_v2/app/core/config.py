"""
Configuration for bflow_ai_v2
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # API
    APP_NAME: str = "bflow_ai_v2 - COA Agent"
    APP_VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8010

    # Ollama/LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GENERATION_MODEL: str = "qwen2.5:3b"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Ollama options
    OLLAMA_OPTIONS: dict = {
        "temperature": 0.3,
        "num_predict": 500,
        "top_k": 20,
        "top_p": 0.9,
    }

    # COA Data paths
    COA_DATA_DIR: str = "./data/coa"
    COA_99_FILE: str = "coa_99.json"
    COA_200_FILE: str = "coa_200.json"
    COA_COMPARE_FILE: str = "coa_compare_99_vs_200.json"

    # COA Search config
    COA_SEARCH_LIMIT: int = 5
    COA_SIMILARITY_THRESHOLD: float = 0.3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
