from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BFLOW AI"

    # MongoDB Configuration
    MONGO_URL: str = "mongodb://localhost:27017/bflow_db"

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"

    # Model config
    CLASSIFIER_MODEL: str = "qwen2.5:0.5b"
    GENERATION_MODEL: str = "qwen2.5:3b"

    # Ollama generation options
    OLLAMA_OPTIONS: dict = {
        "num_ctx": 2048,          # Context window
        "num_predict": 512,       # Max tokens prediction
        "temperature": 0.3,       # Deterministic output
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_thread": 4,          # CPU cores
    }

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""  # Optional password
    USE_REDIS: bool = True  # Enable/disable Redis (fallback to in-memory/file)

    # LLM Cache Configuration
    ENABLE_LLM_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 100  # Maximum cached responses (in-memory fallback)

    # Cache simulate streaming settings
    CACHE_SIMULATE_DELAY: float = 0.02  # Delay between chunks (seconds)
    CACHE_CHARS_PER_CHUNK: int = 1      # Characters per chunk (=1 for char-by-char)

    # Semantic History Matching
    ENABLE_SEMANTIC_HISTORY: bool = True
    SEMANTIC_MODE: str = "hybrid"  # Modes: "sentence", "keyword", "hybrid"
    SEMANTIC_ALPHA: float = 0.7  # Sentence weight for hybrid (0.7 = 70% sentence, 30% keyword)
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.85  # Similarity threshold to match

    class Config:
        env_file = ".env"

settings = Settings()
