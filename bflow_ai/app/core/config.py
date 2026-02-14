from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BFLOW AI"

    # MongoDB Configuration
    MONGO_URL: str = "mongodb://localhost:27017/bflow_db"

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"

    # Model config
    CLASSIFIER_MODEL: str = "qwen2.5:0.5b"
    GENERATION_MODEL: str = "qwen2.5:7b"

    # Ollama generation options
    OLLAMA_OPTIONS: dict = {
        "num_ctx": 8192,          # Context window lớn
        "num_predict": 4096,      # Max tokens prediction - đủ nội dung
        "temperature": 0.3,       # Tăng nhẹ từ 0.2 để tránh early stopping, vẫn giữ chính xác
        "top_p": 0.85,            # Giảm để tập trung vào tokens có xác suất cao
        "top_k": 30,              # Giảm để hạn chế lựa chọn ngẫu nhiên
        "repeat_penalty": 1.15,   # Tăng để tránh lặp lại trong phần giải thích
        "num_thread": 4,          # CPU cores
    }

    # Options cho GeneralFreeAgent - temperature cao hơn để tự nhiên hơn
    GENERAL_FREE_OPTIONS: dict = {
        "num_ctx": 4096,          # Context nhỏ hơn cho chat
        "num_predict": 2048,      # Tăng lên để response dài hơn, chi tiết hơn
        "temperature": 0.75,      # Cao hơn để tự nhiên, sáng tạo
        "top_p": 0.9,             # Tăng để đa dạng hơn
        "top_k": 40,              # Tăng để nhiều lựa chọn hơn
        "repeat_penalty": 1.1,    # Giảm để cho phép lặp lại trong chat
        "num_thread": 4,
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
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.95  # Similarity threshold to match (tăng từ 0.85)

    class Config:
        env_file = ".env"

settings = Settings()
