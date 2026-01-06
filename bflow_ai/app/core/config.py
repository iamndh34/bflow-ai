from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BFLOW AI"
    MONGO_URL: str
    DB_NAME: str = "bflow_ai"
    OLLAMA_HOST: str = "http://localhost:11434"
    # OLLAMA_HOST: str = "http://mis_ollama:11434"

    class Config:
        env_file = ".env"

settings = Settings()