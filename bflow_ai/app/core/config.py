from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BFLOW AI"
    MONGO_URL: str
    DB_NAME: str = "bflow_ai"

    class Config:
        env_file = ".env"

settings = Settings()