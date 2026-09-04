from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    decision_latency_limit_sec: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
