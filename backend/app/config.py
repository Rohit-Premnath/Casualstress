import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "causalstress")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "causalstress")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "db")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Data Sources
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

    # Environment
    ENV: str = os.getenv("ENV", "development")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()