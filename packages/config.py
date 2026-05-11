from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Budget
    budget_total_usd: float = Field(default=20.0, alias="BUDGET_TOTAL_USD")
    budget_kill_usd: float = Field(default=18.0, alias="BUDGET_KILL_USD")

    # Voice
    voice_mode: str = Field(default="text", alias="VOICE_MODE")
    vapi_api_key: str = Field(default="", alias="VAPI_API_KEY")
    vapi_phone_number_id: str = Field(default="", alias="VAPI_PHONE_NUMBER_ID")
    vapi_assistant_id: str = Field(default="", alias="VAPI_ASSISTANT_ID")
    demo_borrower_phone: str = Field(default="", alias="DEMO_BORROWER_PHONE")
    public_webhook_url: str = Field(default="", alias="PUBLIC_WEBHOOK_URL")

    # Storage
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="riverline", alias="POSTGRES_USER")
    postgres_password: str = Field(default="riverline", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="riverline", alias="POSTGRES_DB")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Temporal
    temporal_host: str = Field(default="localhost:7233", alias="TEMPORAL_HOST")
    temporal_namespace: str = Field(default="default", alias="TEMPORAL_NAMESPACE")
    temporal_task_queue: str = Field(default="riverline", alias="TEMPORAL_TASK_QUEUE")

    # Reproducibility
    rng_seed: int = Field(default=20260512, alias="RNG_SEED")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_async_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
