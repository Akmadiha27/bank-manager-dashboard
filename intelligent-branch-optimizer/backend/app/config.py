from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Intelligent Branch Optimizer API"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Supabase (configure when wiring live data; no secrets in repo)
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None

    # Paths relative to backend/ working directory
    data_csv_path: str = "../data/bank_branch_synthetic_dataset.csv"
    models_dir: str = "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
