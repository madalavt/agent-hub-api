from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/decision_tree.db"
    # When DATABASE_URL goes through a transaction-pooling PgBouncer: a direct connection to
    # Postgres for startup work (schema upgrades, advisory locks). Empty = use DATABASE_URL.
    direct_database_url: str | None = None
    # Internal services (extraction, annotations, worker) send this as X-API-Key.
    api_key: str | None = None
    cors_origins: list[str] = ["http://localhost:3000"]

    # Sign-in. Providers are named in order; "local" = username and password stored here.
    # Future providers (entra, oauth) are added in app/auth/providers.py and listed here.
    auth_providers: list[str] = ["local"]
    session_secret: str | None = None  # signs session tokens; required wherever it is shared
    session_ttl_s: int = 8 * 3600
    # First-run password for the seeded admin ("admin"). Unset: generated and logged once.
    seed_admin_password: str | None = None

    # Example customers/projects/trees (one JSON file per project in seed_dir). Each seed is
    # applied once: it is skipped when the customer already has a project with that name.
    seed_on_start: bool = True
    seed_dir: Path = Path(__file__).resolve().parent.parent / "seed"

    # ML models nodes may use. GPT via Azure OpenAI / OpenAI; Ollama for local development.
    supported_model_types: list[str] = ["azure_openai", "openai", "ollama"]
    # Ollama models added to the catalog on start (set SEED_LOCAL_MODELS=false in hosted envs).
    seed_local_models: bool = True
    local_models: list[str] = ["qwen3:8b", "llama3.2:3b", "gemma3:4b"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
