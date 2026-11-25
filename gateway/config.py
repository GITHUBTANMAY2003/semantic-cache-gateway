"""Gateway settings, read from environment variables or .env.
Keeps URLs and keys out of the code, so moving from Ollama to vLLM is a config change."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": an unknown line in .env should not stop the server from starting.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_api_key: str = ""
    upstream_timeout_seconds: float = 120.0
