"""Gateway settings, read from environment variables or .env.
Keeps URLs and keys out of the code, so moving from Ollama to vLLM is a config change."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": an unknown line in .env should not stop the server from starting.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_api_key: str = ""
    upstream_timeout_seconds: float = 120.0
    # Above this temperature the caller wants variety, so a stored answer would be wrong.
    max_cache_temperature: float = 0.3
    # 24 h: answers can go stale, so entries expire.
    cache_ttl_seconds: int = 86400
    # Similarity at or above this is a direct hit (the founder's 98%).
    tau_high: float = 0.98
    # 384-d vectors, small enough to run on CPU.
    embed_model: str = "BAAI/bge-small-en-v1.5"
