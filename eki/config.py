from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EKI_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data/runtime")
    embedding: Literal["hash", "fastembed"] = "hash"
    reranker: Literal["lexical", "fastembed"] = "lexical"
    generator: Literal["extractive", "openai"] = "extractive"
    llm_model: str = "gpt-4.1-mini"
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    api_key: str = "local-demo-change-me"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    redis_url: str | None = None
    web_hosts: list[str] = []
    langfuse: bool = False
    max_upload_bytes: int = 5_000_000
    max_text_chars: int = 1_000_000
    max_chunks: int = 2000
    max_corpus_chunks: int = 20000
    cache_ttl: int = 300
    session_ttl: int = 1800
