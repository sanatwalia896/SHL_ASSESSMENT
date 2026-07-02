from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    app_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = app_root / "data"

    normalized_catalog_path: Path = data_dir / "normalized_catalog.json"
    embedding_documents_path: Path = data_dir / "embedding_documents.json"
    synonym_index_path: Path = data_dir / "synonym_index.json"

    max_recommendations: int = 5
    faiss_top_k: int = 20
    keyword_top_k: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()