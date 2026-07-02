from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    app_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = app_root / "data"

    artifacts_dir: Path = data_dir / "artifacts"

    normalized_catalog_path: Path = artifacts_dir / "normalized_catalog.json"
    embedding_documents_path: Path = artifacts_dir / "embedding_documents.json"
    synonym_index_path: Path = artifacts_dir / "synonym_index.json"
    faiss_index_path: Path = artifacts_dir / "faiss.index"

    max_recommendations: int = 5
    faiss_top_k: int = 20
    keyword_top_k: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()