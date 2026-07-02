from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.schemas import CatalogProduct
from app.settings import get_settings


class CatalogStore:
    def __init__(self, catalog_path: Path, synonym_path: Path | None = None) -> None:
        self.catalog_path = catalog_path
        self.synonym_path = synonym_path
        self.products = self._load_products(catalog_path)
        self.by_id = {product.id: product for product in self.products}
        self.by_name = {product.name.lower(): product for product in self.products}
        self.synonyms = self._load_synonyms(synonym_path)

    def _load_products(self, path: Path) -> list[CatalogProduct]:
        with path.open("r", encoding="utf-8") as file:
            raw_products = json.load(file)

        products: list[CatalogProduct] = []
        for item in raw_products:
            try:
                product = CatalogProduct.model_validate(item)
                if product.status.lower() == "ok" and product.name and product.url:
                    products.append(product)
            except Exception:
                continue

        return products

    def _load_synonyms(self, path: Path | None) -> dict[str, list[str]]:
        if path is None or not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        synonyms: dict[str, list[str]] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, list):
                    synonyms[str(key).lower()] = [str(v).lower() for v in value]
                elif isinstance(value, str):
                    synonyms[str(key).lower()] = [value.lower()]

        return synonyms

    def get_by_id(self, product_id: str) -> CatalogProduct | None:
        return self.by_id.get(str(product_id))

    def get_by_name(self, name: str) -> CatalogProduct | None:
        return self.by_name.get(name.lower())

    def expand_keywords(self, keywords: list[str]) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set()

        for keyword in keywords:
            key = keyword.lower().strip()
            if not key or key in seen:
                continue

            expanded.append(key)
            seen.add(key)

            for synonym in self.synonyms.get(key, []):
                synonym = synonym.lower().strip()
                if synonym and synonym not in seen:
                    expanded.append(synonym)
                    seen.add(synonym)

        return expanded

    def product_text(self, product: CatalogProduct) -> str:
        parts = [
            product.name,
            product.description,
            product.product_type,
            product.category,
            " ".join(product.skills),
            " ".join(product.languages),
            " ".join(product.tags),
            " ".join(product.keys),
            " ".join(product.job_levels),
            product.duration or "",
        ]
        return " ".join(part for part in parts if part).lower()


@lru_cache(maxsize=1)
def get_catalog_store() -> CatalogStore:
    settings = get_settings()
    return CatalogStore(
        catalog_path=settings.normalized_catalog_path,
        synonym_path=settings.synonym_index_path,
    )