from __future__ import annotations

import re
from collections import defaultdict

from app.catalog import CatalogStore, get_catalog_store
from app.schemas import CatalogProduct, QueryPlan, ScoredProduct
from app.settings import get_settings
from app.vector_store import VectorStore, get_vector_store


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def keyword_in_text(keyword: str, text: str) -> bool:
    keyword = normalize_text(keyword)
    text = normalize_text(text)
    return keyword in text


class RetrievalEngine:
    def __init__(
        self,
        catalog: CatalogStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = get_settings()
        self.catalog = catalog or get_catalog_store()
        self.vector_store = vector_store or get_vector_store()

    def retrieve(self, plan: QueryPlan) -> list[ScoredProduct]:
        keyword_results = self._keyword_search(plan)
        vector_results = self._vector_search(plan)

        merged: dict[str, ScoredProduct] = {}

        for item in keyword_results + vector_results:
            product_id = item.product.id
            if product_id not in merged:
                merged[product_id] = item
            else:
                merged[product_id].score += item.score
                merged[product_id].reasons.extend(item.reasons)
                merged[product_id].source += f"+{item.source}"

        results = list(merged.values())
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(self.settings.keyword_top_k, self.settings.faiss_top_k)]

    def _keyword_search(self, plan: QueryPlan) -> list[ScoredProduct]:
        direct_keywords = self.catalog.expand_keywords(plan.direct_keywords)
        related_keywords = self.catalog.expand_keywords(plan.related_keywords)

        results: list[ScoredProduct] = []

        for product in self.catalog.products:
            text = self.catalog.product_text(product)
            score = 0.0
            reasons: list[str] = []

            for keyword in direct_keywords:
                if keyword_in_text(keyword, product.name):
                    score += 40
                    reasons.append(f"direct title match: {keyword}")
                elif keyword_in_text(keyword, text):
                    score += 18
                    reasons.append(f"direct keyword match: {keyword}")

            for keyword in related_keywords:
                if keyword_in_text(keyword, product.name):
                    score += 18
                    reasons.append(f"related title match: {keyword}")
                elif keyword_in_text(keyword, text):
                    score += 7
                    reasons.append(f"related keyword match: {keyword}")

            for keyword in plan.must_include:
                if keyword and not keyword_in_text(keyword, text):
                    score -= 20
                    reasons.append(f"missing requested signal: {keyword}")

            for keyword in plan.must_exclude:
                if keyword and keyword_in_text(keyword, text):
                    score -= 50
                    reasons.append(f"excluded signal: {keyword}")

            score += self._general_penalties(plan, product)

            if score > 0:
                results.append(
                    ScoredProduct(
                        product=product,
                        score=score,
                        reasons=reasons,
                        source="keyword",
                    )
                )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: self.settings.keyword_top_k]

    def _vector_search(self, plan: QueryPlan) -> list[ScoredProduct]:
        if not plan.semantic_query.strip():
            return []

        raw_results = self.vector_store.search(
            plan.semantic_query,
            top_k=self.settings.faiss_top_k,
        )

        results: list[ScoredProduct] = []

        for item in raw_results:
            product = self.catalog.get_by_id(str(item["id"]))
            if product is None:
                continue

            score = float(item["score"]) * 20
            score += self._general_penalties(plan, product)

            if score <= 0:
                continue

            results.append(
                ScoredProduct(
                    product=product,
                    score=score,
                    reasons=["semantic match"],
                    source="faiss",
                )
            )

        return results

    def _general_penalties(self, plan: QueryPlan, product: CatalogProduct) -> float:
        text = self.catalog.product_text(product)
        query_terms = " ".join(plan.direct_keywords + plan.related_keywords).lower()

        penalty = 0.0

        dotnet_terms = [".net", "mvc", "wcf", "mvvm"]

        if any(term in query_terms for term in ["rust", "excel", "word"]):
            if any(term in text for term in dotnet_terms):
                penalty -= 100

        if "rust" in query_terms:
            if "executive" in text and "leadership" not in query_terms:
                penalty -= 40

        if "excel" in query_terms or "word" in query_terms:
            if "office" not in text and "excel" not in text and "word" not in text:
                penalty -= 20

        return penalty


def get_retrieval_engine() -> RetrievalEngine:
    return RetrievalEngine()