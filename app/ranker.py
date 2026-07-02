from __future__ import annotations

from app.schemas import QueryPlan, ScoredProduct
from app.settings import get_settings


class Ranker:
    def __init__(self) -> None:
        self.settings = get_settings()

    def rank(
    self,
    candidates: list[ScoredProduct],
    plan: QueryPlan,
    limit: int | None = None,) -> list[ScoredProduct]:
        ranked: list[ScoredProduct] = []

        for item in candidates:
            item.score += self._general_boost(item, plan)
            ranked.append(item)

        ranked.sort(key=lambda item: item.score, reverse=True)
        limit = limit or self.settings.max_recommendations
        return self._diversify(ranked, limit=limit)

    def _general_boost(self, item: ScoredProduct, plan: QueryPlan) -> float:
        product = item.product
        name = product.name.lower()

        text = " ".join(
            [
                product.name,
                product.description,
                product.product_type,
                " ".join(product.keys),
                " ".join(product.tags),
                " ".join(product.skills),
                " ".join(product.job_levels),
            ]
        ).lower()

        direct_keywords = [k.lower().strip() for k in plan.direct_keywords if k.strip()]
        related_keywords = [k.lower().strip() for k in plan.related_keywords if k.strip()]
        query_terms = " ".join(direct_keywords + related_keywords)

        boost = 0.0

        # Strongest signal: product title match.
        for keyword in direct_keywords:
            if keyword in name:
                boost += 100
            elif keyword in text:
                boost += 35

        for keyword in related_keywords:
            if keyword in name:
                boost += 40
            elif keyword in text:
                boost += 15

        # Generic technical-hiring preference.
        is_technical_query = any(
            term in query_terms
            for term in [
                "engineer",
                "developer",
                "software",
                "backend",
                "frontend",
                "programming",
                "coding",
                "infrastructure",
                "networking",
                "linux",
                "cloud",
                "java",
                "python",
                "rust",
                "aws",
                "docker",
                "kubernetes",
            ]
        )

        if is_technical_query:
            if "knowledge & skills" in text:
                boost += 30
            if "simulation" in text or "live coding" in text or "coding" in name:
                boost += 35
            if "ability & aptitude" in text or "verify" in name:
                boost += 12

            # Generic weak-fit penalties for technical IC hiring.
            weak_terms = [
                "hipo",
                "high potential",
                "executive",
                "managerial",
                "leadership report",
                "process monitoring",
                "administration",
            ]
            if any(term in name for term in weak_terms):
                boost -= 70

            if "personality" in text and not any(
                term in query_terms
                for term in ["personality", "behavior", "behaviour", "opq", "leadership", "culture"]
            ):
                boost -= 25

        # Office/admin query preference.
        is_office_query = any(
            term in query_terms
            for term in ["excel", "word", "office", "admin", "administrative", "clerical"]
        )

        if is_office_query:
            if "microsoft" in name:
                boost += 35
            if "excel" in name or "word" in name:
                boost += 80
            if "knowledge & skills" in text or "simulation" in text:
                boost += 20

            if any(term in text for term in [".net", "mvc", "wcf", "mvvm", "java", "python"]):
                boost -= 80

        # Contact center / customer service preference.
        is_contact_query = any(
            term in query_terms
            for term in ["contact center", "contact centre", "call center", "customer service", "inbound"]
        )

        if is_contact_query:
            if "contact" in name or "customer service" in name or "svar" in name:
                boost += 70
            if "simulation" in text:
                boost += 35

        # Leadership query preference.
        is_leadership_query = any(
            term in query_terms
            for term in ["executive", "cxo", "director", "leadership", "senior leadership"]
        )

        if is_leadership_query:
            if "opq" in name:
                boost += 60
            if "leadership" in name:
                boost += 50
            if "executive" in name:
                boost += 35

        # Graduate query preference.
        is_graduate_query = any(
            term in query_terms
            for term in ["graduate", "entry level", "entry-level", "fresher", "campus"]
        )

        if is_graduate_query:
            if "graduate" in name:
                boost += 60
            if "verify" in name:
                boost += 25
            if "opq" in name:
                boost += 15

        # Safety/industrial query preference.
        is_safety_query = any(
            term in query_terms
            for term in ["safety", "dependability", "plant", "operator", "industrial", "manufacturing"]
        )

        if is_safety_query:
            if "safety" in name or "dependability" in name or "dsi" in name:
                boost += 75
            if "workplace health" in name:
                boost += 45

        return boost

    def _diversify(self, ranked: list[ScoredProduct], limit: int) -> list[ScoredProduct]:
        selected: list[ScoredProduct] = []
        seen_names: set[str] = set()
        type_counts: dict[str, int] = {}

        for item in ranked:
            name_key = item.product.name.lower().replace(" (new)", "").strip()
            product_type = item.product.product_type or "unknown"

            if name_key in seen_names:
                continue

            if type_counts.get(product_type, 0) >= 5:
                continue

            if item.score <= 0:
                continue

            selected.append(item)
            seen_names.add(name_key)
            type_counts[product_type] = type_counts.get(product_type, 0) + 1

            if len(selected) >= limit:
                break

        return selected


def get_ranker() -> Ranker:
    return Ranker()