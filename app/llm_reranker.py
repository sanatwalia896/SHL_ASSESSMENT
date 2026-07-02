from __future__ import annotations

import json

from groq import Groq

from app.schemas import QueryPlan, ScoredProduct
from app.settings import get_settings


class LLMReranker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = (
            Groq(api_key=self.settings.groq_api_key)
            if self.settings.groq_api_key
            else None
        )

        self.last_metadata = {
            "confidence": "unknown",
            "catalog_gap": "",
            "missing_exact_terms": [],
        }

    def rerank(self, plan: QueryPlan, candidates: list[ScoredProduct]) -> list[ScoredProduct]:
        self.last_metadata = {
            "confidence": "unknown",
            "catalog_gap": "",
            "missing_exact_terms": [],
        }

        if self.client is None or not candidates:
            return candidates[: self.settings.max_recommendations]

        shortlist = candidates[:12]
        product_by_id = {item.product.id: item for item in shortlist}

        try:
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._build_prompt(plan, shortlist)},
                ],
            )

            content = response.choices[0].message.content or "{}"
            data = self._parse_json(content)

            self.last_metadata = {
                "confidence": data.get("confidence", "unknown"),
                "catalog_gap": data.get("catalog_gap", ""),
                "missing_exact_terms": data.get("missing_exact_terms", []),
            }

            ordered_ids = [str(x) for x in data.get("selected_ids", [])]

            reranked: list[ScoredProduct] = []
            seen: set[str] = set()

            for product_id in ordered_ids:
                if product_id in product_by_id and product_id not in seen:
                    reranked.append(product_by_id[product_id])
                    seen.add(product_id)

            # Important:
            # Do not auto-fill weak products just to reach max_recommendations.
            # If the LLM selected only 3 or 4 strong products, return only those.
            if reranked:
                return reranked[: self.settings.max_recommendations]

            return candidates[: self.settings.max_recommendations]

        except Exception:
            self.last_metadata = {
                "confidence": "unknown",
                "catalog_gap": "",
                "missing_exact_terms": [],
            }
            return candidates[: self.settings.max_recommendations]

    def _system_prompt(self) -> str:
        return """
You are a safe SHL catalog reranker.

You can ONLY select from the provided product IDs.
Do not invent products.
Do not invent URLs.
Return ONLY valid JSON.

Output schema:
{
  "selected_ids": ["id1", "id2", "id3"],
  "confidence": "exact_match | closest_available | weak_match",
  "catalog_gap": "short explanation of any catalog gap",
  "missing_exact_terms": ["term1"]
}

Rules:
- Select at most 5 products.
- Prefer products that directly support the user's hiring requirement.
- Do not add weak products just to fill the list.
- If only 3 or 4 products are strong, return only 3 or 4.
- Remove products that are weak, generic, leadership-only, manager-only, or unrelated.
- Keep useful complementary assessments if relevant.
- For technical roles, prefer technical knowledge, simulations, coding, networking, Linux, cloud, and reasoning products.
- For office roles, prefer Microsoft Office, Excel, Word, typing, and clerical products.
- For leadership roles, prefer OPQ, leadership reports, executive scenarios, and cognitive/personality assessments.
- For safety roles, prefer safety and dependability products.

Confidence rules:
- Use "exact_match" only when selected products directly match the requested role, skill, or assessment purpose.
- Use "closest_available" when the catalog lacks an exact technology-specific or role-specific test, but selected products are useful substitutes.
- Use "weak_match" when the candidates only loosely match the requirement.
- If the user asks for a specific technology such as Rust, Go, Golang, Kotlin, TypeScript, Swift, Scala, Ruby, PHP, React, Angular, Vue, or Node.js, and no selected product name exactly contains that technology, include it in missing_exact_terms.
- If confidence is "closest_available" or "weak_match", write a clear catalog_gap.
""".strip()

    def _build_prompt(self, plan: QueryPlan, candidates: list[ScoredProduct]) -> str:
        rows = []

        for item in candidates:
            p = item.product
            rows.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "url": p.url,
                    "test_type": ",".join(p.keys) if p.keys else p.product_type,
                    "product_type": p.product_type,
                    "keys": p.keys,
                    "job_levels": p.job_levels,
                    "languages": p.languages,
                    "duration": p.duration,
                    "description": p.description[:600],
                    "deterministic_score": round(item.score, 2),
                }
            )

        return json.dumps(
            {
                "query_plan": plan.model_dump(),
                "candidate_products": rows,
            },
            ensure_ascii=False,
        )

    def _parse_json(self, text: str) -> dict:
        text = text.strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            text = text[start : end + 1]

        return json.loads(text)


def get_llm_reranker() -> LLMReranker:
    return LLMReranker()