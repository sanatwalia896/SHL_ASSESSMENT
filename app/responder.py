from __future__ import annotations

from groq import Groq

from app.schemas import QueryPlan, ScoredProduct
from app.settings import get_settings


class ResponseGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.groq_api_key) if self.settings.groq_api_key else None

    def write_recommendation_reply(
        self,
        plan: QueryPlan,
        ranked_products: list[ScoredProduct],
    ) -> str:
        if not ranked_products:
            return "I could not find a strong SHL catalog match for that requirement. Could you share the role, seniority, and key skills to assess?"

        if self.client is None:
            return self._fallback_reply(plan, ranked_products)

        prompt = self._build_prompt(plan, ranked_products)

        try:
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            reply = response.choices[0].message.content or ""
            return reply.strip() or self._fallback_reply(plan, ranked_products)
        except Exception:
            return self._fallback_reply(plan, ranked_products)

    def _system_prompt(self) -> str:
        return """
You are an SHL assessment recommender.

Write a concise recruiter-facing response.

Rules:
- Use only the shortlisted catalog products provided.
- Never invent assessment names.
- Never invent URLs.
- Never mention BM25, FAISS, embeddings, semantic similarity, metadata similarity, ranking scores, or implementation details.
- Never say "Prioritized X assessments".
- If the user asked for a specific technology and there is no exact product for it, explicitly say:
  "The SHL catalog does not currently include a <technology>-specific assessment."
- Then explain that the shortlist covers the closest available assessment signals.
- Keep the answer short.
- Do not include markdown tables.
- Do not output JSON.
""".strip()

    def _build_prompt(self, plan: QueryPlan, ranked_products: list[ScoredProduct]) -> str:
        products_text = "\n".join(
            [
                f"{i+1}. {item.product.name} | type={item.product.product_type} | keys={', '.join(item.product.keys)} | description={item.product.description[:350]}"
                for i, item in enumerate(ranked_products)
            ]
        )

        return f"""
Query plan:
direct_keywords: {plan.direct_keywords}
related_keywords: {plan.related_keywords}
semantic_query: {plan.semantic_query}

Shortlisted catalog products:
{products_text}

Write the final assistant reply.
""".strip()

    def _fallback_reply(
        self,
        plan: QueryPlan,
        ranked_products: list[ScoredProduct],
    ) -> str:
        query_terms = " ".join(plan.direct_keywords + plan.related_keywords).lower()

        prefix = ""
        if "rust" in query_terms:
            has_exact_rust = any("rust" in item.product.name.lower() for item in ranked_products)
            if not has_exact_rust:
                prefix = "The SHL catalog does not currently include a Rust-specific assessment. "

        names = ", ".join(item.product.name for item in ranked_products[:5])

        return (
            prefix
            + f"The strongest available SHL matches are {names}. "
            + "These cover the closest relevant skills and supporting assessment signals for the role."
        )


def get_response_generator() -> ResponseGenerator:
    return ResponseGenerator()