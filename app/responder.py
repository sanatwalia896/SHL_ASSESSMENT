from __future__ import annotations

from groq import Groq

from app.schemas import QueryPlan, ScoredProduct
from app.settings import get_settings


class ResponseGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.groq_api_key:
            self.client = Groq(api_key=self.settings.groq_api_key)
        else:
            self.client = None
            print("⚠️  GROQ_API_KEY not found. Using fallback responder.")

    def write_recommendation_reply(
        self,
        plan: QueryPlan,
        ranked_products: list[ScoredProduct],
        context=None,
        rerank_metadata=None,
    ) -> str:
        if not ranked_products:
            return self.write_clarification_reply(
                "I could not find a strong SHL catalog match. Could you share the role, seniority, and key skills to assess?"
            )

        if self.client is None:
            return self._fallback_recommendation_reply(
                plan=plan,
                ranked_products=ranked_products,
                context=context,
                rerank_metadata=rerank_metadata,
            )

        try:
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": self._recommendation_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._build_recommendation_prompt(
                            plan=plan,
                            ranked_products=ranked_products,
                            context=context,
                            rerank_metadata=rerank_metadata,
                        ),
                    },
                ],
            )

            reply = response.choices[0].message.content or ""
            return reply.strip() or self._fallback_recommendation_reply(
                plan=plan,
                ranked_products=ranked_products,
                context=context,
                rerank_metadata=rerank_metadata,
            )

        except Exception:
            return self._fallback_recommendation_reply(
                plan=plan,
                ranked_products=ranked_products,
                context=context,
                rerank_metadata=rerank_metadata,
            )

    def write_clarification_reply(self, question: str) -> str:
        if not question.strip():
            question = "What role or hiring situation should I match SHL assessments to?"
        return question.strip()

    def write_refusal_reply(self) -> str:
        return (
            "I can only help with SHL assessment selection from the SHL catalog, "
            "not legal, compliance, general hiring advice, or unrelated requests."
        )

    def write_confirmation_reply(self) -> str:
        return "Confirmed. The shortlist above is the recommended SHL assessment set."

    def _recommendation_system_prompt(self) -> str:
        return """
You are an SHL assessment recommender.

Write a concise recruiter-facing response.

Rules:
- Use only the shortlisted catalog products provided.
- Never invent assessment names.
- Never invent URLs.
- Never mention BM25, FAISS, embeddings, semantic similarity, metadata similarity, ranking scores, deterministic scores, or implementation details.
- Never say "Prioritized X assessments".
- If this is a follow-up turn, respond as a continuation of the conversation, not as a fresh first-turn answer.
- Directly address what the user added, removed, compared, questioned, or confirmed.
- If the user asks whether to add a cognitive test, explicitly say whether to add it and name the cognitive test.
- If reranker evidence says confidence is closest_available or weak_match, state the catalog limitation clearly before giving closest matches.
- If missing_exact_terms contains a technology or skill, say:
  "The SHL catalog does not currently include a <term>-specific assessment."
- Then explain that the shortlist covers the closest available assessment signals.
- Keep the answer short.
- Do not include markdown tables.
- Do not output JSON.
""".strip()

    def _build_recommendation_prompt(
        self,
        plan: QueryPlan,
        ranked_products: list[ScoredProduct],
        context=None,
        rerank_metadata=None,
    ) -> str:
        products_text = "\n".join(
            [
                (
                    f"{i + 1}. {item.product.name} | "
                    f"type={item.product.product_type} | "
                    f"keys={', '.join(item.product.keys)} | "
                    f"description={item.product.description[:350]}"
                )
                for i, item in enumerate(ranked_products)
            ]
        )

        context_text = ""
        if context is not None:
            context_text = f"""
Conversation context:
is_follow_up: {context.is_follow_up}
latest_user_request: {context.latest_user_request}
previous_requirements: {context.previous_requirements}
previous_recommendations: {context.previous_recommendations}
user_change_request: {context.user_change_request}
should_end: {context.should_end}
""".strip()

        rerank_text = ""
        if rerank_metadata is not None:
            rerank_text = f"""
Reranker evidence:
confidence: {rerank_metadata.get("confidence", "unknown")}
catalog_gap: {rerank_metadata.get("catalog_gap", "")}
missing_exact_terms: {rerank_metadata.get("missing_exact_terms", [])}
""".strip()

        return f"""
{context_text}

{rerank_text}

Query plan:
intent: {plan.intent}
direct_keywords: {plan.direct_keywords}
related_keywords: {plan.related_keywords}
semantic_query: {plan.semantic_query}

Shortlisted catalog products:
{products_text}

Write the final assistant reply.

Important:
- If confidence is closest_available or weak_match, mention the catalog gap clearly before explaining the closest matches.
- If missing_exact_terms is not empty, explicitly say:
  "The SHL catalog does not currently include a <term>-specific assessment."
- If this is a follow-up, answer the latest user request first.
- Do not restart the conversation.
- Do not repeat the full original recommendation unless useful.
- Explain what changed from the previous shortlist when relevant.
- If the user asks whether to add something, clearly say yes/no and name the product being added.
- Use only the shortlisted catalog products.
""".strip()

    def _fallback_recommendation_reply(
        self,
        plan: QueryPlan,
        ranked_products: list[ScoredProduct],
        context=None,
        rerank_metadata=None,
    ) -> str:
        missing_terms = []
        confidence = "unknown"

        if rerank_metadata is not None:
            missing_terms = rerank_metadata.get("missing_exact_terms", []) or []
            confidence = rerank_metadata.get("confidence", "unknown")

        prefix = ""

        if missing_terms:
            term = str(missing_terms[0]).strip()
            if term:
                prefix = (
                    f"The SHL catalog does not currently include a "
                    f"{term}-specific assessment. "
                )

        if not prefix:
            for keyword in plan.direct_keywords:
                key = keyword.lower().strip()

                if key and not any(key in item.product.name.lower() for item in ranked_products):
                    if key in {
                        "rust",
                        "go",
                        "golang",
                        "typescript",
                        "kotlin",
                        "swift",
                        "scala",
                        "ruby",
                        "php",
                        "react",
                        "angular",
                        "vue",
                        "node.js",
                    }:
                        prefix = (
                            f"The SHL catalog does not currently include a "
                            f"{keyword}-specific assessment. "
                        )
                    break

        if not prefix and confidence in {"closest_available", "weak_match"}:
            prefix = "The shortlisted products are the closest available SHL catalog matches. "

        names = ", ".join(
            item.product.name
            for item in ranked_products[: self.settings.max_recommendations]
        )

        if context is not None and context.is_follow_up:
            return (
                prefix
                + f"Based on the previous context, the updated SHL shortlist is {names}. "
                + "These cover the closest relevant skills and supporting assessment signals."
            )

        return (
            prefix
            + f"The strongest available SHL matches are {names}. "
            + "These cover the closest relevant skills and supporting assessment signals for the role."
        )


def get_response_generator() -> ResponseGenerator:
    return ResponseGenerator()