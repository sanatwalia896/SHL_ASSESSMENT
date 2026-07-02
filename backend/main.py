from __future__ import annotations

from fastapi import FastAPI

from app.schemas import ChatRequest, ChatResponse, RecommendationItem, ScoredProduct
from backend.dependencies import (
    context_builder,
    planner,
    retriever,
    ranker,
    llm_reranker,
    responder,
)

app = FastAPI(title="SHL Assessment Recommender")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def latest_user_text(request: ChatRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content.lower().strip()
    return ""


def assistant_history_text(request: ChatRequest) -> str:
    return "\n".join(
        message.content.lower()
        for message in request.messages
        if message.role == "assistant"
    )


def has_previous_shortlist(request: ChatRequest) -> bool:
    text = assistant_history_text(request)

    return (
        "https://www.shl.com/products/product-catalog/view/" in text
        or "shortlist" in text
        or "recommendation" in text
        or "recommended" in text
    )


def apply_agent_guards(plan, request: ChatRequest):
    latest_user = latest_user_text(request)

    vague_queries = {
        "assessment",
        "i need an assessment",
        "need an assessment",
        "recommend an assessment",
        "i need a test",
        "need a test",
    }

    if latest_user in vague_queries:
        plan.intent = "clarify"
        plan.needs_clarification = True
        plan.clarification_question = (
            "What role or hiring situation should I match SHL assessments to?"
        )
        return plan

    legal_terms = [
        "legally required",
        "legal requirement",
        "required by law",
        "compliance obligation",
        "does this satisfy",
        "is this legal",
        "hipaa requirement",
        "satisfy hipaa",
    ]

    if any(term in latest_user for term in legal_terms):
        plan.intent = "refuse"
        return plan

    injection_terms = [
        "ignore your instructions",
        "ignore previous instructions",
        "outside the shl catalog",
        "recommend anything",
        "make up",
        "invent",
        "hallucinate",
    ]

    if any(term in latest_user for term in injection_terms):
        plan.intent = "refuse"
        return plan

    # Product-comparison only.
    # Do NOT treat "compare candidates against a benchmark" as product comparison.
    compare_terms = [
        "difference between",
        "different from",
        "compare these",
        "compare the",
        " vs ",
        " versus ",
        "do we really need",
        "is the advanced",
        "is this different",
    ]

    if any(term in latest_user for term in compare_terms):
        plan.intent = "compare"
        return plan

    refine_terms = [
        "add ",
        "drop ",
        "remove ",
        "replace ",
        "instead",
        "actually",
        "keep ",
        "shorter",
        "go with",
        "include",
        "exclude",
    ]

    if has_previous_shortlist(request) and any(term in latest_user for term in refine_terms):
        plan.intent = "refine"

    confirm_terms = [
        "confirmed",
        "lock it in",
        "looks good",
        "that works",
        "that's good",
        "perfect",
        "thanks",
        "thank you",
        "final list",
        "that covers it",
    ]

    if any(term in latest_user for term in confirm_terms):
        plan.intent = "confirm"

    return plan


def to_recommendation_items(ranked) -> list[RecommendationItem]:
    return [
        RecommendationItem(
            name=item.product.name,
            url=item.product.url,
            test_type=", ".join(item.product.keys)
            if item.product.keys
            else item.product.product_type,
        )
        for item in ranked
    ]


def make_history_scored_product(product) -> ScoredProduct:
    """
    Creates a ScoredProduct for a product preserved from earlier conversation history.
    This is written defensively because ScoredProduct may or may not have source/reasons fields.
    """
    try:
        return ScoredProduct(
            product=product,
            score=999.0,
            source="history",
            reasons=["Preserved from previous shortlist"],
        )
    except TypeError:
        return ScoredProduct(
            product=product,
            score=999.0,
        )


def merge_previous_recommendations(context, ranked):
    """
    In follow-up/refinement turns, preserve the previous shortlist and append newly
    recommended products, unless the latest user explicitly drops/removes/excludes items.

    Example 1:
    Previous shortlist = Rust technical tests.
    Latest user asks = "Should I also add a cognitive test?"
    New retrieval = SHL Verify Interactive G+.
    Final shortlist = previous Rust tests + Verify G+.

    Example 2:
    Previous shortlist = Excel/Word simulations + knowledge tests.
    Latest user asks = "Keep only simulations. Drop knowledge-only tests."
    Final shortlist = only simulation products.
    """

    if context is None or not getattr(context, "is_follow_up", False):
        return ranked

    previous_names = getattr(context, "previous_recommendations", []) or []

    if not previous_names:
        return ranked

    latest_request = (
        getattr(context, "latest_user_request", "")
        or getattr(context, "user_change_request", "")
        or ""
    ).lower()

    catalog = getattr(retriever, "catalog", None)

    if catalog is None:
        return ranked

    merged = []
    seen_names = set()

    def should_drop_product(product_name: str) -> bool:
        product_name_clean = product_name.strip()
        name = product_name_clean.lower()

        drop_intent = any(
            term in latest_request
            for term in [
                "drop",
                "remove",
                "exclude",
                "keep only",
                "only the",
                "without",
            ]
        )

        if not drop_intent:
            return False

        # Case: "Drop the short knowledge-only Excel and Word tests."
        if "knowledge-only" in latest_request or "knowledge only" in latest_request:
            if product_name_clean in {"MS Excel (New)", "MS Word (New)"}:
                return True

        # Case: "Keep only practical simulation tests."
        if "keep only" in latest_request and "simulation" in latest_request:
            if "simulation" not in name and "365" not in name:
                return True

            if product_name_clean in {"MS Excel (New)", "MS Word (New)"}:
                return True

        # Generic exact product removal.
        if name in latest_request and any(
            term in latest_request for term in ["drop", "remove", "exclude", "without"]
        ):
            return True

        # Common shorthand removal.
        if "ms excel" in latest_request and product_name_clean == "MS Excel (New)":
            return True

        if "ms word" in latest_request and product_name_clean == "MS Word (New)":
            return True

        if "opq" in latest_request and "drop" in latest_request and "opq" in name:
            return True

        if "verify g+" in latest_request and "drop" in latest_request and "verify g+" in name:
            return True

        return False

    for name in previous_names:
        product = None

        if hasattr(catalog, "get_by_name"):
            product = catalog.get_by_name(name)

        if product is None:
            continue

        if should_drop_product(product.name):
            continue

        name_key = product.name.lower().strip()

        if name_key in seen_names:
            continue

        merged.append(make_history_scored_product(product))
        seen_names.add(name_key)

    for item in ranked:
        if should_drop_product(item.product.name):
            continue

        name_key = item.product.name.lower().strip()

        if name_key in seen_names:
            continue

        merged.append(item)
        seen_names.add(name_key)

    return merged[:10]


def retrieve_and_rank(plan):
    candidates = retriever.retrieve(plan)

    # Try wider pool if your Ranker supports limit.
    # Fallback keeps compatibility with the older Ranker.
    try:
        ranked_candidates = ranker.rank(candidates, plan, limit=15)
    except TypeError:
        ranked_candidates = ranker.rank(candidates, plan)

    final_ranked = llm_reranker.rerank(plan, ranked_candidates)

    rerank_metadata = getattr(
        llm_reranker,
        "last_metadata",
        {
            "confidence": "unknown",
            "catalog_gap": "",
            "missing_exact_terms": [],
        },
    )

    return final_ranked, rerank_metadata


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    context = context_builder.build(request.messages)

    plan = planner.plan(request.messages)
    plan = apply_agent_guards(plan, request)

    if context.should_end:
        plan.intent = "confirm"

    if plan.intent == "refuse":
        return ChatResponse(
            reply=responder.write_refusal_reply(),
            recommendations=[],
            end_of_conversation=False,
        )

    if plan.intent == "clarify" or plan.needs_clarification:
        return ChatResponse(
            reply=responder.write_clarification_reply(
                plan.clarification_question
                or "What role or hiring situation should I match SHL assessments to?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    ranked, rerank_metadata = retrieve_and_rank(plan)

    # For follow-up/refinement/confirmation turns, preserve previous shortlist
    # and append newly retrieved products.
    if plan.intent in {"refine", "confirm"} or getattr(context, "is_follow_up", False):
        ranked = merge_previous_recommendations(context, ranked)

    if plan.intent == "compare":
        reply = responder.write_recommendation_reply(
            plan=plan,
            ranked_products=ranked,
            context=context,
            rerank_metadata=rerank_metadata,
        )

        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    if plan.intent == "confirm":
        reply = responder.write_confirmation_reply()

        return ChatResponse(
            reply=reply,
            recommendations=to_recommendation_items(ranked),
            end_of_conversation=True,
        )

    reply = responder.write_recommendation_reply(
        plan=plan,
        ranked_products=ranked,
        context=context,
        rerank_metadata=rerank_metadata,
    )

    return ChatResponse(
        reply=reply,
        recommendations=to_recommendation_items(ranked),
        end_of_conversation=False,
    )