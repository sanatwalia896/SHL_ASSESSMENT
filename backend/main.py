from __future__ import annotations

from fastapi import FastAPI

from app.schemas import ChatRequest, ChatResponse, RecommendationItem
from backend.dependencies import planner, retriever, ranker, responder

app = FastAPI(title="SHL Assessment Recommender")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    plan = planner.plan(request.messages)

    if plan.intent == "refuse":
        return ChatResponse(
            reply="I can only help with SHL assessment selection from the SHL catalog.",
            recommendations=[],
            end_of_conversation=False,
        )

    if plan.needs_clarification or plan.intent == "clarify":
        return ChatResponse(
            reply=plan.clarification_question
            or "What role or hiring situation should I match SHL assessments to?",
            recommendations=[],
            end_of_conversation=False,
        )

    if plan.intent == "confirm":
        return ChatResponse(
            reply="Glad this works. The shortlist above is confirmed.",
            recommendations=[],
            end_of_conversation=True,
        )

    candidates = retriever.retrieve(plan)
    ranked = ranker.rank(candidates, plan)

    reply = responder.write_recommendation_reply(plan, ranked)

    recommendations = [
        RecommendationItem(
            name=item.product.name,
            url=item.product.url,
            test_type=",".join(item.product.keys) if item.product.keys else item.product.product_type,
        )
        for item in ranked
    ]

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=False,
    )