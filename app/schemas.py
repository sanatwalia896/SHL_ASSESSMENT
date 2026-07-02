from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


Role = Literal["user", "assistant", "system"]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[Message] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reply: str
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    end_of_conversation: bool = False


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Literal[
        "clarify",
        "recommend",
        "refine",
        "compare",
        "refuse",
        "confirm",
    ] = "recommend"

    direct_keywords: list[str] = Field(default_factory=list)
    related_keywords: list[str] = Field(default_factory=list)
    semantic_query: str = ""

    must_include: list[str] = Field(default_factory=list)
    must_exclude: list[str] = Field(default_factory=list)

    needs_clarification: bool = False
    clarification_question: str = ""


class CatalogProduct(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    description: str = ""
    product_type: str = ""
    category: str = ""
    skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    job_levels: list[str] = Field(default_factory=list)
    duration: str | None = None
    duration_minutes: int | None = None
    remote: bool | None = None
    adaptive: bool | None = None
    status: str = ""
    url: str = ""


class ScoredProduct(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product: CatalogProduct
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    source: str = ""