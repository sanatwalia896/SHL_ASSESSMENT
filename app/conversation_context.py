from __future__ import annotations

import json

from groq import Groq

from app.schemas import Message
from app.settings import get_settings


class ConversationContext:
    def __init__(
        self,
        latest_user_request: str = "",
        previous_requirements: list[str] | None = None,
        previous_recommendations: list[str] | None = None,
        user_change_request: str = "",
        is_follow_up: bool = False,
        should_end: bool = False,
    ) -> None:
        self.latest_user_request = latest_user_request
        self.previous_requirements = previous_requirements or []
        self.previous_recommendations = previous_recommendations or []
        self.user_change_request = user_change_request
        self.is_follow_up = is_follow_up
        self.should_end = should_end


class ConversationContextBuilder:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.groq_api_key) if self.settings.groq_api_key else None
    def _looks_like_final_confirmation(self, text: str) -> bool:
        text = text.lower().strip()
        confirmation_terms = [
        "confirmed",
        "confirm",
        "lock it in",
        "that works",
        "that's good",
        "perfect",
        "thanks",
        "thank you",
        "that covers it",
        "final list",
        ]
        return any(term in text for term in confirmation_terms)

    def build(self, messages: list[Message]) -> ConversationContext:
        latest_user = self._latest_user(messages)

    # First turn: no previous context exists.
        if len(messages) <= 1:
            return ConversationContext(
            latest_user_request=latest_user,
            previous_requirements=[],
            previous_recommendations=[],
            user_change_request="",
            is_follow_up=False,
            should_end=False,
        )

        if self.client is None:
            return ConversationContext(
            latest_user_request=latest_user,
            previous_requirements=[],
            previous_recommendations=[],
            user_change_request=latest_user,
            is_follow_up=True,
            should_end=self._looks_like_final_confirmation(latest_user),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                temperature=0,
                messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._history_text(messages)},
                ],
            )

            content = response.choices[0].message.content or "{}"
            data = self._parse_json(content)

            return ConversationContext(
            latest_user_request=data.get("latest_user_request", latest_user),
            previous_requirements=data.get("previous_requirements", []),
            previous_recommendations=data.get("previous_recommendations", []),
            user_change_request=data.get("user_change_request", latest_user),
            is_follow_up=bool(data.get("is_follow_up", True)),
            should_end=bool(data.get("should_end", False)),
            )

        except Exception:
            return ConversationContext(
            latest_user_request=latest_user,
            previous_requirements=[],
            previous_recommendations=[],
            user_change_request=latest_user,
            is_follow_up=True,
            should_end=self._looks_like_final_confirmation(latest_user),
            )
    def _system_prompt(self) -> str:
        return """
You analyze a stateless SHL recommender conversation.

Return ONLY valid JSON.

Extract:
{
  "latest_user_request": "...",
  "previous_requirements": ["..."],
  "previous_recommendations": ["..."],
  "user_change_request": "...",
  "is_follow_up": true,
  "should_end": false
}

Rules:
- Use the entire conversation history.
- previous_requirements should include role, seniority, skills, language, location, constraints, and earlier decisions.
- previous_recommendations should include product names already recommended by the assistant.
- user_change_request should capture what the latest user wants changed, added, removed, compared, or confirmed.
- should_end is true only if the latest user clearly finalizes the shortlist, such as "confirmed", "that works", "lock it in", "final list", or "thanks".
""".strip()

    def _history_text(self, messages: list[Message]) -> str:
        return "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in messages
        )

    def _latest_user(self, messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

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


def get_conversation_context_builder() -> ConversationContextBuilder:
    return ConversationContextBuilder()