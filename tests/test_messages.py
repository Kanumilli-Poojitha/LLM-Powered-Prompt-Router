"""Basic tests and sample message coverage for the prompt router."""

from __future__ import annotations

from app.classifier import DEFAULT_INTENT, classify_intent
from app.router import CLARIFICATION_QUESTION, route_and_respond


class DummyLLM:
    """Small test double to avoid real API calls in unit tests."""

    def __init__(self, response: str) -> None:
        self.response = response

    def chat(self, system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
        return self.response


TEST_MESSAGES = [
    "how do i sort a list of objects in python?",
    "explain this sql query for me",
    "This paragraph sounds awkward, can you help me fix it?",
    "I'm preparing for a job interview, any tips?",
    "what's the average of these numbers: 12, 45, 23, 67, 34",
    "Help me make this better.",
    "I need to write a function that takes a user id and returns their profile, but also i need help with my resume.",
    "hey",
    "Can you write me a poem about clouds?",
    "Rewrite this sentence to be more professional.",
    "I'm not sure what to do with my career.",
    "what is a pivot table",
    "fxi thsi bug pls: for i in range(10) print(i)",
    "How do I structure a cover letter?",
    "My boss says my writing is too verbose.",
]


def test_at_least_fifteen_messages_present() -> None:
    assert len(TEST_MESSAGES) >= 15


def test_classifier_handles_malformed_json() -> None:
    llm = DummyLLM("this is not json")
    result = classify_intent("hello", llm_client=llm)  # type: ignore[arg-type]
    assert result == DEFAULT_INTENT


def test_classifier_parses_valid_json() -> None:
    llm = DummyLLM('{"intent": "code", "confidence": 0.92}')
    result = classify_intent("fix my bug", llm_client=llm)  # type: ignore[arg-type]
    assert result["intent"] == "code"
    assert result["confidence"] == 0.92


def test_router_asks_clarification_for_unclear() -> None:
    response = route_and_respond("help", {"intent": "unclear", "confidence": 0.1})
    assert response == CLARIFICATION_QUESTION


def test_router_asks_clarification_for_low_confidence() -> None:
    response = route_and_respond("help", {"intent": "code", "confidence": 0.2})
    assert response == CLARIFICATION_QUESTION


def test_router_uses_persona_for_high_confidence_intent() -> None:
    llm = DummyLLM("Use try/except and return structured errors.")
    response = route_and_respond(
        "fix my python function",
        {"intent": "code", "confidence": 0.95},
        llm_client=llm,  # type: ignore[arg-type]
    )
    assert "try/except" in response
