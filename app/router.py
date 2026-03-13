"""Routing logic for selecting specialized personas and generating responses."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.llm_client import LLMClient
from app.prompts import PERSONA_PROMPTS

CLARIFICATION_QUESTION = (
    "Could you clarify your request? Are you asking about coding, data analysis, "
    "writing improvement, or career advice?"
)


def route_and_respond(
    message: str,
    intent_data: Dict[str, Any],
    llm_client: Optional[LLMClient] = None,
    confidence_threshold: float = 0.7,
) -> str:
    """Route to persona prompt and return generated response text."""
    intent = str(intent_data.get("intent", "unclear")).strip().lower()
    confidence = float(intent_data.get("confidence", 0.0) or 0.0)

    if intent == "unclear" or confidence < confidence_threshold:
        return CLARIFICATION_QUESTION

    system_prompt = PERSONA_PROMPTS.get(intent)
    if not system_prompt:
        return CLARIFICATION_QUESTION

    client = llm_client or LLMClient()

    try:
        return client.chat(system_prompt=system_prompt, user_message=message)
    except Exception:
        return "I hit a temporary issue while generating a response. Please try again in a moment."
