"""Intent classification logic for prompt routing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from app.llm_client import LLMClient
from app.prompts import ALLOWED_INTENTS, CLASSIFIER_SYSTEM_PROMPT

DEFAULT_INTENT: Dict[str, Any] = {"intent": "unclear", "confidence": 0.0}


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON from a raw model response safely."""
    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_intent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize classifier payload into target schema."""
    intent = str(payload.get("intent", "unclear")).strip().lower()

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if intent not in ALLOWED_INTENTS:
        intent = "unclear"

    confidence = max(0.0, min(1.0, confidence))

    return {"intent": intent, "confidence": confidence}


def classify_intent(message: str, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """Classify user intent using an LLM and return structured JSON data."""
    if not message or not message.strip():
        return DEFAULT_INTENT.copy()

    client = llm_client or LLMClient()

    try:
        raw = client.chat(system_prompt=CLASSIFIER_SYSTEM_PROMPT, user_message=message)
    except Exception:
        return DEFAULT_INTENT.copy()

    parsed = _extract_json_object(raw)
    if parsed is None:
        return DEFAULT_INTENT.copy()

    return _normalize_intent_payload(parsed)
