"""FastAPI entrypoint for the LLM prompt router service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.classifier import classify_intent
from app.logger import log_route_event
from app.router import route_and_respond

app = FastAPI(title="LLM Prompt Router", version="1.0.0")


class ChatRequest(BaseModel):
    """Request payload for chat routing."""

    message: str = Field(..., min_length=1, description="User message to classify and route")


class ChatResponse(BaseModel):
    """Response payload with intent metadata and generated output."""

    intent: str
    confidence: float
    response: str


@app.get("/health")
def health() -> dict[str, str]:
    """Health probe endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Classify, route, respond, and log each user request."""
    intent_data = classify_intent(request.message)
    final_response = route_and_respond(request.message, intent_data)

    log_route_event(
        {
            "intent": intent_data.get("intent", "unclear"),
            "confidence": intent_data.get("confidence", 0.0),
            "user_message": request.message,
            "final_response": final_response,
        }
    )

    return ChatResponse(
        intent=str(intent_data.get("intent", "unclear")),
        confidence=float(intent_data.get("confidence", 0.0) or 0.0),
        response=final_response,
    )
