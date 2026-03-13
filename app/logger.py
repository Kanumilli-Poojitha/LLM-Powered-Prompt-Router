"""JSONL logging for prompt routing requests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path("logs/route_log.jsonl")


def log_route_event(event: Dict[str, Any]) -> None:
    """Append one JSON object as a single line for each routing event."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intent": event.get("intent", "unclear"),
        "confidence": float(event.get("confidence", 0.0) or 0.0),
        "user_message": str(event.get("user_message", "")),
        "final_response": str(event.get("final_response", "")),
    }

    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True) + "\n")
