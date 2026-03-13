"""LLM client wrapper for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        load_dotenv()
        resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        resolved_model = (model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()

        env_base_url = os.getenv("OPENAI_BASE_URL", "")
        candidate_base_url = base_url if base_url is not None else env_base_url
        resolved_base_url = candidate_base_url.strip() or None

        if not resolved_api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY. Set it in your environment or .env file."
            )

        self.model = resolved_model
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    def chat(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        """Send a chat completion request and return text content."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""
