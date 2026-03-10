"""Shared OpenAI and rendering helpers for the agentic modules."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from services.rendering import render_markdown

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Lazily initialize a shared OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
) -> str:
    """Run a chat completion and return plain text content."""
    params: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if response_format is not None:
        params["response_format"] = response_format
    if seed is not None:
        params["seed"] = seed

    response = get_openai_client().chat.completions.create(**params)
    return response.choices[0].message.content


def call_text_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int | None = 800,
) -> str:
    """Run a standard text completion."""
    return chat_completion(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def call_json_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int | None = 500,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a JSON-formatted completion and parse the result."""
    raw = chat_completion(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        seed=seed,
    )
    return json.loads(raw)


def render_formats(raw_text: str) -> tuple[str, str]:
    """Return HTML and plain-text variants for model output."""
    html = render_markdown(raw_text, tables=True)
    plain = re.sub(r"<[^>]+>", "", html)
    return html, plain
