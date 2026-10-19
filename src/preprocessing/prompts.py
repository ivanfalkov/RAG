"""Prompt rendering shared by the API and local generators."""

from __future__ import annotations

from beartype import beartype


@beartype
def render_prompt(template: str, **fields: str) -> str:
    """Fill a prompt template. The result is one user turn, not a raw completion."""
    return template.format(**fields).strip()


@beartype
def render_user_message(template: str, question: str) -> str:
    """Fill a zero-shot prompt that has a single {question} field."""
    return render_prompt(template, question=question)
