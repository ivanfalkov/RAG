"""Prompt rendering shared by the API and local generators."""

from __future__ import annotations

from beartype import beartype


@beartype
def render_user_message(template: str, question: str) -> str:
    """Fill the experiment prompt. The result is one user turn, not a raw completion."""
    return template.format(question=question).strip()
