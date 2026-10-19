"""Zero-shot generation through the Groq OpenAI-compatible API."""

from __future__ import annotations

import os
import time
from typing import Any

from beartype import beartype
from openai import OpenAI

from src.data.qa import QAExample
from src.preprocessing.prompts import render_user_message
from src.utils.logging_utils import get_logger
from src.utils.rate_limit import DailyRateLimitExceeded, is_daily_rate_limit

logger = get_logger(__name__)


class GroqGenerator:
    """Sequential chat completions. Instruct mode keeps reasoning tokens out of the answer."""

    def __init__(self, config: dict[str, Any]) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")

        model_cfg = config["model"]
        gen_cfg = config["generation"]
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            max_retries=0,
        )
        self._model = str(model_cfg["name"])
        self._reasoning_effort = str(model_cfg.get("reasoning_effort", "none"))
        self._reasoning_format = str(model_cfg.get("reasoning_format", "hidden"))
        self._temperature = float(gen_cfg["temperature"])
        self._max_tokens = int(gen_cfg["max_tokens"])
        self._sleep_between = float(gen_cfg.get("sleep_between", 0.0))
        self._max_retries = int(gen_cfg.get("max_retries", 3))
        self._template = str(config["prompt"])

    def _complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={
                "reasoning_effort": self._reasoning_effort,
                "reasoning_format": self._reasoning_format,
            },
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    @beartype
    def predict(self, examples: list[QAExample]) -> list[str]:
        """Generate one short answer per question, retrying transient Groq errors."""
        predictions: list[str] = []
        total = len(examples)
        for index, item in enumerate(examples, start=1):
            prompt = render_user_message(self._template, item.question)
            answer = self._complete_with_retry(prompt, item.id)
            predictions.append(answer)
            logger.info("predicted %d/%d id=%s", index, total, item.id)
            if self._sleep_between > 0 and index < total:
                time.sleep(self._sleep_between)
        return predictions

    def _complete_with_retry(self, prompt: str, example_id: str) -> str:
        for attempt in range(self._max_retries):
            try:
                return self._complete(prompt)
            except Exception as exc:
                if is_daily_rate_limit(exc):
                    raise DailyRateLimitExceeded(str(exc)) from exc
                logger.warning(
                    "groq call failed for %s (attempt %d/%d): %s",
                    example_id,
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(self._sleep_between * (2**attempt) or 1.0)
        raise RuntimeError(f"groq retries exhausted for {example_id}")
