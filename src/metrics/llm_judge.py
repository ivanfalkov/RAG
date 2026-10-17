"""LLM-as-judge evaluation of QA answers via Groq."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from beartype import beartype
from openai import OpenAI

from src.utils.logging_utils import get_logger
from src.utils.rate_limit import DailyRateLimitExceeded, is_daily_rate_limit

logger = get_logger(__name__)


def _truncate(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class Judge:
    def __init__(self, config: dict[str, Any]) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        cfg = config["judge"]
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            max_retries=0,
        )
        self.model: str = cfg["model"]
        self.max_retries: int = cfg.get("max_retries", 3)
        self.sleep_between: float = cfg.get("sleep_between", 4.0)
        self.temperature: float = cfg.get("temperature", 0.0)
        self.max_tokens: int = cfg.get("max_tokens", 2048)
        self.reasoning_effort: str = cfg.get("reasoning_effort", "low")
        self.preview_chars: int = cfg.get("question_preview_chars", 200)
        self.score_min: int = cfg.get("score_min", 1)
        self.score_max: int = cfg.get("score_max", 5)
        self.metrics: dict[str, str] = cfg["metrics"]
        self.metric_names: list[str] = list(self.metrics)

    def _prompt(self, question: str, reference: str, prediction: str) -> str:
        criteria = "\n".join(f"- {name}: {desc}" for name, desc in self.metrics.items())
        example = {
            name: f"<int {self.score_min}-{self.score_max}>"
            for name in self.metric_names
        }
        example["reasoning"] = "<short explanation>"
        return (
            "You are evaluating an answer to a question about a corporate knowledge base.\n"
            "Score only against the question and the reference answer. "
            "Do not use outside knowledge. "
            "If the candidate says it does not know while the reference contains the answer, "
            "correctness and completeness are 1.\n\n"
            f"Rate the candidate on a scale of {self.score_min}-{self.score_max} "
            "for each criterion:\n\n"
            f"{criteria}\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference}\n\n"
            f"Candidate answer:\n{prediction}\n\n"
            "IMPORTANT: Respond with ONLY a valid JSON object. Do NOT include any reasoning, "
            "explanations, or markdown formatting. Your entire response must be parseable as JSON.\n\n"
            f"{json.dumps(example)}"
        )

    def _json_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            name: {
                "type": "integer",
                "minimum": self.score_min,
                "maximum": self.score_max,
            }
            for name in self.metric_names
        }
        properties["reasoning"] = {"type": "string"}
        required = [*self.metric_names, "reasoning"]
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _failed_result(self) -> dict[str, Any]:
        result: dict[str, Any] = {name: None for name in self.metric_names}
        result["reasoning"] = "failed"
        return result

    def _call_api(self, question: str, reference: str, prediction: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self._prompt(question, reference, prediction),
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_scores",
                    "strict": True,
                    "schema": self._json_schema(),
                },
            },
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        content = response.choices[0].message.content
        return json.loads(content)

    @beartype
    def score(self, question: str, reference: str, prediction: str) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                return self._call_api(question, reference, prediction)
            except Exception as e:
                if is_daily_rate_limit(e):
                    raise DailyRateLimitExceeded(str(e)) from e
                logger.warning(
                    "judge call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.sleep_between * (2**attempt))
        return self._failed_result()


@beartype
def make_record(
    row: Any,
    model_name: str,
    index: int,
    result: dict[str, Any],
    metric_names: list[str],
) -> dict[str, Any]:
    row_id = row.get("id", index)
    if hasattr(row_id, "item"):
        row_id = row_id.item()
    record: dict[str, Any] = {
        "id": row_id,
        "model": model_name,
        "question": row["question"],
        "reference": row["reference"],
        "prediction": row["prediction"],
    }
    for name in metric_names:
        record[name] = result.get(name)
    record["reasoning"] = result.get("reasoning", "")
    return record


@beartype
def log_record(
    record: dict[str, Any],
    index: int,
    total: int,
    metric_names: list[str],
    preview_chars: int,
) -> None:
    scores = " ".join(f"{name}={record.get(name)}" for name in metric_names)
    logger.info(
        "\n--- [%s] row %d/%d ---\nQUESTION:   %s\nREFERENCE:  %s\nPREDICTION: %s\nSCORES:     %s\nREASONING:  %s",
        record["model"],
        index,
        total,
        _truncate(str(record["question"]), preview_chars),
        str(record["reference"]).strip(),
        str(record["prediction"]).strip(),
        scores,
        str(record.get("reasoning") or "").strip(),
    )


@beartype
def summarize(
    records: list[dict[str, Any]], metric_names: list[str]
) -> dict[str, float | int]:
    scored = [
        r for r in records if all(r.get(name) is not None for name in metric_names)
    ]
    n_scored = len(scored)

    averages: dict[str, float | int] = {}
    for name in metric_names:
        averages[f"judge_{name}"] = (
            sum(r[name] for r in scored) / n_scored if n_scored else 0.0
        )
    averages["n_scored"] = n_scored
    averages["n_failed"] = len(records) - n_scored
    return averages
