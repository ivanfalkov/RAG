"""Load the golden question-answer set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beartype import beartype


@dataclass(frozen=True)
class QAExample:
    """One question aligned with its reference answer."""

    id: str
    question: str
    answer: str


def _read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {index} must be an object")
        records.append(item)
    return records


@beartype
def load_qa(questions_path: Path, answers_path: Path) -> list[QAExample]:
    """Join questions and ground truth on id. Reject duplicates and orphans."""
    questions = _read_records(questions_path)
    answers = _read_records(answers_path)

    answers_by_id: dict[str, str] = {}
    for row in answers:
        example_id = str(row["id"])
        if example_id in answers_by_id:
            raise ValueError(f"duplicate ground-truth id: {example_id}")
        answers_by_id[example_id] = str(row["answer"])

    examples: list[QAExample] = []
    seen: set[str] = set()
    missing: list[str] = []
    for row in questions:
        example_id = str(row["id"])
        if example_id in seen:
            raise ValueError(f"duplicate question id: {example_id}")
        seen.add(example_id)
        answer = answers_by_id.get(example_id)
        if answer is None:
            missing.append(example_id)
            continue
        examples.append(
            QAExample(
                id=example_id,
                question=str(row["question"]),
                answer=answer,
            )
        )

    extra = sorted(set(answers_by_id) - seen)
    if missing or extra:
        raise ValueError(
            "question and ground-truth ids do not match: "
            f"missing answers for {missing}, extra answers for {extra}"
        )
    return examples


@beartype
def load_qa_from_config(config: dict[str, Any]) -> list[QAExample]:
    """Load the golden set using paths from the merged YAML config."""
    paths = config["paths"]
    return load_qa(Path(paths["questions"]), Path(paths["ground_truth"]))
