"""Shared zero-shot run: config snapshot, predictions, ROUGE/BLEU."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from beartype import beartype

from src.data.qa import QAExample
from src.metrics.compute_metrics import compute_all
from src.utils import artifacts
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@beartype
def run_zero_shot(
    config: dict[str, Any],
    examples: list[QAExample],
    predict: Callable[[list[QAExample]], list[str]],
) -> Path:
    """Run one zero-shot experiment and return the artifact directory."""
    experiment_name = str(config["experiment_name"])
    logger.info("starting experiment: %s (%d examples)", experiment_name, len(examples))

    run_dir = artifacts.create_run_dir(Path(config["paths"]["artifacts_dir"]), experiment_name)
    artifacts.save_config(config, run_dir / "config.json")
    artifacts.save_jsonl(
        [
            {"id": item.id, "question": item.question, "answer": item.answer}
            for item in examples
        ],
        run_dir / "data.jsonl",
    )

    predictions = predict(examples)
    if len(predictions) != len(examples):
        raise ValueError(
            f"predictor returned {len(predictions)} rows for {len(examples)} questions"
        )

    artifacts.save_jsonl(
        [
            {
                "id": item.id,
                "question": item.question,
                "reference": item.answer,
                "prediction": prediction,
            }
            for item, prediction in zip(examples, predictions, strict=True)
        ],
        run_dir / "predictions.jsonl",
    )

    metrics = compute_all(predictions, [item.answer for item in examples])
    artifacts.save_metrics(metrics, run_dir / "metrics.json")
    logger.info("[%s] %s", experiment_name, metrics)
    return run_dir
