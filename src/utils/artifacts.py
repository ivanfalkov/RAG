"""Local artifact I/O."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from beartype import beartype

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@beartype
def create_run_dir(root: Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("created run dir: %s", run_dir)
    return run_dir


@beartype
def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("saved json: %s", path)


@beartype
def save_jsonl(records: list[dict[str, str]], path: Path) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("saved jsonl: %d rows -> %s", len(records), path)


@beartype
def save_metrics(metrics: dict[str, float], path: Path) -> None:
    save_json(metrics, path)
    logger.info("saved metrics artifact -> %s", path)


@beartype
def save_config(config: dict[str, Any], path: Path) -> None:
    save_json(config, path)
    logger.debug("saved config artifact -> %s", path)
