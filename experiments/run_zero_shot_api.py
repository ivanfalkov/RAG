"""Entry point: zero-shot qwen/qwen3.8-27b via Groq."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.data.qa import load_qa_from_config
from src.experiments.zero_shot import run_zero_shot
from src.models.groq_generator import GroqGenerator
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/general_config.yaml"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/zero_shot_api.yaml"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    config = {**load_config(args.config), **load_config(args.experiment_config)}
    examples = load_qa_from_config(config)
    generator = GroqGenerator(config)
    run_dir = run_zero_shot(config, examples, generator.predict)
    logging.getLogger(__name__).info("artifacts: %s", run_dir)


if __name__ == "__main__":
    main()
