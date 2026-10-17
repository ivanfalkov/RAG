"""Entry point: vanilla RAG with local Qdrant and Qwen2.5-1.5B via vLLM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.data.articles import load_articles_from_config
from src.data.qa import load_qa_from_config
from src.experiments.rag import RagPipeline
from src.experiments.runner import predict_evaluate_save_artifacts
from src.models.vllm_generator import VllmGenerator
from src.retrieval.index import build_retriever
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/general_config.yaml"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/rag_local.yaml"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    config = {**load_config(args.config), **load_config(args.experiment_config)}
    examples = load_qa_from_config(config)
    documents = load_articles_from_config(config)
    retriever = build_retriever(config, documents)
    generator = VllmGenerator(config)
    pipeline = RagPipeline(config, retriever, generator)
    run_dir = predict_evaluate_save_artifacts(config, examples, pipeline.predict)
    logging.getLogger(__name__).info("artifacts: %s", run_dir)


if __name__ == "__main__":
    main()
