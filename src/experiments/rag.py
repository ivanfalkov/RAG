"""Vanilla RAG: retrieve two articles, then generate with vLLM."""

from __future__ import annotations

from typing import Any

from beartype import beartype
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langsmith import traceable

from src.data.qa import QAExample
from src.models.vllm_generator import VllmGenerator
from src.preprocessing.prompts import render_prompt
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _format_context(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


class RagPipeline:
    """One dense lookup per question, then one batched vLLM call."""

    def __init__(
        self,
        config: dict[str, Any],
        retriever: VectorStoreRetriever,
        generator: VllmGenerator,
    ) -> None:
        self._template = str(config["prompt"])
        self._retriever = retriever
        self._generator = generator

    @traceable(name="retrieve_article", run_type="retriever")
    def retrieve_context(self, question: str) -> str:
        """Return the text of the nearest articles."""
        documents = self._retriever.invoke(question)
        titles = [str(document.metadata.get("title", "")) for document in documents]
        logger.info("retrieved %s", titles)
        return _format_context(documents)

    @traceable(name="rag_local", run_type="chain")
    @beartype
    def predict(self, examples: list[QAExample]) -> list[str]:
        """Answer every question from retrieved context only."""
        messages = [
            render_prompt(
                self._template,
                context=self.retrieve_context(item.question),
                question=item.question,
            )
            for item in examples
        ]
        return self._generator.complete(messages)
