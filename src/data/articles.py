"""Load knowledge-base articles as one document each."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from langchain_core.documents import Document


def _read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"{path} must be a JSON list")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"{path} item {index} must be an object")
        records.append(item)
    return records


@beartype
def load_articles(path: Path) -> list[Document]:
    """Turn each article into one document. The texts are already short."""
    documents: list[Document] = []
    seen: set[str] = set()
    for row in _read_records(path):
        article_id = str(row["id"])
        if article_id in seen:
            raise ValueError(f"duplicate article id: {article_id}")
        seen.add(article_id)
        title = str(row["title"]).strip()
        text = str(row["text"]).strip()
        documents.append(
            Document(
                page_content=f"{title}\n\n{text}",
                metadata={"id": article_id, "title": title},
            )
        )
    if not documents:
        raise ValueError(f"no articles in {path}")
    return documents


@beartype
def load_articles_from_config(config: dict[str, Any]) -> list[Document]:
    """Load articles using the path from the merged YAML config."""
    return load_articles(Path(config["paths"]["articles"]))
