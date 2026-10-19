"""Build a local Qdrant collection and a similarity retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore

from src.retrieval.embeddings import E5Embeddings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@beartype
def build_embeddings(model_name: str) -> E5Embeddings:
    """Load the embedder on CPU so vLLM can keep the GPU."""
    logger.info("loading embeddings %s on CPU", model_name)
    return E5Embeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@beartype
def build_vector_store(
    documents: list[Document],
    embeddings: E5Embeddings,
    path: Path,
    collection_name: str,
) -> QdrantVectorStore:
    """Recreate the local collection from the given documents."""
    path.mkdir(parents=True, exist_ok=True)
    logger.info("indexing %d documents into %s", len(documents), collection_name)
    return QdrantVectorStore.from_documents(
        documents,
        embeddings,
        path=str(path),
        collection_name=collection_name,
        force_recreate=True,
    )


@beartype
def build_retriever(config: dict[str, Any], documents: list[Document]) -> VectorStoreRetriever:
    """Index articles and return a dense similarity retriever."""
    retrieval = config["retrieval"]
    embeddings = build_embeddings(str(retrieval["embedding_model"]))
    store = build_vector_store(
        documents,
        embeddings,
        Path(retrieval["qdrant_path"]),
        str(retrieval["collection_name"]),
    )
    top_k = int(retrieval["top_k"])
    return store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
