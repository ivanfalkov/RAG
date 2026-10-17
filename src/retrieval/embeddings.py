"""E5 embeddings with the prefixes the model was trained on."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


class E5Embeddings(HuggingFaceEmbeddings):
    """multilingual-e5 expects 'passage: ' for documents and 'query: ' for questions."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {text}" for text in texts]
        vectors: list[list[float]] = super().embed_documents(prefixed)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector: list[float] = super().embed_query(f"query: {text}")
        return vector
