"""Embedding package module exports."""

from src.embedding.embedder import embed_chunks, embed_query, get_model
from src.embedding.faiss_store import (
    build_index,
    save_index,
    load_index,
    search,
    get_or_create_index,
)

__all__ = [
    "embed_chunks",
    "embed_query",
    "get_model",
    "build_index",
    "save_index",
    "load_index",
    "search",
    "get_or_create_index",
]
