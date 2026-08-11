"""FAISS vector store manager for creating, persisting, and querying per-repo indices."""

import os
import json
from typing import Any
import numpy as np
import faiss

from config import CACHE_DIR, TOP_K_RETRIEVAL
from src.embedding.embedder import embed_chunks


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a FAISS L2 index from a 2D numpy embedding array.
    
    Args:
        embeddings: 2D numpy array of shape (num_vectors, dim) and dtype float32.
        
    Returns:
        Populated faiss.IndexFlatL2 instance.
    """
    if embeddings.size == 0:
        raise ValueError("Cannot build FAISS index from empty embeddings array.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(repo_name: str, index: faiss.Index, chunks: list[dict[str, Any]]) -> tuple[str, str]:
    """Save FAISS binary index and chunk metadata JSON to data/cache/.
    
    Args:
        repo_name: Identifier for the repository (e.g. 'psf_requests').
        index: Populated FAISS index.
        chunks: List of chunk metadata dicts in exact order matching index vectors.
        
    Returns:
        Tuple of (index_file_path, metadata_file_path).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    index_path = os.path.join(CACHE_DIR, f"{repo_name}.index")
    meta_path = os.path.join(CACHE_DIR, f"{repo_name}_metadata.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return index_path, meta_path


def load_index(repo_name: str) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load FAISS index and chunk metadata list from data/cache/.
    
    Args:
        repo_name: Identifier for the repository.
        
    Returns:
        Tuple of (faiss.Index, list of chunk dicts).
    """
    index_path = os.path.join(CACHE_DIR, f"{repo_name}.index")
    meta_path = os.path.join(CACHE_DIR, f"{repo_name}_metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"FAISS index or metadata not found for '{repo_name}' in {CACHE_DIR}.")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def search(index: faiss.Index, query_vector: np.ndarray, k: int = TOP_K_RETRIEVAL) -> list[int]:
    """Perform L2 similarity search and return top-k nearest chunk indices.
    
    Args:
        index: FAISS index instance.
        query_vector: 2D numpy vector array of shape (1, dim) and dtype float32.
        k: Top-k nearest neighbors to retrieve.
        
    Returns:
        List of integer indices corresponding to position in chunk metadata list.
    """
    if query_vector.ndim == 1:
        query_vector = np.expand_dims(query_vector, axis=0)

    distances, indices = index.search(query_vector, k)
    return indices[0].tolist()


def get_or_create_index(repo_name: str, chunks: list[dict[str, Any]]) -> tuple[faiss.Index, list[dict[str, Any]], np.ndarray | None]:
    """Check cache for existing FAISS index & metadata, loading if present, or building & saving if missing.
    
    Returns:
        Tuple of (index, chunks, embeddings_or_None).
    """
    index_path = os.path.join(CACHE_DIR, f"{repo_name}.index")
    meta_path = os.path.join(CACHE_DIR, f"{repo_name}_metadata.json")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print(f"Loading cached FAISS index and metadata for '{repo_name}'...")
        index, cached_chunks = load_index(repo_name)
        return index, cached_chunks, None
    else:
        print(f"Embedding {len(chunks)} chunks and building FAISS index for '{repo_name}'...")
        embeddings = embed_chunks(chunks)
        index = build_index(embeddings)
        save_index(repo_name, index, chunks)
        return index, chunks, embeddings
