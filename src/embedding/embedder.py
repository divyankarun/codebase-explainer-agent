"""FastEmbed wrapper for generating dense text vector embeddings."""

from typing import Any
import numpy as np
from fastembed import TextEmbedding
from config import EMBEDDING_MODEL

# Global model instance initialized lazily
_model_instance = None


def get_model() -> TextEmbedding:
    """Get or initialize the FastEmbed TextEmbedding singleton model."""
    global _model_instance
    if _model_instance is None:
        print(f"Initializing FastEmbed model ('{EMBEDDING_MODEL}')...")
        _model_instance = TextEmbedding(EMBEDDING_MODEL)
    return _model_instance


def embed_chunks(chunks: list[dict[str, Any]]) -> np.ndarray:
    """Extract text from chunks and generate dense float32 vector embeddings.
    
    Args:
        chunks: List of chunk dicts containing a 'text' key.
        
    Returns:
        2D numpy array of shape (num_chunks, embedding_dim) with dtype float32.
    """
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    texts = [c.get("text", "")[:2000] for c in chunks]
    model = get_model()
    print(f"Generating embeddings for {len(texts)} chunks in batches of 16...")
    embeddings = list(model.embed(texts, batch_size=16))
    print(f"Finished generating {len(embeddings)} embeddings.")
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string into a float32 vector.
    
    Args:
        query: User search query string.
        
    Returns:
        2D numpy array of shape (1, embedding_dim) with dtype float32.
    """
    model = get_model()
    embeddings = list(model.embed([query]))
    return np.array(embeddings, dtype=np.float32)
