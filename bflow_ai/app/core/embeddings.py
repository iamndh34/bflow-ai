"""
Embedding Service - Singleton cho SentenceTransformer model

Thay vì load model nhiều lần ở nhiều nơi (tốn RAM + thời gian),
dùng singleton này để reuse model.

Features:
1. Singleton model instance
2. LRU cache cho embedding computation
3. Batch encoding support

Usage:
    from app.core.embeddings import get_embed_model

    model = get_embed_model()
    embedding = model.encode("text")
    # hoặc dùng cached:
    from app.core.embeddings import encode_cached
    embedding = encode_cached("text")
"""
import threading
from functools import lru_cache
from typing import Optional, List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Singleton embedding model service"""

    _model: Optional[SentenceTransformer] = None
    _lock = threading.Lock()
    _model_name = "bkai-foundation-models/vietnamese-bi-encoder"

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Get singleton embedding model instance"""
        if cls._model is None:
            with cls._lock:
                # Double-check locking
                if cls._model is None:
                    print(f"[EmbeddingService] Loading model: {cls._model_name}")
                    cls._model = SentenceTransformer(cls._model_name)
                    print(f"[EmbeddingService] Model loaded successfully")
        return cls._model

    @classmethod
    def reset(cls):
        """Reset model instance (cho testing)"""
        with cls._lock:
            cls._model = None


# Convenience function
def get_embed_model() -> SentenceTransformer:
    """Get singleton embedding model"""
    return EmbeddingService.get_model()


# Cached encoding with LRU cache
@lru_cache(maxsize=2000)
def encode_cached(text: str) -> np.ndarray:
    """
    Encode text với LRU cache.

    Args:
        text: Input text

    Returns:
        Embedding vector

    Note: Cache key là text string, chỉ cache exact matches.
    Cho approximate matching, dùng batch encode thay thế.
    """
    model = get_embed_model()
    return model.encode(text)


def encode_batch(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    Encode batch texts efficiently.

    Args:
        texts: List of input texts
        normalize: Normalize embeddings (recommended for cosine similarity)

    Returns:
        Embedding matrix (n_texts, embedding_dim)
    """
    model = get_embed_model()
    return model.encode(texts, normalize_embeddings=normalize)


def encode_single(text: str, normalize: bool = True) -> np.ndarray:
    """
    Encode single text without cache.

    Args:
        text: Input text
        normalize: Normalize embedding

    Returns:
        Embedding vector
    """
    model = get_embed_model()
    return model.encode(text, normalize_embeddings=normalize)


def batch_cosine_similarity(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray
) -> np.ndarray:
    """
    Batch compute cosine similarities between query and corpus.

    Vectorized operation - nhanh hơn loop rất nhiều.

    Args:
        query_embedding: (embedding_dim,) query vector
        corpus_embeddings: (n_docs, embedding_dim) corpus matrix

    Returns:
        (n_docs,) similarity scores
    """
    # Assume embeddings are normalized
    return np.dot(corpus_embeddings, query_embedding)
