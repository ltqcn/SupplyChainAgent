"""Simple vectorizer that doesn't require downloading models.

Uses hash-based or random embeddings for offline testing.
"""

import hashlib
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np

warnings.filterwarnings("ignore")


class SimpleVectorizer:
    """Simple vectorizer that works offline without model download.
    
    Uses hash-based embeddings (deterministic) for testing.
    Not suitable for production but allows the system to run offline.
    """
    
    def __init__(self, dimension: int = 384):
        """Initialize simple vectorizer.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.model_name = f"simple-hash-{dimension}d"
        print(f"Using SimpleVectorizer (offline mode): {self.dimension}d")
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector using hash.
        
        This creates deterministic vectors based on text content.
        Not semantically meaningful but consistent for testing.
        """
        # Use hash to seed random generator for deterministic output
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(hash_val % (2**32))
        
        # Generate random vector
        vec = rng.randn(self.dimension).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Ignored (for API compatibility)
            
        Returns:
            Array of embeddings
        """
        embeddings = np.array([self._text_to_vector(t) for t in texts])
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text.
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        return self._text_to_vector(text)


def get_vectorizer(use_simple: bool = False, dimension: int = 384):
    """Get appropriate vectorizer.
    
    Args:
        use_simple: Force use of simple vectorizer
        dimension: Embedding dimension
        
    Returns:
        Vectorizer instance
    """
    if use_simple:
        return SimpleVectorizer(dimension)
    
    # Try to use sentence-transformers
    try:
        from src.data.vectorizer import DocumentVectorizer
        import os
        
        # Check if offline mode is requested
        if os.environ.get('USE_SIMPLE_VECTORIZER', '0') == '1':
            raise ImportError("Offline mode requested")
        
        return DocumentVectorizer()
    except Exception as e:
        print(f"Warning: Could not load sentence-transformers ({e})")
        print("Falling back to SimpleVectorizer (offline mode)")
        return SimpleVectorizer(dimension)
