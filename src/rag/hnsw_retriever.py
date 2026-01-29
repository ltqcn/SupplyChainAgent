"""HNSW (Hierarchical Navigable Small World) retriever implementation.

Implements graph-based approximate nearest neighbor search using FAISS.
Provides O(log N) search complexity with high recall rates.

Reference: PRD Section 4.1.2 - HNSW Graph Index
"""

import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import settings
from src.models import RAGResult, RetrievalAlgorithm

# Suppress multiprocessing warnings on macOS
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")


class HNSWRetriever:
    """HNSW retriever for semantic similarity search.
    
    Configuration:
    - M=16: Maximum connections per layer
    - efConstruction=200: Build-time search depth
    - efSearch=64: Query-time search depth (adjustable)
    
    Performance: Recall@10 >95%, <20ms query latency for 100K vectors
    """
    
    def __init__(
        self,
        dimension: int | None = None,
        m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
    ):
        """Initialize HNSW retriever.
        
        Args:
            dimension: Vector dimension (auto-detected from settings if None)
            m: Max connections per layer (default from settings)
            ef_construction: Build-time search depth (default from settings)
            ef_search: Query-time search depth (default from settings)
        """
        self.dimension = dimension or settings.embedding_dimension
        self.m = m or settings.HNSW_M
        self.ef_construction = ef_construction or settings.HNSW_EF_CONSTRUCTION
        self.ef_search = ef_search or settings.HNSW_EF_SEARCH
        
        self.index: faiss.IndexHNSWFlat | None = None
        self.doc_ids: list[str] = []
        self.doc_metadata: list[dict[str, Any]] = []
    
    def build_index(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str],
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build HNSW index from embeddings.
        
        Args:
            embeddings: Array of shape (n_docs, dimension) with normalized vectors
            doc_ids: Document identifiers
            doc_metadata: Optional document metadata
        """
        n_docs = len(embeddings)
        self.doc_ids = doc_ids
        self.doc_metadata = doc_metadata or [{} for _ in doc_ids]
        
        print(f"Building HNSW index: {n_docs} vectors, dim={self.dimension}")
        print(f"  M={self.m}, efConstruction={self.ef_construction}")
        
        try:
            # Create HNSW index
            # IndexHNSWFlat uses L2 distance, but we use normalized vectors
            # so cosine similarity = 1 - 0.5 * L2_distance^2
            self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
            self.index.hnsw.efConstruction = self.ef_construction
            
            # Add vectors to index (ensure float32)
            embeddings_f32 = embeddings.astype(np.float32)
            if not embeddings_f32.flags.c_contiguous:
                embeddings_f32 = np.ascontiguousarray(embeddings_f32)
            
            print(f"  Adding vectors to index...")
            self.index.add(embeddings_f32)
            
            # Set search parameter
            self.index.hnsw.efSearch = self.ef_search
            
            print(f"  HNSW index built successfully!")
            print(f"    Vectors: {n_docs}")
            print(f"    Levels: {self.index.hnsw.max_level}")
            
        except Exception as e:
            print(f"  Error building HNSW index: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        ef_search: int | None = None,
    ) -> tuple[list[RAGResult], int]:
        """Search for nearest neighbors using HNSW.
        
        Args:
            query_embedding: Query vector (normalized)
            top_k: Number of results to return
            ef_search: Optional override for search depth
            
        Returns:
            Tuple of (results list, number of visited nodes)
        """
        if self.index is None:
            raise RuntimeError("HNSW index not built. Call build_index() first.")
        
        # Set search depth if provided
        if ef_search is not None:
            self.index.hnsw.efSearch = ef_search
        
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32 and contiguous
        query_f32 = query_embedding.astype(np.float32)
        if not query_f32.flags.c_contiguous:
            query_f32 = np.ascontiguousarray(query_f32)
        
        # Search
        try:
            distances, indices = self.index.search(
                query_f32,
                min(top_k * 2, len(self.doc_ids)),  # Search extra for filtering
            )
        except Exception as e:
            print(f"HNSW search error: {e}")
            return [], 0
        
        # Get search stats (approximate visited nodes)
        visited_nodes = self.index.hnsw.efSearch * 2  # Approximation
        
        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            
            # Convert L2 distance to cosine similarity
            # For normalized vectors: cosine_sim = 1 - 0.5 * L2_dist^2
            l2_dist = float(dist)
            cosine_sim = max(0, 1 - 0.5 * l2_dist ** 2)
            
            # Normalize to [0, 1]
            normalized_score = cosine_sim
            
            result = RAGResult(
                doc_id=self.doc_ids[idx],
                content="",  # Will be filled by retriever
                retrieval_algo=RetrievalAlgorithm.HNSW,
                raw_score=cosine_sim,
                normalized_score=normalized_score,
                doc_type=self.doc_metadata[idx].get("doc_type", "unknown"),
                timestamp=self.doc_metadata[idx].get("timestamp"),
                source_authority=self.doc_metadata[idx].get("source_authority", 0.8),
                confidence=normalized_score,
                chunk_idx=0,
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results, visited_nodes
    
    def save_index(self, filepath: Path | str) -> None:
        """Save HNSW index to disk.
        
        Args:
            filepath: Path to save index
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index is None:
            raise RuntimeError("Index not built, nothing to save")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(filepath))
            
            # Save metadata separately
            metadata_path = filepath.with_suffix(".meta")
            metadata = {
                "doc_ids": self.doc_ids,
                "doc_metadata": self.doc_metadata,
                "dimension": self.dimension,
                "m": self.m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
            }
            np.save(metadata_path, metadata, allow_pickle=True)
            
            print(f"HNSW index saved to {filepath}")
        except Exception as e:
            print(f"Error saving HNSW index: {e}")
            raise
    
    def load_index(self, filepath: Path | str) -> None:
        """Load HNSW index from disk.
        
        Args:
            filepath: Path to load index from
        """
        filepath = Path(filepath)
        metadata_path = filepath.with_suffix(".meta.npy")
        
        if not filepath.exists():
            raise FileNotFoundError(f"HNSW index not found: {filepath}")
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(filepath))
            
            # Load metadata
            if metadata_path.exists():
                metadata = np.load(metadata_path, allow_pickle=True).item()
                self.doc_ids = metadata["doc_ids"]
                self.doc_metadata = metadata["doc_metadata"]
                self.dimension = metadata["dimension"]
                self.m = metadata["m"]
                self.ef_construction = metadata["ef_construction"]
                self.ef_search = metadata["ef_search"]
            
            print(f"HNSW index loaded: {len(self.doc_ids)} vectors from {filepath}")
        except Exception as e:
            print(f"Error loading HNSW index: {e}")
            raise
