"""IVF-PQ (Inverted File with Product Quantization) retriever.

Implements compressed vector storage for large-scale retrieval.
Reduces memory usage by 192:1 compression ratio while maintaining accuracy.

Reference: PRD Section 4.1.3 - IVF-PQ Quantization Index
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


class IVFPQRetriever:
    """IVF-PQ retriever for large-scale vector storage.
    
    Configuration:
    - nlist=4096: Number of Voronoi cells (clusters)
    - nprobe=16: Clusters to search during query
    - m=16: Number of subquantizers
    - nbits=8: Bits per quantization code (256 centroids per subquantizer)
    
    Compression: 768-dim float32 (3KB) -> 16 bytes (192:1 ratio)
    Suitable for: Long-term memory, historical records (1M+ vectors)
    """
    
    def __init__(
        self,
        dimension: int | None = None,
        nlist: int | None = None,
        m: int | None = None,
        nbits: int | None = None,
        nprobe: int | None = None,
    ):
        """Initialize IVF-PQ retriever.
        
        Args:
            dimension: Vector dimension (auto-detected from settings if None)
            nlist: Number of clusters (default from settings)
            m: Number of subquantizers (default from settings)
            nbits: Bits per code (default from settings)
            nprobe: Clusters to probe during search (default from settings)
        """
        self.dimension = dimension or settings.embedding_dimension
        self.nlist = nlist or settings.IVF_PQ_NLIST
        self.m = m or settings.IVF_PQ_M
        self.nbits = nbits or settings.IVF_PQ_NBITS
        self.nprobe = nprobe or settings.IVF_PQ_NPROBE
        
        self.index: faiss.IndexIVFPQ | None = None
        self.doc_ids: list[str] = []
        self.doc_metadata: list[dict[str, Any]] = []
    
    def build_index(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str],
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build IVF-PQ index from embeddings.
        
        Note: Requires training data (subsample of embeddings for clustering).
        
        Args:
            embeddings: Array of shape (n_docs, dimension) with normalized vectors
            doc_ids: Document identifiers
            doc_metadata: Optional document metadata
        """
        n_docs = len(embeddings)
        self.doc_ids = doc_ids
        self.doc_metadata = doc_metadata or [{} for _ in doc_ids]
        
        print(f"Building IVF-PQ index: {n_docs} vectors, dim={self.dimension}")
        print(f"  nlist={self.nlist}, m={self.m}, nbits={self.nbits}")
        
        # Ensure float32 and contiguous
        embeddings_f32 = embeddings.astype(np.float32)
        if not embeddings_f32.flags.c_contiguous:
            embeddings_f32 = np.ascontiguousarray(embeddings_f32)
        
        # Adjust nlist if dataset is small
        effective_nlist = min(self.nlist, max(1, n_docs // 39))
        if effective_nlist < self.nlist:
            print(f"  Adjusted nlist to {effective_nlist} (need 39*nlist <= n_docs)")
        
        try:
            # Create quantizer (flat index for cluster centroids)
            quantizer = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            
            # Create IVF-PQ index
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                effective_nlist,
                self.m,
                self.nbits,
            )
            
            # Training required for IVF-PQ
            # Use subset for training if dataset is large
            train_size = min(n_docs, max(256 * effective_nlist, 10000))
            train_indices = np.random.choice(n_docs, train_size, replace=False)
            train_vectors = embeddings_f32[train_indices]
            
            print(f"  Training on {train_size} vectors...")
            self.index.train(train_vectors)
            
            # Add all vectors to index
            print(f"  Adding {n_docs} vectors to index...")
            self.index.add(embeddings_f32)
            
            # Set search parameter
            self.index.nprobe = self.nprobe
            
            print(f"  IVF-PQ index built successfully!")
            print(f"    Compression ratio: {self.dimension * 4 / (self.m * self.nbits / 8):.1f}:1")
            
        except Exception as e:
            print(f"  Error building IVF-PQ index: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        nprobe: int | None = None,
    ) -> list[RAGResult]:
        """Search for nearest neighbors using IVF-PQ.
        
        Args:
            query_embedding: Query vector (normalized)
            top_k: Number of results to return
            nprobe: Optional override for clusters to probe
            
        Returns:
            List of RAGResult with approximate distances
        """
        if self.index is None:
            raise RuntimeError("IVF-PQ index not built. Call build_index() first.")
        
        # Set nprobe if provided
        if nprobe is not None:
            self.index.nprobe = nprobe
        
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
                min(top_k * 2, len(self.doc_ids)),
            )
        except Exception as e:
            print(f"IVF-PQ search error: {e}")
            return []
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            
            # IVF-PQ returns inner product (for cosine similarity)
            # For normalized vectors, inner product = cosine similarity
            inner_product = float(dist)
            
            # Normalize to [0, 1] (already in this range for normalized vectors)
            normalized_score = max(0, min(1, (inner_product + 1) / 2))
            
            result = RAGResult(
                doc_id=self.doc_ids[idx],
                content="",  # Will be filled by retriever
                retrieval_algo=RetrievalAlgorithm.IVF_PQ,
                raw_score=inner_product,
                normalized_score=normalized_score,
                doc_type=self.doc_metadata[idx].get("doc_type", "unknown"),
                timestamp=self.doc_metadata[idx].get("timestamp"),
                source_authority=self.doc_metadata[idx].get("source_authority", 0.8),
                confidence=normalized_score * 0.95,  # Slight penalty for quantization
                chunk_idx=0,
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save_index(self, filepath: Path | str) -> None:
        """Save IVF-PQ index to disk.
        
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
                "nlist": self.nlist,
                "m": self.m,
                "nbits": self.nbits,
                "nprobe": self.nprobe,
            }
            np.save(metadata_path, metadata, allow_pickle=True)
            
            print(f"IVF-PQ index saved to {filepath}")
        except Exception as e:
            print(f"Error saving IVF-PQ index: {e}")
            raise
    
    def load_index(self, filepath: Path | str) -> None:
        """Load IVF-PQ index from disk.
        
        Args:
            filepath: Path to load index from
        """
        filepath = Path(filepath)
        metadata_path = filepath.with_suffix(".meta.npy")
        
        if not filepath.exists():
            raise FileNotFoundError(f"IVF-PQ index not found: {filepath}")
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(filepath))
            
            # Load metadata
            if metadata_path.exists():
                metadata = np.load(metadata_path, allow_pickle=True).item()
                self.doc_ids = metadata["doc_ids"]
                self.doc_metadata = metadata["doc_metadata"]
                self.dimension = metadata["dimension"]
                self.nlist = metadata["nlist"]
                self.m = metadata["m"]
                self.nbits = metadata["nbits"]
                self.nprobe = metadata["nprobe"]
                
                # Restore nprobe setting
                self.index.nprobe = self.nprobe
            
            print(f"IVF-PQ index loaded: {len(self.doc_ids)} vectors from {filepath}")
        except Exception as e:
            print(f"Error loading IVF-PQ index: {e}")
            raise
