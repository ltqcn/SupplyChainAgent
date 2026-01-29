"""Pure Python implementation of IVF-PQ (Inverted File + Product Quantization).

No FAISS dependency - uses only NumPy.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.models import RAGResult, RetrievalAlgorithm

warnings.filterwarnings("ignore")


class PureIVFPQRetriever:
    """Pure Python IVF-PQ implementation without FAISS.
    
    Combines:
    - IVF (Inverted File): K-means clustering for coarse quantization
    - PQ (Product Quantization): Fine quantization within each cluster
    
    Compression: dimension * 4 bytes -> m bytes (typically 192:1 ratio)
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
            dimension: Vector dimension
            nlist: Number of clusters (coarse quantizer)
            m: Number of subquantizers (PQ)
            nbits: Bits per subquantizer code
            nprobe: Number of clusters to search
        """
        self.dimension = dimension or settings.embedding_dimension
        self.nlist = nlist or settings.IVF_PQ_NLIST
        self.m = m or settings.IVF_PQ_M
        self.nbits = nbits or settings.IVF_PQ_NBITS
        self.nprobe = nprobe or settings.IVF_PQ_NPROBE
        
        # Derived parameters
        self.ncentroids = 2 ** self.nbits  # 256 for 8 bits
        self.dsub = self.dimension // self.m  # Dimension per subvector
        
        # IVF: Coarse quantizer (k-means centroids)
        self.coarse_centroids: np.ndarray | None = None  # shape: (nlist, dimension)
        
        # Inverted lists: cluster_id -> list of vector indices
        self.inverted_lists: list[list[int]] = []
        
        # PQ: Sub-quantizer centroids
        # Shape: (m, ncentroids, dsub)
        self.pq_centroids: np.ndarray | None = None
        
        # Encoded vectors
        # Shape: (n_vectors, m) - each entry is a code (0-255)
        self.codes: np.ndarray | None = None
        
        # Metadata
        self.doc_ids: list[str] = []
        self.doc_metadata: list[dict[str, Any]] = []
        self.n_vectors = 0
    
    def _kmeans(
        self,
        data: np.ndarray,
        k: int,
        max_iter: int = 20,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """K-means clustering.
        
        Args:
            data: Vectors to cluster (n, dim)
            k: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (centroids, assignments)
        """
        n, dim = data.shape
        
        # Handle edge case: k > n
        if k > n:
            k = n
        
        # Initialize with random points
        if k == n:
            indices = np.arange(n)
        else:
            indices = np.random.choice(n, k, replace=False)
        centroids = data[indices].copy()
        
        for iteration in range(max_iter):
            # Assign to nearest centroid
            distances = np.linalg.norm(data[:, None] - centroids[None], axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if np.any(mask):
                    new_centroids[i] = data[mask].mean(axis=0)
                else:
                    # Empty cluster, reinitialize
                    new_centroids[i] = data[np.random.choice(n)]
            
            # Check convergence
            change = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            
            if change < tol:
                break
        
        return centroids, assignments
    
    def _train_pq(self, data: np.ndarray) -> np.ndarray:
        """Train product quantizer.
        
        Args:
            data: Training vectors (n, dimension)
            
        Returns:
            PQ centroids of shape (m, ncentroids, dsub)
        """
        n = len(data)
        pq_centroids = np.zeros((self.m, self.ncentroids, self.dsub), dtype=np.float32)
        
        for i in range(self.m):
            # Extract subvectors
            start = i * self.dsub
            end = start + self.dsub
            subvectors = data[:, start:end]
            
            # K-means on subvectors
            # Make sure k is not larger than n
            k = min(self.ncentroids, n)
            centroids, _ = self._kmeans(subvectors, k, max_iter=10)
            
            # If k < ncentroids, duplicate the last centroid
            if k < self.ncentroids:
                for j in range(k, self.ncentroids):
                    centroids = np.vstack([centroids, centroids[-1:]])
            
            pq_centroids[i] = centroids
        
        return pq_centroids
    
    def _encode_pq(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors using PQ.
        
        Args:
            vectors: Vectors to encode (n, dimension)
            
        Returns:
            Codes of shape (n, m)
        """
        n = len(vectors)
        codes = np.zeros((n, self.m), dtype=np.uint8)
        
        for i in range(self.m):
            start = i * self.dsub
            end = start + self.dsub
            subvectors = vectors[:, start:end]
            
            # Find nearest centroid for each subvector
            distances = np.linalg.norm(
                subvectors[:, None] - self.pq_centroids[i][None],
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1).astype(np.uint8)
        
        return codes
    
    def _pq_distance(self, query: np.ndarray, codes: np.ndarray) -> float:
        """Compute approximate distance using PQ codes.
        
        Args:
            query: Query vector (dimension,)
            codes: PQ codes (m,)
            
        Returns:
            Approximate squared distance
        """
        distance = 0.0
        for i in range(self.m):
            start = i * self.dsub
            end = start + self.dsub
            query_sub = query[start:end]
            centroid = self.pq_centroids[i, codes[i]]
            distance += np.sum((query_sub - centroid) ** 2)
        return distance
    
    def build_index(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str],
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build IVF-PQ index.
        
        Args:
            embeddings: Vectors (n, dimension)
            doc_ids: Document IDs
            doc_metadata: Metadata
        """
        n = len(embeddings)
        self.n_vectors = n
        self.doc_ids = doc_ids
        self.doc_metadata = doc_metadata or [{} for _ in doc_ids]
        
        print(f"Building Pure IVF-PQ index: {n} vectors, dim={self.dimension}")
        print(f"  nlist={self.nlist}, m={self.m}, nbits={self.nbits}")
        
        # Adjust nlist for small datasets
        effective_nlist = min(self.nlist, max(1, n // 39))
        if effective_nlist < self.nlist:
            print(f"  Adjusted nlist to {effective_nlist}")
            self.nlist = effective_nlist
        
        # Normalize vectors
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Train coarse quantizer (IVF)
        print(f"  Training coarse quantizer (k-means with {self.nlist} clusters)...")
        train_size = min(n, max(self.nlist * 10, 1000))
        train_indices = np.random.choice(n, train_size, replace=False)
        train_data = embeddings_norm[train_indices]
        
        self.coarse_centroids, coarse_assignments = self._kmeans(train_data, self.nlist)
        
        # Assign all vectors to clusters
        all_distances = np.linalg.norm(
            embeddings_norm[:, None] - self.coarse_centroids[None],
            axis=2
        )
        cluster_assignments = np.argmin(all_distances, axis=1)
        
        # Build inverted lists
        self.inverted_lists = [[] for _ in range(self.nlist)]
        for idx, cluster_id in enumerate(cluster_assignments):
            self.inverted_lists[cluster_id].append(idx)
        
        # Train PQ
        print(f"  Training product quantizer...")
        # Use a subset for PQ training
        pq_train_size = min(n, 10000)
        pq_train_indices = np.random.choice(n, pq_train_size, replace=False)
        pq_train_data = embeddings_norm[pq_train_indices]
        
        self.pq_centroids = self._train_pq(pq_train_data)
        
        # Encode all vectors
        print(f"  Encoding vectors...")
        self.codes = self._encode_pq(embeddings_norm)
        
        print(f"  IVF-PQ index built!")
        print(f"    Compression ratio: {self.dimension * 4 / self.m:.1f}:1")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        nprobe: int | None = None,
    ) -> list[RAGResult]:
        """Search index.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            nprobe: Number of clusters to probe
            
        Returns:
            List of RAGResult
        """
        if self.coarse_centroids is None or self.codes is None:
            raise RuntimeError("Index not built")
        
        probe = nprobe or self.nprobe
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Find nearest clusters
        cluster_distances = np.linalg.norm(
            self.coarse_centroids - query_norm[None],
            axis=1
        )
        nearest_clusters = np.argsort(cluster_distances)[:probe]
        
        # Search within selected clusters
        candidates = []
        for cluster_id in nearest_clusters:
            for idx in self.inverted_lists[cluster_id]:
                # Compute approximate distance
                dist = self._pq_distance(query_norm, self.codes[idx])
                candidates.append((dist, idx))
        
        # Sort by distance (ascending, lower is closer)
        candidates.sort(key=lambda x: x[0])
        
        # Build results
        results = []
        for dist, idx in candidates[:top_k]:
            # Convert distance to similarity score
            # Using negative distance normalized to [0, 1]
            similarity = float(1.0 / (1.0 + dist))
            
            result = RAGResult(
                doc_id=self.doc_ids[idx],
                content="",
                retrieval_algo=RetrievalAlgorithm.IVF_PQ,
                raw_score=-float(dist),  # Negative distance
                normalized_score=similarity,
                doc_type=self.doc_metadata[idx].get("doc_type", "unknown"),
                timestamp=self.doc_metadata[idx].get("timestamp"),
                source_authority=self.doc_metadata[idx].get("source_authority", 0.8),
                confidence=similarity * 0.95,
                chunk_idx=0,
            )
            results.append(result)
        
        return results
    
    def save_index(self, filepath: Path | str) -> None:
        """Save index to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "coarse_centroids": self.coarse_centroids,
            "inverted_lists": self.inverted_lists,
            "pq_centroids": self.pq_centroids,
            "codes": self.codes,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "nprobe": self.nprobe,
            "doc_ids": self.doc_ids,
            "doc_metadata": self.doc_metadata,
            "n_vectors": self.n_vectors,
        }
        
        np.save(filepath, data, allow_pickle=True)
        print(f"Pure IVF-PQ index saved to {filepath}")
    
    def load_index(self, filepath: Path | str) -> None:
        """Load index from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"IVF-PQ index not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True).item()
        
        self.coarse_centroids = data["coarse_centroids"]
        self.inverted_lists = data["inverted_lists"]
        self.pq_centroids = data["pq_centroids"]
        self.codes = data["codes"]
        self.dimension = data["dimension"]
        self.nlist = data["nlist"]
        self.m = data["m"]
        self.nbits = data["nbits"]
        self.nprobe = data["nprobe"]
        self.doc_ids = data["doc_ids"]
        self.doc_metadata = data["doc_metadata"]
        self.n_vectors = data["n_vectors"]
        
        print(f"Pure IVF-PQ index loaded: {self.n_vectors} vectors from {filepath}")
