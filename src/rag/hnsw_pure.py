"""Pure Python implementation of HNSW (Hierarchical Navigable Small World).

No FAISS dependency - uses only NumPy for vector operations.

Reference: "Efficient and robust approximate nearest neighbor search using 
Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2016)
"""

import heapq
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.models import RAGResult, RetrievalAlgorithm

warnings.filterwarnings("ignore")


class HNSWNode:
    """Node in HNSW graph."""
    
    def __init__(self, idx: int, vector: np.ndarray, level: int):
        self.idx = idx
        self.vector = vector
        self.level = level
        self.neighbors: list[list[int]] = [[] for _ in range(level + 1)]


class PureHNSWRetriever:
    """Pure Python HNSW implementation without FAISS.
    
    This is slower than FAISS for large datasets but provides:
    - No external dependencies beyond NumPy
    - Works on all platforms including macOS ARM
    - Educational value for understanding the algorithm
    
    Time Complexity:
    - Search: O(log N) average case
    - Insert: O(log N) average case
    
    Space Complexity: O(N * M * levels)
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
            dimension: Vector dimension
            m: Max connections per layer (default 16)
            ef_construction: Search depth during construction (default 200)
            ef_search: Search depth during query (default 64)
        """
        self.dimension = dimension or settings.embedding_dimension
        self.m = m or settings.HNSW_M
        self.ef_construction = ef_construction or settings.HNSW_EF_CONSTRUCTION
        self.ef_search = ef_search or settings.HNSW_EF_SEARCH
        
        # HNSW parameters
        self.m_max = self.m  # Max connections per node
        self.m_max0 = self.m * 2  # Max connections for layer 0
        self.level_mult = 1.0 / np.log(self.m)  # For level generation
        
        # Graph storage
        self.nodes: list[HNSWNode] = []
        self.entry_point: int | None = None
        self.max_level = 0
        
        # Metadata
        self.doc_ids: list[str] = []
        self.doc_metadata: list[dict[str, Any]] = []
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))
    
    def _get_random_level(self) -> int:
        """Generate random level for new node using exponential distribution."""
        # Level ~ exp(-level / level_mult)
        r = np.random.random()
        level = int(-np.log(r) * self.level_mult)
        return min(level, self.max_level + 1)
    
    def _search_layer(
        self,
        query: np.ndarray,
        entry_idx: int,
        ef: int,
        level: int,
    ) -> list[tuple[float, int]]:
        """Search a single layer using greedy beam search.
        
        Args:
            query: Query vector
            entry_idx: Starting node index
            ef: Beam width (number of candidates to track)
            level: Which layer to search
            
        Returns:
            List of (distance, idx) for closest neighbors
        """
        # Visited set to avoid cycles
        visited = {entry_idx}
        
        # Priority queue for candidates (max heap using negative distance)
        # We want closest points, so use min heap with distance
        candidates = [(-self._cosine_similarity(query, self.nodes[entry_idx].vector), entry_idx)]
        
        # Result set (min heap with distance)
        results = [(self._cosine_similarity(query, self.nodes[entry_idx].vector), entry_idx)]
        heapq.heapify(results)
        
        while candidates:
            # Get closest candidate
            neg_dist, curr_idx = heapq.heappop(candidates)
            curr_dist = -neg_dist
            
            # Early termination: if candidate is worse than worst result
            if len(results) >= ef:
                worst_result_dist = results[0][0]  # Min heap, so smallest is worst
                if curr_dist < worst_result_dist:
                    break
            
            # Explore neighbors
            for neighbor_idx in self.nodes[curr_idx].neighbors[level]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    neighbor_dist = self._cosine_similarity(query, self.nodes[neighbor_idx].vector)
                    
                    # Add to results if better than worst
                    if len(results) < ef:
                        heapq.heappush(results, (neighbor_dist, neighbor_idx))
                    elif neighbor_dist > results[0][0]:
                        heapq.heapreplace(results, (neighbor_dist, neighbor_idx))
                    
                    # Add to candidates for further exploration
                    heapq.heappush(candidates, (-neighbor_dist, neighbor_idx))
        
        # Convert to list sorted by distance (descending)
        return sorted(results, key=lambda x: x[0], reverse=True)
    
    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: list[tuple[float, int]],
        m: int,
    ) -> list[int]:
        """Select M neighbors from candidates.
        
        Uses simple heuristic: select closest M points.
        More sophisticated: use heuristic from paper to maintain graph connectivity.
        """
        # Simple version: just take closest M
        return [idx for _, idx in candidates[:m]]
    
    def _connect_node(
        self,
        new_idx: int,
        neighbors: list[int],
        level: int,
    ) -> None:
        """Connect new node to its neighbors at given level."""
        new_node = self.nodes[new_idx]
        
        for neighbor_idx in neighbors:
            neighbor_node = self.nodes[neighbor_idx]
            
            # Add bidirectional connection
            new_node.neighbors[level].append(neighbor_idx)
            neighbor_node.neighbors[level].append(new_idx)
            
            # Shrink connections if exceeding limit
            m_max = self.m_max0 if level == 0 else self.m_max
            
            if len(neighbor_node.neighbors[level]) > m_max:
                # Keep only closest m_max connections
                distances = [
                    (self._cosine_similarity(self.nodes[new_idx].vector, self.nodes[n].vector), n)
                    for n in neighbor_node.neighbors[level]
                ]
                distances.sort(reverse=True)
                neighbor_node.neighbors[level] = [n for _, n in distances[:m_max]]
    
    def add_vector(self, vector: np.ndarray, idx: int) -> None:
        """Add a single vector to the index.
        
        Args:
            vector: Vector to add
            idx: Index in the nodes list
        """
        # Generate random level for new node
        level = self._get_random_level()
        
        # Create node
        node = HNSWNode(idx, vector, level)
        
        # Ensure level structure exists
        for existing_node in self.nodes:
            while len(existing_node.neighbors) <= level:
                existing_node.neighbors.append([])
        
        self.nodes.append(node)
        
        # Update max level
        if level > self.max_level:
            self.max_level = level
        
        # Insert into graph
        if self.entry_point is None:
            # First node
            self.entry_point = idx
            return
        
        # Find entry point for insertion
        curr_idx = self.entry_point
        curr_dist = self._cosine_similarity(vector, self.nodes[curr_idx].vector)
        
        # Navigate from top level down
        for l in range(self.max_level, level + 1, -1):
            changed = True
            while changed:
                changed = False
                for neighbor_idx in self.nodes[curr_idx].neighbors[l]:
                    neighbor_dist = self._cosine_similarity(vector, self.nodes[neighbor_idx].vector)
                    if neighbor_dist > curr_dist:
                        curr_idx = neighbor_idx
                        curr_dist = neighbor_dist
                        changed = True
        
        # Insert at each level
        for l in range(min(level, self.max_level), -1, -1):
            # Find neighbors at this level
            results = self._search_layer(vector, curr_idx, self.ef_construction, l)
            neighbors = self._select_neighbors(vector, results, self.m)
            
            # Connect
            self._connect_node(idx, neighbors, l)
            
            # Update entry point for next level
            if results:
                curr_idx = results[0][1]
    
    def build_index(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str],
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build HNSW index from embeddings.
        
        Args:
            embeddings: Array of shape (n_docs, dimension)
            doc_ids: Document identifiers
            doc_metadata: Optional metadata
        """
        n_docs = len(embeddings)
        self.doc_ids = doc_ids
        self.doc_metadata = doc_metadata or [{} for _ in doc_ids]
        
        print(f"Building Pure HNSW index: {n_docs} vectors, dim={self.dimension}")
        print(f"  M={self.m}, efConstruction={self.ef_construction}")
        
        # Reset
        self.nodes = []
        self.entry_point = None
        self.max_level = 0
        
        # Normalize vectors for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Insert vectors one by one
        for i in range(n_docs):
            if i % 100 == 0 and i > 0:
                print(f"  Progress: {i}/{n_docs}")
            self.add_vector(embeddings_norm[i], i)
        
        print(f"  HNSW index built: {n_docs} vectors, levels={self.max_level + 1}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        ef_search: int | None = None,
    ) -> tuple[list[RAGResult], int]:
        """Search for nearest neighbors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            ef_search: Search depth (default self.ef_search)
            
        Returns:
            Tuple of (results, visited_count)
        """
        if not self.nodes or self.entry_point is None:
            return [], 0
        
        ef = ef_search or self.ef_search
        
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Navigate from top level
        curr_idx = self.entry_point
        
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = self._cosine_similarity(query_norm, self.nodes[curr_idx].vector)
                
                for neighbor_idx in self.nodes[curr_idx].neighbors[l]:
                    neighbor_dist = self._cosine_similarity(query_norm, self.nodes[neighbor_idx].vector)
                    if neighbor_dist > curr_dist:
                        curr_idx = neighbor_idx
                        changed = True
        
        # Search at level 0 with ef
        results = self._search_layer(query_norm, curr_idx, ef, 0)
        
        # Build RAGResults
        rag_results = []
        for score, idx in results[:top_k]:
            result = RAGResult(
                doc_id=self.doc_ids[idx],
                content="",
                retrieval_algo=RetrievalAlgorithm.HNSW,
                raw_score=float(score),
                normalized_score=float(max(0, score)),
                doc_type=self.doc_metadata[idx].get("doc_type", "unknown"),
                timestamp=self.doc_metadata[idx].get("timestamp"),
                source_authority=self.doc_metadata[idx].get("source_authority", 0.8),
                confidence=float(max(0, score)),
                chunk_idx=0,
            )
            rag_results.append(result)
        
        return rag_results, len(results)
    
    def save_index(self, filepath: Path | str) -> None:
        """Save index to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "nodes": [
                {
                    "idx": n.idx,
                    "vector": n.vector,
                    "level": n.level,
                    "neighbors": n.neighbors,
                }
                for n in self.nodes
            ],
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "dimension": self.dimension,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "doc_ids": self.doc_ids,
            "doc_metadata": self.doc_metadata,
        }
        
        np.save(filepath, data, allow_pickle=True)
        print(f"Pure HNSW index saved to {filepath}")
    
    def load_index(self, filepath: Path | str) -> None:
        """Load index from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"HNSW index not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True).item()
        
        self.nodes = [
            HNSWNode(n["idx"], n["vector"], n["level"])
            for n in data["nodes"]
        ]
        for i, n in enumerate(data["nodes"]):
            self.nodes[i].neighbors = n["neighbors"]
        
        self.entry_point = data["entry_point"]
        self.max_level = data["max_level"]
        self.dimension = data["dimension"]
        self.m = data["m"]
        self.ef_construction = data["ef_construction"]
        self.ef_search = data["ef_search"]
        self.doc_ids = data["doc_ids"]
        self.doc_metadata = data["doc_metadata"]
        
        print(f"Pure HNSW index loaded: {len(self.nodes)} vectors from {filepath}")
