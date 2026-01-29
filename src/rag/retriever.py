"""Unified retriever orchestrating multi-way RAG retrieval.

Coordinates BM25, HNSW, and IVF-PQ retrievers with RFF fusion.
Provides a single interface for retrieval with full provenance tracking.

Pure Python implementation - no FAISS dependency.

Reference: PRD Section 4 - RAG System Implementation
"""

import pickle
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.config import settings
from src.data.vectorizer import DocumentVectorizer
from src.models import RAGResult, RAGRetrievalLog, RetrievalAlgorithm
from src.rag.bm25_retriever import BM25Retriever
from src.rag.hnsw_pure import PureHNSWRetriever
from src.rag.ivf_pq_pure import PureIVFPQRetriever
from src.rag.rff_fusion import RFFFusion

warnings.filterwarnings("ignore", category=UserWarning)


class UnifiedRetriever:
    """Unified retriever with multi-way recall and RFF fusion.
    
    This is the main entry point for RAG retrieval in the system.
    It manages all three retrieval algorithms and fuses their results.
    
    Uses pure Python implementations (no FAISS).
    """
    
    def __init__(
        self,
        use_bm25: bool = True,
        use_hnsw: bool = True,
        use_ivf_pq: bool = True,
        use_rff_fusion: bool = True,
    ):
        """Initialize unified retriever.
        
        Args:
            use_bm25: Enable BM25 retrieval
            use_hnsw: Enable HNSW retrieval (Pure Python)
            use_ivf_pq: Enable IVF-PQ retrieval (Pure Python)
            use_rff_fusion: Use RFF fusion
        """
        self.use_bm25 = use_bm25
        self.use_hnsw = use_hnsw
        self.use_ivf_pq = use_ivf_pq
        self.use_rff_fusion = use_rff_fusion
        
        # Initialize retrievers (Pure Python versions)
        self.bm25_retriever = BM25Retriever() if use_bm25 else None
        self.hnsw_retriever = PureHNSWRetriever() if use_hnsw else None
        self.ivf_pq_retriever = PureIVFPQRetriever() if use_ivf_pq else None
        
        # Fusion
        self.rff_fusion = RFFFusion() if use_rff_fusion else None
        
        # Vectorizer for query encoding
        self.vectorizer = DocumentVectorizer()
        
        # Document storage
        self.doc_texts: dict[str, str] = {}
        self.doc_metadata: dict[str, dict[str, Any]] = {}
    
    def build_indices(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        doc_embeddings: np.ndarray,
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build all retrieval indices.
        
        Args:
            doc_ids: Document identifiers
            doc_texts: Document text content
            doc_embeddings: Pre-computed embeddings (n_docs, dimension)
            doc_metadata: Optional document metadata
        """
        doc_metadata = doc_metadata or [{} for _ in doc_ids]
        
        # Store documents
        self.doc_texts = {doc_id: text for doc_id, text in zip(doc_ids, doc_texts)}
        self.doc_metadata = {doc_id: meta for doc_id, meta in zip(doc_ids, doc_metadata)}
        
        print(f"\nBuilding indices for {len(doc_ids)} documents (Pure Python)...")
        
        # Build BM25 index
        if self.bm25_retriever:
            self.bm25_retriever.build_index(doc_ids, doc_texts, doc_metadata)
        
        # Build HNSW index (Pure Python)
        if self.hnsw_retriever:
            self.hnsw_retriever.build_index(doc_embeddings, doc_ids, doc_metadata)
        
        # Build IVF-PQ index (Pure Python)
        if self.ivf_pq_retriever:
            self.ivf_pq_retriever.build_index(doc_embeddings, doc_ids, doc_metadata)
        
        print("All indices built successfully!\n")
    
    def save_indices(self, output_dir: Path | str) -> None:
        """Save all indices to disk.
        
        Args:
            output_dir: Directory to save indices
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.bm25_retriever:
            self.bm25_retriever.save_index(output_dir / "bm25.pkl")
        
        if self.hnsw_retriever:
            self.hnsw_retriever.save_index(output_dir / "hnsw.npy")
        
        if self.ivf_pq_retriever:
            self.ivf_pq_retriever.save_index(output_dir / "ivf_pq.npy")
        
        # Save document store
        doc_store = {
            "doc_texts": self.doc_texts,
            "doc_metadata": self.doc_metadata,
        }
        with open(output_dir / "doc_store.pkl", "wb") as f:
            pickle.dump(doc_store, f)
        
        print(f"All indices saved to {output_dir}")
    
    def load_indices(self, index_dir: Path | str | None = None) -> None:
        """Load all indices from disk.
        
        Args:
            index_dir: Directory containing saved indices
        """
        if index_dir is None:
            index_dir = settings.PROJECT_ROOT / "data" / "indices"
        else:
            index_dir = Path(index_dir)
        
        print(f"\nLoading indices from {index_dir}...")
        
        # Load BM25
        if self.bm25_retriever and (index_dir / "bm25.pkl").exists():
            self.bm25_retriever.load_index(index_dir / "bm25.pkl")
        
        # Load HNSW (Pure Python)
        if self.hnsw_retriever and (index_dir / "hnsw.npy").exists():
            self.hnsw_retriever.load_index(index_dir / "hnsw.npy")
        
        # Load IVF-PQ (Pure Python)
        if self.ivf_pq_retriever and (index_dir / "ivf_pq.npy").exists():
            self.ivf_pq_retriever.load_index(index_dir / "ivf_pq.npy")
        
        # Load document store
        doc_store_path = index_dir / "doc_store.pkl"
        if doc_store_path.exists():
            with open(doc_store_path, "rb") as f:
                doc_store = pickle.load(f)
            self.doc_texts = doc_store["doc_texts"]
            self.doc_metadata = doc_store["doc_metadata"]
        
        print("All indices loaded successfully!\n")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_type: Literal["exact_lookup", "semantic_search", "hybrid"] = "hybrid",
    ) -> tuple[list[RAGResult], RAGRetrievalLog]:
        """Execute multi-way retrieval with fusion.
        
        Args:
            query: User query string
            top_k: Number of results to return
            query_type: Query type for algorithm selection
            
        Returns:
            Tuple of (fused results, retrieval log)
        """
        query_id = f"rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        start_time = time.time()
        
        # Adjust weights based on query type
        if query_type == "exact_lookup":
            weights = {"BM25": 0.6, "HNSW": 0.3, "IVF_PQ": 0.1}
        elif query_type == "semantic_search":
            weights = {"BM25": 0.1, "HNSW": 0.6, "IVF_PQ": 0.3}
        else:  # hybrid
            weights = {"BM25": 0.3, "HNSW": 0.4, "IVF_PQ": 0.3}
        
        # Encode query
        query_embedding = self.vectorizer.encode([query])[0]
        
        # Collect retrieval paths for logging
        retrieval_paths = []
        
        # BM25 retrieval
        bm25_results = []
        if self.bm25_retriever and self.bm25_retriever.bm25 is not None:
            t0 = time.time()
            bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
            latency_ms = int((time.time() - t0) * 1000)
            
            retrieval_paths.append({
                "algo": "BM25",
                "params": {"k1": self.bm25_retriever.k1, "b": self.bm25_retriever.b},
                "recall_count": len(bm25_results),
                "latency_ms": latency_ms,
                "top_results": [{"doc_id": r.doc_id, "score": r.raw_score} for r in bm25_results[:3]],
            })
        
        # HNSW retrieval (Pure Python)
        hnsw_results = []
        visited_nodes = 0
        if self.hnsw_retriever and self.hnsw_retriever.nodes:
            t0 = time.time()
            hnsw_results, visited_nodes = self.hnsw_retriever.search(
                query_embedding, top_k=top_k * 2
            )
            latency_ms = int((time.time() - t0) * 1000)
            
            retrieval_paths.append({
                "algo": "HNSW",
                "params": {"ef": self.hnsw_retriever.ef_search, "M": self.hnsw_retriever.m},
                "recall_count": len(hnsw_results),
                "latency_ms": latency_ms,
                "visited_nodes": visited_nodes,
                "top_results": [{"doc_id": r.doc_id, "score": r.raw_score} for r in hnsw_results[:3]],
            })
        
        # IVF-PQ retrieval (Pure Python)
        ivf_pq_results = []
        if self.ivf_pq_retriever and self.ivf_pq_retriever.codes is not None:
            t0 = time.time()
            ivf_pq_results = self.ivf_pq_retriever.search(query_embedding, top_k=top_k * 2)
            latency_ms = int((time.time() - t0) * 1000)
            
            retrieval_paths.append({
                "algo": "IVF_PQ",
                "params": {
                    "nprobe": self.ivf_pq_retriever.nprobe,
                    "nlist": self.ivf_pq_retriever.nlist,
                },
                "recall_count": len(ivf_pq_results),
                "latency_ms": latency_ms,
                "top_results": [{"doc_id": r.doc_id, "score": r.raw_score} for r in ivf_pq_results[:3]],
            })
        
        # Fill content from doc store
        for results in [bm25_results, hnsw_results, ivf_pq_results]:
            for r in results:
                if not r.content and r.doc_id in self.doc_texts:
                    r.content = self.doc_texts[r.doc_id]
        
        # Fuse results
        if self.use_rff_fusion and self.rff_fusion:
            fused_results = self.rff_fusion.fuse(
                bm25_results, hnsw_results, ivf_pq_results,
                weights=weights, top_k=top_k
            )
            fusion_method = "RFF"
        else:
            from src.rag.rff_fusion import WeightedFusion
            fused_results = WeightedFusion.fuse(
                bm25_results, hnsw_results, ivf_pq_results,
                weights=weights, top_k=top_k
            )
            fusion_method = "Weighted"
        
        # Fill content for fused results
        for r in fused_results:
            if not r.content and r.doc_id in self.doc_texts:
                r.content = self.doc_texts[r.doc_id]
        
        # Build retrieval log
        log = RAGRetrievalLog(
            query_id=query_id,
            timestamp=datetime.now(),
            query_text=query,
            retrieval_paths=retrieval_paths,
            fusion_method=fusion_method,
            fusion_weights=list(weights.values()),
            top_k=top_k,
            final_results=fused_results,
        )
        
        total_latency_ms = int((time.time() - start_time) * 1000)
        
        # Print retrieval summary for terminal logging
        print(f"\n[RAG] Query: '{query[:50]}...' (Type: {query_type})")
        for path in retrieval_paths:
            print(f"  [{path['algo']}] Recall: {path['recall_count']}, Latency: {path['latency_ms']}ms")
        print(f"  [Fusion] Method: {fusion_method}, Top-1 Confidence: {fused_results[0].confidence:.2f} if fused_results else 0")
        print(f"  [Total] {len(fused_results)} results in {total_latency_ms}ms\n")
        
        return fused_results, log
    
    def retrieve_with_fallback(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.7,
        max_retries: int = 2,
    ) -> tuple[list[RAGResult], RAGRetrievalLog]:
        """Retrieve with automatic fallback on low confidence."""
        results, log = self.retrieve(query, top_k=top_k)
        
        # Check if retry is needed
        if not results or results[0].confidence < min_confidence:
            if max_retries > 0:
                print(f"[RAG] Low confidence ({results[0].confidence if results else 0:.2f}), retrying...")
                
                expanded_query = f"{query} 相关信息"
                results, log = self.retrieve(
                    expanded_query,
                    top_k=top_k * 2,
                    query_type="semantic_search",
                )
        
        return results[:top_k], log


# Singleton instance for application use
_unified_retriever: UnifiedRetriever | None = None


def get_retriever() -> UnifiedRetriever:
    """Get or create the global unified retriever instance."""
    global _unified_retriever
    if _unified_retriever is None:
        _unified_retriever = UnifiedRetriever()
    return _unified_retriever
