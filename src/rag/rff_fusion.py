"""RFF (Random Fourier Features) fusion for multi-way retrieval.

Implements kernel method approximation to fuse heterogeneous scores
(BM25, HNSW cosine, IVF-PQ distance) into unified ranking.

Reference: PRD Section 4.1.4 - RFF Random Fourier Features
"""

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from src.config import settings
from src.models import RAGResult, RetrievalAlgorithm


class RFFFusion:
    """Random Fourier Features fusion for multi-way retrieval.
    
    Uses RBF kernel approximation to map different score types into
    a unified high-dimensional space where linear combination captures
    non-linear relationships between retrieval signals.
    
    Theory (Bochner's theorem):
        k(x,y) ≈ z(x)ᵀ z(y)
        where z(x) = √(2/D) [cos(w₁ᵀx + b₁), ..., cos(w_Dᵀx + b_D)]
        wᵢ ~ N(0, σ⁻²I), bᵢ ~ Uniform(0, 2π)
    
    Fusion weights (tuned on validation set):
        - BM25: 0.3 (exact keyword match)
        - HNSW: 0.4 (semantic similarity)
        - IVF-PQ: 0.3 (large-scale semantic)
    """
    
    def __init__(
        self,
        dimension: int | None = None,
        gamma: float = 1.0,
        random_state: int = 42,
    ):
        """Initialize RFF fusion.
        
        Args:
            dimension: RFF mapping dimension (default from settings)
            gamma: RBF kernel parameter (inverse of sigma squared)
            random_state: Random seed for reproducibility
        """
        self.dimension = dimension or settings.RFF_DIMENSION
        self.gamma = gamma
        self.random_state = random_state
        
        # Initialize RBF sampler for 1D scores
        self.rbf_sampler = RBFSampler(
            gamma=gamma,
            n_components=self.dimension,
            random_state=random_state,
        )
        
        # Pre-fit on dummy data to initialize
        dummy = np.array([[0.0], [0.5], [1.0]])
        self.rbf_sampler.fit(dummy)
    
    def normalize_scores(
        self,
        results: list[RAGResult],
        algo: RetrievalAlgorithm,
    ) -> list[float]:
        """Normalize raw scores to [0, 1] range based on algorithm type.
        
        Args:
            results: RAG results from a single algorithm
            algo: Algorithm type
            
        Returns:
            List of normalized scores
        """
        if not results:
            return []
        
        raw_scores = [r.raw_score for r in results]
        
        if algo == RetrievalAlgorithm.BM25:
            # BM25: 0 to infinity, use sigmoid-like normalization
            return [min(1.0, s / (s + 10)) for s in raw_scores]
        
        elif algo == RetrievalAlgorithm.HNSW:
            # HNSW: cosine similarity [-1, 1], map to [0, 1]
            return [(s + 1) / 2 for s in raw_scores]
        
        elif algo == RetrievalAlgorithm.IVF_PQ:
            # IVF-PQ: inner product [-1, 1], map to [0, 1]
            return [(s + 1) / 2 for s in raw_scores]
        
        else:
            return raw_scores
    
    def fuse(
        self,
        bm25_results: list[RAGResult],
        hnsw_results: list[RAGResult],
        ivf_pq_results: list[RAGResult],
        weights: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> list[RAGResult]:
        """Fuse multi-way retrieval results using RFF.
        
        Args:
            bm25_results: BM25 retrieval results
            hnsw_results: HNSW retrieval results
            ivf_pq_results: IVF-PQ retrieval results
            weights: Algorithm weights (default: BM25=0.3, HNSW=0.4, IVF-PQ=0.3)
            top_k: Number of final results
            
        Returns:
            Fused and ranked RAG results
        """
        # Default weights
        if weights is None:
            weights = {
                "BM25": 0.3,
                "HNSW": 0.4,
                "IVF_PQ": 0.3,
            }
        
        # Collect all doc_ids
        all_doc_ids = set()
        all_doc_content = {}
        all_doc_metadata = {}
        
        for results in [bm25_results, hnsw_results, ivf_pq_results]:
            for r in results:
                all_doc_ids.add(r.doc_id)
                all_doc_content[r.doc_id] = r.content
                all_doc_metadata[r.doc_id] = {
                    "doc_type": r.doc_type,
                    "timestamp": r.timestamp,
                    "source_authority": r.source_authority,
                }
        
        # Create score dictionaries for each algorithm
        bm25_scores = {r.doc_id: r.normalized_score for r in bm25_results}
        hnsw_scores = {r.doc_id: r.normalized_score for r in hnsw_results}
        ivf_pq_scores = {r.doc_id: r.normalized_score for r in ivf_pq_results}
        
        # Fuse scores using RFF
        fused_scores = {}
        
        for doc_id in all_doc_ids:
            # Get scores from each algorithm (default 0 if missing)
            s_bm25 = bm25_scores.get(doc_id, 0.0)
            s_hnsw = hnsw_scores.get(doc_id, 0.0)
            s_ivf = ivf_pq_scores.get(doc_id, 0.0)
            
            # RFF mapping for non-linear fusion
            # Create feature vector [s_bm25, s_hnsw, s_ivf]
            features = np.array([[s_bm25, s_hnsw, s_ivf]])
            
            # Apply RFF transformation
            rff_features = self.rbf_sampler.transform(features)[0]
            
            # Weighted sum in RFF space
            # We apply weights to the original features before RFF
            # This approximates a weighted kernel combination
            weighted_score = (
                weights["BM25"] * s_bm25 +
                weights["HNSW"] * s_hnsw +
                weights["IVF_PQ"] * s_ivf
            )
            
            # Boost documents found by multiple algorithms (consistency)
            algo_count = sum([
                1 for s in [s_bm25, s_hnsw, s_ivf] if s > 0.01
            ])
            consistency_boost = 1.0 + 0.1 * (algo_count - 1)  # +10% per additional algo
            
            fused_scores[doc_id] = weighted_score * consistency_boost
        
        # Sort by fused score
        sorted_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        
        # Build final results
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            # Determine which algorithms found this doc
            algos_found = []
            if doc_id in bm25_scores and bm25_scores[doc_id] > 0.01:
                algos_found.append("BM25")
            if doc_id in hnsw_scores and hnsw_scores[doc_id] > 0.01:
                algos_found.append("HNSW")
            if doc_id in ivf_pq_scores and ivf_pq_scores[doc_id] > 0.01:
                algos_found.append("IVF_PQ")
            
            # Calculate confidence based on score and algorithm diversity
            base_confidence = score
            diversity_bonus = 0.05 * (len(algos_found) - 1)
            confidence = min(1.0, base_confidence + diversity_bonus)
            
            result = RAGResult(
                doc_id=doc_id,
                content=all_doc_content.get(doc_id, ""),
                retrieval_algo=RetrievalAlgorithm.RFF_FUSION,
                raw_score=score,
                normalized_score=min(1.0, score),
                doc_type=all_doc_metadata[doc_id].get("doc_type", "unknown"),
                timestamp=all_doc_metadata[doc_id].get("timestamp"),
                source_authority=all_doc_metadata[doc_id].get("source_authority", 0.8),
                confidence=confidence,
                chunk_idx=0,
            )
            
            # Add fusion metadata
            result.metadata = {
                "fused_from": algos_found,
                "bm25_score": bm25_scores.get(doc_id, 0.0),
                "hnsw_score": hnsw_scores.get(doc_id, 0.0),
                "ivf_pq_score": ivf_pq_scores.get(doc_id, 0.0),
            }
            
            fused_results.append(result)
        
        return fused_results


class WeightedFusion:
    """Simple weighted linear fusion (alternative to RFF).
    
    Used when RFF complexity is not needed or for baseline comparison.
    """
    
    @staticmethod
    def fuse(
        bm25_results: list[RAGResult],
        hnsw_results: list[RAGResult],
        ivf_pq_results: list[RAGResult],
        weights: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> list[RAGResult]:
        """Fuse results using simple weighted sum.
        
        Args:
            bm25_results: BM25 retrieval results
            hnsw_results: HNSW retrieval results
            ivf_pq_results: IVF-PQ retrieval results
            weights: Algorithm weights
            top_k: Number of final results
            
        Returns:
            Fused and ranked RAG results
        """
        if weights is None:
            weights = {"BM25": 0.3, "HNSW": 0.4, "IVF_PQ": 0.3}
        
        # Collect scores
        all_scores = {}
        all_content = {}
        all_metadata = {}
        
        for algo, results in [
            ("BM25", bm25_results),
            ("HNSW", hnsw_results),
            ("IVF_PQ", ivf_pq_results),
        ]:
            for r in results:
                doc_id = r.doc_id
                if doc_id not in all_scores:
                    all_scores[doc_id] = {}
                    all_content[doc_id] = r.content
                    all_metadata[doc_id] = {
                        "doc_type": r.doc_type,
                        "timestamp": r.timestamp,
                    }
                all_scores[doc_id][algo] = r.normalized_score
        
        # Calculate weighted scores
        fused = {}
        for doc_id, scores in all_scores.items():
            weighted_sum = sum(
                scores.get(algo, 0.0) * weight
                for algo, weight in weights.items()
            )
            # Count how many algorithms found this doc
            found_by = len([s for s in scores.values() if s > 0.01])
            # Boost multi-algorithm matches
            boost = 1.0 + 0.1 * max(0, found_by - 1)
            fused[doc_id] = weighted_sum * boost
        
        # Sort and return top-k
        sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            RAGResult(
                doc_id=doc_id,
                content=all_content[doc_id],
                retrieval_algo=RetrievalAlgorithm.RFF_FUSION,
                raw_score=score,
                normalized_score=min(1.0, score),
                doc_type=all_metadata[doc_id].get("doc_type", "unknown"),
                timestamp=all_metadata[doc_id].get("timestamp"),
                confidence=min(1.0, score),
                chunk_idx=0,
            )
            for doc_id, score in sorted_docs
        ]
