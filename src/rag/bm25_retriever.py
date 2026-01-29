"""BM25 sparse retriever implementation.

Implements Best Match 25 algorithm for keyword-based retrieval.
Optimized for supply chain entities (SKU codes, batch numbers, etc.).

Reference: PRD Section 4.1.1 - BM25 Sparse Retrieval
"""

import pickle
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.models import RAGResult, RetrievalAlgorithm


class BM25Retriever:
    """BM25 retriever for keyword-based document search.
    
    Configuration:
    - k1=1.5: Term frequency saturation (prevents long document bias)
    - b=0.75: Length normalization factor
    - Uses Jieba for Chinese tokenization with supply chain domain terms
    
    Performance: <20ms latency for 10,000 documents, <50MB memory
    """
    
    # Supply chain domain terms for tokenization
    DOMAIN_TERMS = {
        "应援棒", "亚克力", "灯牌", "手幅", "立牌", "挂件", "演唱会",
        "物料", "库存", "批次", "质检", "供应商", "物流", "仓库",
        "生产", "发货", "入库", "出库", "采购", "订单", "交期",
    }
    
    def __init__(
        self,
        k1: float | None = None,
        b: float | None = None,
    ):
        """Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (default from settings)
            b: Length normalization parameter (default from settings)
        """
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B
        
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_metadata: list[dict[str, Any]] = []
        
        # Add domain terms to Jieba dictionary
        for term in self.DOMAIN_TERMS:
            jieba.add_word(term, freq=1000)
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Chinese text using Jieba.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Use Jieba cut for Chinese text
        tokens = list(jieba.cut(text.lower()))
        # Filter out very short tokens and punctuation
        tokens = [t for t in tokens if len(t.strip()) > 1 or t.isalnum()]
        return tokens
    
    def build_index(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build BM25 index from documents.
        
        Args:
            doc_ids: Document identifiers
            doc_texts: Document text content
            doc_metadata: Optional document metadata
        """
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.doc_metadata = doc_metadata or [{} for _ in doc_ids]
        
        # Tokenize all documents
        print(f"Tokenizing {len(doc_texts)} documents for BM25...")
        tokenized_docs = [self._tokenize(text) for text in doc_texts]
        
        # Build BM25 index
        print(f"Building BM25 index (k1={self.k1}, b={self.b})...")
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        print(f"BM25 index built: {len(doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 10) -> list[RAGResult]:
        """Search documents using BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of RAGResult with BM25 scores
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            
            # Normalize score to [0, 1] using sigmoid-like transformation
            # BM25 scores can be 0-infinity, we use arctan for normalization
            normalized_score = min(1.0, score / (score + 10))
            
            result = RAGResult(
                doc_id=self.doc_ids[idx],
                content=self.doc_texts[idx],
                retrieval_algo=RetrievalAlgorithm.BM25,
                raw_score=score,
                normalized_score=normalized_score,
                doc_type=self.doc_metadata[idx].get("doc_type", "unknown"),
                timestamp=self.doc_metadata[idx].get("timestamp"),
                source_authority=self.doc_metadata[idx].get("source_authority", 0.8),
                confidence=normalized_score * 0.9,  # Slight penalty for keyword-only
                chunk_idx=0,
            )
            results.append(result)
        
        return results
    
    def save_index(self, filepath: Path | str) -> None:
        """Save BM25 index to disk.
        
        Args:
            filepath: Path to save index
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "k1": self.k1,
            "b": self.b,
            "bm25": self.bm25,
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "doc_metadata": self.doc_metadata,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"BM25 index saved to {filepath}")
    
    def load_index(self, filepath: Path | str) -> None:
        """Load BM25 index from disk.
        
        Args:
            filepath: Path to load index from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"BM25 index not found: {filepath}")
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.k1 = data["k1"]
        self.b = data["b"]
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        self.doc_metadata = data["doc_metadata"]
        
        print(f"BM25 index loaded: {len(self.doc_ids)} documents from {filepath}")
