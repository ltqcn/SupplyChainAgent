"""Simple retriever using only BM25 (no FAISS required).

For systems where FAISS causes segmentation faults (e.g., macOS ARM).
"""

import warnings
from pathlib import Path
from typing import Any

from src.config import settings
from src.models import RAGResult, RetrievalAlgorithm
from src.rag.bm25_retriever import BM25Retriever
from src.data.simple_vectorizer import SimpleVectorizer

warnings.filterwarnings("ignore")


class SimpleRetriever:
    """Simple retriever using BM25 only.
    
    This is a fallback for systems where FAISS causes issues.
    It only uses BM25 for text retrieval and simple hash-based embeddings
    for compatibility with the rest of the system.
    """
    
    def __init__(self):
        """Initialize simple retriever."""
        self.bm25_retriever = BM25Retriever()
        self.vectorizer = SimpleVectorizer(settings.embedding_dimension)
        self.doc_texts: dict[str, str] = {}
        self.doc_metadata: dict[str, dict[str, Any]] = {}
    
    def build_indices(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        embeddings: Any,  # Ignored, for API compatibility
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build BM25 index.
        
        Args:
            doc_ids: Document identifiers
            doc_texts: Document text content
            embeddings: Ignored (for API compatibility)
            doc_metadata: Optional document metadata
        """
        print("Building SimpleRetriever index (BM25 only)...")
        
        self.doc_texts = {doc_id: text for doc_id, text in zip(doc_ids, doc_texts)}
        self.doc_metadata = {doc_id: meta for doc_id, meta in zip(doc_ids, doc_metadata or [{} for _ in doc_ids])}
        
        # Build BM25 index
        self.bm25_retriever.build_index(doc_ids, doc_texts, doc_metadata)
        
        print(f"SimpleRetriever index built: {len(doc_ids)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_type: str = "hybrid",
    ) -> tuple[list[RAGResult], Any]:
        """Retrieve documents using BM25 only.
        
        Args:
            query: User query string
            top_k: Number of results to return
            query_type: Ignored (for API compatibility)
            
        Returns:
            Tuple of (results list, None)
        """
        results = self.bm25_retriever.search(query, top_k=top_k)
        
        # Fill content from doc_texts
        for r in results:
            if not r.content and r.doc_id in self.doc_texts:
                r.content = self.doc_texts[r.doc_id]
        
        # Create a simple log object
        log = {
            "query": query,
            "algorithm": "BM25_only",
            "results_count": len(results),
        }
        
        return results, log
    
    def save_indices(self, output_dir: Path | str) -> None:
        """Save index to disk.
        
        Args:
            output_dir: Directory to save index
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save BM25
        self.bm25_retriever.save_index(output_dir / "bm25.pkl")
        
        print(f"SimpleRetriever index saved to {output_dir}")
    
    def load_indices(self, index_dir: Path | str) -> None:
        """Load index from disk.
        
        Args:
            index_dir: Directory containing saved index
        """
        index_dir = Path(index_dir)
        
        # Load BM25
        bm25_path = index_dir / "bm25.pkl"
        if bm25_path.exists():
            self.bm25_retriever.load_index(bm25_path)
            # Rebuild doc_texts from retriever
            self.doc_texts = {doc_id: text for doc_id, text in zip(self.bm25_retriever.doc_ids, self.bm25_retriever.doc_texts)}
            self.doc_metadata = {doc_id: meta for doc_id, meta in zip(self.bm25_retriever.doc_ids, self.bm25_retriever.doc_metadata)}
        
        print(f"SimpleRetriever index loaded from {index_dir}")
