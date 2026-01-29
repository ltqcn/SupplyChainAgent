#!/usr/bin/env python3
"""Quick test script for index building."""

import warnings
import sys
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).parent))

print("Testing index building...")
print("-" * 40)

# Test imports
try:
    from src.config import settings
    print(f"✓ Config loaded")
    print(f"  Model: {settings.EMBEDDING_MODEL}")
    print(f"  Dimension: {settings.embedding_dimension}")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

# Test vectorizer
try:
    from src.data.vectorizer import DocumentVectorizer
    vectorizer = DocumentVectorizer()
    print(f"✓ Vectorizer initialized")
    print(f"  Dimension: {vectorizer.dimension}")
except Exception as e:
    print(f"✗ Vectorizer error: {e}")
    sys.exit(1)

# Test retrievers
try:
    from src.rag.bm25_retriever import BM25Retriever
    from src.rag.hnsw_retriever import HNSWRetriever
    from src.rag.ivf_pq_retriever import IVFPQRetriever
    
    bm25 = BM25Retriever()
    hnsw = HNSWRetriever()
    ivf = IVFPQRetriever()
    
    print(f"✓ Retrievers initialized")
    print(f"  HNSW dim: {hnsw.dimension}")
    print(f"  IVF-PQ dim: {ivf.dimension}")
except Exception as e:
    print(f"✗ Retriever error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 40)
print("✅ All components loaded successfully!")
print("You can now run: ./entry.sh <api_key>")
