#!/usr/bin/env python3
"""Build RAG indices from synthetic data using Pure Python implementations.

No FAISS dependency - uses NumPy only.
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description="Build RAG indices (Pure Python)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing synthetic data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indices"),
        help="Output directory for indices",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="bm25,hnsw,ivf_pq",
        help="Comma-separated list of algorithms to build",
    )
    
    args = parser.parse_args()
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("=" * 60)
    print("SupplyChainRAG Index Builder (Pure Python)")
    print("=" * 60)
    
    algorithms = args.algorithms.split(",")
    print(f"\n1. 配置信息")
    print(f"   数据目录: {args.data_dir}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   算法: {algorithms}")
    print(f"   实现: Pure Python (no FAISS)")
    
    from src.data.vectorizer import DocumentVectorizer, SupplyChainDocumentBuilder
    from src.rag.retriever import UnifiedRetriever
    from src.config import settings
    
    print(f"   Embedding模型: {settings.EMBEDDING_MODEL}")
    print(f"   Embedding维度: {settings.embedding_dimension}")
    
    # Build documents
    print(f"\n2. 构建文档...")
    builder = SupplyChainDocumentBuilder(args.data_dir)
    doc_ids, texts, metadata = builder.build_all_documents()
    print(f"   生成了 {len(texts)} 个文档")
    
    # Save documents
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_dir / "documents.pkl", "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "texts": texts, "metadata": metadata}, f)
    print(f"   文档已保存到 {args.output_dir / 'documents.pkl'}")
    
    # Generate embeddings
    print(f"\n3. 生成Embedding向量...")
    print(f"   这将下载模型（如果尚未缓存）...")
    vectorizer = DocumentVectorizer()
    
    try:
        embeddings = vectorizer.encode(texts, batch_size=32)
        np.save(args.output_dir / "embeddings.npy", embeddings)
        print(f"   向量已保存，形状: {embeddings.shape}")
        print(f"   实际维度: {vectorizer.dimension}")
    except Exception as e:
        print(f"   ⚠️  使用简单向量器 (离线模式)")
        from src.data.simple_vectorizer import SimpleVectorizer
        vectorizer = SimpleVectorizer(settings.embedding_dimension)
        embeddings = vectorizer.encode(texts, batch_size=32)
        np.save(args.output_dir / "embeddings.npy", embeddings)
        print(f"   向量已生成，形状: {embeddings.shape}")
    
    # Build indices
    print(f"\n4. 构建检索索引 (Pure Python)...")
    print(f"   - BM25: 稀疏文本检索")
    print(f"   - HNSW: 图索引 (Python实现)")
    print(f"   - IVF-PQ: 量化索引 (Python实现)")
    
    retriever = UnifiedRetriever(
        use_bm25="bm25" in algorithms,
        use_hnsw="hnsw" in algorithms,
        use_ivf_pq="ivf_pq" in algorithms,
    )
    
    try:
        retriever.build_indices(doc_ids, texts, embeddings, metadata)
        retriever.save_indices(args.output_dir)
        
        print(f"\n" + "=" * 60)
        print("✅ 索引构建成功！")
        print(f"   索引目录: {args.output_dir}")
        print(f"   实现方式: Pure Python (NumPy only, no FAISS)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
