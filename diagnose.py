#!/usr/bin/env python3
"""Diagnose script to identify where the process hangs."""

import sys
import warnings
import traceback

warnings.filterwarnings("ignore")

def log_step(step_name):
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print('='*60)
    sys.stdout.flush()

try:
    log_step("1. Import config")
    from src.config import settings
    print(f"✓ Config loaded: embedding_dim={settings.embedding_dimension}")
    
    log_step("2. Import data generator")
    from src.data.synthetic_generator import SupplyChainDataGenerator
    print("✓ Data generator imported")
    
    log_step("3. Generate small dataset")
    generator = SupplyChainDataGenerator(seed=42)
    data = generator.generate_all(scale='small')
    print(f"✓ Generated: {len(data['batches'])} batches, {len(data['skus'])} SKUs")
    
    log_step("4. Save to JSON")
    generator.save_to_json('data')
    print("✓ Data saved")
    
    log_step("5. Import database")
    from src.data.database import db_manager
    print("✓ Database module imported")
    
    log_step("6. Create tables")
    db_manager.create_tables()
    print("✓ Tables created")
    
    log_step("7. Load data to database")
    db_manager.load_from_synthetic_data('data')
    print("✓ Data loaded to database")
    
    log_step("8. Import vectorizer")
    from src.data.vectorizer import DocumentVectorizer
    print("✓ Vectorizer imported")
    
    log_step("9. Build documents")
    from src.data.vectorizer import SupplyChainDocumentBuilder
    from pathlib import Path
    builder = SupplyChainDocumentBuilder(Path('data'))
    doc_ids, texts, metadata = builder.build_all_documents()
    print(f"✓ Built {len(texts)} documents")
    
    log_step("10. Initialize vectorizer (may download model)")
    vectorizer = DocumentVectorizer()
    print(f"✓ Vectorizer ready: dim={vectorizer.dimension}")
    
    log_step("11. Encode documents")
    import numpy as np
    embeddings = vectorizer.encode(texts[:10], batch_size=10)  # Test with small batch first
    print(f"✓ Encoded {len(embeddings)} vectors, shape={embeddings.shape}")
    
    log_step("12. Import retrievers")
    from src.rag.bm25_retriever import BM25Retriever
    from src.rag.hnsw_retriever import HNSWRetriever
    from src.rag.ivf_pq_retriever import IVFPQRetriever
    print("✓ Retrievers imported")
    
    log_step("13. Test BM25 index")
    bm25 = BM25Retriever()
    bm25.build_index(doc_ids[:10], texts[:10], metadata[:10])
    print(f"✓ BM25 index built: {len(bm25.doc_ids)} docs")
    
    log_step("14. Test HNSW index")
    hnsw = HNSWRetriever(dimension=vectorizer.dimension)
    hnsw.build_index(embeddings, doc_ids[:10], metadata[:10])
    print(f"✓ HNSW index built: {hnsw.index.ntotal} vectors")
    
    log_step("15. Test IVF-PQ index (this may take time)")
    ivf = IVFPQRetriever(dimension=vectorizer.dimension)
    # Need at least 39*nlist vectors for training, use small nlist for test
    ivf.nlist = 1  # Override for small dataset
    ivf.build_index(embeddings, doc_ids[:10], metadata[:10])
    print(f"✓ IVF-PQ index built: {ivf.index.ntotal} vectors")
    
    log_step("SUCCESS")
    print("\n✅ All steps completed successfully!")
    print("You can now run: ./start_simple.sh <api_key>")
    
except Exception as e:
    print(f"\n❌ ERROR at step: {e}")
    traceback.print_exc()
    sys.exit(1)
