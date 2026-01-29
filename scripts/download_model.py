#!/usr/bin/env python3
"""Download embedding model with HuggingFace mirror support.

Usage:
    python scripts/download_model.py [model_name]

Examples:
    python scripts/download_model.py
    python scripts/download_model.py sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set mirror before importing transformers
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from sentence_transformers import SentenceTransformer

from src.config import settings


def download_model(model_name: str | None = None):
    """Download and cache embedding model.
    
    Args:
        model_name: Model name to download, defaults to settings.EMBEDDING_MODEL
    """
    model_name = model_name or settings.EMBEDDING_MODEL
    
    print(f"=" * 60)
    print(f"Downloading embedding model")
    print(f"=" * 60)
    print(f"Model: {model_name}")
    print(f"Mirror: {os.environ.get('HF_ENDPOINT', 'default')}")
    print(f"")
    
    try:
        # Download model
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()
        
        print(f"")
        print(f"✅ Model downloaded successfully!")
        print(f"   Dimension: {dimension}")
        print(f"   Cache: ~/.cache/huggingface/hub/")
        print(f"")
        
        # Test encoding
        print(f"Testing model...")
        test_embedding = model.encode("测试中文编码")
        print(f"✅ Test passed, embedding shape: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"")
        print(f"❌ Error downloading model: {e}")
        print(f"")
        print(f"Troubleshooting:")
        print(f"1. Check network connection")
        print(f"2. Verify HF_ENDPOINT is set correctly")
        print(f"3. Try manual download from: https://hf-mirror.com")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download embedding model")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model name to download (default: from settings)",
    )
    
    args = parser.parse_args()
    
    success = download_model(args.model)
    sys.exit(0 if success else 1)
