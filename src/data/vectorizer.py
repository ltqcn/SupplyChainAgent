"""Text vectorization module using sentence-transformers.

Generates embeddings for supply chain documents.
Supports multiple lightweight models for faster download.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


# Set HuggingFace mirror for China mainland
def setup_hf_mirror():
    """Setup HuggingFace mirror endpoint for faster download in China."""
    # Apply from settings
    if settings.HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
    
    if settings.HF_OFFLINE:
        os.environ["HF_OFFLINE"] = "1"
    
    # Always set trust remote code for compatibility
    os.environ["HF_TRUST_REMOTE_CODE"] = "1"
    
    # Reduce timeout for faster fail
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "10"


# Apply on module load
setup_hf_mirror()


# Model configurations: name -> (dimension, description)
SUPPORTED_MODELS = {
    # 推荐：轻量级多语言模型，下载快（约42MB）
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (384, "多语言轻量模型"),
    "paraphrase-multilingual-MiniLM-L12-v2": (384, "多语言轻量模型"),
    
    # 更轻量选项（约22MB）
    "sentence-transformers/all-MiniLM-L6-v2": (384, "英文轻量模型"),
    "all-MiniLM-L6-v2": (384, "英文轻量模型"),
    
    # 中文轻量模型（约90MB）
    "BAAI/bge-small-zh": (512, "中文轻量模型"),
    "BAAI/bge-small-zh-v1.5": (512, "中文轻量模型v1.5"),
    
    # 原配置（需要翻墙下载）
    "BAAI/bge-large-zh": (1024, "中文大模型"),
    "BAAI/bge-large-zh-v1.5": (1024, "中文大模型v1.5"),
    
    # 其他可选
    "sentence-transformers/distiluse-base-multilingual-cased-v2": (512, "Distil多语言"),
}


def get_model_dimension(model_name: str) -> int:
    """Get embedding dimension for a model.
    
    Args:
        model_name: Model name or path
        
    Returns:
        Embedding dimension
    """
    # Check known models
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name][0]
    
    # Try to infer from model name
    if "small" in model_name.lower() or "mini" in model_name.lower():
        return 384
    if "large" in model_name.lower():
        return 1024
    
    # Default
    return 384


class DocumentVectorizer:
    """Vectorizes supply chain documents for RAG retrieval.
    
    Supports multiple embedding models with automatic dimension detection.
    Default: paraphrase-multilingual-MiniLM-L12-v2 (fast download, good for Chinese)
    """
    
    def __init__(self, model_name: str | None = None):
        """Initialize vectorizer with embedding model.
        
        Args:
            model_name: HuggingFace model name, defaults to settings.EMBEDDING_MODEL
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model: SentenceTransformer | None = None
        self.dimension: int = get_model_dimension(self.model_name)
    
    def _load_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            print(f"  Dimension: {self.dimension}")
            
            # Check if it's a known fast model
            if self.model_name in SUPPORTED_MODELS:
                desc = SUPPORTED_MODELS[self.model_name][1]
                print(f"  Type: {desc}")
            
            # Check cache first
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_in_cache = any(self.model_name.split('/')[-1] in d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)))
            
            if not model_in_cache:
                print(f"  ⚠️  Model not in cache, will download (~42MB)...")
                print(f"  Using mirror: {os.environ.get('HF_ENDPOINT', 'default')}")
            
            try:
                # Set shorter timeout for download
                import urllib.request
                original_timeout = urllib.request.socket.getdefaulttimeout()
                urllib.request.socket.setdefaulttimeout(60)
                
                self.model = SentenceTransformer(
                    self.model_name,
                    device='cpu',
                    cache_folder=os.environ.get('HF_HOME', None)
                )
                
                # Restore timeout
                urllib.request.socket.setdefaulttimeout(original_timeout)
                
                # Update dimension from actual model
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"  ✓ Loaded successfully, dimension: {self.dimension}")
                
            except Exception as e:
                print(f"  ✗ Error loading model: {e}")
                print("  Falling back to offline simple vectorizer...")
                raise
        
        return self.model
    
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Processing batch size
            
        Returns:
            Array of shape (len(texts), dimension) containing embeddings
        """
        try:
            model = self._load_model()
            
            # Normalize embeddings for cosine similarity
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50,  # Only show for large batches
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            
            return embeddings
            
        except Exception as e:
            print(f"\nModel encode failed: {e}")
            print("Switching to offline simple vectorizer...")
            
            # Fallback to simple vectorizer
            from src.data.simple_vectorizer import SimpleVectorizer
            simple = SimpleVectorizer(self.dimension)
            return simple.encode(texts, batch_size)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text.
        
        Args:
            text: Text string to encode
            
        Returns:
            Embedding vector
        """
        return self.encode([text])[0]


class SupplyChainDocumentBuilder:
    """Builds searchable documents from supply chain entities.
    
    Converts structured data (batches, orders, etc.) into text documents
    suitable for embedding and retrieval.
    """
    
    def __init__(self, data_dir: Path):
        """Initialize with data directory.
        
        Args:
            data_dir: Directory containing synthetic data JSON files
        """
        self.data_dir = Path(data_dir)
    
    def _load_json(self, filename: str) -> list[dict]:
        """Load JSON data file."""
        file_path = self.data_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def build_batch_documents(self) -> Iterator[tuple[str, str, dict]]:
        """Build text documents from production batches.
        
        Yields:
            Tuples of (doc_id, text_content, metadata)
        """
        batches = self._load_json("batches.json")
        skus = {s["sku_id"]: s for s in self._load_json("skus.json")}
        factories = {f["factory_id"]: f for f in self._load_json("factories.json")}
        
        for batch in batches:
            sku = skus.get(batch["sku_id"], {})
            factory = factories.get(batch["factory_id"], {})
            
            # Build natural language description
            text = f"""批次{batch['batch_id']}生产信息：
产品：{sku.get('name', batch['sku_id'])}（{sku.get('category', '未知分类')}）
生产工厂：{factory.get('name', batch['factory_id'])}，位于{factory.get('location', '未知地点')}
计划数量：{batch['planned_qty']}件，实际完成：{batch['actual_qty']}件
当前状态：{batch['status']}
生产进度：裁切{batch['cutting_progress']}%，印刷{batch['printing_progress']}%，组装{batch['assembly_progress']}%，包装{batch['packaging_progress']}%
质检合格率：{batch['fpy_rate']*100:.1f}%，不良品：{batch['defect_count']}件
计划交期：{batch['planned_date']}
"""
            
            metadata = {
                "doc_type": "production_report",
                "batch_id": batch["batch_id"],
                "sku_id": batch["sku_id"],
                "factory_id": batch["factory_id"],
                "status": batch["status"],
                "planned_date": batch["planned_date"],
            }
            
            yield batch["batch_id"], text.strip(), metadata
    
    def build_inventory_documents(self) -> Iterator[tuple[str, str, dict]]:
        """Build text documents from inventory records.
        
        Yields:
            Tuples of (doc_id, text_content, metadata)
        """
        inventory = self._load_json("inventory.json")
        skus = {s["sku_id"]: s for s in self._load_json("skus.json")}
        warehouses = {w["warehouse_id"]: w for w in self._load_json("warehouses.json")}
        
        for inv in inventory:
            sku = skus.get(inv["sku_id"], {})
            warehouse = warehouses.get(inv["warehouse_id"], {})
            
            text = f"""库存记录{inv['record_id']}：
仓库：{warehouse.get('name', inv['warehouse_id'])}（{warehouse.get('location', '未知地点')}）
产品：{sku.get('name', inv['sku_id'])}，分类：{sku.get('category', '未知')}
可用库存：{inv['available_qty']}件
预留库存：{inv['reserved_qty']}件
锁定库存：{inv['locked_qty']}件
总库存：{inv['available_qty'] + inv['reserved_qty'] + inv['locked_qty']}件
库位：{inv['location_code']}
批次：{inv.get('batch_id', '未指定')}
"""
            
            metadata = {
                "doc_type": "inventory_record",
                "record_id": inv["record_id"],
                "warehouse_id": inv["warehouse_id"],
                "sku_id": inv["sku_id"],
                "batch_id": inv.get("batch_id"),
                "available_qty": inv["available_qty"],
            }
            
            yield inv["record_id"], text.strip(), metadata
    
    def build_waybill_documents(self) -> Iterator[tuple[str, str, dict]]:
        """Build text documents from waybill records.
        
        Yields:
            Tuples of (doc_id, text_content, metadata)
        """
        waybills = self._load_json("waybills.json")
        
        for wb in waybills:
            # Format routing nodes
            routing_text = "\n".join([
                f"  {node['timestamp']}: {node['location']} - {node['event']}"
                for node in wb.get("routing_nodes", [])
            ])
            
            text = f"""物流单{wb['waybill_id']}：
承运商：{wb['carrier']}
服务类型：{wb['service_type']}
发货地：{wb['ship_from']}
收货地：{wb['ship_to']}
重量：{wb['weight_kg']:.2f}kg，体积：{wb['volume_cbm']:.4f}m³
当前状态：{wb['status']}
创建时间：{wb['created_at']}
物流轨迹：
{routing_text}
"""
            
            metadata = {
                "doc_type": "logistics_record",
                "waybill_id": wb["waybill_id"],
                "carrier": wb["carrier"],
                "status": wb["status"],
                "ship_to": wb["ship_to"],
                "created_at": wb["created_at"],
            }
            
            yield wb["waybill_id"], text.strip(), metadata
    
    def build_supplier_documents(self) -> Iterator[tuple[str, str, dict]]:
        """Build text documents from supplier records.
        
        Yields:
            Tuples of (doc_id, text_content, metadata)
        """
        suppliers = self._load_json("suppliers.json")
        
        for sup in suppliers:
            text = f"""供应商{sup['supplier_id']}：{sup['name']}
位置：{sup['location']}
评级：{sup['rating']}级供应商
质量评分：{sup['quality_score']}分
交付评分：{sup['delivery_score']}分
成本评分：{sup['cost_score']}分
服务评分：{sup['service_score']}分
ISO9001认证：{'是' if sup['iso9001'] else '否'}
联系方式：{sup['contact_email']}
"""
            
            metadata = {
                "doc_type": "supplier_profile",
                "supplier_id": sup["supplier_id"],
                "rating": sup["rating"],
                "location": sup["location"],
            }
            
            yield sup["supplier_id"], text.strip(), metadata
    
    def build_all_documents(self) -> tuple[list[str], list[str], list[dict]]:
        """Build all documents from supply chain data.
        
        Returns:
            Tuple of (doc_ids, texts, metadata_list)
        """
        doc_ids = []
        texts = []
        metadata_list = []
        
        for builder_method in [
            self.build_batch_documents,
            self.build_inventory_documents,
            self.build_waybill_documents,
            self.build_supplier_documents,
        ]:
            for doc_id, text, metadata in builder_method():
                doc_ids.append(doc_id)
                texts.append(text)
                metadata_list.append(metadata)
        
        return doc_ids, texts, metadata_list
    
    def save_documents(self, output_path: Path) -> None:
        """Save built documents to pickle file.
        
        Args:
            output_path: Path to save documents pickle
        """
        doc_ids, texts, metadata_list = self.build_all_documents()
        
        data = {
            "doc_ids": doc_ids,
            "texts": texts,
            "metadata": metadata_list,
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(texts)} documents to {output_path}")


def main():
    """CLI entry point for document vectorization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorize supply chain documents")
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
        help="Output directory for vector files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name (overrides settings)",
    )
    
    args = parser.parse_args()
    
    # Build documents
    print("Building documents from synthetic data...")
    builder = SupplyChainDocumentBuilder(args.data_dir)
    doc_ids, texts, metadata = builder.build_all_documents()
    
    # Save raw documents
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder.save_documents(args.output_dir / "documents.pkl")
    
    # Generate embeddings
    print(f"\nGenerating embeddings for {len(texts)} documents...")
    vectorizer = DocumentVectorizer(model_name=args.model)
    embeddings = vectorizer.encode(texts, batch_size=32)
    
    # Save embeddings
    np.save(args.output_dir / "embeddings.npy", embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {args.output_dir}/embeddings.npy")
    
    # Print model info
    print(f"\nModel: {vectorizer.model_name}")
    print(f"Dimension: {vectorizer.dimension}")


if __name__ == "__main__":
    main()
