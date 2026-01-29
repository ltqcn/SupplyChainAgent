"""Configuration management for SupplyChainRAG.

All configuration is centralized here using Pydantic Settings for type safety
and environment variable injection.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Project paths
    PROJECT_ROOT: Path = Field(default=Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default=Path("data"))
    LOGS_DIR: Path = Field(default=Path("logs"))
    
    # Kimi API Configuration
    KIMI_API_KEY: str = Field(default="", description="Kimi API Key")
    KIMI_BASE_URL: str = Field(default="https://api.moonshot.cn/v1")
    KIMI_MODEL: str = Field(default="kimi-v1-128k")
    
    # Application Mode
    APP_ENV: Literal["development", "production", "testing"] = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    MODE: Literal["round", "query"] = Field(default="query")
    MAX_ROUNDS: int = Field(default=0, description="Max rounds for round mode")
    
    # Context Budget Management
    MAX_CONTEXT_TOKENS: int = Field(default=8000, description="Soft limit for context tokens")
    SAFETY_CONTEXT_TOKENS: int = Field(default=12000, description="Hard limit for context tokens")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///data/supply_chain.db")
    
    # Vector Index Paths
    BM25_INDEX_PATH: Path = Field(default=Path("data/indices/bm25.pkl"))
    HNSW_INDEX_PATH: Path = Field(default=Path("data/indices/hnsw.faiss"))
    IVF_PQ_INDEX_PATH: Path = Field(default=Path("data/indices/ivf_pq.faiss"))
    
    # Memory Configuration
    SHORT_TERM_MAX_TOKENS: int = Field(default=3200)
    LONG_TERM_SUMMARY_THRESHOLD: int = Field(default=20)
    DISK_OFFLOAD_THRESHOLD: int = Field(default=6000)
    
    # Docker Sandbox
    DOCKER_SANDBOX_IMAGE: str = Field(default="supplychainrag-sandbox")
    DOCKER_SANDBOX_CPU: float = Field(default=1.0)
    DOCKER_SANDBOX_MEMORY: str = Field(default="512m")
    DOCKER_SANDBOX_TIMEOUT: int = Field(default=30)
    
    # HuggingFace Configuration
    HF_ENDPOINT: str = Field(default="https://hf-mirror.com")
    HF_OFFLINE: bool = Field(default=False)
    
    # RAG Configuration
    # 推荐轻量级模型（下载快，适合中文）:
    # - paraphrase-multilingual-MiniLM-L12-v2: 42MB, 384维, 多语言
    # - BAAI/bge-small-zh: 90MB, 512维, 中文优化
    # - BAAI/bge-large-zh: 1.3GB, 1024维, 中文大模型（需翻墙）
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)  # 根据模型自动调整
    RFF_DIMENSION: int = Field(default=256)
    
    # BM25 Parameters
    BM25_K1: float = Field(default=1.5, description="BM25 term frequency saturation parameter")
    BM25_B: float = Field(default=0.75, description="BM25 length normalization parameter")
    
    # HNSW Parameters
    HNSW_M: int = Field(default=16, description="HNSW connections per layer")
    HNSW_EF_CONSTRUCTION: int = Field(default=200, description="HNSW construction search depth")
    HNSW_EF_SEARCH: int = Field(default=64, description="HNSW query search depth")
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension based on model."""
        model_dims = {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "all-MiniLM-L6-v2": 384,
            "BAAI/bge-small-zh": 512,
            "BAAI/bge-small-zh-v1.5": 512,
            "BAAI/bge-large-zh": 1024,
            "BAAI/bge-large-zh-v1.5": 1024,
        }
        return model_dims.get(self.EMBEDDING_MODEL, self.EMBEDDING_DIMENSION)
    
    # IVF-PQ Parameters
    IVF_PQ_NLIST: int = Field(default=4096, description="IVF number of clusters")
    IVF_PQ_M: int = Field(default=16, description="PQ number of subquantizers")
    IVF_PQ_NBITS: int = Field(default=8, description="PQ bits per code")
    IVF_PQ_NPROBE: int = Field(default=16, description="IVF clusters to probe during search")
    
    @property
    def is_round_mode(self) -> bool:
        """Check if running in round demonstration mode."""
        return self.MODE == "round" and self.MAX_ROUNDS > 0


# Global settings instance
settings = Settings()
